"""The measured generation loop shared by the energy and thread-scaling experiments.

One measurement = one prompt processed in two separately measured windows:

  prefill  the prompt tokens are evaluated in one batch (prompt processing)
  decode   exactly `n_predict` tokens are generated greedily, one forward pass each

Tokens/s and the headline energy metric (cpu_J_per_token) use the decode window only.
Prefill time and energy are stored in their own columns, and "*_incl_prefill" columns
reproduce the Phase 1 definition for comparison.

Interleaved design: the campaign measures every configuration once per round, in a shuffled
order. One *round unit* = load the model, run `warmup` warm-up generations with the first prompt
(discarded), measure every prompt once, unload. Model loading happens outside every measurement window; its duration is
stored in `model_load_time_s`. Summaries over the rounds of a session are computed when the
session's experiment is complete (`session_summary_rows`).
"""
import hashlib
import math
import time
from collections import defaultdict

import llama_cpp
import numpy as np
from llama_cpp import Llama

from llm_cpu_inference.models import MODELS, model_path, quantization_of
from llm_cpu_inference.energy import NAN, gco2eq
from llm_cpu_inference.log import get_logger
from llm_cpu_inference.results_io import append_rows, defer_interrupt
from llm_cpu_inference.run_metadata import now_iso
from llm_cpu_inference.stats import summarize
from llm_cpu_inference.sysmon import FrequencySampler, flag_throttling

ITERATION_FIELDS = [
    "session_id", "run_id", "timestamp", "experiment", "round", "order_in_round",
    "model_key", "model_file", "method", "params_b", "quantization",
    "n_threads", "n_ctx", "seed", "prompt_id", "iteration",
    "prompt_tokens", "generated_tokens", "eos_suppressed", "output_sha256",
    "model_load_time_s", "prefill_time_s", "decode_time_s",
    "tokens_per_sec", "prefill_tokens_per_sec", "tokens_per_sec_incl_prefill",
    "cpu_J", "ram_J_est", "gpu_J", "total_J", "prefill_cpu_J",
    "idle_cpu_W", "cpu_J_net",
    "cpu_J_per_token", "cpu_J_per_token_net", "cpu_J_per_token_incl_prefill", "avg_cpu_power_W",
    "freq_mean_mhz", "freq_min_mhz", "throttle_flag",
    "temp_mean_C", "temp_max_C", "temp_at_start_C", "cooldown_wait_s", "cooldown_reached",
    "energy_source", "grid_intensity_gco2_per_kwh", "gco2eq_per_token",
]

# Metrics aggregated into mean / std / 95 % CI in the summary file
SUMMARY_METRICS = [
    "tokens_per_sec", "tokens_per_sec_incl_prefill", "prefill_time_s",
    "cpu_J_per_token", "cpu_J_per_token_net", "cpu_J_per_token_incl_prefill",
    "avg_cpu_power_W", "gco2eq_per_token", "temp_mean_C",
]
SUMMARY_FIELDS = [
    "session_id", "timestamp", "experiment", "model_key", "model_file", "method", "params_b", "quantization",
    "n_threads", "n_ctx", "prompt_id", "n_predict", "n_rounds", "n",
    "model_load_time_s_mean", "idle_cpu_W_mean", "temp_max_C", "n_throttled", "deterministic",
] + [f"{metric}_{stat}" for metric in SUMMARY_METRICS for stat in ("mean", "std", "ci95_low", "ci95_high")]

ALL_PROMPTS = "ALL"


def load_model(model_key, n_threads, inference, logits_all=False, n_batch=None):
    """Load a model fully into RAM on the CPU and return it with the load time."""
    kwargs = {"n_batch": n_batch} if n_batch else {}
    start = time.perf_counter()
    llm = Llama(
        model_path=model_path(model_key),
        n_ctx=inference["n_ctx"],
        n_threads=n_threads,
        n_threads_batch=n_threads,  # Default would be the logical core count; keep it equal to n_threads
        n_gpu_layers=0,  # CPU-only: no layer is ever offloaded
        seed=inference["seed"],
        use_mmap=inference["use_mmap"],  # False: weights are read into RAM, no page faults while measuring
        logits_all=logits_all,
        verbose=False,
        **kwargs,
    )
    return llm, time.perf_counter() - start


def end_of_generation_ids(llm):
    """Token ids that would stop generation (EOS, EOT, ...); they are masked for a fixed output length."""
    try:
        vocab = llama_cpp.llama_model_get_vocab(llm._model.model)
        ids = [t for t in range(llm.n_vocab()) if llama_cpp.llama_vocab_is_eog(vocab, t)]
    except AttributeError as exc:
        get_logger().warning("End-of-generation tokens could not be enumerated (%s); only EOS is masked.", exc)
        ids = [llm.token_eos()]
    return np.array(sorted(set(ids)), dtype=np.int64)


def _last_logits(llm, n_vocab):
    return np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1), shape=(n_vocab,))


def greedy_decode(llm, n_predict, eog_ids, thermal=None):
    """Generate exactly n_predict tokens greedily; returns (tokens, number of suppressed EOG tokens)."""
    n_vocab = llm.n_vocab()
    eog_set = set(int(t) for t in eog_ids)
    generated = []
    suppressed = 0
    for _ in range(n_predict):
        if thermal is not None:
            thermal.check()  # Thermal fuse: stop within one token
        logits = _last_logits(llm, n_vocab)
        token = int(logits.argmax())  # Greedy decoding (temperature 0)
        if token in eog_set:
            # The model wants to stop; force the best non-terminal token so that every
            # model and every run performs exactly n_predict forward passes.
            suppressed += 1
            masked = logits.copy()
            masked[eog_ids] = -np.inf
            token = int(masked.argmax())
        generated.append(token)
        llm.eval([token])
    return generated, suppressed


def measure_once(llm, session, prompt_tokens, n_predict, eog_ids):
    """Run prefill + fixed-length greedy decode once and return the raw measurements."""
    meter, thermal = session.meter, session.thermal
    llm.reset()  # Empty KV cache: every measurement processes the prompt from scratch
    sampler = FrequencySampler(session.probe, session.config["thermal"]["freq_sample_interval_s"]).start()
    window_start = time.perf_counter()
    try:
        thermal.check()
        meter.start()
        start = time.perf_counter()
        llm.eval(prompt_tokens)
        prefill_time = time.perf_counter() - start
        prefill = meter.stop()

        meter.start()
        start = time.perf_counter()
        generated, suppressed = greedy_decode(llm, n_predict, eog_ids, thermal)
        decode_time = time.perf_counter() - start
        decode = meter.stop()
    finally:
        meter.abort()
        freq = sampler.stop()
    temp = thermal.stats(window_start, time.perf_counter())

    return {
        "prompt_tokens": len(prompt_tokens),
        "generated_tokens": len(generated),
        "eos_suppressed": suppressed,
        "output_sha256": hashlib.sha256(np.array(generated, dtype=np.int32).tobytes()).hexdigest()[:16],
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "prefill": prefill,
        "decode": decode,
        "freq": freq,
        "temp": temp,
    }


def _derive_row(raw, idle_cpu_w, intensity):
    """Per-measurement metrics. The decode window defines speed and the headline energy metric."""
    n = raw["generated_tokens"]
    decode, prefill = raw["decode"], raw["prefill"]
    decode_time, prefill_time = raw["decode_time_s"], raw["prefill_time_s"]
    cpu_j = decode.cpu_J
    # Net energy: subtract what the idle machine would have drawn over the same time
    cpu_j_net = cpu_j - idle_cpu_w * decode_time
    cpu_j_per_token = cpu_j / n
    return {
        "prompt_tokens": raw["prompt_tokens"],
        "generated_tokens": n,
        "eos_suppressed": raw["eos_suppressed"],
        "output_sha256": raw["output_sha256"],
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "tokens_per_sec": n / decode_time,
        "prefill_tokens_per_sec": raw["prompt_tokens"] / prefill_time,
        "tokens_per_sec_incl_prefill": n / (prefill_time + decode_time),
        "cpu_J": cpu_j,
        "ram_J_est": decode.ram_J_est,  # Estimate (constant-power model), not a measurement
        "gpu_J": decode.gpu_J,
        "total_J": decode.total_J,
        "prefill_cpu_J": prefill.cpu_J,
        "idle_cpu_W": idle_cpu_w,
        "cpu_J_net": cpu_j_net,
        "cpu_J_per_token": cpu_j_per_token,
        "cpu_J_per_token_net": cpu_j_net / n,
        "cpu_J_per_token_incl_prefill": (prefill.cpu_J + cpu_j) / n,
        "avg_cpu_power_W": cpu_j / decode_time,
        "freq_mean_mhz": raw["freq"].mean_mhz,
        "freq_min_mhz": raw["freq"].min_mhz,
        "temp_mean_C": raw["temp"].mean_c,
        "temp_max_C": raw["temp"].max_c,
        "grid_intensity_gco2_per_kwh": intensity,
        # Energy times a constant grid factor: not an independent metric (see llm_cpu_inference.energy.gco2eq)
        "gco2eq_per_token": gco2eq(cpu_j_per_token, intensity),
    }


def run_round_unit(session, unit, params, iterations_csv, cooldown, on_written=None):
    """Measure one configuration for one round and append its rows.

    Steps: load the model (timed, outside every measurement window), `warmup` discarded
    generations with the first prompt, then every prompt measured once. Rows are written only after the whole
    unit has finished, in one append shielded from Ctrl+C together with `on_written` (the
    campaign's state update), so an interrupted or thermally aborted unit leaves no partial data.
    """
    log = get_logger()
    config = session.config
    inference = config["inference"]
    prompts = config["prompts"]
    n_predict, warmup = params["n_predict"], params["warmup"]
    info = MODELS[unit.model_key]
    tag = f"[{unit.experiment} | {unit.model_key} | {unit.n_threads} thr | round {unit.round}]"
    idle_cpu_w = session.idle["cpu_W"]

    llm, load_time = load_model(unit.model_key, unit.n_threads, inference)
    log.info("%s model loaded in %.2f s", tag, load_time)
    raw_rows = []
    warmup_freqs = []
    try:
        eog_ids = end_of_generation_ids(llm)
        tokens = {}
        for prompt in prompts:
            # Raw completion: BOS + prompt text, no chat template, identical for every model
            tokens[prompt["id"]] = llm.tokenize(prompt["text"].encode("utf-8"), add_bos=True, special=False)
            if len(tokens[prompt["id"]]) + n_predict > inference["n_ctx"]:
                raise ValueError(f"Prompt '{prompt['id']}' ({len(tokens[prompt['id']])} tokens) plus "
                                 f"n_predict={n_predict} does not fit n_ctx={inference['n_ctx']}")

        # Warm-up after every (re)load, with the first prompt only: weights are paged in, caches
        # and clocks settle. Not recorded (its frequency only serves as a throttling reference).
        for _ in range(warmup):
            raw = measure_once(llm, session, tokens[prompts[0]["id"]], n_predict, eog_ids)
            warmup_freqs.append(raw["freq"].mean_mhz)

        for prompt in prompts:
            timestamp = now_iso()
            raw = measure_once(llm, session, tokens[prompt["id"]], n_predict, eog_ids)
            raw_rows.append((prompt["id"], timestamp, raw))
            log.info("%s %s: %.2f tok/s | %.2f J/token (CPU, decode) | %.0f MHz | %.1f C (max %.1f)",
                     tag, prompt["id"], raw["generated_tokens"] / raw["decode_time_s"],
                     raw["decode"].cpu_J / raw["generated_tokens"], raw["freq"].mean_mhz,
                     raw["temp"].mean_c, raw["temp"].max_c)
    finally:
        llm.close()

    # Reference frequency = best mean frequency seen in this unit (warm-up included)
    measured_freqs = [raw["freq"].mean_mhz for _, _, raw in raw_rows]
    valid_freqs = [f for f in measured_freqs + warmup_freqs if not math.isnan(f)]
    flags = flag_throttling(measured_freqs, config["thermal"]["throttle_drop_frac"],
                            reference_mhz=max(valid_freqs) if valid_freqs else NAN)

    rows = []
    for (prompt_id, timestamp, raw), flag in zip(raw_rows, flags):
        rows.append({
            "session_id": session.session_id, "run_id": session.run_id, "timestamp": timestamp,
            "experiment": unit.experiment, "round": unit.round, "order_in_round": unit.order_in_round,
            "model_key": unit.model_key, "model_file": info["filename"], "method": info["method"],
            "params_b": info["params_b"], "quantization": quantization_of(info["filename"]),
            "n_threads": unit.n_threads, "n_ctx": inference["n_ctx"], "seed": inference["seed"],
            "prompt_id": prompt_id, "iteration": unit.round, "model_load_time_s": load_time,
            **_derive_row(raw, idle_cpu_w, session.carbon_intensity),
            "throttle_flag": flag, "temp_at_start_C": cooldown["end_c"],
            "cooldown_wait_s": cooldown["waited_s"], "cooldown_reached": cooldown["reached"],
            "energy_source": session.meter.cpu_source,
        })
    if any(flags):
        log.warning("%s %d of %d measurements are flagged for suspected throttling.", tag,
                    sum(1 for f in flags if f), len(flags))

    with defer_interrupt():
        append_rows(iterations_csv, ITERATION_FIELDS, rows)
        if on_written:
            on_written()
    return rows


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value) if value is not None and not (isinstance(value, float) and math.isnan(value)) else False


def session_summary_rows(rows, n_predict, n_rounds):
    """Summaries of one session's rows of one experiment.

    Per configuration (model, threads): one row per prompt (replicate = round) and one 'ALL'
    row pooled over prompts, in which the replicate is the round (mean over its prompts).
    """
    by_config = defaultdict(list)
    for row in rows:
        by_config[(row["model_key"], int(row["n_threads"]))].append(row)

    summaries = []
    timestamp = now_iso()
    for (_, n_threads), config_rows in sorted(by_config.items(), key=lambda item: (item[0][0], item[0][1])):
        first = config_rows[0]
        base = {k: first[k] for k in ("session_id", "experiment", "model_key", "model_file", "method",
                                      "params_b", "quantization", "n_ctx")}
        base.update({"timestamp": timestamp, "n_threads": n_threads, "n_predict": n_predict, "n_rounds": n_rounds,
                     "model_load_time_s_mean": float(np.mean([float(r["model_load_time_s"]) for r in config_rows])),
                     "idle_cpu_W_mean": float(np.nanmean([float(r["idle_cpu_W"]) for r in config_rows]))})

        by_prompt = defaultdict(list)
        for row in config_rows:
            by_prompt[row["prompt_id"]].append(row)
        # Greedy decoding must reproduce the same tokens in every round of a prompt
        deterministic = all(len({r["output_sha256"] for r in group}) == 1 for group in by_prompt.values())

        def build(prompt_id, values, prompt_rows):
            summary = {**base, "prompt_id": prompt_id, "deterministic": deterministic,
                       "n_throttled": sum(1 for r in prompt_rows if _as_bool(r["throttle_flag"])),
                       "temp_max_C": float(np.nanmax([float(r["temp_max_C"]) for r in prompt_rows]))}
            for metric in SUMMARY_METRICS:
                stats = summarize(values[metric])
                summary["n"] = stats["n"]
                for stat in ("mean", "std", "ci95_low", "ci95_high"):
                    summary[f"{metric}_{stat}"] = stats[stat]
            return summary

        for prompt_id, group in by_prompt.items():
            summaries.append(build(prompt_id, {m: [float(r[m]) for r in group] for m in SUMMARY_METRICS}, group))
        by_round = defaultdict(list)
        for row in config_rows:
            by_round[int(row["round"])].append(row)
        pooled = {m: [float(np.mean([float(r[m]) for r in group])) for group in by_round.values()]
                  for m in SUMMARY_METRICS}
        summaries.append(build(ALL_PROMPTS, pooled, config_rows))
    return summaries


def run_warmup_load(session, warmup_config, n_threads):
    """Unmeasured continuous generation after the idle baseline, before the first round.

    Session 1 showed a transient in its first two rounds (+3-10 % J/token). This load brings
    the machine into its steady working state before anything is recorded. Nothing is written
    to the result files; a summary goes to the run metadata.
    """
    log = get_logger()
    duration = warmup_config["duration_s"]
    if not duration or duration <= 0:
        return None
    thermal = session.thermal
    llm, load_time = load_model(warmup_config["model"], n_threads, session.config["inference"])
    log.info("Warm-up load (not measured): %s, %d threads, %.0f s", warmup_config["model"], n_threads, duration)
    tokens = 0
    start = last_log = time.perf_counter()
    try:
        eog_ids = end_of_generation_ids(llm)
        prompt = llm.tokenize(session.config["prompts"][0]["text"].encode("utf-8"), add_bos=True, special=False)
        while time.perf_counter() - start < duration:
            llm.reset()
            thermal.check()
            llm.eval(prompt)
            generated, _ = greedy_decode(llm, 50, eog_ids, thermal)
            tokens += len(generated)
            if time.perf_counter() - last_log >= 60:
                log.info("Warm-up load: %.0f / %.0f s | %.1f C", time.perf_counter() - start, duration, thermal.current())
                last_log = time.perf_counter()
    finally:
        llm.close()
    end = time.perf_counter()
    temp = thermal.stats(start, end)
    record = {"model": warmup_config["model"], "n_threads": n_threads, "duration_s": end - start, "tokens": tokens,
              "tokens_per_sec": tokens / (end - start), "model_load_time_s": load_time,
              "temp_mean_C": temp.mean_c, "temp_max_C": temp.max_c, "finished_at": now_iso()}
    log.info("Warm-up load done: %.0f s, %d tokens, %.1f C max", end - start, tokens, temp.max_c)
    return record
