"""Analysis of the unattended night run (session night_01, produced by night_run.py).

Reads results/night/ (never modified) and writes results/analysis/night_*.csv:

  night_integrity.csv            per block: status, units, rows, NaN cells, throttle rows, temperatures, Turbo state,
                                 deterministic configurations (one distinct output per configuration x prompt)
  night_run_summary.csv          run level: heat probe, Turbo restore evidence (log + powercfg), thermal peak, exit line
  night_anomalies_mad.csv        median +- 3 scaled MAD rule of analyze_campaign per (block, model, threads, prompt)
  night_anomaly_sensitivity.csv  change of the block means when the flagged J/token rows are removed
  night_thread_scaling.csv       A1: model x threads (2/4/8): J/token, tokens/s, power (mean +- SD over 5 rounds),
                                 values relative to the model's optimum, effective bandwidth utilisation (estimate 2)
  night_thread_optimum.csv       A1: per model: min-energy, saturating (98 %) and fastest thread count; 2->4 / 2->8 changes
  night_quantization_ladder.csv  A5 + A2: 8B Q4_K_M / Q5_K_M / Q6_K / Q8_0 at 8 threads: energy, speed, power, ratios to
                                 Q4_K_M with t confidence intervals over 10 rounds, bytes per token, bandwidth utilisation,
                                 90-chunk perplexity, bits per byte and paired delta-NLL against Q8_0 and Q4_K_M
  night_turbo.csv                B1: Turbo on / off per model: J/token, tokens/s, package power, frequency, temperature
  night_turbo_ratio.csv          B1: paired on/off ratios (same round) with t confidence intervals; first/second half
                                 consistency; Turbo-off values against the A1/A5 blocks of the same night
  night_perplexity.csv           A2: 90-chunk perplexity / bits per byte against the 60-chunk campaign subset; ranks

The night run is one session, so the unit of replication inside it is the round (A1: 5 rounds, A5 and B1: 10 rounds;
one round = the mean of the three prompts). Confidence intervals are Student-t intervals over those round values.
"""

import argparse
import glob
import json
import math
import os
import subprocess

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (makes src/ importable without installing the package)
from analyze_bandwidth import bytes_per_token, load_sizes, memory_peak_from_metadata
from analyze_campaign import anomaly_sensitivity, mad_anomalies, t_interval
from llm_cpu_inference.models import MODELS
from llm_cpu_inference.config import PROJECT_ROOT

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
NIGHT_SUBDIR = "night"
GROUP = ["model_key", "n_threads"]
A1_EXPERIMENT = "night_a1"
A5_EXPERIMENT = "night_a5"
B1_PREFIX = "night_b1_pre"
A2_EXPERIMENT = "night_a2"
BLOCK_EXPERIMENTS = {"a1": [A1_EXPERIMENT], "a5": [A5_EXPERIMENT],
                     "b1_pre": [f"{B1_PREFIX}_turbo_on", f"{B1_PREFIX}_turbo_off"], "a2": [A2_EXPERIMENT],
                     "a2_rest": ["night_a2_rest"], "b1_post": ["night_b1_post_turbo_on", "night_b1_post_turbo_off"]}
COGNITIVE_BLOCKS = ("a2", "a2_rest")
NIGHT_RUNS = (("night", "night_"), ("night_02", "night02_"))  # (results sub-directory, output prefix)
LADDER = ["8B-Q4_K_M", "8B-Q5_K_M", "8B-Q6_K", "8B-Q8_0"]
MODEL_ORDER = ["1B-Q4_K_M", "1B-Q8_0", "3B-Q4_K_M", "3B-Q8_0", "8B-Q4_K_M", "8B-Q5_K_M", "8B-Q6_K", "8B-Q8_0",
               "Sparse8B-Q4_K_M", "Sparse8B-Q8_0"]
SATURATION_FRAC = 0.98
TURBO_OFF_HEX = "0x00000063"  # 99 %


# ---------------------------------------------------------------- loading
def load_night(results_dir=RESULTS_DIR, subdir=NIGHT_SUBDIR):
    night_dir = os.path.join(results_dir, subdir)
    if not os.path.exists(os.path.join(night_dir, "energy_iterations.csv")):
        return None
    energy = pd.read_csv(os.path.join(night_dir, "energy_iterations.csv"))
    cog_path = os.path.join(night_dir, "cognitive_results.csv")
    cognitive = pd.read_csv(cog_path) if os.path.exists(cog_path) else pd.DataFrame()
    full_path = os.path.join(night_dir, "cognitive_results_full.csv")  # merged full-set rows (a2_rest)
    cognitive_full = pd.read_csv(full_path) if os.path.exists(full_path) else pd.DataFrame()
    state_path = os.path.join(night_dir, "campaign_state.json")
    state = json.load(open(state_path, encoding="utf-8")) if os.path.exists(state_path) else {"sessions": {}, "units": {}}
    metas = [json.load(open(p, encoding="utf-8")) for p in glob.glob(os.path.join(night_dir, "metadata", "*.json"))]
    metas = [m for m in metas if not m.get("smoke")]
    meta = max(metas, key=lambda m: m.get("timestamp_start", "")) if metas else {}
    log_text = ""
    for path in sorted(glob.glob(os.path.join(night_dir, "logs", "night_*.log"))):
        log_text += open(path, encoding="utf-8", errors="replace").read()
    session_id = energy["session_id"].iloc[0]
    return {"dir": night_dir, "energy": energy, "cognitive": cognitive, "cognitive_full": cognitive_full, "state": state,
            "meta": meta, "log": log_text, "session_id": session_id, "nll_dir": os.path.join(night_dir, "nll")}


def round_means(df, metric):
    """One value per (model, threads, round): the mean of the metric over the prompts of that round."""
    return df.groupby(GROUP + ["round"], as_index=False)[metric].mean()


def round_pivot(df, metric, index="round", columns="model_key"):
    return round_means(df, metric).pivot_table(index=index, columns=columns, values=metric)


def ratio_ci(numerator, denominator):
    """Mean and 95 % t interval of the paired ratio (same round)."""
    r = (numerator / denominator).dropna()
    mean, _, low, high = t_interval(r.values)
    return mean, low, high, int(len(r))


def powercfg_turbo_off_lines():
    """Number of PROCTHROTTLEMAX index lines equal to 99 % (expected 2: AC and DC); -1 when unavailable."""
    if os.name != "nt":
        return -1
    try:
        raw = subprocess.run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_PROCESSOR", "PROCTHROTTLEMAX"],
                             capture_output=True, timeout=20).stdout or b""
    except (OSError, subprocess.SubprocessError):
        return -1
    # The console code page is locale dependent; the hex value is ASCII and the label may be localised
    return sum(1 for line in raw.decode("ascii", errors="ignore").splitlines() if TURBO_OFF_HEX in line)


# ---------------------------------------------------------------- 1. integrity
def integrity_table(night):
    energy, cognitive, state = night["energy"], night["cognitive"], night["state"]
    sess = state["sessions"].get(night["session_id"], {})
    blocks = sess.get("night", {}).get("blocks", {})
    executed = pd.DataFrame(night["meta"].get("executed_units", []))
    units = pd.DataFrame([{"unit_id": k, **v} for k, v in state["units"].items()
                          if k.startswith(night["session_id"] + "|")])
    rows = []
    for block, experiments in BLOCK_EXPERIMENTS.items():
        if block not in blocks:
            continue
        exp_mask = units["unit_id"].str.split("|").str[1].isin(experiments) if len(units) else pd.Series(dtype=bool)
        done = int((units.loc[exp_mask, "status"] == "completed").sum()) if len(units) else 0
        if block in COGNITIVE_BLOCKS:
            df = night["cognitive_full"] if block == "a2_rest" and len(night["cognitive_full"]) else cognitive
            det = f"{len(df)}/{len(df)}" if len(df) else "0/0"
            throttle = 0
            nan_cells = int(df.isna().sum().sum()) if len(df) else 0
        else:
            df = energy[energy["experiment"].isin(experiments)]
            per_cfg = (df.groupby(GROUP + ["prompt_id", "experiment"])["output_sha256"].nunique() if len(df)
                       else pd.Series(dtype=int))
            det = f"{int((per_cfg == 1).sum())}/{len(per_cfg)}"
            throttle = int(df["throttle_flag"].sum()) if len(df) else 0
            nan_cells = int(df.isna().sum().sum()) if len(df) else 0
        ex = executed[executed["block"] == block] if len(executed) and "block" in executed else pd.DataFrame()
        turbo_states = sorted(ex["turbo_on"].astype(str).unique()) if len(ex) else []
        proc_states = sorted(ex["max_processor_state"].astype(int).unique()) if len(ex) else []
        temp_max = float(df["temp_max_C"].max()) if len(df) and "temp_max_C" in df else float("nan")
        temp_mean = float(df["temp_mean_C"].mean()) if len(df) and "temp_mean_C" in df else float("nan")
        rows.append({"block": block, "status": blocks[block], "experiments": ";".join(experiments),
                     "units_completed": done, "units_in_metadata": int(len(ex)), "rows": int(len(df)),
                     "nan_cells": nan_cells, "throttle_rows": throttle, "temp_max_C": temp_max,
                     "temp_mean_C": round(temp_mean, 2) if not math.isnan(temp_mean) else temp_mean,
                     "turbo_on_states": ";".join(turbo_states),
                     "max_processor_state_pct": ";".join(map(str, proc_states)),
                     "deterministic_configs": det})
    # Turbo-on and Turbo-off peak temperatures separately (B1 block)
    b1 = energy[energy["experiment"].str.startswith(B1_PREFIX)]
    for state_name, mask in (("turbo_on", b1["experiment"].str.endswith("turbo_on")),
                             ("turbo_off", b1["experiment"].str.endswith("turbo_off"))):
        sub = b1[mask]
        if len(sub):
            rows.append({"block": f"b1_pre/{state_name}", "status": "subset", "experiments": sub["experiment"].iloc[0],
                         "units_completed": int(sub.groupby(GROUP + ["round"]).ngroups), "units_in_metadata": "",
                         "rows": int(len(sub)), "nan_cells": int(sub.isna().sum().sum()),
                         "throttle_rows": int(sub["throttle_flag"].sum()), "temp_max_C": float(sub["temp_max_C"].max()),
                         "temp_mean_C": round(float(sub["temp_mean_C"].mean()), 2),
                         "turbo_on_states": str(state_name == "turbo_on"), "max_processor_state_pct": "",
                         "deterministic_configs": ""})
    return pd.DataFrame(rows)


def run_summary(night):
    meta, log, state = night["meta"], night["log"], night["state"]
    sess = state["sessions"].get(night["session_id"], {})
    b1 = sess.get("night", {}).get("b1", {})
    lines = log.splitlines()

    def last(needle):
        hits = [ln for ln in lines if needle in ln]
        return hits[-1].split("| INFO    | ")[-1].strip() if hits else ""

    thermal = meta.get("thermal", {})
    idle = meta.get("idle_baseline", {})
    items = {
        "session_id": night["session_id"], "run_id": meta.get("run_id", ""), "status": meta.get("status", ""),
        "started": meta.get("timestamp_start", ""),
        "finished": meta.get("timestamp_end", sess.get("completed_at", "")),
        "operator_note": meta.get("operator_note", ""),
        "idle_cpu_W": idle.get("cpu_W", float("nan")),
        "idle_cpu_utilisation_pct": idle.get("cpu_utilisation_pct", ""),
        "heat_probe_peak_C": b1.get("heat_peak_c", float("nan")), "heat_probe_limit_C": b1.get("heat_max_c", ""),
        "b1_skipped_thermal": b1.get("skipped", ""), "heat_probe_log": last("Heat probe"),
        "turbo_restored_logged": "Turbo restored" in log, "turbo_restored_log": last("Turbo restored"),
        "powercfg_99pct_lines_now": powercfg_turbo_off_lines(),
        "thermal_max_C": thermal.get("max_temperature_c", float("nan")),
        "thermal_abort_reason": thermal.get("abort_reason", ""),
        "thermal_abort_C": thermal.get("abort_c", ""), "thermal_abort_hold_s": thermal.get("abort_hold_s", ""),
        "blocks": json.dumps(sess.get("night", {}).get("blocks", {})),
        "finished_log": last("Night run finished"),
        "log_warnings": sum(1 for ln in lines if "| WARNING" in ln),
        "log_errors": sum(1 for ln in lines if "| ERROR" in ln),
        "block_failed_lines": sum(1 for ln in lines if "BLOCK FAILED" in ln),
        "thermal_abort_lines": sum(1 for ln in lines if "THERMAL_ABORT" in ln),
    }
    checks = meta.get("cognitive_merge_checks")
    if checks:
        items["merge_checks_total"] = len(checks)
        items["merge_checks_passed"] = sum(1 for c in checks if c["same_text"] and c["overlap_identical"])
        items["merge_overlap_max_abs_diff"] = max(c["overlap_max_abs_diff"] for c in checks)
        items["merge_chunks"] = f"{checks[0]['previous_chunks']} earlier + {checks[0]['new_chunks']} new = {checks[0]['total_chunks']}"
    return pd.DataFrame({"item": list(items), "value": [str(v) for v in items.values()]})


def anomaly_frames(energy):
    return {exp: energy[energy["experiment"] == exp].copy() for exp in sorted(energy["experiment"].unique())}


# ---------------------------------------------------------------- 2a. A1 thread optimum
def thread_scaling_table(energy, sizes, peak_gbs):
    a1 = energy[energy["experiment"] == A1_EXPERIMENT]
    bpt = bytes_per_token(sizes).set_index("model_key")
    rows = []
    for model in [m for m in MODEL_ORDER if m in set(a1["model_key"])]:
        sub = a1[a1["model_key"] == model]
        piv = {m: round_pivot(sub, m, columns="n_threads")
               for m in ("cpu_J_per_token", "tokens_per_sec", "avg_cpu_power_W")}
        j_mean, s_mean = piv["cpu_J_per_token"].mean(), piv["tokens_per_sec"].mean()
        saturation = s_mean[s_mean >= SATURATION_FRAC * s_mean.max()].index.min()
        bytes_est2 = float(bpt.loc[model, "estimate_2_bytes"]) if model in bpt.index else float("nan")
        for n in sorted(piv["cpu_J_per_token"].columns):
            rows.append({
                "model_key": model, "n_threads": int(n), "n_rounds": int(piv["cpu_J_per_token"][n].count()),
                "J_per_token": j_mean[n], "J_per_token_sd": piv["cpu_J_per_token"][n].std(ddof=1),
                "tokens_per_sec": s_mean[n], "tokens_per_sec_sd": piv["tokens_per_sec"][n].std(ddof=1),
                "cpu_power_W": piv["avg_cpu_power_W"][n].mean(),
                "cpu_power_W_sd": piv["avg_cpu_power_W"][n].std(ddof=1),
                "J_rel_to_min": j_mean[n] / j_mean.min(), "speed_rel_to_max": s_mean[n] / s_mean.max(),
                "is_min_energy": bool(n == j_mean.idxmin()), "is_max_speed": bool(n == s_mean.idxmax()),
                "is_saturation": bool(n == saturation),
                "bytes_per_token_est2": bytes_est2,
                "effective_gbs_est2": bytes_est2 * s_mean[n] / 1e9,
            })
    df = pd.DataFrame(rows)
    df["utilisation_est2_pct"] = 100 * df["effective_gbs_est2"] / peak_gbs
    df["theoretical_peak_gbs"] = peak_gbs
    return df


def thread_optimum_table(ts):
    rows = []
    for model, g in ts.groupby("model_key", sort=False):
        g = g.set_index("n_threads").sort_index()
        j, s, w, u = g["J_per_token"], g["tokens_per_sec"], g["cpu_power_W"], g["utilisation_est2_pct"]
        info = MODELS[model]
        rows.append({
            "model_key": model, "params_b": info["params_b"], "quantization": model.split("-", 1)[1],
            "min_energy_threads": int(j.idxmin()), "min_energy_J_per_token": j.min(),
            "saturation_threads_98pct": int(s[s >= SATURATION_FRAC * s.max()].index.min()),
            "max_speed_threads": int(s.idxmax()), "max_speed_tokens_per_sec": s.max(),
            "speed_2_to_4_pct": 100 * (s[4] / s[2] - 1), "speed_4_to_8_pct": 100 * (s[8] / s[4] - 1),
            "speed_2_to_8_pct": 100 * (s[8] / s[2] - 1),
            "energy_2_to_4_pct": 100 * (j[4] / j[2] - 1), "energy_2_to_8_pct": 100 * (j[8] / j[2] - 1),
            "power_2_to_8_pct": 100 * (w[8] / w[2] - 1),
            "utilisation_2_pct": u[2], "utilisation_4_pct": u[4], "utilisation_8_pct": u[8],
            "J_by_threads": ";".join(f"{n}:{x:.3f}" for n, x in j.items()),
            "speed_by_threads": ";".join(f"{n}:{x:.3f}" for n, x in s.items()),
            "power_by_threads": ";".join(f"{n}:{x:.1f}" for n, x in w.items()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2b. A5 quantization ladder (+ A2 quality)
def paired_delta_nll(nll_dir, cog, model, reference):
    """Paired chunk-level delta NLL (model - reference) from the cached per-token NLL arrays; NaN when unavailable."""
    from benchmark_cognitive import nll_cache_path  # local import: that module loads the inference stack

    nan3 = (float("nan"), float("nan"), float("nan"))
    if model not in cog.index or reference not in cog.index:
        return nan3
    paths = [nll_cache_path(nll_dir, cog.loc[k, "model_file"], cog.loc[k, "dataset_sha256"], int(cog.loc[k, "n_ctx"]),
                            int(cog.loc[k, "n_chunks"])) for k in (model, reference)]
    if not all(os.path.exists(p) for p in paths):
        return nan3
    a, b = (np.load(p, allow_pickle=True) for p in paths)
    if a["nll"].shape != b["nll"].shape or str(a["scored_text_sha256"]) != str(b["scored_text_sha256"]):
        return nan3
    delta = (a["nll"] - b["nll"]).mean(axis=1)
    mean, _, low, high = t_interval(delta)
    return mean, low, high


def quantization_ladder(energy, cognitive, sizes, peak_gbs, nll_dir, anchor="8B-Q4_K_M"):
    a5 = energy[energy["experiment"] == A5_EXPERIMENT]
    cog = cognitive.set_index("model_key") if len(cognitive) else pd.DataFrame()
    sz = sizes.set_index("model_key")
    bpt = bytes_per_token(sizes).set_index("model_key")
    piv = {m: round_pivot(a5, m) for m in ("cpu_J_per_token", "tokens_per_sec", "avg_cpu_power_W", "gco2eq_per_token")}
    rows = []
    for model in [m for m in LADDER if m in piv["cpu_J_per_token"].columns]:
        j, s, w = piv["cpu_J_per_token"][model], piv["tokens_per_sec"][model], piv["avg_cpu_power_W"][model]
        jr, jlo, jhi, n = ratio_ci(j, piv["cpu_J_per_token"][anchor])
        sr, slo, shi, _ = ratio_ci(s, piv["tokens_per_sec"][anchor])
        in_sizes = model in sz.index
        row = {
            "model_key": model, "quantization": model.split("-", 1)[1], "n_rounds": n,
            "file_gb": float(sz.loc[model, "file_bytes"]) / 1e9 if in_sizes else float("nan"),
            "bits_per_weight": (8 * float(sz.loc[model, "tensors_bytes"]) / (MODELS[model]["params_b"] * 1e9)
                                if in_sizes else float("nan")),
            "J_per_token": j.mean(), "J_per_token_sd": j.std(ddof=1),
            "tokens_per_sec": s.mean(), "tokens_per_sec_sd": s.std(ddof=1),
            "cpu_power_W": w.mean(), "mgco2eq_per_token": 1000 * piv["gco2eq_per_token"][model].mean(),
            "J_ratio_vs_anchor": jr, "J_ratio_ci95_low": jlo, "J_ratio_ci95_high": jhi,
            "speed_ratio_vs_anchor": sr, "speed_ratio_ci95_low": slo, "speed_ratio_ci95_high": shi,
            "bytes_ratio_vs_anchor": (float(bpt.loc[model, "estimate_2_bytes"]) / float(bpt.loc[anchor, "estimate_2_bytes"])
                                      if model in bpt.index and anchor in bpt.index else float("nan")),
            "bytes_per_token_est2": float(bpt.loc[model, "estimate_2_bytes"]) if model in bpt.index else float("nan"),
        }
        row["effective_gbs_est2"] = row["bytes_per_token_est2"] * row["tokens_per_sec"] / 1e9
        row["utilisation_est2_pct"] = 100 * row["effective_gbs_est2"] / peak_gbs
        if len(cog) and model in cog.index:
            c = cog.loc[model]
            row.update({"n_chunks": int(c["n_chunks"]), "perplexity": c["perplexity"],
                        "ppl_ci95_low": c["ppl_ci95_low"], "ppl_ci95_high": c["ppl_ci95_high"],
                        "bits_per_byte": c["bits_per_byte"], "top1_accuracy": c["top1_accuracy"],
                        "delta_nll_vs_Q8": c["delta_nll_vs_ref"], "delta_nll_vs_Q8_ci95_low": c["delta_nll_ci95_low"],
                        "delta_nll_vs_Q8_ci95_high": c["delta_nll_ci95_high"],
                        "delta_ppl_pct_vs_Q8": c["delta_ppl_pct_vs_ref"]})
            d, dlo, dhi = paired_delta_nll(nll_dir, cog, model, anchor)
            row.update({"delta_nll_vs_anchor": d, "delta_nll_vs_anchor_ci95_low": dlo,
                        "delta_nll_vs_anchor_ci95_high": dhi,
                        "delta_ppl_pct_vs_anchor": (math.exp(d) - 1) * 100 if not math.isnan(d) else float("nan")})
        rows.append(row)
    df = pd.DataFrame(rows)
    if "perplexity" in df:
        # Marginal cost of quality along the ladder: extra J/token per 1 % perplexity reduction relative to the anchor
        extra_j = df["J_per_token"] - df.loc[df["model_key"] == anchor, "J_per_token"].iloc[0]
        gain = -df["delta_ppl_pct_vs_anchor"]
        df["J_per_token_per_ppl_pct_vs_anchor"] = extra_j / gain.where(gain.abs() > 1e-9)
    return df


# ---------------------------------------------------------------- 2c. B1 Turbo on / off
TURBO_METRICS = ("cpu_J_per_token", "tokens_per_sec", "avg_cpu_power_W", "freq_mean_mhz", "temp_mean_C")


def _b1(energy):
    b1 = energy[energy["experiment"].str.startswith(B1_PREFIX)].copy()
    b1["turbo"] = np.where(b1["experiment"].str.endswith("turbo_on"), "on", "off")
    return b1


def turbo_table(energy):
    b1 = _b1(energy)
    rows = []
    for model in [m for m in MODEL_ORDER if m in set(b1["model_key"])]:
        for turbo in ("off", "on"):
            sub = b1[(b1["model_key"] == model) & (b1["turbo"] == turbo)]
            rm = {m: round_means(sub, m)[m] for m in TURBO_METRICS}
            rows.append({
                "model_key": model, "turbo": turbo, "n_threads": int(sub["n_threads"].iloc[0]),
                "n_rounds": int(len(rm["cpu_J_per_token"])),
                "J_per_token": rm["cpu_J_per_token"].mean(), "J_per_token_sd": rm["cpu_J_per_token"].std(ddof=1),
                "tokens_per_sec": rm["tokens_per_sec"].mean(), "tokens_per_sec_sd": rm["tokens_per_sec"].std(ddof=1),
                "cpu_power_W": rm["avg_cpu_power_W"].mean(), "cpu_power_W_sd": rm["avg_cpu_power_W"].std(ddof=1),
                "freq_mean_mhz": rm["freq_mean_mhz"].mean(), "freq_min_mhz": float(sub["freq_min_mhz"].min()),
                "temp_mean_C": rm["temp_mean_C"].mean(), "temp_max_C": float(sub["temp_max_C"].max()),
                "temp_at_start_C": float(sub["temp_at_start_C"].mean()),
                "throttle_rows": int(sub["throttle_flag"].sum()),
                "mgco2eq_per_token": 1000 * sub["gco2eq_per_token"].mean(),
            })
    return pd.DataFrame(rows)


def turbo_ratio_table(energy):
    b1 = _b1(energy)
    a1 = energy[energy["experiment"] == A1_EXPERIMENT]
    a5 = energy[energy["experiment"] == A5_EXPERIMENT]
    rows = []
    for model in [m for m in MODEL_ORDER if m in set(b1["model_key"])]:
        sub = b1[b1["model_key"] == model]
        n_threads = int(sub["n_threads"].iloc[0])
        row = {"model_key": model, "n_threads": n_threads}
        for metric, short in (("cpu_J_per_token", "J"), ("tokens_per_sec", "speed"), ("avg_cpu_power_W", "power"),
                              ("freq_mean_mhz", "freq")):
            piv = pd.DataFrame({state: round_means(sub[sub["turbo"] == state], metric).set_index("round")[metric]
                                for state in ("on", "off")})
            r, lo, hi, n = ratio_ci(piv["on"], piv["off"])
            row.update({f"{short}_ratio_on_off": r, f"{short}_ratio_ci95_low": lo, f"{short}_ratio_ci95_high": hi})
            if metric == "cpu_J_per_token":
                row["n_rounds"] = n
                ratios = (piv["on"] / piv["off"]).sort_index()
                half = len(ratios) // 2
                row["J_ratio_rounds_first_half"] = ratios.iloc[:half].mean()
                row["J_ratio_rounds_second_half"] = ratios.iloc[half:].mean()
                row["J_ratio_halves_diff_pct"] = 100 * (ratios.iloc[half:].mean() / ratios.iloc[:half].mean() - 1)
        # Turbo-off block against the other Turbo-off blocks of the night with the same configuration
        off_j = round_means(sub[sub["turbo"] == "off"], "cpu_J_per_token")["cpu_J_per_token"].mean()
        for name, ref in (("a1", a1), ("a5", a5)):
            r = ref[(ref["model_key"] == model) & (ref["n_threads"] == n_threads)]
            if len(r):
                ref_j = round_means(r, "cpu_J_per_token")["cpu_J_per_token"].mean()
                row[f"J_off_vs_{name}_pct"] = 100 * (off_j / ref_j - 1)
        row["temp_max_on_C"] = float(sub.loc[sub["turbo"] == "on", "temp_max_C"].max())
        row["temp_max_off_C"] = float(sub.loc[sub["turbo"] == "off", "temp_max_C"].max())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2c'. B1 over two independent sessions (night_01 + night_02)
def turbo_combined_table(energies):
    """Per model x Turbo state: mean of the session means, between-session SD and the session values."""
    rows = []
    sessions = sorted(energies)
    frames = {sid: _b1(e) for sid, e in energies.items()}
    models = [m for m in MODEL_ORDER if all(m in set(f["model_key"]) for f in frames.values())]
    for model in models:
        for turbo in ("off", "on"):
            per = {}
            for sid in sessions:
                sub = frames[sid][(frames[sid]["model_key"] == model) & (frames[sid]["turbo"] == turbo)]
                per[sid] = {m: round_means(sub, m)[m].mean() for m in TURBO_METRICS}
                per[sid]["temp_max_C"] = float(sub["temp_max_C"].max())
                per[sid]["freq_min_mhz"] = float(sub["freq_min_mhz"].min())
                per[sid]["throttle_rows"] = int(sub["throttle_flag"].sum())
                per[sid]["mgco2eq_per_token"] = 1000 * sub["gco2eq_per_token"].mean()
                per[sid]["n_rounds"] = int(sub["round"].nunique())
                per[sid]["n_threads"] = int(sub["n_threads"].iloc[0])
            df = pd.DataFrame(per).T
            row = {"model_key": model, "turbo": turbo, "n_sessions": len(sessions), "n_threads": int(df["n_threads"].iloc[0]),
                   "n_rounds": int(df["n_rounds"].sum())}
            for metric, name in (("cpu_J_per_token", "J_per_token"), ("tokens_per_sec", "tokens_per_sec"),
                                 ("avg_cpu_power_W", "cpu_power_W"), ("freq_mean_mhz", "freq_mean_mhz"),
                                 ("temp_mean_C", "temp_mean_C")):
                row[name] = df[metric].mean()
                row[name + "_sd"] = df[metric].std(ddof=1)
                row[name + "_by_session"] = ";".join(f"{sid}:{df.loc[sid, metric]:.4f}" for sid in sessions)
            row.update({"freq_min_mhz": df["freq_min_mhz"].min(), "temp_max_C": df["temp_max_C"].max(),
                        "throttle_rows": int(df["throttle_rows"].sum()), "mgco2eq_per_token": df["mgco2eq_per_token"].mean(),
                        "temp_max_by_session": ";".join(f"{sid}:{df.loc[sid, 'temp_max_C']:.0f}" for sid in sessions)})
            rows.append(row)
    return pd.DataFrame(rows)


def turbo_combined_ratio_table(energies):
    """Paired on/off ratios pooled over the rounds of all sessions (t interval, df = n_rounds - 1), the per-session
    ratio means and their relative difference (between-session consistency)."""
    sessions = sorted(energies)
    frames = {sid: _b1(e) for sid, e in energies.items()}
    models = [m for m in MODEL_ORDER if all(m in set(f["model_key"]) for f in frames.values())]
    rows = []
    for model in models:
        row = {"model_key": model, "n_sessions": len(sessions)}
        for metric, short in (("cpu_J_per_token", "J"), ("tokens_per_sec", "speed"), ("avg_cpu_power_W", "power"),
                              ("freq_mean_mhz", "freq")):
            ratios = {}
            for sid in sessions:
                sub = frames[sid][frames[sid]["model_key"] == model]
                piv = pd.DataFrame({state: round_means(sub[sub["turbo"] == state], metric).set_index("round")[metric]
                                    for state in ("on", "off")})
                ratios[sid] = (piv["on"] / piv["off"]).dropna()
            pooled = pd.concat(ratios.values())
            mean, _, low, high = t_interval(pooled.values)
            per = {sid: r.mean() for sid, r in ratios.items()}
            row.update({f"{short}_ratio_on_off": mean, f"{short}_ratio_ci95_low": low, f"{short}_ratio_ci95_high": high,
                        f"{short}_ratio_by_session": ";".join(f"{sid}:{v:.4f}" for sid, v in per.items()),
                        f"{short}_ratio_session_diff_pct": 100 * (max(per.values()) / min(per.values()) - 1)})
            if metric == "cpu_J_per_token":
                row["n_rounds"] = int(len(pooled))
                # Session as the unit of replication (n = 2): mean of the session ratios and its t interval
                smean, _, slow, shigh = t_interval(list(per.values()))
                row.update({"J_ratio_session_mean": smean, "J_ratio_session_ci95_low": slow, "J_ratio_session_ci95_high": shigh})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2d. A2 perplexity vs the campaign subset
def perplexity_table(cog_night, cog_campaign, label_night="night", label_campaign="subset"):
    """Side by side: the perplexity run `cog_night` against the earlier run `cog_campaign` (ranks, change)."""
    cn = cog_night.set_index("model_key")
    cc = cog_campaign.set_index("model_key") if len(cog_campaign) else pd.DataFrame()
    rank_n = cn["perplexity"].rank().astype(int)
    rank_c = cc["perplexity"].rank().astype(int) if len(cc) else pd.Series(dtype=int)
    a, b = label_night, label_campaign
    rows = []
    for model in [m for m in MODEL_ORDER if m in cn.index]:
        c = cn.loc[model]
        row = {"model_key": model, "method": MODELS[model]["method"], "base": MODELS[model]["base"],
               f"n_chunks_{a}": int(c["n_chunks"]), f"n_scored_tokens_{a}": int(c["n_scored_tokens"]),
               f"perplexity_{a}": c["perplexity"], f"ppl_ci95_low_{a}": c["ppl_ci95_low"],
               f"ppl_ci95_high_{a}": c["ppl_ci95_high"], f"bits_per_byte_{a}": c["bits_per_byte"],
               f"top1_accuracy_{a}": c["top1_accuracy"],
               "delta_nll_vs_ref": c["delta_nll_vs_ref"], "delta_nll_ci95_low": c["delta_nll_ci95_low"],
               "delta_nll_ci95_high": c["delta_nll_ci95_high"], "delta_ppl_pct_vs_ref": c["delta_ppl_pct_vs_ref"],
               "reference": c["reference"], f"rank_{a}": int(rank_n[model]), f"eval_time_s_{a}": c["eval_time_s"]}
        if model in cc.index:
            o = cc.loc[model]
            row.update({f"n_chunks_{b}": int(o["n_chunks"]), f"perplexity_{b}": o["perplexity"],
                        f"ppl_ci95_low_{b}": o["ppl_ci95_low"], f"ppl_ci95_high_{b}": o["ppl_ci95_high"],
                        f"bits_per_byte_{b}": o["bits_per_byte"],
                        f"delta_ppl_pct_vs_ref_{b}": o["delta_ppl_pct_vs_ref"],
                        "ppl_change_pct": 100 * (c["perplexity"] / o["perplexity"] - 1),
                        f"rank_{b}": int(rank_c[model])})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Analysis of the unattended night run (results/night).")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args()
    runs = {subdir: (prefix, load_night(args.results_dir, subdir)) for subdir, prefix in NIGHT_RUNS}
    runs = {k: v for k, v in runs.items() if v[1] is not None}
    if "night" not in runs:
        raise SystemExit("Error: results/night/energy_iterations.csv not found (no night run).")
    out_dir = os.path.join(args.results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    sizes = load_sizes(out_dir)
    peak_gbs = float(memory_peak_from_metadata(args.results_dir).loc[0, "theoretical_peak_gbs"])
    campaign_cog_path = os.path.join(args.results_dir, "cognitive_results.csv")
    campaign_cog = pd.read_csv(campaign_cog_path) if os.path.exists(campaign_cog_path) else pd.DataFrame()

    def save(name, df):
        df.to_csv(os.path.join(out_dir, name), index=False)
        print(f"written: {os.path.join(out_dir, name)} ({len(df)} rows)")

    # Per run: integrity, anomalies, and the blocks it contains
    for subdir, (prefix, night) in runs.items():
        experiments = set(night["energy"]["experiment"])
        save(f"{prefix}integrity.csv", integrity_table(night))
        save(f"{prefix}run_summary.csv", run_summary(night))
        frames = anomaly_frames(night["energy"])
        anomalies = mad_anomalies(frames)
        save(f"{prefix}anomalies_mad.csv", anomalies)
        save(f"{prefix}anomaly_sensitivity.csv", anomaly_sensitivity(frames, anomalies))
        if A1_EXPERIMENT in experiments:
            ts = thread_scaling_table(night["energy"], sizes, peak_gbs)
            save(f"{prefix}thread_scaling.csv", ts)
            save(f"{prefix}thread_optimum.csv", thread_optimum_table(ts))
        if any(e.startswith(B1_PREFIX) for e in experiments):
            save(f"{prefix}turbo.csv", turbo_table(night["energy"]))
            save(f"{prefix}turbo_ratio.csv", turbo_ratio_table(night["energy"]))
        if len(night["cognitive"]) and A2_EXPERIMENT not in experiments and not len(night["cognitive_full"]):
            pass
    night1 = runs["night"][1]
    night2 = runs.get("night_02", (None, None))[1]
    full = night2["cognitive_full"] if night2 is not None and len(night2["cognitive_full"]) else pd.DataFrame()

    # A5 ladder with the best available quality data (full set when night_02 exists, else the 90 chunks)
    quality, nll_dir = (full, night2["nll_dir"]) if len(full) else (night1["cognitive"], night1["nll_dir"])
    save("night_quantization_ladder.csv",
         quantization_ladder(night1["energy"], quality, sizes, peak_gbs, nll_dir))
    # A2: 90 chunks vs the 60-chunk campaign subset; full set (564) vs 90 chunks
    if len(night1["cognitive"]):
        save("night_perplexity.csv", perplexity_table(night1["cognitive"], campaign_cog, "night", "subset"))
    if len(full):
        save("night_perplexity_full.csv", perplexity_table(full, night1["cognitive"], "full", "night"))
    # B1 over both sessions
    if night2 is not None:
        energies = {night1["session_id"]: night1["energy"], night2["session_id"]: night2["energy"]}
        save("night_turbo_combined.csv", turbo_combined_table(energies))
        save("night_turbo_combined_ratio.csv", turbo_combined_ratio_table(energies))


if __name__ == "__main__":
    main()
