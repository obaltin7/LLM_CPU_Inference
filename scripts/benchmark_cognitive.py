"""Language-modelling quality on WikiText-2: perplexity, bits-per-byte and top-1 accuracy (Phase 2 schema).

Protocol (the llama.cpp `perplexity` convention): the test text is tokenized and cut into
non-overlapping chunks of n_ctx tokens (stride = n_ctx); the first token of each chunk is
replaced by BOS and only the second half of the chunk is scored, so every scored token has
at least n_ctx/2 tokens of context.

Besides perplexity (a per-token metric that depends on the tokenizer), bits-per-byte is
reported: total NLL in bits divided by the number of UTF-8 bytes of the scored text. The
SHA256 of the scored bytes is stored per model so byte-level identity of the evaluated text
across models can be verified rather than assumed.

Perplexity is deterministic, so it is measured once per campaign (session_01 only, see
cognitive.first_session_only). The thermal fuse is checked between chunks.

Outputs (appended, never rewritten):
  results/cognitive_results.csv     one row per model
  results/nll/*.npz                 per-token NLL (chunks x scored tokens) for paired comparisons
  results/metadata/<run_id>.json

Run through the campaign (python run_campaign.py --experiments cognitive); this script is a shortcut.
"""
import csv
import hashlib
import math
import os
import time
import zipfile

import numpy as np
from huggingface_hub import hf_hub_download

import _paths  # noqa: F401  (makes src/ importable without installing the package)
from llm_cpu_inference.models import MODELS, quantization_of
from llm_cpu_inference.config import PROJECT_ROOT, resolve_thread_count
from llm_cpu_inference.generation import load_model
from llm_cpu_inference.log import get_logger
from llm_cpu_inference.results_io import append_rows, defer_interrupt
from llm_cpu_inference.run_metadata import now_iso
from llm_cpu_inference.stats import summarize

EXPERIMENT = "cognitive"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NAN = float("nan")

COGNITIVE_FIELDS = [
    "session_id", "run_id", "timestamp", "model_key", "model_file", "method", "params_b", "quantization", "base_model",
    "dataset", "dataset_sha256", "n_ctx", "stride", "n_chunks", "n_threads", "n_vocab",
    "n_eval_tokens", "n_scored_tokens", "scored_bytes", "scored_text_sha256", "text_roundtrip_ok",
    "nll_mean", "nll_se_token", "perplexity", "ppl_ci95_low", "ppl_ci95_high", "bits_per_byte", "top1_accuracy",
    "reference", "paired_with_reference", "delta_nll_vs_ref", "delta_nll_ci95_low", "delta_nll_ci95_high",
    "delta_ppl_pct_vs_ref", "delta_bits_per_byte_vs_ref",
    "model_load_time_s", "eval_time_s", "eval_tokens_per_sec",
    "temp_mean_C", "temp_max_C", "temp_at_start_C", "cooldown_wait_s",
]


def load_dataset(cognitive):
    """Return the raw test text (bytes) and its SHA256; downloads the archive on first use."""
    zip_path = os.path.join(DATA_DIR, cognitive["dataset_file"])
    if not os.path.exists(zip_path):
        zip_path = hf_hub_download(cognitive["dataset_repo"], cognitive["dataset_file"],
                                   repo_type="dataset", local_dir=DATA_DIR)
    with zipfile.ZipFile(zip_path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(cognitive["dataset_member"]))
        raw = archive.read(name)
    return raw, hashlib.sha256(raw).hexdigest()


def log_softmax(logits):
    """Row-wise log-softmax (numerically stable)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def nll_cache_path(nll_dir, model_file, dataset_sha, n_ctx, n_chunks, chunk_start=0):
    """Cache file of a per-token NLL array; a part evaluated from chunk `chunk_start` > 0 carries an `_s<start>` tag."""
    stem = os.path.splitext(model_file)[0]
    part = f"_s{chunk_start}" if chunk_start else ""
    return os.path.join(nll_dir, f"{stem}__wt2-{dataset_sha[:8]}_ctx{n_ctx}{part}_c{n_chunks}.npz")


def evaluate_model(session, model_key, raw_text, chunk_start=0, n_chunks=None, hash_checkpoint=None):
    """Per-token NLL and top-1 accuracy of one model on chunks [chunk_start, chunk_start + n) of the text.

    n defaults to cognitive.chunks (0 = every remaining chunk). The scored-text hash always runs from chunk 0,
    so a part evaluated later hashes to the same value as a single pass over the same range; `hash_checkpoint`
    = k additionally records the hash and byte count after chunk k (the end of an earlier part) so that the
    two parts can be proven to cover the identical text before they are merged (see complete_full_set).
    """
    log = get_logger()
    config = session.config
    cognitive = config["cognitive"]
    n_ctx = cognitive["n_ctx"]
    n_threads = resolve_thread_count(cognitive["n_threads"], session.topology)
    inference = {**config["inference"], "n_ctx": n_ctx}
    tag = f"[cognitive | {model_key}]"

    log.info("%s loading model...", tag)
    llm, load_time = load_model(model_key, n_threads, inference, logits_all=True, n_batch=n_ctx)
    try:
        tokens = llm.tokenize(raw_text, add_bos=False, special=False)
        total_chunks = len(tokens) // n_ctx
        if chunk_start < 0 or chunk_start >= total_chunks:
            raise ValueError(f"chunk_start {chunk_start} outside the {total_chunks} chunks of the text")
        requested = cognitive["chunks"] if n_chunks is None else n_chunks
        available = total_chunks - chunk_start
        n_chunks = available if requested <= 0 else min(requested, available)
        chunk_end = chunk_start + n_chunks
        half = n_ctx // 2
        n_scored = n_ctx - half
        eval_tokens = tokens[:chunk_end * n_ctx]  # the whole prefix: the hash and the byte check cover it
        log.info("%s %d tokens in the test text, %d chunks; evaluating chunks %d-%d (%d)", tag, len(tokens),
                 total_chunks, chunk_start, chunk_end - 1, n_chunks)

        # Byte length of every token: lets the scored text be measured in bytes (tokenizer-independent)
        byte_len = {t: len(llm.detokenize([t])) for t in set(eval_tokens)}
        eval_bytes = llm.detokenize(eval_tokens)
        roundtrip_ok = (sum(byte_len[t] for t in eval_tokens) == len(eval_bytes)
                        and raw_text.startswith(eval_bytes))
        if not roundtrip_ok:
            log.warning("%s detokenized text does not reproduce the raw bytes exactly: bits_per_byte is "
                        "written as NaN.", tag)

        scored_sha = hashlib.sha256()
        scored_bytes_total = 0
        checkpoint = {}

        def hash_chunk(i):
            nonlocal scored_bytes_total
            scored_ids = tokens[i * n_ctx + half:(i + 1) * n_ctx]
            scored_sha.update(llm.detokenize(scored_ids))
            scored_bytes_total += sum(byte_len[t] for t in scored_ids)
            if hash_checkpoint is not None and i + 1 == hash_checkpoint:
                checkpoint.update({"checkpoint_chunks": hash_checkpoint, "checkpoint_scored_sha256": scored_sha.hexdigest(),
                                   "checkpoint_scored_bytes": scored_bytes_total})

        for i in range(chunk_start):  # hash of the chunks evaluated earlier (no inference)
            hash_chunk(i)
        prefix_scored_bytes = scored_bytes_total

        thermal = session.thermal
        nll = np.empty((n_chunks, n_scored), dtype=np.float64)
        correct_per_chunk = np.zeros(n_chunks, dtype=np.int64)
        bos = llm.token_bos()
        start = time.perf_counter()
        for j, i in enumerate(range(chunk_start, chunk_end)):
            thermal.check()  # Thermal fuse: stop between chunks
            original = tokens[i * n_ctx:(i + 1) * n_ctx]
            scored_ids = original[half:]

            llm.reset()
            llm.eval([bos] + original[1:])

            # The logits at position j predict token j + 1
            logits = llm.scores[half - 1:n_ctx - 1]
            targets = np.array(scored_ids)
            log_probs = log_softmax(logits.astype(np.float64))
            nll[j] = -log_probs[np.arange(n_scored), targets]
            correct_per_chunk[j] = int((logits.argmax(axis=1) == targets).sum())
            hash_chunk(i)

            if (j + 1) % 10 == 0 or j + 1 == n_chunks:
                log.info("%s chunk %d/%d | running PPL %.4f | %.0f s", tag, j + 1, n_chunks,
                         math.exp(nll[:j + 1].mean()), time.perf_counter() - start)
        eval_time = time.perf_counter() - start
        temp = thermal.stats(start, start + eval_time)
        n_vocab = llm.n_vocab()
    finally:
        llm.close()

    return {
        "nll": nll, "n_ctx": n_ctx, "n_chunks": n_chunks, "n_threads": n_threads, "n_vocab": n_vocab,
        "n_eval_tokens": n_chunks * n_ctx, "scored_bytes": scored_bytes_total - prefix_scored_bytes,
        "scored_text_sha256": scored_sha.hexdigest(), "text_roundtrip_ok": roundtrip_ok,
        "top1_accuracy": int(correct_per_chunk.sum()) / nll.size, "model_load_time_s": load_time,
        "eval_time_s": eval_time, "eval_tokens_per_sec": n_chunks * n_ctx / eval_time,
        "temp_mean_C": temp.mean_c, "temp_max_C": temp.max_c,
        # Part bookkeeping (not CSV columns): used when two parts are merged
        "chunk_start": chunk_start, "total_chunks": total_chunks, "correct_per_chunk": correct_per_chunk,
        "prefix_scored_bytes": prefix_scored_bytes, "full_scored_bytes": scored_bytes_total, **checkpoint,
    }


def summarize_quality(result):
    """Perplexity with a chunk-level 95 % t-interval, and bits-per-byte.

    Tokens inside a chunk are correlated, so the interval is computed from the chunk means
    (one replicate per chunk) instead of treating every token as independent.
    """
    nll = result["nll"]
    chunks = summarize(nll.mean(axis=1))
    bits_per_byte = (nll.sum() / (math.log(2) * result["scored_bytes"])
                     if result["text_roundtrip_ok"] and result["scored_bytes"] else NAN)
    return {
        "n_scored_tokens": int(nll.size),
        "nll_mean": float(nll.mean()),
        "nll_se_token": float(nll.std(ddof=1) / math.sqrt(nll.size)),  # i.i.d.-token SE, for reference only
        "perplexity": math.exp(nll.mean()),
        "ppl_ci95_low": math.exp(chunks["ci95_low"]),
        "ppl_ci95_high": math.exp(chunks["ci95_high"]),
        "bits_per_byte": bits_per_byte,
    }


def compare_to_reference(result, reference):
    """Paired NLL difference (model - reference) over the same scored tokens."""
    unpaired = {"paired_with_reference": False, "delta_nll_vs_ref": NAN, "delta_nll_ci95_low": NAN,
                "delta_nll_ci95_high": NAN, "delta_ppl_pct_vs_ref": NAN, "delta_bits_per_byte_vs_ref": NAN}
    if reference is None:
        get_logger().warning("No reference NLL is available: paired columns are NaN.")
        return unpaired
    ref_nll, ref_sha = reference
    if ref_nll.shape != result["nll"].shape or ref_sha != result["scored_text_sha256"]:
        get_logger().warning("Reference was scored on different text or tokens (tokenizer or protocol "
                             "mismatch): paired columns are NaN.")
        return unpaired

    delta = result["nll"] - ref_nll
    chunks = summarize(delta.mean(axis=1))
    delta_bpb = (delta.sum() / (math.log(2) * result["scored_bytes"])
                 if result["text_roundtrip_ok"] and result["scored_bytes"] else NAN)
    return {
        "paired_with_reference": True,
        "delta_nll_vs_ref": float(delta.mean()),
        "delta_nll_ci95_low": chunks["ci95_low"],
        "delta_nll_ci95_high": chunks["ci95_high"],
        # The PPL ratio is the exponential of the mean NLL difference
        "delta_ppl_pct_vs_ref": (math.exp(delta.mean()) - 1) * 100,
        "delta_bits_per_byte_vs_ref": delta_bpb,
    }


def load_reference_nll(session, dataset_sha, n_chunks, chunk_start=0):
    """NLL array of the reference model, from this or an earlier run with the identical protocol."""
    cognitive = session.config["cognitive"]
    path = nll_cache_path(session.paths["nll"], MODELS[cognitive["reference"]]["filename"], dataset_sha,
                          cognitive["n_ctx"], n_chunks, chunk_start)
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        return data["nll"], str(data["scored_text_sha256"])


def _write_result(session, model_key, result, dataset_sha, cooldown, nll_path, csv_path, on_written=None):
    """Store the NLL array and append the result row (one interrupt-shielded block)."""
    cognitive = session.config["cognitive"]
    info = MODELS[model_key]
    with defer_interrupt():
        # Saved first: the reference model is paired with its own array (difference 0)
        np.savez(nll_path, nll=result["nll"], scored_text_sha256=result["scored_text_sha256"])
        row = {
            "session_id": session.session_id, "run_id": session.run_id, "timestamp": now_iso(),
            "model_key": model_key, "temp_at_start_C": cooldown["end_c"], "cooldown_wait_s": cooldown["waited_s"],
            "model_file": info["filename"], "method": info["method"], "params_b": info["params_b"],
            "quantization": quantization_of(info["filename"]), "base_model": info["base"],
            "dataset": cognitive["dataset_name"], "dataset_sha256": dataset_sha,
            "stride": result["n_ctx"], "reference": cognitive["reference"],
            **{k: v for k, v in result.items() if k != "nll"},
            **summarize_quality(result),
            **compare_to_reference(result, load_reference_nll(session, dataset_sha, result["n_chunks"],
                                                              result.get("chunk_start", 0))),
        }
        append_rows(csv_path, COGNITIVE_FIELDS, [row])
        if on_written:
            on_written()
    return row


def _log_done(model_key, row, label="DONE"):
    get_logger().info("[cognitive | %s] %s: PPL %.4f [%.4f, %.4f] | %.4f bits/byte | top-1 %.2f %% | "
                      "dPPL vs %s: %+.2f %% | %.0f s | %d chunks", model_key, label, row["perplexity"],
                      row["ppl_ci95_low"], row["ppl_ci95_high"], row["bits_per_byte"], row["top1_accuracy"] * 100,
                      row["reference"], row["delta_ppl_pct_vs_ref"], row["eval_time_s"], row["n_chunks"])


def run_cognitive_unit(session, model_key, raw_text, dataset_sha, cooldown, on_written=None):
    """Evaluate one model, store its NLL array and append its row to the results file.

    `on_written` (the campaign's state update) runs inside the same interrupt-shielded block
    as the file writes.
    """
    info = MODELS[model_key]
    result = evaluate_model(session, model_key, raw_text)
    nll_path = nll_cache_path(session.paths["nll"], info["filename"], dataset_sha, result["n_ctx"], result["n_chunks"])
    row = _write_result(session, model_key, result, dataset_sha, cooldown, nll_path, session.paths["cognitive"], on_written)
    _log_done(model_key, row)
    return row


def _previous_row(csv_path, model_key, n_chunks):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["model_key"] == model_key and int(r["n_chunks"]) == n_chunks]
    if not rows:
        raise RuntimeError(f"MERGE_MISMATCH: no row of {model_key} with {n_chunks} chunks in {csv_path}")
    return rows[-1]


def complete_full_set(session, model_key, raw_text, dataset_sha, cooldown, previous, on_written=None):
    """Evaluate the chunks an earlier run did not cover and merge both parts into a full-set result.

    `previous` = {"nll_dir", "results_csv", "n_chunks", "overlap"} describes the earlier part (chunks
    [0, n_chunks)). The new pass starts `overlap` chunks before its end, so the overlapping chunks are
    evaluated twice; the merge is accepted only when (a) the scored-text hash and byte count after chunk
    n_chunks equal the earlier part's values (identical tokenization and text) and (b) the overlapping NLL
    rows are bit-for-bit identical (identical inference). Otherwise MERGE_MISMATCH is raised and the unit fails.
    Writes: the new part (row in cognitive_results.csv, `_s<start>` NLL file) and the merged full set
    (row in cognitive_results_full.csv, NLL file of all chunks).
    """
    log = get_logger()
    info = MODELS[model_key]
    n_prev, overlap = int(previous["n_chunks"]), int(previous["overlap"])
    if not 0 <= overlap <= n_prev:
        raise ValueError(f"overlap {overlap} must be within [0, {n_prev}]")
    start = n_prev - overlap
    result = evaluate_model(session, model_key, raw_text, chunk_start=start, hash_checkpoint=n_prev)
    prev_path = nll_cache_path(previous["nll_dir"], info["filename"], dataset_sha, result["n_ctx"], n_prev)
    if not os.path.exists(prev_path):
        raise RuntimeError(f"MERGE_MISMATCH: earlier NLL file missing: {prev_path}")
    with np.load(prev_path) as data:
        prev_nll, prev_sha = data["nll"], str(data["scored_text_sha256"])
    prev_row = _previous_row(previous["results_csv"], model_key, n_prev)

    # (a) same text: hash and bytes after chunk n_prev; (b) same inference: overlapping chunks identical
    same_text = (result.get("checkpoint_scored_sha256") == prev_sha
                 and int(result.get("checkpoint_scored_bytes", -1)) == int(prev_row["scored_bytes"]))
    overlap_prev, overlap_new = prev_nll[start:n_prev], result["nll"][:overlap]
    identical = bool(np.array_equal(overlap_prev, overlap_new))
    max_diff = float(np.abs(overlap_prev - overlap_new).max()) if overlap else 0.0
    log.info("[cognitive | %s] MERGE CHECK: text hash/bytes after chunk %d %s; overlap chunks %d-%d bit-identical: %s "
             "(max |dNLL| %.3g)", model_key, n_prev, "MATCH" if same_text else "DIFFER", start, n_prev - 1,
             identical, max_diff)
    verdict = {"model_key": model_key, "previous_chunks": n_prev, "overlap_chunks": overlap, "same_text": same_text,
               "overlap_identical": identical, "overlap_max_abs_diff": max_diff,
               "new_chunks": int(result["n_chunks"]) - overlap, "total_chunks": int(result["total_chunks"])}
    session.metadata.setdefault("cognitive_merge_checks", []).append(verdict)
    if not (same_text and identical):
        raise RuntimeError(f"MERGE_MISMATCH for {model_key}: same_text={same_text}, overlap_identical={identical}")

    # The new part as measured (its own row and NLL file)
    part_path = nll_cache_path(session.paths["nll"], info["filename"], dataset_sha, result["n_ctx"], result["n_chunks"], start)
    part_row = _write_result(session, model_key, result, dataset_sha, cooldown, part_path, session.paths["cognitive"])
    _log_done(model_key, part_row, label=f"PART chunks {start}-{start + result['n_chunks'] - 1}")

    # The merged full set: earlier chunks + the new non-overlapping chunks
    merged_nll = np.concatenate([prev_nll, result["nll"][overlap:]])
    n_full = merged_nll.shape[0]
    prev_correct = round(float(prev_row["top1_accuracy"]) * prev_nll.size)
    eval_time = float(prev_row["eval_time_s"]) + result["eval_time_s"]
    merged = {
        **{k: v for k, v in result.items() if k not in ("nll", "correct_per_chunk")},
        "nll": merged_nll, "n_chunks": n_full, "n_eval_tokens": n_full * result["n_ctx"],
        "scored_bytes": result["full_scored_bytes"], "scored_text_sha256": result["scored_text_sha256"],
        "text_roundtrip_ok": result["text_roundtrip_ok"] and prev_row["text_roundtrip_ok"] in ("True", "1", "true"),
        "top1_accuracy": (prev_correct + int(result["correct_per_chunk"][overlap:].sum())) / merged_nll.size,
        "eval_time_s": eval_time, "eval_tokens_per_sec": n_full * result["n_ctx"] / eval_time,
        "temp_max_C": max(float(prev_row["temp_max_C"]), result["temp_max_C"]), "chunk_start": 0,
    }
    full_path = nll_cache_path(session.paths["nll"], info["filename"], dataset_sha, result["n_ctx"], n_full)
    row = _write_result(session, model_key, merged, dataset_sha, cooldown, full_path, session.paths["cognitive_full"],
                        on_written)
    _log_done(model_key, row, label=f"FULL SET ({n_prev} earlier + {n_full - n_prev} new chunks)")
    return row


def order_reference_first(model_keys, reference):
    """The reference must be evaluated first so the other models can be paired with it."""
    return sorted(model_keys, key=lambda key: key != reference)


if __name__ == "__main__":
    import sys

    from run_campaign import main as campaign_main

    sys.exit(campaign_main(default_experiments=["cognitive"]))
