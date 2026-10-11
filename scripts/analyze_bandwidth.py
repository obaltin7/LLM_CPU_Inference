"""Memory-bandwidth utilisation of the decode phase, from the existing campaign data (no new measurement).

Effective bandwidth = (bytes read per generated token) x (decode tokens/s). Two estimates of the bytes
per token (both from the GGUF tensor headers):

    estimate_1  every weight tensor once per token (the whole file minus the GGUF header)
    estimate_2  estimate_1 corrected for the embedding table and the KV cache:
                - models with a separate output.weight (Llama-3-8B, Sparse-Llama-3.1-8B): token_embd is
                  an embedding lookup, only one row is read per token -> token_embd removed;
                - models with tied embeddings (Llama-3.2-1B/3B, no output.weight): token_embd IS the
                  output projection and is read in full every token -> nothing removed;
                - plus the KV-cache bytes read per token: context_length x layers x 2 (K and V)
                  x n_head_kv x head_dim x 2 bytes (f16), averaged over the generated tokens.

The theoretical peak is channels x bus width x transfer rate, from the memory modules recorded in the run
metadata (Win32_PhysicalMemory): 2 x 64 bit x 2133 MT/s = 34.1 GB/s (decimal GB).

    python analyze_bandwidth.py            # writes results/analysis/bandwidth_*.csv and gguf_tensor_sizes.csv
"""
import argparse
import glob
import json
import os

import pandas as pd

import _paths  # noqa: F401  (makes src/ importable without installing the package)
from analyze_campaign import session_values
from llm_cpu_inference.models import MODELS, model_path
from llm_cpu_inference.config import PROJECT_ROOT

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_ORDER = ["1B-Q4_K_M", "1B-Q8_0", "3B-Q4_K_M", "3B-Q8_0", "8B-Q4_K_M", "8B-Q5_K_M", "8B-Q6_K", "8B-Q8_0",
               "Sparse8B-Q4_K_M", "Sparse8B-Q8_0"]
KV_BYTES_PER_ELEMENT = 2  # llama.cpp default KV cache type: f16
GENERATED_TOKENS = 100

# Rough compute roof for the roofline sketch (AVX2, Turbo off): cores x 3.5 GHz x 2 FMA units x 8 fp32 lanes x 2 FLOP
PEAK_GFLOPS_FP32 = 8 * 3.5 * 2 * 8 * 2  # = 896 GFLOP/s, an upper bound; quantized kernels do not reach it


# ---------------------------------------------------------------- hardware
def memory_peak_from_metadata(results_dir):
    """Channels, bus width and transfer rate of the installed modules -> theoretical peak in GB/s."""
    meta_paths = sorted(glob.glob(os.path.join(results_dir, "metadata", "*.json")))
    meta = json.load(open(meta_paths[-1], encoding="utf-8"))
    modules = meta["hardware"]["memory"]["modules"]
    channels = len({m["slot"].split("-")[1] for m in modules})  # "Controller0-ChannelA-DIMM1" -> ChannelA
    mt_s = meta["hardware"]["memory"]["configured_speed_mhz"]
    bus_bytes = 8  # 64-bit DDR4 channel
    peak_gbs = channels * bus_bytes * mt_s * 1e6 / 1e9
    return pd.DataFrame([{
        "modules": len(modules), "slots": "; ".join(m["slot"] for m in modules), "channels": channels,
        "bus_width_bits": bus_bytes * 8, "transfer_rate_mt_s": mt_s,
        "formula": f"{channels} ch x {bus_bytes} B x {mt_s} MT/s", "theoretical_peak_gbs": peak_gbs,
        "part_number": modules[0]["part_number"], "source_metadata": os.path.basename(meta_paths[-1]),
    }])


# ---------------------------------------------------------------- GGUF
def gguf_tensor_sizes(keys=MODEL_ORDER):
    """Weight bytes and architecture parameters from the GGUF headers (needs the model files)."""
    from gguf import GGUFReader  # local import: only needed when the models are on disk
    rows = []
    for key in keys:
        path = model_path(key)
        if not os.path.exists(path):
            continue
        reader = GGUFReader(path)
        fields = {f.name: f for f in reader.fields.values()}
        arch = reader.fields["general.architecture"].parts[-1].tobytes().decode()

        def scalar(name):
            f = fields.get(f"{arch}.{name}")
            return int(f.parts[f.data[0]].tolist()[0]) if f is not None else None

        by_name = {t.name: int(t.n_bytes) for t in reader.tensors}
        rows.append({
            "model_key": key, "file_bytes": os.path.getsize(path), "tensors_bytes": sum(by_name.values()),
            "token_embd_bytes": by_name.get("token_embd.weight", 0),
            "output_weight_bytes": by_name.get("output.weight", 0),
            "tied_embeddings": "output.weight" not in by_name,
            "layers": scalar("block_count"), "n_embd": scalar("embedding_length"),
            "n_head": scalar("attention.head_count"), "n_head_kv": scalar("attention.head_count_kv"),
        })
    return pd.DataFrame(rows)


def bytes_per_token(sizes, prompt_tokens=15, generated=GENERATED_TOKENS):
    """Both estimates of the bytes read per generated token (see module docstring)."""
    s = sizes.copy()
    s["estimate_1_bytes"] = s["tensors_bytes"]
    embedding_read = s["token_embd_bytes"].where(~s["tied_embeddings"], 0)  # removed only when untied
    head_dim = s["n_embd"] // s["n_head"]
    # Mean context length over the generated tokens: prompt + (1 + generated) / 2
    mean_ctx = prompt_tokens + (generated + 1) / 2
    s["kv_cache_bytes"] = (mean_ctx * s["layers"] * 2 * s["n_head_kv"] * head_dim * KV_BYTES_PER_ELEMENT).round()
    s["estimate_2_bytes"] = s["estimate_1_bytes"] - embedding_read + s["kv_cache_bytes"]
    s["embedding_correction_bytes"] = -embedding_read
    return s


# ---------------------------------------------------------------- effective bandwidth
def speed_by_model(energy):
    v = session_values(energy, "tokens_per_sec").pivot_table(index="model_key", columns="session_id",
                                                              values="tokens_per_sec")
    return pd.DataFrame({"tokens_per_sec": v.mean(axis=1), "tokens_per_sec_sd": v.std(axis=1, ddof=1)})


def speed_by_threads(thread):
    v = session_values(thread, "tokens_per_sec").pivot_table(index="n_threads", columns="session_id",
                                                              values="tokens_per_sec")
    return pd.DataFrame({"tokens_per_sec": v.mean(axis=1), "tokens_per_sec_sd": v.std(axis=1, ddof=1)})


def bandwidth_models(sizes, energy, peak_gbs):
    bpt = bytes_per_token(sizes).set_index("model_key")
    speed = speed_by_model(energy)
    df = bpt.join(speed, how="inner")
    for i in (1, 2):
        df[f"effective_gbs_est{i}"] = df[f"estimate_{i}_bytes"] * df["tokens_per_sec"] / 1e9
        df[f"effective_gbs_est{i}_sd"] = df[f"estimate_{i}_bytes"] * df["tokens_per_sec_sd"] / 1e9
        df[f"utilisation_est{i}_pct"] = 100 * df[f"effective_gbs_est{i}"] / peak_gbs
    # Roofline sketch: FLOP per token ~ 2 x parameters (matrix-vector products), intensity on estimate 2
    df["params_b"] = [MODELS[k]["params_b"] for k in df.index]
    df["flop_per_token"] = 2 * df["params_b"] * 1e9
    df["arithmetic_intensity_flop_per_byte"] = df["flop_per_token"] / df["estimate_2_bytes"]
    df["attained_gflops"] = df["flop_per_token"] * df["tokens_per_sec"] / 1e9
    df["bandwidth_roof_gflops"] = df["arithmetic_intensity_flop_per_byte"] * peak_gbs
    df["theoretical_peak_gbs"] = peak_gbs
    return df.loc[[k for k in MODEL_ORDER if k in df.index]].reset_index()


def bandwidth_threads(sizes, thread, peak_gbs, model_key="8B-Q4_K_M"):
    bpt = bytes_per_token(sizes).set_index("model_key").loc[model_key]
    df = speed_by_threads(thread)
    for i in (1, 2):
        df[f"effective_gbs_est{i}"] = bpt[f"estimate_{i}_bytes"] * df["tokens_per_sec"] / 1e9
        df[f"effective_gbs_est{i}_sd"] = bpt[f"estimate_{i}_bytes"] * df["tokens_per_sec_sd"] / 1e9
        df[f"utilisation_est{i}_pct"] = 100 * df[f"effective_gbs_est{i}"] / peak_gbs
    df["model_key"] = model_key
    df["theoretical_peak_gbs"] = peak_gbs
    return df.reset_index()


def load_sizes(analysis_dir):
    """Tensor sizes from the cached CSV; computed from the model files when the cache is missing."""
    cache = os.path.join(analysis_dir, "gguf_tensor_sizes.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    sizes = gguf_tensor_sizes()
    if sizes.empty:
        raise SystemExit("Error: neither gguf_tensor_sizes.csv nor the GGUF files under models/ were found.")
    return sizes


def main():
    parser = argparse.ArgumentParser(description="Memory-bandwidth utilisation of the decode phase.")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--refresh-gguf", action="store_true", help="Re-read the GGUF headers even if cached")
    args = parser.parse_args()
    analysis_dir = os.path.join(args.results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    cache = os.path.join(analysis_dir, "gguf_tensor_sizes.csv")
    if args.refresh_gguf or not os.path.exists(cache):
        gguf_tensor_sizes().to_csv(cache, index=False)
        print("written:", cache)
    sizes = load_sizes(analysis_dir)

    hardware = memory_peak_from_metadata(args.results_dir)
    peak = float(hardware.loc[0, "theoretical_peak_gbs"])
    energy = pd.read_csv(os.path.join(args.results_dir, "energy_iterations.csv"))
    thread = pd.read_csv(os.path.join(args.results_dir, "thread_scaling_iterations.csv"))

    outputs = {"bandwidth_hardware.csv": hardware,
               "bandwidth_models.csv": bandwidth_models(sizes, energy, peak),
               "bandwidth_threads.csv": bandwidth_threads(sizes, thread, peak)}
    for name, df in outputs.items():
        path = os.path.join(analysis_dir, name)
        df.to_csv(path, index=False, float_format="%.6g")
        print("written:", path)

    pd.set_option("display.width", 200)
    print(f"\nTeorik tepe: {hardware.loc[0, 'formula']} = {peak:.2f} GB/s")
    m = outputs["bandwidth_models.csv"]
    print(m[["model_key", "estimate_1_bytes", "estimate_2_bytes", "tokens_per_sec", "effective_gbs_est1",
             "effective_gbs_est2", "utilisation_est2_pct", "arithmetic_intensity_flop_per_byte"]].round(3).to_string(index=False))
    t = outputs["bandwidth_threads.csv"]
    print(t[["n_threads", "tokens_per_sec", "effective_gbs_est2", "utilisation_est2_pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
