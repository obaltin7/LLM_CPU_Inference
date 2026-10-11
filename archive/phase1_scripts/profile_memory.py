import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import psutil

from llm_cpu_inference.models import MODELS_DIR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_CSV = os.path.join(PROJECT_ROOT, "results", "phase1", "benchmark_results.csv")  # Phase 1 archive (old schema)


def measure_read_bandwidth(n_threads, array_mb=512, repeats=5):
    """Measures the attainable memory bandwidth (GB/s) with multi-threaded sequential reads."""
    # Each thread is given its own array far larger than the L3 cache, so that the
    # reads are served from DRAM rather than the cache (STREAM-like read-only test).
    arrays = [np.ones(array_mb * 1024 * 1024, dtype=np.uint8) for _ in range(n_threads)]
    total_bytes = sum(a.nbytes for a in arrays)

    # The best pass is taken: scheduler noise can only slow the measurement down
    best_time = float("inf")
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        for _ in range(repeats):
            start = time.perf_counter()
            # np.max on uint8 runs with SIMD and releases the GIL: the compute cost per
            # byte is so low that the speed is set by memory, not by the processor.
            list(pool.map(np.max, arrays))
            best_time = min(best_time, time.perf_counter() - start)

    return total_bytes / best_time / 1e9


def measure_peak_bandwidth():
    """Returns the bandwidths measured with different thread counts and the highest of them."""
    n_physical = psutil.cpu_count(logical=False)
    by_threads = {n: measure_read_bandwidth(n) for n in sorted({1, 2, 4, n_physical}) if n <= n_physical}
    return by_threads, max(by_threads.values())


def model_size_gb(model_name):
    return os.path.getsize(os.path.join(MODELS_DIR, model_name)) / 1e9


def effective_bandwidth(model_name, tokens_per_sec):
    """Approximate memory traffic required by inference (GB/s).

    For every generated token the full set of weights is read once, so the
    traffic is ~ model size x tokens/s. Since the input embedding table is not
    read in full, the value slightly overestimates the real traffic (by about 6%).
    """
    return model_size_gb(model_name) * tokens_per_sec


def run_bandwidth_analysis(csv_path, run_id=None):
    """Compares the measured speeds with the machine's peak memory bandwidth."""
    print("Measuring memory bandwidth...")
    by_threads, peak = measure_peak_bandwidth()
    for n_threads, gbs in by_threads.items():
        print(f"{n_threads:>2} threads: {gbs:.1f} GB/s")
    print(f"Peak read bandwidth: {peak:.1f} GB/s\n")

    df = pd.read_csv(csv_path, dtype={"run_id": str})
    if run_id is None:
        run_id = df["run_id"].iloc[-1]
    run_df = df[df["run_id"] == run_id]

    print(f"Test run: {run_id}")
    print(f"{'Model':<12}{'Size (GB)':>12}{'Tokens/s':>10}{'Required (GB/s)':>16}{'Usage':>10}")
    for (model_name, quant), group in run_df.groupby(["model", "quantization"], sort=False):
        tps = group["tokens_per_sec"].mean()
        needed = effective_bandwidth(model_name, tps)
        print(f"{quant:<12}{model_size_gb(model_name):>12.2f}{tps:>10.2f}{needed:>16.1f}"
              f"{needed / peak:>10.0%}")

    print("\nAs usage approaches 100%, speed is limited by memory bandwidth rather than by the processor.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory bandwidth bottleneck analysis.")
    sub = parser.add_subparsers(dest="command", required=True)

    bw = sub.add_parser("bandwidth", help="Measure peak bandwidth and compare it with the benchmark results")
    bw.add_argument("--input", default=RESULTS_CSV, help="Output of benchmark_energy.py")
    bw.add_argument("--run-id", default=None, help="Test run to examine (default: the most recent run)")

    args = parser.parse_args()
    run_bandwidth_analysis(args.input, args.run_id)
