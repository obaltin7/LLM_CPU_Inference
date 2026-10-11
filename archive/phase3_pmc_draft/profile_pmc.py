# DRAFT: This script has not yet been validated against a real xperf trace.
# Requirements: administrator privileges and the Windows Performance Toolkit (xperf.exe on PATH).
# On first run, the xperf command line and the dump format (parse_pmc_dump) must be checked.
# See docs/profiling_design.md for the design rationale.
import argparse
import csv
import ctypes
import os
import re
import shutil
import subprocess
import time
from datetime import datetime

import psutil
from llama_cpp import Llama

from llm_cpu_inference.models import MODELS, available_models, model_path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROFILE_CSV = os.path.join(PROJECT_ROOT, "profile_results.csv")
TRACE_DIR = os.path.join(PROJECT_ROOT, "traces")

# Built-in Windows PMC source names (listed with "wpr -pmcsources")
PMC_COUNTERS = ["LLCMisses", "LLCReference", "InstructionRetired"]

PROFILE_FIELDS = [
    "timestamp", "quantization", "n_threads", "max_tokens",
    "generated_tokens", "tokens_per_sec",
    "llc_misses", "llc_references", "instructions",
    "llc_miss_rate", "llc_mpki", "llc_misses_per_token",
]

PID_PATTERN = re.compile(r"\(\s*(\d+)\)")


class PmcTrace:
    """ETW kernel session that records the hardware counters at every context switch (CSWITCH)."""

    def __init__(self, etl_path, counters=PMC_COUNTERS):
        self.etl_path = etl_path
        self.counters = counters

    def __enter__(self):
        subprocess.run(
            ["xperf", "-on", "PROC_THREAD+LOADER+CSWITCH",
             "-pmc", ",".join(self.counters), "CSWITCH", "strict"],
            check=True
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        subprocess.run(["xperf", "-d", self.etl_path], check=True)
        return False


def parse_pmc_dump(dump_path, pid):
    """Extracts the counter totals belonging to the given process from an xperf text dump.

    PMCs are cumulative per-core counters: the difference between two consecutive
    context switches on the same CPU belongs to the thread that ran in that
    interval (the one leaving the CPU at the switch).
    """
    totals = {}
    last_by_cpu = {}
    pmc_names = None
    pending = None  # Values of the Pmc line immediately preceding a CSwitch line
    old_process_col = cpu_col = None

    with open(dump_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]

            if parts[0] == "Pmc":
                if parts[1] == "TimeStamp":  # Header line: gives the counter names
                    pmc_names = parts[3:]
                    totals = dict.fromkeys(pmc_names, 0)
                elif pmc_names:
                    pending = [int(v) for v in parts[3:3 + len(pmc_names)]]

            elif parts[0] == "CSwitch":
                if parts[1] == "TimeStamp":  # Header line: gives the column positions
                    old_process_col = next(i for i, p in enumerate(parts)
                                           if p.startswith("Old Process Name"))
                    cpu_col = parts.index("CPU")
                    continue
                if pending is None or cpu_col is None:
                    continue

                cpu = parts[cpu_col]
                previous = last_by_cpu.get(cpu)
                match = PID_PATTERN.search(parts[old_process_col])
                if previous is not None and match and int(match.group(1)) == pid:
                    for name, now, before in zip(pmc_names, pending, previous):
                        totals[name] += now - before

                last_by_cpu[cpu] = pending
                pending = None

    return totals


def profile_model(key, prompt, max_tokens, n_threads):
    """Runs a single inference pass under a PMC trace (no energy is measured)."""
    print(f"--- Cache Profile: {key} ---")
    llm = Llama(
        model_path=model_path(key),
        n_ctx=512,
        n_threads=n_threads,
        n_threads_batch=n_threads,
        seed=42,
        use_mmap=False,
        verbose=False
    )

    # Warm-up pass outside the trace: model loading and first-touch misses are not counted
    llm(prompt, max_tokens=max_tokens, temperature=0.0, echo=False)
    llm.reset()

    os.makedirs(TRACE_DIR, exist_ok=True)
    etl_path = os.path.join(TRACE_DIR, f"pmc_{key}.etl")

    with PmcTrace(etl_path):
        start = time.perf_counter()
        output = llm(prompt, max_tokens=max_tokens, temperature=0.0, echo=False)
        elapsed = time.perf_counter() - start
    llm.close()

    dump_path = etl_path.replace(".etl", ".txt")
    subprocess.run(["xperf", "-i", etl_path, "-o", dump_path], check=True)
    totals = parse_pmc_dump(dump_path, os.getpid())

    tokens = output['usage']['completion_tokens']
    misses = totals["LLCMisses"]

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "quantization": key,
        "n_threads": n_threads,
        "max_tokens": max_tokens,
        "generated_tokens": tokens,
        # Compared against the speed in the energy test: a marked difference indicates
        # that the tracing overhead distorts the workload.
        "tokens_per_sec": round(tokens / elapsed, 4),
        "llc_misses": misses,
        "llc_references": totals["LLCReference"],
        "instructions": totals["InstructionRetired"],
        "llc_miss_rate": round(misses / totals["LLCReference"], 4),
        "llc_mpki": round(misses / totals["InstructionRetired"] * 1000, 4),
        "llc_misses_per_token": round(misses / tokens),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L3 cache miss profile of LLM inference (draft).")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=None,
                        help="Models to profile (default: all registered models present on disk)")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--output", default=PROFILE_CSV)
    args = parser.parse_args()

    if not ctypes.windll.shell32.IsUserAnAdmin():
        raise SystemExit("Error: PMC tracing requires administrator privileges. Open the terminal as administrator.")
    if shutil.which("xperf") is None:
        raise SystemExit("Error: xperf not found. Install the Windows Performance Toolkit (Windows ADK).")

    test_prompt = "Explain the architecture of a CPU in detail."
    n_threads = psutil.cpu_count(logical=False)

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_FIELDS)
        if write_header:
            writer.writeheader()

        for key in available_models(args.models):
            row = profile_model(key, test_prompt, args.max_tokens, n_threads)
            writer.writerow(row)
            print(f"L3 miss rate: {row['llc_miss_rate']:.2%} | "
                  f"MPKI: {row['llc_mpki']:.2f} | "
                  f"Misses per token: {row['llc_misses_per_token']:,} | "
                  f"Speed: {row['tokens_per_sec']:.2f} tokens/s")

    print(f"Results appended to: {args.output}")
