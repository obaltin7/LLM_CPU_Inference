"""Thread-scaling experiment: shortcut for `python run_campaign.py --experiments thread_scaling`.

The thread list is taken from configs/experiment.yaml and resolved against the detected core
topology ([1, 2, 4, 6, 8, 12, 16] on the i9-11900K). Every (model, thread count) pair is one
configuration; configurations are interleaved round by round like the energy experiment. The
highest thread counts get a stricter thermal check (see thermal.high_thread_* in the config).

Outputs (appended, never rewritten):
  results/thread_scaling_iterations.csv   one row per (session, round, model, threads, prompt)
  results/thread_scaling_summary.csv      per session: mean / std / 95 % t-interval per configuration
  results/metadata/<run_id>.json
"""
import sys

import _paths  # noqa: F401  (makes src/ importable without installing the package)
from run_campaign import main as campaign_main

if __name__ == "__main__":
    sys.exit(campaign_main(default_experiments=["thread_scaling"]))
