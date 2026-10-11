"""Energy and speed benchmark: shortcut for `python run_campaign.py --experiments energy`.

The measurement itself lives in src/llm_cpu_inference/generation.py (prefill and decode measured separately,
interleaved rounds); sessions, resume and thermal safety are handled by run_campaign.py.
Every parameter comes from configs/experiment.yaml.

Outputs (appended, never rewritten):
  results/energy_iterations.csv    one row per (session, round, model, prompt)
  results/energy_summary.csv       per session: mean / std / 95 % t-interval per model and prompt
  results/metadata/<run_id>.json
"""
import sys

import _paths  # noqa: F401  (makes src/ importable without installing the package)
from run_campaign import main as campaign_main

if __name__ == "__main__":
    sys.exit(campaign_main(default_experiments=["energy"]))
