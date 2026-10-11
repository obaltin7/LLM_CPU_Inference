# Phase 1 archive (frozen)

The files in this folder were produced during Phase 1 and Work Packages 4–6 and are **not modified**.
Phase 2 measurements are written with the new schema to separate files under `results/`.

| File | Content |
|---|---|
| `benchmark_results.csv` | Energy and speed; runs `05b54e9b` (4 October 2026) and `33be986f` (6 October 2026) |
| `cognitive_results.csv` | WikiText-2 perplexity; runs `65074524` and `83541291` |
| `thread_scaling_results.csv` | Thread scaling; run `652d900c` (summary rows only) |
| `combined_results.csv` | Per-model merge of the files above |
| `other_machine/` | CodeCarbon output left over from another machine (an i5-12450H laptop); not thesis data |

The definitions in these files differ from Phase 2:

- `tokens_per_sec` and `joules_per_token` include the prompt-processing (prefill) time and energy.
- Idle power is not subtracted.
- Single prompt, 5 measurement passes (3 in thread scaling); no confidence intervals.
- The measurements were taken with the Windows "Balanced" power plan and with background applications open.
- The sparse model is labelled `2:4 sparse pruning`; according to its model card, knowledge
  distillation was also applied after pruning.
- `scripts/`: the Phase 1 scripts `benchmark_cpu.py` and `profile_memory.py` (old schema) are kept in
  `archive/phase1_scripts/`; they import the model registry from `src/llm_cpu_inference/models.py`, so
  add `src/` to `PYTHONPATH` to run them.
