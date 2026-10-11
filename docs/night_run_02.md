# Night Run Protocol 2 (night_02)

Starts with a single command and runs unattended; **there is no time limit**, and no block is cut short for reasons of
duration. Blocks: **B1** (exactly the same design as B1-pre in night_01: heat test with Turbo on, 80 °C check,
1B-Q4/8B-Q4/8B-Q8 × {Turbo on, off}, 8 threads, interleaved random order, 1 warm-up + 10 rounds, fuse at 85 °C with
Turbo on) → **A2-completion** (Turbo off; the remaining chunks of the WikiText-2 test for 10 models: the 90 chunks of
night_01 are not rerun, chunks 88–89 are re-evaluated as an overlap check and compared bit for bit with night_01,
chunks 90–563 are added, merged at chunk level → `cognitive_results_full.csv`). Estimated duration ≈ **16 hours**
(B1 1.7 / A2-completion 14.3 / start-up 0.4). All settings in `configs/experiment.yaml` → `night_02`.
Turbo is restored to 99 % on every exit path (normal completion, Ctrl+C, thermal abort, error).

| # | What it does | Command (PowerShell, single line) | Expected output |
|---|---|---|---|
| 1 | Restart the computer; wait **20 min** idle. IDE, browser, Discord, Steam, OneDrive closed; Windows Update paused. | — | — |
| 2 | Open an **administrator** PowerShell (Start → PowerShell → right-click → Run as administrator) and change to the project directory. | `cd <repo>` | The prompt shows the project directory. |
| 3 | Verify that Turbo is off (99 %). | `powercfg /query SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX \| Select-String "0x00000063"` | Two lines of `0x00000063` (= 99; AC and DC). If not: `powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99; powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99; powercfg /setactive SCHEME_CURRENT` |
| 4 | Is the repository clean and on the `master` branch? Are the night_01 outputs in place? | `git status --short; git branch --show-current; Test-Path results\night\cognitive_results.csv` | Empty list, `master` and `True`. (`--final` does not start with a dirty tree; A2-completion is refused without `results\night\`.) |
| 5 | View the plan and the duration estimate (no measurement). | `.\.venv\Scripts\python.exe scripts/night_run.py --list --night-config night_02` | `night_02 (new): 70 units planned ... total ~16.0 h` and minutes per block (b1_pre ~100, a2_rest ~858). |
| 6 | Read the room temperature and start the run (replace `__` with the value). | `New-Item -ItemType Directory -Force results\night_02\logs \| Out-Null; .\.venv\Scripts\python.exe -u scripts/night_run.py --final --night-config night_02 --note "room __ C" *> "results\night_02\logs\console_$(Get-Date -Format yyyyMMdd_HHmmss).log"` | The window stays empty; the output goes under `results\night_02\logs\`. |
| 7 | Monitor in a **separate (non-administrator) window**. | `Get-Content (Get-ChildItem <repo>\results\night_02\logs\night_*.log \| Sort-Object LastWriteTime \| Select-Object -Last 1).FullName -Wait -Tail 40 \| Select-String "Thermal monitor\|System is quiet\|Idle baseline\|Warm-up load\|=== Block\|>>> \[\|<<< \[\|Heat probe\|B1_SKIPPED\|Turbo\|THERMAL\|REFUSED\|FAILED\|MERGE\|FULL SET\|complete\|finished"` | In the first 20 minutes, in order: `Thermal monitor` → `System is quiet` → `Idle baseline` → `Warm-up load done` → `Heat probe (Turbo on): ... peak XX C` (if XX ≤ 80, the B1 units; if > 80, `B1_SKIPPED_THERMAL`) → `=== Block b1_pre` and `>>> [b1_pre] Unit 1/60`. After ~1.7 hours: `=== Block a2_rest` and, for each model, `MERGE CHECK: ... MATCH; overlap chunks 88-89 bit-identical: True` followed by `FULL SET (90 earlier + 474 new chunks)`. |
| 8 | In the morning / the next day: verify that the run has finished and that Turbo is off. | `Get-Content (Get-ChildItem results\night_02\logs\night_*.log \| Sort-Object LastWriteTime \| Select-Object -Last 1).FullName -Tail 5; powercfg /query SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX \| Select-String "0x00000063"` | `Turbo restored: maximum processor state 99 %.` and `Night run finished: exit 0 ...`; powercfg shows two lines of `0x00000063`. |
| 9 | Record the outcome and update the analysis. | — | night_02 finished: exit code __, blocks __ (last line of the log), room temperature __ °C; then run the analysis and figure scripts. |

## Troubleshooting

| Situation | What you see | One-line fix |
|---|---|---|
| Quiet gate | `REFUSED: QUIET_TIMEOUT ...` (exit 2) | Close the application using CPU in Task Manager; rerun the command from step 6 unchanged (it resumes from where it left off). |
| night_01 output not found | `REFUSED: a2_rest needs the earlier perplexity run in ...` | `results\night\cognitive_results.csv` and `results\night\nll\*_c90.npz` must be in place (tracked in git; restore with `git checkout -- results/night`). |
| Thermal abort | `THERMAL_ABORT: ...` (exit 3); Turbo is still restored to 99 % | Wait 10 min for cool-down, verify Turbo with step 3, rerun the command from step 6. If it stopped during the B1 heat test, the B1 blocks are skipped anyway. |
| Merge mismatch | `MERGE_MISMATCH for <model>: same_text=..., overlap_identical=...`; the unit is `failed`, the run continues | Not expected (same model file, same tokenizer, deterministic inference). If it happens, note it in the morning; no full-set row is written for that model, only the remaining-chunks row stays in `cognitive_results.csv`. |
| Internet outage | No trace in the log; the run is unaffected (no network needed) | Do nothing; note it in the morning. |
| Interruption (power loss, Ctrl+C, error) | The log ends with `Interrupted` / `BLOCK FAILED` | After a restart, do step 3, then run the command from step 6 **unchanged**: completed units are skipped (per model in A2-completion) and the block resumes from where it left off. |
| Suspicion that Turbo was left on | `0x00000064` (= 100) in step 3 | `powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99; powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99; powercfg /setactive SCHEME_CURRENT` (this check is performed automatically at the start of the run). |
| Plan mismatch | `REFUSED: the night config no longer produces the plan ...` | Do not change the `night_02` section of `configs/experiment.yaml`; if you changed it, revert it, or move `results/night_02/campaign_state.json` aside (new run). |
