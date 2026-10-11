# Night Run Protocol (night_01)

Starts with a single command and runs unattended until morning. Blocks: **A1** (8 models × 2/4/8 threads, 5 rounds) →
**A5** (8B Q4/Q5/Q6/Q8, 10 rounds) → **B1-pre** (heat test with Turbo on; if it stays below 80 °C, 1B-Q4/8B-Q4/8B-Q8 ×
{Turbo on, off}, 10 rounds) → **A2** (perplexity, 10 models, 90 chunks) → B1-post (off; budget). Estimated duration
≈ 7.9 hours (A1 2.2 / A5 1.0 / B1-pre 1.6 / A2 2.6 / start-up 0.5). All settings in `configs/experiment.yaml` → `night`.
Turbo is restored to 99 % on every exit path (normal completion, Ctrl+C, thermal abort, error).

| # | What it does | Command (PowerShell, single line) | Expected output |
|---|---|---|---|
| 1 | Restart the computer; wait **20 min** idle. IDE, browser, Discord, Steam, OneDrive closed; Windows Update paused. | — | — |
| 2 | Open an **administrator** PowerShell (Start → PowerShell → right-click → Run as administrator) and change to the project directory. | `cd <repo>` | The prompt shows the project directory. |
| 3 | Verify that Turbo is off (99 %). | `powercfg /query SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX \| Select-String "0x00000063"` | Two lines of `0x00000063` (= 99; AC and DC). If not: `powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99; powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99; powercfg /setactive SCHEME_CURRENT` |
| 4 | Is the repository clean and on the `master` branch? | `git status --short; git branch --show-current` | Empty list and `master`. (`--final` does not start with a dirty tree.) |
| 5 | View the plan and the duration estimate (no measurement). | `.\.venv\Scripts\python.exe scripts/night_run.py --list` | `night_01 (new): N units planned ... total ~7.9 h` and minutes per block. |
| 6 | Read the room temperature and start the night run (replace `__` with the value). | `New-Item -ItemType Directory -Force results\night\logs \| Out-Null; .\.venv\Scripts\python.exe -u scripts/night_run.py --final --note "room __ C" *> "results\night\logs\console_$(Get-Date -Format yyyyMMdd_HHmmss).log"` | The window stays empty; the output goes under `results\night\logs\`. |
| 7 | Monitor in a **separate (non-administrator) window**. | `Get-Content (Get-ChildItem <repo>\results\night\logs\night_*.log \| Sort-Object LastWriteTime \| Select-Object -Last 1).FullName -Wait -Tail 40 \| Select-String "Thermal monitor\|System is quiet\|Idle baseline\|Warm-up load\|=== Block\|>>> \[\|<<< \[\|Heat probe\|B1_SKIPPED\|Turbo\|THERMAL\|REFUSED\|FAILED\|complete\|finished"` | In the first 20 minutes, in order: `Thermal monitor: ... now ~30 C` → `System is quiet` → `Idle baseline: CPU package ~14 W` → `Warm-up load ... 300 s` → `Warm-up load done` → `=== Block a1` and `>>> [a1] Unit 1/120`. When B1-pre is reached (after ~3.5 hours): `Heat probe (Turbo on): ... peak XX C`; if XX ≤ 80, the B1 units; if > 80, `B1_SKIPPED_THERMAL`. |
| 8 | In the morning: verify that the run has finished and that Turbo is off. | `Get-Content (Get-ChildItem results\night\logs\night_*.log \| Sort-Object LastWriteTime \| Select-Object -Last 1).FullName -Tail 5; powercfg /query SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX \| Select-String "0x00000063"` | `Turbo restored: maximum processor state 99 %.` and `Night run finished: exit 0 ...`; powercfg shows two lines of `0x00000063`. |
| 9 | Record the outcome and update the analysis. | — | Exit code __, blocks __ (last line of the log), room temperature in the morning __ °C; then run the analysis and figure scripts. |

## Troubleshooting

| Situation | What you see | One-line fix |
|---|---|---|
| Quiet gate | `REFUSED: QUIET_TIMEOUT ...` (exit 2) | Close the application using CPU in Task Manager; rerun the command from step 6 unchanged (it resumes from where it left off). |
| Thermal abort | `THERMAL_ABORT: ...` (exit 3); Turbo is still restored to 99 % | Wait 10 min for cool-down, verify Turbo with step 3, rerun the command from step 6. If it stopped during the B1 heat test, the B1 blocks are skipped anyway. |
| Internet outage | No trace in the log; the run is unaffected (no network needed) | Do nothing; note it in the morning for `results/analysis` (as in session 03). |
| Interruption (power loss, Ctrl+C, error) | The log ends with `Interrupted` / `BLOCK FAILED` | After a restart, do step 3, then run the command from step 6 **unchanged**: completed units are skipped and the block resumes from where it left off. |
| Suspicion that Turbo was left on | `0x00000064` (= 100) in step 3 | `powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99; powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99; powercfg /setactive SCHEME_CURRENT` (this check is performed automatically at the start of the run). |
| Plan mismatch | `REFUSED: the night config no longer produces the plan ...` | Do not change the `night` section of `configs/experiment.yaml`; if you changed it, revert it, or move `results/night/campaign_state.json` aside (new night). |
