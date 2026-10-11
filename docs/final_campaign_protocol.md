# Final Campaign Protocol (Phase 2 data collection)

Final measurements are taken on a system in which nothing but the measurement tool itself is running.
All parameters are read from `configs/experiment.yaml`.

**Thermal safety comes before reproducibility.** The "High performance" plan is not used;
the "Balanced" plan is used with maximum processor state 99 % (Turbo Boost off). The campaign does not
start if it cannot read the CPU package temperature; it stops safely if the temperature limit is exceeded.

## 0. One-time preparation

1. **PawnIO driver** (for temperature readings; asks for administrator approval):
   `winget install --id namazso.PawnIO --exact`
2. **LibreHardwareMonitor library** (version and SHA256 pinned; placed under `third_party/`):
   `.\.venv\Scripts\python.exe scripts/fetch_lhm.py`
3. **Heat test** (in an administrator PowerShell, 16 threads, 2 minutes). The final campaign does not
   start until the peak temperature has been reported:
   `.\.venv\Scripts\python.exe scripts/heat_test.py`
   Result (6 October 2026, Balanced + 99 %, 8B-Q4_K_M, 16 threads, 123 s): peak package
   temperature **48 °C**, mean 43.6 °C, mean package power 65 W, frequency constant at 3504 MHz.
   The fuse did not trip (16-thread limit 80 °C).

## 1. Protocol: multi-session design

- The campaign is run in **at least 2, preferably 3 independent sessions** (`session_01`, `session_02`, `session_03`).
- Sessions take place **on different days**, and the **computer is restarted** before each session.
- Because the cognitive (perplexity) experiment is deterministic, it is run only in `session_01`.
- If a session is interrupted (Ctrl+C, thermal abort, power loss), **the same command** resumes that session
  from where it left off. A new session starts only with `--new-session` and only if the previous session is complete.
- Within a session, models and thread settings are measured **interleaved by round**: each round measures every
  configuration once, in an order shuffled with a session-specific seed.

## 2. Pre-run checklist (every session)

- [ ] Computer **restarted** and left idle for at least **20 minutes** after boot
      (the campaign itself does not start measuring until 20 minutes after boot).
- [ ] **Power plan "Balanced" + maximum processor state 99 % (Turbo off).** Set and verify:
      `powercfg /setactive SCHEME_BALANCED; powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99; powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99; powercfg /setactive SCHEME_CURRENT`
      To revert (after the campaign is finished; 100 % = Turbo on):
      `powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100; powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100; powercfg /setactive SCHEME_CURRENT`
      In `--final` mode the campaign does not start if this setting differs from what is expected.
- [ ] **IDE closed.** Applications that start at boot, such as the browser, Discord, Steam, Epic Games, OneDrive
      and Kaspersky Password Manager, are closed as well (Task Manager → Startup).
      Idle CPU utilisation should be below 2–3 %.
- [ ] **Windows Update paused** (Settings → Windows Update → Pause updates).
      Because the system is on an Insider build (26300), updates arrive frequently.
- [ ] **Screen saver off, sleep "Never"** (Settings → System → Power). While the campaign is running it also
      blocks sleep itself (`SetThreadExecutionState`) and restores the previous setting when it finishes.
- [ ] Power source stable: desktop, plugged into the mains.
- [ ] BIOS unchanged: XMP off, RAM at 2133 MHz (written to the metadata automatically).
- [ ] **Room temperature measured and written into the command with `--note`**; read again and noted when the run finishes.
- [ ] Repository clean and on the correct branch: `git status` empty, branch `phase2-measurement-hardening`.
      In `--final` mode the campaign does not start with a dirty working tree.
- [ ] The campaign will be started from an **administrator PowerShell** (Start → PowerShell → right-click →
      "Run as administrator"). The temperature sensor cannot be read without administrator privileges.
- [ ] Mouse and keyboard are not touched during the run.

## 3. Starting

Session 1 (in an administrator PowerShell, single line):

```powershell
cd C:\path\to\LLM_CPU_Inference; .\.venv\Scripts\Activate.ps1; New-Item -ItemType Directory -Force results\logs | Out-Null; python -u scripts/run_campaign.py --final --note "room 23 C" *> "results\logs\console_$(Get-Date -Format yyyyMMdd_HHmmss).log"
```

Sessions 2 and 3 (on other days, after a restart): on the same line, add `--new-session`
after `--final`.

The window appears empty; the output goes to two files:

- `results\logs\campaign_session_XX_<run_id>.log` — time-stamped run log (the file to watch)
- `results\logs\console_<date>.log` — the complete console output (including unexpected error messages)

To run in chunks, add `--experiments energy`, `--experiments thread_scaling` or
`--experiments cognitive`. The chunks are parts of the same session; the session counts as complete
once all chunks are finished. To view the plan without measuring: `python scripts/run_campaign.py --list`

Exit codes: 0 OK, 1 some units failed, 2 refused to start, 3 thermal abort,
130 Ctrl+C.

## 4. Monitoring progress

In a second PowerShell window (administrator not required):

```powershell
Get-Content (Get-ChildItem C:\path\to\LLM_CPU_Inference\results\logs\campaign_*.log | Sort-Object LastWriteTime | Select-Object -Last 1).FullName -Wait -Tail 50 | Select-String ">>> Unit|<<< Unit|Cool|THERMAL|REFUSED|WARNING|ERROR|Idle baseline:|Power:|Thermal monitor|complete|Run finished|Interrupted"
```

In the first minutes after starting, the log should show the following; if any of them is wrong, stop with Ctrl+C:

1. `Power: plan Balanced (balanced), max processor state AC 99 % / DC 99 %, turbo disabled: True`
2. `Thermal monitor: ... / CPU Package | now ... C` — a value close to room temperature.
3. `Energy tracking: CPU source = windows_emi (measured: True)`
4. `System is quiet: mean CPU utilisation ... %` — the quiet gate has been passed. If it stops with
   `QUIET_TIMEOUT`, something is running in the background: find it in Task Manager, close it, and restart the
   same command (the session resumes from where it left off).
5. No `> 5 %` warning should appear after the `Idle baseline: ...` line.
6. `Warm-up load (not measured): ... 300 s` and, 5 minutes later, `Warm-up load done`; then the first unit.

At the start of the run, in order: wait until 20 minutes have passed since boot → 15 s wait and cool-down to 55 °C →
quiet gate (mean of the last 60 s < 2 %, at most 15 min) → 60 s idle measurement → unmeasured 5 min
warm-up load → rounds.

## 5. Duration estimate (interleaved order, Turbo off, single-prompt warm-up)

Each unit reloads the model, performs **one warm-up generation** with the first prompt (not recorded) and then
**three measured generations**: 4 generations (100 tokens each) + loading per unit.

Basis:

- **Generation times** are taken from the 100-token measurements of Phase 1. Because this workload is bound by
  memory bandwidth, turning Turbo off does not noticeably change the speed at 8 threads:
  in the smoke test 8B-Q4_K_M measured 6.4 tok/s and 1B-Q4_K_M 35.8 tok/s (20 tokens).
- **Cool-down wait ≈ 0.** In the heat test (16 threads, 2 minutes) the package rose to at most 48 °C,
  and in the smoke test no unit waited for the 55 °C target. This may change if the room temperature is high;
  in that case at most 5 minutes per unit are added to the duration.
- The **1- and 2-thread** durations are estimates (assumed slower than Phase 1 because the compute share
  increases with Turbo off, ±20 %).
- The **cognitive** evaluation measured about 7 % slower than Phase 1 in the smoke test with real sensors.

**Energy — 8 models × 10 rounds = 80 units**

| Model | Per generation | Loading | Unit (4 generations) |
|---|---|---|---|
| 8B-Q4_K_M | ~17.3 s | ~5 s | ~74 s |
| 8B-Q8_0 | ~28.9 s | ~7 s | ~123 s |
| 3B-Q4_K_M | ~7.9 s | ~2.5 s | ~34 s |
| 3B-Q8_0 | ~12.7 s | ~3.5 s | ~54 s |
| 1B-Q4_K_M | ~3.1 s | ~1 s | ~14 s |
| 1B-Q8_0 | ~5.2 s | ~1.5 s | ~22 s |
| Sparse8B-Q4_K_M | ~17.3 s | ~5 s | ~74 s |
| Sparse8B-Q8_0 | ~29.0 s | ~7 s | ~123 s |
| One round | | | ~8.7 min |
| 10 rounds + start-up (15 s wait, 60 s idle measurement) | | | **~1 h 30 min** |

**Thread scaling — 8B-Q4_K_M, 7 settings × 10 rounds = 70 units**

| Threads | Per generation | Unit (4 generations + loading) |
|---|---|---|
| 1 | ~46 s | ~3.1 min |
| 2 | ~24 s | ~1.7 min |
| 4 | ~19 s | ~1.4 min |
| 6, 8, 12, 16 | ~17.5 s | ~1.3 min (each) |
| One round | | ~11.3 min |
| 10 rounds + start-up | | **~1 h 55 min** |

**Cognitive — session 1 only, 8 models × 60 chunks**

| Model | Duration |
|---|---|
| 8B-Q8_0 (reference) | ~21 min |
| 8B-Q4_K_M | ~12.5 min |
| 3B-Q4_K_M | ~6 min |
| 3B-Q8_0 | ~10 min |
| 1B-Q4_K_M | ~2.5 min |
| 1B-Q8_0 | ~4 min |
| Sparse8B-Q4_K_M | ~12 min |
| Sparse8B-Q8_0 | ~19.5 min |
| **Total (including loads)** | **~1 h 30 min** |

**Actual (session_01, 7 October 2026):** 4 h 20 min in total — energy 82 min, thread
scaling 98 min, cognitive 78 min; no cool-down wait occurred at all. The estimate (4 h 55 min) was 12 % too high.

| Session | Content | Estimated duration | To reserve in the calendar |
|---|---|---|---|
| Session 1 | energy + thread scaling + cognitive | actual **4 h 20 min** | — |
| Session 2 | restart + 20 min wait, then: start-up guards (~8 min: quiet gate ≥1 min, idle measurement 1 min, warm-up load 5 min) + energy (~82 min) + thread scaling (~98 min) | **~3 h 10 min** (+20 min wait) | 4 hours |
| Session 3 (preferably) | same as Session 2 | **~3 h 10 min** (+20 min wait) | 4 hours |

The quiet gate can wait up to 15 minutes; if something is running in the background, start-up takes longer or
the run stops with `QUIET_TIMEOUT`.

Session 1 can be split into chunks (same session, different times): `--experiments energy`, then
`--experiments thread_scaling`, then `--experiments cognitive`. Each chunk takes its own idle measurement.

## 6. Interruption, thermal abort and resumption

- **Ctrl+C is safe:** no half-written line is produced; the unit running at that moment is discarded.
- **Thermal fuse:** if the package temperature stays above 85 °C for 10 s (80 °C at 16 threads),
  reaches 95 °C in a single reading, or cannot be read for 5 s, the run stops, `THERMAL_ABORT` is written
  to the log, and the exit code is 3. A 16-thread unit does not start until the package has dropped to
  55 °C (if it does not drop within 5 min, again `THERMAL_ABORT`).
- After a thermal abort: wait for the machine to cool down, check the cooling (fans, airflow, room
  temperature), then run **the same command** again; it resumes from where it left off.
- If a unit fails for any other reason, the campaign does not stop; the error is written to the log and the
  unit is retried on the next start.
- If the measurement parameters in `configs/experiment.yaml` are changed while a session is in progress,
  resumption is refused (a session is measured with a single protocol).

## 7. After it finishes

- The last lines of the log: `session_XX is complete.` and `Run finished ...: N completed, 0 failed`
- In `results\energy_summary.csv` and `results\thread_scaling_summary.csv`, the rows of the relevant session
  should have `n_throttled` 0 and `deterministic` `True`.
- Within-session and between-session variability (after at least 2 sessions):
  `python scripts/analyze_sessions.py` and `python scripts/analyze_sessions.py --experiment thread_scaling`
- Note the room temperature again. Once the campaign is completely finished, use the revert command in
  Section 2 to turn Turbo back on.
