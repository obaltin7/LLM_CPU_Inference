# results/analysis — cross-session analysis

Produced by `python scripts/analyze_campaign.py` from `results/energy_iterations.csv` and
`results/thread_scaling_iterations.csv` (8 October 2026; sessions 01–03).
The raw measurement files are not modified. In the main comparisons the unit of replication is the
**session**: a session value is the mean of that session's passes (each pass = the mean of three prompts).

| File | Content |
|---|---|
| `session_integrity.csv` | Per-session integrity: completed units, row count, NaN cells, throttle flag, maximum temperature, cool-down waits, warning/error/thermal-stop counts, idle power, number of outlier measurements (\|z\| > 2.5). |
| `session_means_energy.csv`, `session_means_thread_scaling.csv` | Configuration × session: mean J/token (raw and net) and tokens/s side by side; cross-session CV (%). The `_long` versions give the same data in long format. |
| `variance_components.csv` | One-way random-effects ANOVA per configuration (`scripts/analyze_sessions.py`): within-session and between-session CV, share of between-session variance (ICC). Subsets: all sessions; all sessions, passes 3–10; sessions 02–03 only. |
| `session1_deviation.csv` | Percentage deviation of session 01 from the mean of sessions 02–03 (all passes and passes 3–10); for comparison, the difference between sessions 02 and 03. |
| `ranking_by_session.csv`, `ranking_sparse_dense_order.csv` | J/token ranking of the models in the energy experiment, per session; order of the sparse/dense 8B pairs. |
| `quantization_ratio_ci.csv` | Q8_0/Q4_K_M J/token ratio (raw and net) and Q4_K_M/Q8_0 speed ratio; per-session ratios, mean and Student-t 95% confidence interval (direct and on the log scale); n = 3 and n = 2 (sessions 02–03). |
| `sparse_vs_dense_tost.csv` | Equivalence test between sparse 8B and dense 8B (TOST, bounds ±3%, log ratio, 90% confidence interval = two one-sided 95% tests). Verdict: `equivalent` / `not equivalent` / `inconclusive`. |
| `thread_scaling_optimum.csv` | Per session: the thread count giving the lowest J/token (raw and net), the highest speed and the saturation point (the smallest thread count reaching 98% of the highest speed). |
| `session_03_outage_window.csv` | Measurements between 00:00 and 00:06 in session 03 and their within-configuration z-scores (internet outage window). |
| `anomalies_mad.csv` | Rows falling outside median ± 3 × scaled MAD within each (configuration, prompt, session) group (J/token and tokens/s); with deviation percentage, throttle flag, temperature and frequency. |
| `anomaly_sensitivity.csv` | Change in the session means when the flagged J/token rows are removed (sensitivity analysis). |
| `bandwidth_hardware.csv`, `bandwidth_models.csv`, `bandwidth_threads.csv`, `gguf_tensor_sizes.csv` | `scripts/analyze_bandwidth.py`: memory modules and theoretical peak; bytes/token estimates per model and per thread count, effective bandwidth, ratio to peak, roofline quantities; GGUF tensor sizes (cache). |
| `night_integrity.csv`, `night_run_summary.csv` | `scripts/analyze_night.py` (night run `night_01`, 9 October 2026): per-block status, unit/row counts, NaN, throttle, temperatures (Turbo on/off separately), processor state, deterministic configuration; run-level heat-up trial, evidence of the Turbo rollback (log + powercfg), thermal peak, final row. |
| `night_anomalies_mad.csv`, `night_anomaly_sensitivity.csv` | The same MAD rule on the night blocks (groups: block × configuration × prompt; A1 n = 5, A5/B1 n = 10) and sensitivity. |
| `night_thread_scaling.csv`, `night_thread_optimum.csv` | A1: 8 models × 2/4/8 threads; J/token, tokens/s, power (pass mean ± SD), within-model relative values, bandwidth utilisation; per model, the lowest energy / saturation / highest speed and the 2→4, 2→8 changes. |
| `night_quantization_ladder.csv` | A5 + A2: 8B Q4_K_M/Q5_K_M/Q6_K/Q8_0; energy, speed, power, ratios to Q4 (pass-paired Student-t CI, df = 9), bytes/token, bandwidth utilisation, 90-chunk perplexity, bits/byte, ΔNLL paired to Q8 and Q4, extra J per 1% PPL. |
| `night_turbo.csv`, `night_turbo_ratio.csv` | B1: Turbo on/off per-model means (J/token, tokens/s, power, frequency, temperature) and pass-paired on/off ratios with CI; first/last 5 pass consistency; difference between the Turbo-off values and the A1/A5 blocks. |
| `night_perplexity.csv` | A2: 90-chunk perplexity/bits/byte ↔ the 60-chunk campaign subset; rankings, ΔNLL paired to the reference. |
| `night02_integrity.csv`, `night02_run_summary.csv`, `night02_anomalies_mad.csv`, `night02_anomaly_sensitivity.csv`, `night02_turbo.csv`, `night02_turbo_ratio.csv` | The same tables for the second night run (`night_02`, 10 October 2026); `run_summary` includes the merge checks (10/10 digest/byte matches, overlapping chunks bit-identical). |
| `night_turbo_combined.csv`, `night_turbo_combined_ratio.csv` | B1 over two sessions (night_01 + night_02): per state, the mean of the session means ± between-session SD (n = 2); on/off ratios over 20 paired passes (Student-t, df = 19), per-session ratios and their difference, CI with the session as the unit of replication. |
| `night_perplexity_full.csv` | A2 full set (564 chunks, night_02 `cognitive_results_full.csv`) ↔ 90 chunks; rankings, ΔNLL paired to the reference. `night_quantization_ladder.csv` also uses the full-set quality values. |
| `session_03_network_events.csv` | Network connected/disconnected records from the Windows event log (`Microsoft-Windows-NetworkProfile/Operational`, events 10000/10001). Command: `Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-NetworkProfile/Operational'; StartTime='2026-10-07 22:00'; EndTime='2026-10-08 01:30'}`. |
