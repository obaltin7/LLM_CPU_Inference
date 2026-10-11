# Findings Report — Phase 2 Final Campaign

**Date:** 10 October 2026 (first version 8 October). **Data:** `results/` (sessions 01–03, 7–8 October 2026; 158 + 150 + 150
units), `results/night/` (night run `night_01`, 9 October 00:04–07:27; 230 units: A1 120, A5 40, B1-pre 60, A2 10) and
`results/night_02/` (`night_02`, 10 October 00:36–15:33; 70 units: B1 60, A2-completion 10).
**Analysis:** `scripts/analyze_campaign.py`, `scripts/analyze_bandwidth.py`, `scripts/analyze_night.py`,
`scripts/make_figures.py` (`figures/en/`, `tables/en/`; Turkish versions under `figures/tr/`, `tables/tr/`). This document
is the input to the thesis write-up: every finding is given with its number, confidence interval and the associated
figure/table name.

The unit of replication is the **session**: a session value is the mean of 10 rounds, and each round is the mean of
three prompts. "± x" is the standard deviation of the session means (n = 3); "[a; b]" is the Student-t 95 % confidence
interval (for ratios n = 3 sessions, df = 2). Night-run blocks A1/A5 are single sessions; there the unit of replication
is the **round** (A1 5 rounds, A5 10 rounds; CIs are Student-t over rounds). B1 (Turbo) was run in two independent
sessions (night_01, night_02); ratios are computed over 20 paired rounds, and between-session consistency is reported
separately. Perplexity confidence intervals are at chunk level: the main text uses the **full** WikiText-2 test set
(564 chunks, 144,384 scored tokens; the first 90 chunks from night_01, the remainder from night_02, merged at chunk
level); the 60- and 90-chunk subsets are given in Appendix A. Raw energy metrics (J/token, token/s) are reported with
three sessions, **net** energy (idle power subtracted) only with sessions 02–03 (rationale: §4).

Because the research questions are not written out explicitly in the repository, they were derived as follows from
the aim stated in README §1; they may be renumbered according to the thesis numbering.

---

## 1. Research questions

### RQ1 — How does the quantization level (Q4_K_M ↔ Q8_0) change speed, energy and quality?

**Answer.** On the same model, Q8_0 consumes 52–60 % more CPU energy per token and is 37–41 % slower; in return,
perplexity improves by only 1.3 % (8B), 2.4 % (3B) and 3.2 % (1B) (full set, 564 chunks). When the intermediate levels
(Q5_K_M, Q6_K) are measured on 8B, the trade-off curve turns out to be **non-linear** and forms a knee at Q6_K: Q6_K
delivers the quality of Q8_0 with 19 % less energy; Q8_0 has no measurable quality gain over Q6_K
(ΔNLL 0.0000 [−0.0007; 0.0008]).

**Numbers** (T1, T2; `quantization_ratio_ci.csv`, `night_quantization_ladder.csv`, `night_perplexity_full.csv`):

| Family | J/token Q8_0 / Q4_K_M [95 % CI] | Speed Q4_K_M / Q8_0 [95 % CI] | PPL Q4_K_M → Q8_0 (564 chunks) | ΔNLL (Q4 − Q8, token-paired) |
|---|---|---|---|---|
| 1B | 1.521 [1.475; 1.566] | 1.580 [1.549; 1.612] | 14.47 → 14.00 (−3.2 %) | +0.033 |
| 3B | 1.558 [1.525; 1.591] | 1.636 [1.603; 1.670] | 10.80 → 10.54 (−2.4 %) | +0.024 |
| 8B | 1.600 [1.591; 1.609] | 1.681 [1.675; 1.686] | 8.478 → 8.367 (−1.3 %) | +0.0131 [0.0114; 0.0149] |

- 8B: Q4_K_M 9.728 ± 0.087 J/token, 6.40 ± 0.05 token/s; Q8_0 15.562 ± 0.151 J/token,
  3.81 ± 0.03 token/s. The additional cost of +5.83 J/token (+60 %) buys a perplexity reduction of 1.3 %; the marginal
  cost is ≈ 4.4 J/token per 1 % PPL improvement. Top-1 next-token accuracy 54.56 % → 54.95 %.

**Trade-off curve between Q4 and Q8** (A5 energy: night_01, 8 threads, 10 rounds, round-based CIs with df = 9; quality:
full set, 564 chunks; T2 † rows, fig02 † points):

| 8B | bits/weight | J/token | Ratio J / Q4 [95 % CI] | Token/s | PPL (564 chunks) | ΔPPL / Q4 | ΔNLL / Q8 [95 % CI] | Extra J per 1 % PPL |
|---|---|---|---|---|---|---|---|---|
| Q4_K_M | 4.91 | 9.579 ± 0.026 | 1 | 6.42 | 8.478 | — | +0.0131 [0.0114; 0.0149] | — |
| Q5_K_M | 5.73 | 11.242 ± 0.031 | 1.174 [1.171; 1.177] | 5.62 | 8.403 | −0.88 % | +0.0043 [0.0033; 0.0053] | 1.89 J |
| Q6_K | 6.59 | 12.337 ± 0.076 | 1.288 [1.282; 1.294] | 4.91 | 8.368 | −1.30 % | +0.0000 [−0.0007; 0.0008] | 2.12 J |
| Q8_0 | 8.53 | 15.268 ± 0.027 | 1.594 [1.591; 1.597] | 3.83 | 8.367 | −1.31 % | 0 (reference) | 4.36 J |

- Energy follows the byte count: the J/token ratio (1.17 / 1.29 / 1.59) stays slightly below the corrected bytes/token
  ratio (1.16 / 1.33 / 1.73); Q8_0 is the cheapest per byte (simpler dequantization kernel, 89.5 % bandwidth
  utilization; T7).
- Quality, by contrast, saturates: the Q4 → Q5 step closes two thirds of the ΔNLL (0.0089 [0.0071; 0.0106]), Q5 → Q6
  the remainder (0.0042); the Q6 → Q8 step is zero (0.0000 [−0.0007; 0.0008] over 144,384 tokens) but costs
  +2.93 J/token (+24 %). **The knee is at Q6_K**: for 8B the "lossless" option is Q6_K, not Q8_0 (−19 % energy,
  +28 % speed). On the full set the Q6_K and Q8_0 perplexities differ only in the fourth decimal (8.3675 ↔ 8.3673): a
  statistical tie; on 90 chunks the order was reversed (8.801 ↔ 8.803), which shows that the difference is noise.
- Bandwidth utilization of the new models lies in the same band: Q5_K_M 88.1 %, Q6_K 88.3 % (with speed rescaled to
  the campaign scale; T7), so the finding is consistent with RQ4.
- The Q8_0 / Q4_K_M ratio measured in the same night (1.594 [1.591; 1.597]) overlaps the campaign ratio
  (1.600 [1.591; 1.609]); the anchor coefficients for transferring night data to the campaign scale are J ×1.018,
  speed ×0.996 (mean of the two anchors).
- The speed ratio follows the byte ratio: 8B file-size ratio 8.54 / 4.92 = 1.74, speed ratio 1.68; 3B 1.69 / 1.64;
  1B 1.63 / 1.58 (see RQ4).
- The energy ratio is smaller than the speed ratio (1.52–1.60 ↔ 1.58–1.68): package power is slightly lower with Q8_0
  (59–61 W ↔ 62–63 W; `energy_iterations.csv`), because the kernel spends more time waiting on memory.
- Quantization sensitivity increases as the model gets smaller (PPL loss 1.3 % → 2.4 % → 3.2 %); the energy saving is
  ~35–39 % at every size.

**Figures/tables:** fig01, fig02, T1, T2. **Limitation:** PPL was measured on a single dataset (the full WikiText-2
test set, 564 chunks, 144,384 scored tokens) (deterministic; the ranking and the paired differences are the same with
the 60- and 90-chunk subsets, Appendix A). The Q5_K_M/Q6_K energy values come from a single night session. Q8_0 is the
reference; FP16 was not measured.

### RQ2 — What is the energy–quality trade-off among quantization, pruning + distillation and 2:4 sparse pruning?

**Answer.** Structural downsizing (Llama-3.2 1B/3B) reduces energy almost proportionally to model size but lowers
quality markedly; all six instruct models lie on the Pareto front (none is dominated by another in both energy and
quality). The 2:4 sparse 8B is equivalent to the dense 8B in energy and speed (±3 % TOST); its lower perplexity stems
from its being a base model.

**Numbers** (T1, T3; fig02; perplexity on the full set, 564 chunks, `night_perplexity_full.csv`):

| Model | J/token | Token/s | PPL [95 % CI] | PPL relative to reference | Bits/byte |
|---|---|---|---|---|---|
| 1B-Q4_K_M | 1.758 ± 0.003 | 35.80 ± 0.14 | 14.47 [14.00; 14.96] | +72.9 % | 0.863 |
| 1B-Q8_0 | 2.673 ± 0.035 | 22.66 ± 0.27 | 14.00 [13.54; 14.48] | +67.4 % | 0.852 |
| 3B-Q4_K_M | 4.397 ± 0.042 | 14.24 ± 0.17 | 10.80 [10.45; 11.16] | +29.0 % | 0.768 |
| 3B-Q8_0 | 6.850 ± 0.017 | 8.70 ± 0.04 | 10.54 [10.20; 10.89] | +26.0 % | 0.761 |
| 8B-Q4_K_M | 9.728 ± 0.087 | 6.40 ± 0.05 | 8.478 [8.20; 8.76] | +1.32 % | 0.690 |
| 8B-Q5_K_M † | 11.44 ± 0.03 | 5.60 ± 0.02 | 8.403 [8.13; 8.69] | +0.43 % | 0.687 |
| 8B-Q6_K † | 12.55 ± 0.08 | 4.89 ± 0.02 | 8.368 [8.09; 8.65] | +0.00 % | 0.686 |
| 8B-Q8_0 (reference) | 15.562 ± 0.151 | 3.81 ± 0.03 | 8.367 [8.09; 8.65] | — | 0.686 |
| Sparse8B-Q4_K_M (base) | 9.742 ± 0.133 | 6.39 ± 0.08 | 8.271 [8.02; 8.53] | −1.15 % | 0.682 |
| Sparse8B-Q8_0 (base) | 15.491 ± 0.165 | 3.80 ± 0.03 | 8.117 [7.87; 8.37] | −2.99 % | 0.676 |

† Measured in the night run (single session, 10 rounds); J/token and token/s rescaled to the campaign scale via the
anchor models (RQ1).

- Pareto front (instruct, J/token ↔ PPL): 1B-Q4 → 1B-Q8 → 3B-Q4 → 3B-Q8 → 8B-Q4 → 8B-Q5 → 8B-Q6 → 8B-Q8;
  8B-Q8_0 remains on the front only by a PPL difference of 0.0002 (the ΔNLL CI includes zero; on 90 chunks Q6_K was
  ahead), i.e. the Q6_K ↔ Q8_0 step is a tie. The marginal cost per step increases: 1B-Q4 → 3B-Q4 PPL −25.4 % for
  +2.64 J/token; 3B-Q4 → 8B-Q4 −21.5 % for +5.33 J; 8B-Q4 → 8B-Q6 −1.3 % for +2.82 J; 8B-Q6 → 8B-Q8 0.0 for +3.01 J.
- Size scaling: 1B-Q4 has 1/6.1 of the file size of 8B-Q4, uses 1/5.5 of its energy and runs 5.6 times as fast.
- Sparse ↔ dense (T3, TOST ±3 %, 90 % CI): J/token Q4 +0.13 % [−0.75; +1.02], Q8 −0.46 % [−0.91; 0.00];
  speed Q4 −0.22 % [−0.92; +0.49], Q8 −0.20 % [−0.51; +0.12]. All four comparisons are **equivalent**.
  In session 03 the ranking of 8B-Q4 and Sparse8B-Q4 swaps (difference 0.4 %) — the expected behaviour of
  equivalence.
- The sparse model's ΔNLL is −0.012 [−0.022; −0.001] (Q4) and −0.030 [−0.041; −0.020] (Q8; 564 chunks): this is the
  advantage on WikiText of a base model that has not undergone instruct fine-tuning; on the full set this advantage is
  less than half of that on the subset (−0.040 / −0.059). It does not enter the method comparison.

**Figures/tables:** fig02, T1, T3. **Limitation:** there is no "pruning-only" or "distillation-only" arm
(Llama-3.2 applies both together); the sparse model is a base model and cannot be compared with the instruct models on
the quality axis; GGUF/llama.cpp processes sparsity as dense, so this result means "2:4 sparsity has no payoff in this
deployment pipeline", not a statement about the general value of the method.

### RQ3 — How does the thread count affect speed and energy efficiency?

**Answer.** On 8B-Q4_K_M, speed saturates at 4 threads (6.39 token/s, 98 % of the maximum), the peak speed is at
6 threads (6.47), but **the lowest energy is at 2 threads** (6.70 J/token). The physical core count (8) and
Hyper-Threading (12, 16) bring no speed gain and increase energy.

**Numbers** (T4; the same optimum in all three sessions, `thread_scaling_optimum.csv`):

| Threads | Token/s | J/token | Package power (W) | Bandwidth utilization (RQ4) |
|---|---|---|---|---|
| 1 | 2.919 ± 0.014 | 9.152 ± 0.202 | 26.7 | 40 % |
| 2 | 5.256 ± 0.012 | **6.700 ± 0.045** | 35.2 | 71 % |
| 4 | 6.390 ± 0.011 | 7.102 ± 0.057 | 45.4 | 87 % |
| 6 | **6.473 ± 0.022** | 8.269 ± 0.022 | 53.5 | 88 % |
| 8 | 6.427 ± 0.015 | 9.539 ± 0.065 | 61.3 | 87 % |
| 12 | 6.341 ± 0.037 | 9.970 ± 0.043 | 63.2 | 86 % |
| 16 | 6.293 ± 0.037 | 10.126 ± 0.051 | 63.7 | 85 % |

- 2 → 8 threads: speed +22 %, energy/token +42 %, power +74 %. 8 → 16: speed −2 %, energy +6 %.
- Pareto-optimal settings: 2, 4, 6 (fig04); 1 is both slow and inefficient (the constant uncore power is spread over a
  long duration); 8/12/16 are dominated.
- In all three sessions: lowest J/token at 2 threads (6.66 / 6.75 / 6.69), saturation at 4 threads, peak speed at
  6 threads.

**Thread optimum across all models** (A1, night run: 8 models × 2/4/8 threads, 5 rounds; T10, fig11,
`night_thread_optimum.csv`):

| Model | J/token 2 / 4 / 8 | Token/s 2 / 4 / 8 | Speed 2→4 | Energy 2→8 | BW utilization 2 / 4 / 8 |
|---|---|---|---|---|---|
| 1B-Q4_K_M | 1.19 / 1.29 / 1.68 | 29.2 / 35.3 / 35.5 | +20.9 % | +41 % | 69 / 83 / 83 % |
| 1B-Q8_0 | 1.69 / 1.89 / 2.54 | 20.0 / 22.7 / 22.8 | +13.7 % | +50 % | 77 / 88 / 88 % |
| 3B-Q4_K_M | 3.01 / 3.21 / 4.23 | 11.6 / 14.1 / 14.3 | +21.6 % | +41 % | 69 / 83 / 85 % |
| 3B-Q8_0 | 4.38 / 4.88 / 6.50 | 7.65 / 8.78 / 8.75 | +14.8 % | +48 % | 77 / 88 / 88 % |
| 8B-Q4_K_M | 6.95 / 7.10 / 9.58 | 5.18 / 6.37 / 6.31 | +23.1 % | +38 % | 70 / 86 / 85 % |
| 8B-Q8_0 | 10.01 / 11.15 / 15.12 | 3.34 / 3.83 / 3.81 | +14.6 % | +51 % | 78 / 90 / 89 % |
| Sparse8B-Q4_K_M | 6.68 / 7.13 / 9.48 | 5.24 / 6.34 / 6.37 | +21.0 % | +42 % | 71 / 86 / 86 % |
| Sparse8B-Q8_0 | 9.88 / 11.07 / 14.93 | 3.36 / 3.83 / 3.83 | +14.0 % | +51 % | 79 / 90 / 89 % |

- **The "2 threads is most efficient" finding holds for all eight models**; speed saturation (98 % of the maximum) is
  at 4 threads in all eight; the 4 → 8 speed difference is −1.1 % … +1.7 % (within noise).
- There is **no** relationship with model size: the 2 → 4 speed gain is the same from 1B to 8B (Q4: 21–23 %,
  Q8: 14–15 %). Small models are not closer to compute-bound; even 1B-Q4 uses 69 % of the bandwidth at 2 threads.
- The decisive factor is the quantization format: Q8_0 models are already at 77–79 % bandwidth utilization at
  2 threads (Q4_K_M: 69–71 %), so they gain less speed from additional cores (+14 % ↔ +22 %) and pay a larger energy
  penalty (+48–51 % ↔ +38–42 %); at 8 threads all models sit at the same 83–90 % ceiling (consistent with RQ4).
- Power from 2 → 8 threads rises by +68–74 % on every model (33–36 W → 57–61 W); speed by +14–24 %. The energy penalty
  is the ratio of the two.
- Note: the 8B-Q4_K_M 2-thread mean (6.95 ± 0.63) is inflated by a single outlier round (round 2, `cpu_arch`, +54 %);
  with the outlier rows removed it becomes 6.66 J/token (campaign: 6.70). The optimum and the ranking do not change (§3).

**Figures/tables:** fig03, fig04, fig08, fig11, T4, T10. **Limitation:** 1/6/12/16 threads only on 8B-Q4_K_M
(campaign); the other models at 2/4/8 in a single night session (5 rounds). Turbo off; with Turbo on the power curve
is different (RQ5).

### RQ4 — What is the bottleneck of the decode phase on the CPU; where does the energy go?

**Answer.** Decode is bound by memory bandwidth: all eight models run at an effective read rate of 28.7–30.4 GB/s,
84–89 % of the theoretical peak (34.1 GB/s); thread saturation coincides with the bandwidth ceiling. Once the ceiling
is reached, the cores draw power while waiting for data; this explains the 2-thread optimum and the energy gain of
quantization (fewer bytes → less time).

**Numbers** (T7, `bandwidth_models.csv`, `bandwidth_threads.csv`; method in README §5.13):

- Theoretical peak: 2 channels × 64 bit × 2133 MT/s = 34.13 GB/s (modules `Controller0-ChannelA/B-DIMM1`).
- Effective bandwidth (corrected bytes/token × token/s): 1B-Q4 28.7, 1B-Q8 29.8, 3B-Q4 28.8,
  3B-Q8 29.8, 8B-Q4 29.6, 8B-Q8 30.4, Sparse8B-Q4 29.5, Sparse8B-Q8 30.3 GB/s → 84–89 %.
- By thread count (8B-Q4): 13.5 (40 %) → 24.3 (71 %) → 29.6 (87 %) → 29.9 (88 %) → 29.7 → 29.3 → 29.1 GB/s.
- Night-run additions (T7 †, T10): 8B-Q5_K_M 88.1 % and 8B-Q6_K 88.3 % (30.1 GB/s) lie in the same band; from
  2 → 4 → 8 threads all eight models rise to 69–79 % → 83–90 % → 83–89 % utilization (RQ3 table).
- Roofline: arithmetic intensity 1.8–3.5 FLOP/byte, ridge point ≈ 26 FLOP/byte; all points lie on the bandwidth
  roof (fig09, qualitative).
- Difference from expectation: the 1B models did not show markedly lower utilization (84 % ↔ 87–89 %); rationale in
  `docs/profiling_design.md` §7.2 (the embedding matrix is read in full because it is tied; it does not fit in L3).

**Figures/tables:** fig07, fig08, fig09, T7. **Limitation:** bytes/token is an estimate derived from the GGUF header
(not a measurement); the practically achievable peak was not measured under Phase 2 conditions (Phase 1, Turbo on:
30.9 GB/s); hardware counters were not used (§7.3).

---

### RQ5 — How does Turbo Boost change decode speed and energy? (B1, controlled experiment)

**Answer.** With Turbo Boost enabled (maximum processor state 100 %; measured 4.80 GHz ↔ 3.50 GHz at 99 %), decode
speed increases by **0–2 %**, while CPU package power and energy/token rise to **2.30–2.35 times**. In a workload bound
by memory bandwidth, frequency brings no speed, only power (the RQ4 mechanism). Across three models (1B-Q4, 8B-Q4,
8B-Q8) and two independent sessions the ratios are too close to be distinguished (between-session difference
≤ 1.8 %); disabling Turbo means ≈ −57 % energy for this workload.

**Design.** Two independent night sessions (`night_01` 9 October, `night_02` 10 October; both after a reboot, 20 min
idle wait and a warm-up load), with the same block design: first a 120 s Turbo-on heat trial (8B-Q4, 8 threads;
peak 75 °C / 77 °C < 80 °C threshold → block proceeded), then 3 models × {on, off} × 10 rounds, interleaved within the
same round in shuffled order; a 30 s settling and cool-down wait between units. Ratios are paired within round
(on / off) and pooled over the 20 rounds of the two sessions, CIs are Student-t (df = 19); per-session ratios and their
differences are reported separately. The Turbo state was set with `powercfg` for every unit and written to the
metadata (`max_processor_state` 100 / 99); its restoration to 99 % at the end of every run was verified via the log
(`Turbo restored`) and `powercfg /query` (AC and DC `0x00000063`).

**Numbers** (T9, fig10; `night_turbo_combined.csv`, `night_turbo_combined_ratio.csv`; values are the mean of the two
sessions):

| Model (8 threads) | J/token off → on | Ratio J [95 % CI] | Ratio J session 1 / 2 | Token/s off → on | Ratio speed [95 % CI] | Power (W) off → on | Ratio power [95 % CI] | Temperature mean / peak (°C) off → on |
|---|---|---|---|---|---|---|---|---|
| 1B-Q4_K_M | 1.752 → 4.029 | 2.300 [2.282; 2.317] | 2.279 / 2.320 | 35.8 → 36.5 | 1.020 [1.017; 1.023] | 62.7 → 147.2 | 2.346 [2.334; 2.358] | 46 / 49 → 64 / 75 |
| 8B-Q4_K_M | 9.658 → 22.18 | 2.296 [2.286; 2.307] | 2.291 / 2.302 | 6.43 → 6.52 | 1.014 [1.012; 1.016] | 62.1 → 144.5 | 2.328 [2.320; 2.336] | 46 / 52 → 64 / 81 |
| 8B-Q8_0 | 15.40 → 35.87 | 2.328 [2.323; 2.334] | 2.332 / 2.325 | 3.83 → 3.84 | 1.004 [1.002; 1.005] | 59.0 → 137.8 | 2.337 [2.334; 2.341] | 45 / 54 → 63 / 87 |

- Between-session consistency: J-ratio difference +1.8 % (1B-Q4), +0.5 % (8B-Q4), −0.3 % (8B-Q8); speed ratio
  ≤ 0.5 %; power ratio ≤ 1.3 %; frequency ratio 1.371, identical in every session and model (difference ≤ 0.02 %). If
  the session is taken as the unit of replication (n = 2, df = 1), the J-ratio CIs are 1B [2.03; 2.57],
  8B-Q4 [2.23; 2.36], 8B-Q8 [2.29; 2.37] — the conclusion is unchanged.
- Measured frequency: off 3,504 MHz (minimum 3,498), on 4,803–4,805 MHz (minimum 4,706–4,775). Frequency +37 %,
  speed +0–2 %: decode is insensitive to frequency.
- Power ×2.33–2.35 ↔ frequency ×1.37: power reflects the rise in voltage along with frequency (P ∝ f·V²); on 1B,
  147 W for 36.5 token/s, i.e. all of Turbo's additional power goes into waiting.
- The speed gain decreases from small to large (1B +2.0 % → 8B-Q4 +1.4 % → 8B-Q8 +0.4 %): in the model with the highest
  bandwidth utilization (Q8, 89 %) no contribution of frequency remains.
- Turbo-off values are within +0.5 % / +1.2 % (8B) and +3.9 % (1B-Q4; warmer start after interleaved Turbo-on units)
  of the A1/A5 blocks of the same night (night_01); in night_02 the Turbo-off J/token is +0.6 % / +0.6 % / +1.3 %
  relative to night_01.
- Thermal: with Turbo on, the peaks of 85 °C (night_01) and 87 °C (night_02, 8B-Q8) touched the thermal-stop threshold
  (85 °C, 10 s) momentarily but did not persist for 10 s; throttle_flag 0/360; Turbo-off peak 54 °C in both sessions.
- **The cross-phase comparison is supporting evidence only**: the −59 % energy and 140 → 62 W finding for
  Phase 1 (Turbo on, different window) ↔ Phase 2 (off) (§2) agrees with the controlled experiment (−57 %,
  145 → 62 W).

**Figures/tables:** fig10, T9. **Limitation:** two sessions (20 rounds), three models, 8 threads; the Turbo-on
measurements were taken at 63–65 °C mean temperature (off: 45–46 °C), so the leakage-power difference is included in
the power ratio. The "Turbo off" mechanism used is Windows' 99 % maximum processor state setting (not the BIOS Turbo
switch).

---

## 2. Unexpected and notable findings

1. **Lowest energy at 2 threads** (RQ3): using all 8 cores increases energy/token by 42 % and speed by 22 %.
   For edge devices, the rule is "fewer cores, less energy".
2. **Turbo Boost brings no speed, only power (RQ5, controlled, two sessions):** with Turbo on, frequency ×1.37, speed
   ×1.00–1.02, energy/token ×2.30–2.33 (three models, two independent sessions with ≤ 1.8 % difference). Supporting
   cross-phase observation: Phase 1 (Turbo on, IDE open, different measurement window) gives 24.39 J/token including
   prefill and 5.78 token/s for 8B-Q4; Phase 2 (Turbo off) 10.08 J/token and 6.24 token/s: −59 % energy, package
   power ~140 W → 62 W — the same magnitude as the controlled result.
2b. **Q8_0 has no quality gain over Q6_K (RQ1):** on 8B, Q6_K delivers the same perplexity as Q8_0 (ΔNLL 0.0000
   [−0.0007; 0.0008], 144,384 tokens) with 19 % less energy; the Q4 → Q8 trade-off curve has its knee at Q6_K.
3. **Sparse ↔ dense equivalence** (RQ2): 2:4 sparsity is stored densely in GGUF; the energy and speed differences
   are below 1 % and equivalent within the ±3 % bound.
4. **Bandwidth utilization independent of the model** (RQ4): from 0.8 GB to 8.5 GB all models sit at the same
   ~29–30 GB/s ceiling; Q8_0 kernels show ~3 points higher utilization than Q4_K_M.
5. **Prefill 5–7 times faster than decode** (fig06): 1B-Q4 227 ↔ 36 token/s, 8B-Q4 38 ↔ 6.4 token/s;
   prefill is compute-bound (prompts of 11–18 tokens). Hence, for short prompts, the contribution of prefill to total
   energy is small (8B-Q4: 10.08 including prefill ↔ 9.73 J/token decode, 3.6 %).
6. **High between-session reproducibility**: the difference between sessions 02 and 03 is ≤ 0.43 % (J/token) for all
   models; the generated texts are bit-identical across the three sessions (SHA-256, 24/24 + 21/21
   configurations).
7. **Net energy ratios lower than raw ratios** (8B: 1.575 ↔ 1.600): subtracting idle power (13.8 W) slightly reduces
   the relative cost of Q8_0; reports should state which definition is used.

---

## 3. Anomaly scan (`anomalies_mad.csv`, `anomaly_sensitivity.csv`)

**Method.** In every (configuration, prompt, session) group (n = 10 rounds), rows outside median ± 3 × scaled MAD
(1.4826 × MAD) were flagged. Because the groups are very tight (within-session CV 0.3–2.5 %), the rule also captures
small deviations; therefore deviations ≥ 5 % are additionally reported as a practical threshold.

**Result.**
- throttle_flag: **0 / 1,350** rows (frequency 3504 MHz in all measurements); no thermal stop; highest package
  temperature 59 °C (session 01, cognitive), 53 °C (energy), 50 °C (thread).
- 127 rows (9.4 %) flagged in J/token; **40** deviate by ≥ 5 %, **9** by ≥ 10 %. 121 are positive (excess energy),
  6 negative: the anomalies are almost exclusively energy added by background activity.
- Largest: session 01 round 1 Sparse8B-Q8 `transformer` +23.0 % (00:19, start-up transient); session 03 round 2
  1-thread `transformer` +21.5 % (00:03, network outage); session 01 round 10 8B-Q8 `transformer` +20.2 %
  (01:32); session 01 round 1 Sparse8B-Q4 `story` +19.5 %; session 03 round 1 Sparse8B-Q8 `cpu_arch` +14.2 %
  (22:27, first unit).
- **Temporal clusters:** flagged rows occur in minute-scale clusters, affecting unrelated models in the same
  minutes — session 01 rounds 1–2 (00:19–00:33, the known start-up transient, 15 rows), session 02 round 4
  (17:15), round 7 (19:11–19:16), round 10 (19:38–19:44), session 03 round 7 (23:12–23:18), round 10 (23:36–23:38
  and 01:13–01:17). Likely cause: periodic background processes (the source cannot be identified because no
  process-level CPU log was kept). The network outage (session 03, 00:02:13–00:03:44) is documented via the Windows
  event log.
- **Ranking:** across sessions only the equivalent pair (8B-Q4 ↔ Sparse8B-Q4) swaps places.
- **Sensitivity:** removing the flagged rows changes the session means by at most −1.96 % (Sparse8B-Q4,
  session 01), and in sessions 02–03 by at most −0.88 % (1-thread, session 03); the thread optimum, Pareto
  memberships and ranking do not change; the effect on the Q8/Q4 ratios is bounded below ±2 % (sum of the two
  models' largest changes). **Decision: no rows were removed**; the effect stays within the between-session
  uncertainty shown in the tables.

**Night run** (`night_anomalies_mad.csv`, `night_anomaly_sensitivity.csv`, `night_integrity.csv`; same rule,
groups (block, configuration, prompt), n = 5 rounds in A1, n = 10 in A5/B1):
- Integrity: 230/230 units completed, no failed/skipped block (B1-post disabled by design); NaN 0;
  throttle_flag **0/660**; generated texts identical across all configurations (102/102). Peak temperature: A1 51 °C,
  A5 52 °C, B1 Turbo off 54 °C, Turbo on 85 °C, A2 60 °C. Idle power 13.69 W (0.9 % load, 31 °C).
- 53 rows (8.0 %) flagged in J/token; 14 deviate by ≥ 5 %, all in A1 (rounds 1–2 and 5) plus one in B1-off.
  Largest: A1 8B-Q4 2 threads round 2 `cpu_arch` **+53.8 %** (single prompt; `transformer` in the same round +9.5 %) —
  a single event, around 00:45; A1 8B-Q4 8 threads rounds 1–2 (+9–12 %, three prompts), Sparse8B-Q8 4 threads
  round 5 (+5–6 %).
- Sensitivity: removing the flagged rows changes a block mean by at most −4.1 % (8B-Q4, 2 threads, A1; 6.95 → 6.66),
  the next largest −1.9 % (8B-Q4, 8 threads, A1); ≤ 0.4 % in A5 and B1. The thread optimum, the saturation point, the
  quantization ratios and the Turbo ratios do not change. **Decision: no rows were removed**; the 8B-Q4 2-thread value
  is given with a note in RQ3.

**Second night run** (`night_02`; `night02_integrity.csv`, `night02_run_summary.csv`, `night02_anomalies_mad.csv`):
- Integrity: 70/70 units (B1 60, A2-completion 10), no failed/skipped block; NaN 0; throttle_flag **0/180**;
  generated texts identical (18/18). Peak temperature: B1 Turbo off 54 °C, Turbo on 87 °C (8B-Q8, momentary),
  A2 62 °C. Idle power 13.80 W (1.0 % load, 32 °C). Turbo restored to 99 % (log + `powercfg`).
- Merge evidence: for 10/10 models the text digest and byte count after chunk 90 are **identical** to night_01, and
  the NLL of the re-evaluated chunks 88–89 is **bit-identical** to night_01 (max |ΔNLL| = 0); hence the full set
  (90 + 474 chunks) is exactly what a single pass would have produced.
- Anomalies: 7 rows (3.9 %) flagged in J/token, only one ≥ 5 % (+5.9 %); sensitivity ≤ 0.17 %. No rows removed.

---

## 4. Protocol deviations and their handling

| Deviation | Evidence | Handling |
|---|---|---|
| Session 01 ran without a reboot (uptime ~11.5 h); transient +3–10 % degradation in the first two rounds; idle measurement taken at 4.1 % load (16.39 W ↔ 13.8 W) | `session1_deviation.csv`: session 01 is +0.2…+2.35 % (all rounds), +0.2…+1.14 % (rounds 3–10) above the mean of sessions 02–03 | Included in the raw metrics (n = 3; the deviation is 30 times smaller than the effects, the ranking does not change, the CIs narrow); **net energy with sessions 02–03** (idle measurement unreliable). From session 02 on, a post-boot wait, a quiescence gate and a 300 s warm-up load were added. |
| Internet outage in session 03 (00:02:13–00:03:44, ~20 disconnect/reconnect events; Kaspersky DNS timeout) | `session_03_network_events.csv`, `session_03_outage_window.csv` | The run was not interrupted (the campaign does not depend on the network); one prompt of unit 94 +21.5 %; effect on the 1-thread session mean 1.0 %. No data deleted; documented. |
| Minute-scale background clusters (§3) | `anomalies_mad.csv` | Effect on session means ≤ 2 %; the interleaved order distributes the effect evenly across models. Recommendation for future sessions: process-level CPU usage logging. |

---

## 5. Limitations (single list)

1. **Instruction set:** llama-cpp-python prebuilt CPU wheel (AVX2/FMA/F16C); the AVX-512 of the i9-11900K was not used.
   Since decode is bandwidth-bound, its effect is expected to be small at ≥ 4 threads; it was not measured for
   1–2 threads and prefill.
2. **Memory speed:** DDR4 2133 MT/s (XMP off, thermal-safety decision). The results are specific to this bandwidth;
   with faster memory all absolute speeds and energy/token values change (ratios largely preserved, RQ4).
3. **Turbo Boost off** (Balanced plan, 99 %): the absolute power and speed of the main campaign do not transfer directly
   to a Turbo-on system; the Turbo effect was measured under control (RQ5: speed ×1.00–1.02, energy ×2.3; two sessions)
   but is limited to three models and 8 threads.
4. **RAM energy not measured:** CodeCarbon estimates RAM power with a constant model; the main metric is CPU package
   energy only (RAPL PKG, Windows EMI). DRAM and system-wide (wall) energy are not included.
5. **Machine-wide counter:** the RAPL package counter covers all processes; background activity enters the measurement (§3).
6. **Single hardware platform:** i9-11900K / Z590 / Windows 11; no generalization to other microarchitectures (E-core, ARM).
7. **Single dataset:** perplexity only on the WikiText-2 test set (full, 564 chunks, 144,384 tokens; CIs ±3.3 %).
   Task accuracy (MMLU etc.) was not measured; perplexity is a limited quality metric for instruct models.
8. **No pure-pruning / pure-distillation arm;** the sparse model is a base model (RQ2).
9. **Short prompts, fixed 100 tokens:** prefill was measured at 11–18 tokens; in long contexts the share of KV-cache
   and prefill energy increases.
10. **Three sessions:** the t-intervals of the ratios are wide with df = 2; session 01 is off-protocol.
11. **Thread scaling:** 1/2/4/6/8/12/16 only on 8B-Q4_K_M (3 sessions); 2/4/8 on all models but in a single night
    session (5 rounds). The RQ1 Q5/Q6 energy and RQ3 all-model findings were not verified by between-session
    replication (the RQ5 Turbo effect was verified in two sessions).
12. **Bytes/token estimate** from the GGUF header; the effect of L3 (16 MB) and activation traffic were neglected.
13. **The carbon metric is not independent:** gCO2eq/token = J/token × 464.586 g/kWh (CodeCarbon TUR value).
14. **Room temperature** was noted manually as 23 °C in every session; it was not measured continuously.

---

## 6. Sources (for the thesis bibliography)

### 6.1 Models (GGUF, Hugging Face)

| Key | File | Repository | Size | SHA-256 (first 16) | Origin |
|---|---|---|---|---|---|
| 8B-Q4_K_M | Meta-Llama-3-8B-Instruct.Q4_K_M.gguf | QuantFactory/Meta-Llama-3-8B-Instruct-GGUF | 4.921 GB | 86c8ea6c8b755687 | Meta Llama 3 8B Instruct (dense) |
| 8B-Q5_K_M | Meta-Llama-3-8B-Instruct.Q5_K_M.gguf | QuantFactory/Meta-Llama-3-8B-Instruct-GGUF | 5.733 GB | 5a8d6562833be7c8 | same (night run A5/A2; tokenizer identical to 8B-Q4) |
| 8B-Q6_K | Meta-Llama-3-8B-Instruct.Q6_K.gguf | QuantFactory/Meta-Llama-3-8B-Instruct-GGUF | 6.596 GB | bd7efd73f9fb67e4 | same (night run A5/A2; tokenizer identical to 8B-Q4) |
| 8B-Q8_0 | Meta-Llama-3-8B-Instruct.Q8_0.gguf | QuantFactory/Meta-Llama-3-8B-Instruct-GGUF | 8.541 GB | e39eaa0b315c34d2 | same |
| 3B-Q4_K_M | Llama-3.2-3B-Instruct.Q4_K_M.gguf | QuantFactory/Llama-3.2-3B-Instruct-GGUF | 2.019 GB | 668d556a35ebfae0 | Meta Llama 3.2 3B Instruct (pruning + distillation; teacher Llama 3.1 8B/70B) |
| 3B-Q8_0 | Llama-3.2-3B-Instruct.Q8_0.gguf | QuantFactory/Llama-3.2-3B-Instruct-GGUF | 3.422 GB | 6c17f17637737118 | same |
| 1B-Q4_K_M | Llama-3.2-1B-Instruct.Q4_K_M.gguf | QuantFactory/Llama-3.2-1B-Instruct-GGUF | 0.808 GB | f3cdd84d4a33483d | Meta Llama 3.2 1B Instruct (pruning + distillation) |
| 1B-Q8_0 | Llama-3.2-1B-Instruct.Q8_0.gguf | QuantFactory/Llama-3.2-1B-Instruct-GGUF | 1.321 GB | 1b7f984d853990d1 | same |
| Sparse8B-Q4_K_M | Sparse-Llama-3.1-8B-2of4.Q4_K_M.gguf | QuantFactory/Sparse-Llama-3.1-8B-2of4-GGUF | 4.921 GB | c481e7089ffaedd5 | Neural Magic, Llama 3.1 8B base; SparseGPT 2:4 + SquareHead distillation |
| Sparse8B-Q8_0 | Sparse-Llama-3.1-8B-2of4.Q8_0.gguf | QuantFactory/Sparse-Llama-3.1-8B-2of4-GGUF | 8.541 GB | 36a463d4fe9d6780 | same |

Full SHA-256 values are in `results/metadata/<run_id>.json` → `models.*.sha256`. Licences: Llama 3 Community
License (8B), Llama 3.2 Community License (1B/3B), Llama 3.1 Community License (Sparse-Llama derivative).

### 6.2 Dataset

- WikiText-2 (raw, test split): `wikitext-2-raw-v1.zip` from the `ggml-org/ci` repository → `wiki.test.raw`
  (the file used by the llama.cpp perplexity tool); SHA-256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.
  Original: Merity et al., *Pointer Sentinel Mixture Models* (2016), CC BY-SA 3.0.

### 6.3 Software

| Component | Version / source |
|---|---|
| Python | 3.12.10 (Windows 11 Home 10.0.26300) |
| llama-cpp-python | 0.3.36, prebuilt CPU wheel (`https://abetlen.github.io/llama-cpp-python/whl/cpu`; AVX2, FMA, F16C, OpenMP, LLAMAFILE, REPACK) — llama.cpp / ggml (MIT) |
| CodeCarbon | 3.3.1 (`OfflineEmissionsTracker`, task mode; CPU: Windows Energy Meter Interface = RAPL PKG; GPU: NVML; RAM: constant model) |
| LibreHardwareMonitorLib | 0.9.6 (MPL-2.0), via pythonnet 3.0.5; PawnIO driver (namazso) for the temperature sensor |
| numpy / pandas / matplotlib | 2.5.3 / 3.0.6 / 3.11.2 |
| psutil / PyYAML / huggingface_hub | 7.2.2 / 6.0.3 / 2.1.1 |
| gguf (tensor sizes) / python-docx | 0.19.0 / 1.2.0 |
| Frequency counter | Windows PDH `\Processor Information(_Total)\% Processor Performance` |
| Power plan | Balanced, maximum processor state 99 % (Turbo Boost off) |

### 6.4 Hardware

Intel Core i9-11900K (8C/16T, L3 16 MB, 3.5 GHz nominal), 2 × 16 GB DDR4 G.Skill F4-3200C16-16GSXFB @ 2133 MT/s
(dual channel, XMP off), MSI Z590 PRO WIFI (MS-7D09), NVIDIA GeForce RTX 2060 (not used for inference).
Room temperature 23 °C (three sessions). Heat trial (6 October, 16 threads, 123 s): highest package 48 °C, 65 W.

---

## 7. Figure and table index

`figures/en/`: fig01 energy/speed (per model), fig02 energy–accuracy Pareto (full-set PPL; Q5/Q6 †), fig03 thread
scaling, fig04 thread Pareto, fig05 session consistency (supplementary), fig06 prefill/decode, fig07 bandwidth
(per model), fig08 bandwidth (per thread), fig09 roofline, fig10 Turbo on/off (two sessions), fig11 model × thread
heat map. `tables/en/`: T1 main results (PPL 564 chunks), T2 quantization (+ night † rows), T3 sparse–dense TOST,
T4 thread scaling, T5 variance decomposition, T6/T6b conditions and sessions, T7 bandwidth (+ Q5/Q6 †), T8
prefill/decode, T9 Turbo (two sessions), T10 thread optimum. Turkish versions (same file names) are in `figures/tr/`,
`tables/tr/`. Night-run analysis tables: `results/analysis/night_*.csv`, `night02_*.csv` (`scripts/analyze_night.py`).

---

## Appendix A. Perplexity subset results (60 and 90 chunks) and comparison with the full set

The three measurements are nested prefixes of the same text: session 01 first 60 chunks (15,360 tokens), night_01
first 90 chunks (23,040 tokens), and the 564 chunks completed with night_02 (144,384 tokens; `night_perplexity.csv`,
`night_perplexity_full.csv`). The ranking of the eight models did not change along 60 → 90 → 564; the new Q5_K_M and
Q6_K settled between Q4_K_M and Q8_0, and the Q6_K ↔ Q8_0 order swapped between 90 and 564 chunks (difference in the
fourth decimal, a tie).

| Model | PPL 60 chunks [95 % CI] | PPL 90 chunks [95 % CI] | PPL 564 chunks [95 % CI] | Bits/byte 564 | 90 → 564 | Rank 60 / 90 / 564 |
|---|---|---|---|---|---|---|
| Sparse8B-Q8_0 (base) | 8.052 [7.35; 8.82] | 8.300 [7.73; 8.92] | 8.117 [7.87; 8.37] | 0.676 | −2.2 % | 1 / 1 / 1 |
| Sparse8B-Q4_K_M (base) | 8.213 [7.51; 8.99] | 8.454 [7.88; 9.08] | 8.271 [8.02; 8.53] | 0.682 | −2.2 % | 2 / 2 / 2 |
| 8B-Q8_0 | 8.595 [7.78; 9.50] | 8.803 [8.11; 9.55] | 8.367 [8.09; 8.65] | 0.686 | −5.0 % | 3 / 4 / 3 |
| 8B-Q6_K | — | 8.801 [8.12; 9.54] | 8.368 [8.09; 8.65] | 0.686 | −4.9 % | — / 3 / 4 |
| 8B-Q5_K_M | — | 8.843 [8.16; 9.59] | 8.403 [8.13; 8.69] | 0.687 | −5.0 % | — / 5 / 5 |
| 8B-Q4_K_M | 8.708 [7.89; 9.61] | 8.932 [8.25; 9.67] | 8.478 [8.20; 8.76] | 0.690 | −5.1 % | 4 / 6 / 6 |
| 3B-Q8_0 | 10.94 [9.97; 12.00] | 11.24 [10.44; 12.10] | 10.54 [10.20; 10.89] | 0.761 | −6.2 % | 5 / 7 / 7 |
| 3B-Q4_K_M | 11.23 [10.23; 12.33] | 11.53 [10.71; 12.42] | 10.80 [10.45; 11.16] | 0.768 | −6.4 % | 6 / 8 / 8 |
| 1B-Q8_0 | 14.79 [13.51; 16.20] | 15.17 [14.08; 16.34] | 14.00 [13.54; 14.48] | 0.852 | −7.7 % | 7 / 9 / 9 |
| 1B-Q4_K_M | 15.29 [13.96; 16.75] | 15.60 [14.48; 16.79] | 14.47 [14.00; 14.96] | 0.863 | −7.2 % | 8 / 10 / 10 |

Paired differences (Q4 − Q8 ΔNLL, 8B): 0.0130 [0.0077; 0.0183] (60) ↔ 0.0145 [0.0095; 0.0196] (90) ↔
0.0131 [0.0114; 0.0149] (564); relative PPL differences (8B / 3B / 1B): +1.3 % / +2.6 % / +3.4 % (60),
+1.4 % / +2.5 % / +2.8 % (90), +1.3 % / +2.4 % / +3.2 % (564). The first 90 chunks are harder than the rest of the set
(full-set PPL 2–8 % lower, in the same direction for all models; the difference is larger for the small models);
because the CIs of the subsets overlap across models, between-model inferences rest on the paired ΔNLL and yield the
same conclusion in all three measurements. The correctness of the merge was demonstrated in night_02 (§3: digest/byte
match and bit-identity of chunks 88–89, 10/10 models).
