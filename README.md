# Running Large Language Models on CPUs: Speed, Energy and Quality

Measurement pipeline and data of a master's thesis study on the inference speed, CPU energy and
output quality of quantized, pruned and distilled Llama models running **entirely on a desktop CPU**.
The repository contains the experiment code, the configuration, the raw measurements of every session,
the analysis scripts and the generated figures and tables.

## 1. Motivation and scope

On edge devices, large language models are often served without a GPU, on general-purpose
processors. This study measures, in a reproducible way, the **generation speed** (tokens/s) and the
**energy efficiency** (CPU package joules per generated token, read from RAPL counters) of
GGUF-quantized Llama models on a desktop Intel CPU, together with their **quality** (WikiText-2
perplexity). The comparison covers three compression routes of the same model family:

| Registry key | Model | Compression | Note |
|---|---|---|---|
| `8B-Q4_K_M`, `8B-Q8_0` | Meta-Llama-3-8B-Instruct | quantization | dense reference; `8B-Q8_0` is the quality reference |
| `8B-Q5_K_M`, `8B-Q6_K` | Meta-Llama-3-8B-Instruct | quantization | intermediate levels, measured in the night runs |
| `3B-Q4_K_M`, `3B-Q8_0` | Llama-3.2-3B-Instruct | structured pruning + distillation | pruned and then distilled by Meta (teachers: Llama-3.1 8B/70B) |
| `1B-Q4_K_M`, `1B-Q8_0` | Llama-3.2-1B-Instruct | structured pruning + distillation | same |
| `Sparse8B-Q4_K_M`, `Sparse8B-Q8_0` | Sparse-Llama-3.1-8B-2of4 | 2:4 sparse pruning + distillation | SparseGPT 2:4 followed by SquareHead distillation; **base model**; GGUF stores the sparsity densely |

All models are QuantFactory GGUF conversions run with `llama-cpp-python`. The eight campaign models
share the same tokenizer (Llama-3 BPE, 128,256 tokens), so perplexity can be compared token by token.
The registry lives in `src/llm_cpu_inference/models.py`; the models used by an experiment are listed in
`configs/experiment.yaml`.

## 2. Reference system and requirements

| Component | Value |
|---|---|
| CPU | Intel Core i9-11900K (8 cores / 16 threads, non-hybrid, 16 MB L3) |
| Memory | 32 GB DDR4 (2 × 16 GB, dual channel), 2133 MHz (JEDEC, XMP off) |
| Motherboard | MSI Z590 PRO WIFI (MS-7D09) |
| GPU | NVIDIA GeForce RTX 2060 (drives the display only; not used for inference) |
| Operating system | Windows 11 Home, build 26300 (no virtualisation layer) |
| Python | 3.12 |
| Inference engine | llama-cpp-python 0.3.36 (pre-built CPU wheel: AVX2, FMA, F16C, OpenMP) |
| Energy measurement | CodeCarbon 3.3.1, Windows Energy Meter Interface (RAPL) |
| Temperature measurement | LibreHardwareMonitorLib 0.9.6 (pythonnet) + PawnIO driver; administrator rights required |
| Power setting (Phase 2) | Windows "Balanced" plan, maximum processor state 99 % (Turbo Boost off, 3.5 GHz) |

Minimum requirements to repeat the experiment on another system:

- An x86-64 CPU with AVX2. Real energy measurement needs an Intel CPU with RAPL counters and
  Windows 11 (see Section 5.3).
- At least 16 GB of RAM. Models are loaded fully into memory (`use_mmap=False`); Q8_0 alone takes
  about 8.5 GB.
- About 35 GB of free disk space for the models (about 13.5 GB for the 8B pair only).

Three configuration details matter when interpreting the results:

- **Memory speed.** The RAM of the reference system runs at 2133 MHz (XMP off). LLM decoding is
  memory-bandwidth bound, so this setting directly affects speed (see
  [docs/profiling_design.md](docs/profiling_design.md)).
- **Instruction set.** The pre-built wheel is limited to AVX2; the AVX-512 units of the i9-11900K
  are not used.
- **Power plan and Turbo Boost.** For thermal safety all Phase 2 measurements use the "Balanced"
  plan with the maximum processor state at 99 % (Turbo Boost off; the cores stay at 3.5 GHz).
  Phase 1 was measured with Turbo on (about 4.8 GHz under load), so the absolute values of the
  two phases are not directly comparable. In the smoke test, switching Turbo off for 8B-Q4_K_M
  reduced speed by only about 3 % while package power fell from about 138 W to 61 W and energy
  from about 23.9 to 10.9 J/token. The active plan and processor state are written to the metadata
  file of every run.

## 3. Installation

```powershell
# 1) Create and activate a virtual environment with Python 3.12
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# 2) Install the dependencies (no C++ compiler needed)
pip install -r requirements.txt

# 3) Download the missing models into models/ (8 models, ~35 GB; the 8B pair alone ~13.5 GB)
python scripts/download_models.py
```

`scripts/download_models.py` only fetches files that are missing from `models/`; an interrupted
download resumes when the script is run again. A subset can be selected with
`python scripts/download_models.py --models 8B-Q4_K_M 8B-Q8_0`. The model files exceed GitHub's
file-size limit and are excluded by `.gitignore`.

The scripts import the shared package from `src/` on their own, so no installation step is needed.
If you prefer an installed package (for example to import it from a notebook), run
`pip install -e .` in the repository root.

For the CPU temperature sensor, run the campaign from an **administrator PowerShell** after the
one-time setup `winget install --id namazso.PawnIO --exact` and `python scripts/fetch_lhm.py`
(downloads LibreHardwareMonitor into `third_party/`, pinned by version and SHA-256).

## 4. Usage

Phase 2 measurements are taken with `scripts/run_campaign.py`. Every experiment parameter (model
list, prompts, rounds and warm-ups, generation length, thread list, shuffle seed, thermal limits,
idle-baseline duration, perplexity subset) lives in one file, `configs/experiment.yaml`. The
operating protocol of the final campaign is described in
[docs/final_campaign_protocol.md](docs/final_campaign_protocol.md).

```powershell
python scripts/run_campaign.py --list                          # session plan and progress (no measurement)
python scripts/run_campaign.py --smoke                         # short end-to-end trial -> results/smoke/
python scripts/heat_test.py                                    # 2-minute heat test at 16 threads
python scripts/run_campaign.py --final --note "room 23 C"      # session 1 (created if missing, resumed otherwise)
python scripts/run_campaign.py --final --new-session           # next session (another day, after a reboot)
python scripts/run_campaign.py --final --experiments energy    # one part of a session only
python scripts/analyze_sessions.py                             # within- / between-session variance
python scripts/night_run.py --list                             # night-run plan and time estimate (docs/night_run_01.md)
python scripts/night_run.py --final --note "room 23 C"         # unattended night run (night_01) -> results/night/
python scripts/night_run.py --final --night-config night_02    # second run: B1 repeat + full-set perplexity -> results/night_02/
python -m pytest                                               # unit tests (fast); night-run flow: python -m pytest -m slow
```

A campaign is split into sessions and units (Section 5.2). Completed units are recorded in
`results/campaign_state.json`; an interrupted session resumes from where it stopped when the same
command is run again. Result rows are written once per unit in a single call that is protected
from Ctrl+C, so no partial line can remain. The `--final` mode refuses to start with a dirty git
working tree, an unexpected power setting or a non-real temperature source.

`scripts/benchmark_energy.py`, `scripts/thread_scaling_experiment.py` and
`scripts/benchmark_cognitive.py` are shortcuts that start one part of the campaign
(the same as `--experiments`).

Outputs are appended under `results/`; existing files are never rewritten:

| File | Content |
|---|---|
| `energy_iterations.csv` | Energy experiment; one row per (session, round, model, prompt) |
| `energy_summary.csv` | Per session and model: mean, SD and 95 % CI (per prompt and `ALL`) |
| `thread_scaling_iterations.csv`, `thread_scaling_summary.csv` | Same structure, by thread count |
| `cognitive_results.csv` | Per model: perplexity, bits/byte, top-1 accuracy, paired difference against the reference |
| `nll/*.npz` | Token-level NLL arrays (for paired comparisons) |
| `metadata/<run_id>.json` | Hardware, software, configuration and idle-baseline record of the run (Section 5.10) |
| `logs/campaign_<session_id>_<run_id>.log` | Time-stamped run log |
| `campaign_state.json` | Sessions, shuffle seeds, planned order and completed units |

Phase 1 results are frozen under `results/phase1/` with the old schema (figures in
`results/phase1/figures/`, scripts in `archive/phase1_scripts/`). Phase 2 figures and tables are
produced from a single entry point (Section 8): `python scripts/make_figures.py --lang en`.
The `logs/console_<date>.log` files are the complete console output and are **for debugging only**.

## 5. Measurement methodology

### 5.1 Unit of measurement and workload

One measurement is the processing of a single prompt in two separate windows:

1. **Prefill (prompt processing):** the prompt tokens are evaluated in one batch.
2. **Decode (generation):** exactly `n_predict` = 100 tokens are generated greedily (argmax),
   one forward pass each.

To keep the length fixed, the end-of-generation tokens (EOS, EOT) are suppressed and the model's
next most likely token is taken instead; the number of suppressions is written to the
`eos_suppressed` column. Every model and every measurement therefore performs the same number of
forward passes.

- Prompts are sent as raw completions (BOS + text, no chat template); the workload is identical
  for every model.
- Three fixed prompts are used (`configs/experiment.yaml`: `cpu_arch`, `transformer`, `story`);
  every round runs all three in the same order.
- The model is loaded with `use_mmap=False` (weights read into RAM, no disk access during a
  measurement), `n_gpu_layers=0` (CPU only), `n_ctx=512` and `seed=42`. The seed has no effect on
  greedy decoding but is fixed and recorded.
- `n_threads = n_threads_batch` = number of physical cores (8); thread scaling uses the list in
  the configuration.
- The KV cache is cleared before every measurement; every measurement processes the prompt from
  scratch.
- **Determinism** is checked: all rounds of the same prompt must produce the same token sequence
  (`output_sha256`; `deterministic` column in the summary file).

### 5.2 Interleaved order and multi-session design

Day-to-day repeats of Phase 1 differed by 6–7 %. So that time-dependent drift (warm-up,
background activity, thermal state) is not confounded with the model effect:

- **Interleaving.** The energy and thread-scaling experiments measure the models (or thread
  settings) **round by round**, not block by block: every round measures every configuration
  once. The order within a round is reshuffled for each round with a session-specific seed
  (derived from `campaign.shuffle_seed` and the session id). The seed, the planned order
  (`campaign_state.json`, metadata `session.planned_orders`) and the realised order (metadata
  `executed_units`; `round` and `order_in_round` in the rows) are recorded.
- **Unit.** One round of one configuration: the model is loaded (duration in
  `model_load_time_s`, outside the measurement windows), **one warm-up generation** with the
  first prompt is run and discarded, then every prompt is measured once and the model is
  unloaded. Reload cost therefore never enters a measurement.
- **Sessions.** The campaign is run in at least 2, preferably 3 **independent sessions** on
  different days, each after a reboot. Every row and every metadata file carries `session_id`.
  An interrupted session is resumed under the same id; a session can only be completed with a
  single measurement protocol (parameter fingerprint).
- The **cognitive experiment** is deterministic, is not interleaved and is measured only in the
  first session (`cognitive.first_session_only`).

### 5.3 Energy source and tracking mode

Energy is measured with CodeCarbon 3.3.1's `OfflineEmissionsTracker` in **task mode**. Each
measurement window (prefill and decode separately) is one task; the energy is the difference of the
hardware's cumulative energy counter between the start and the end of the window, so it does not
depend on the sampling interval. The overhead of the window start/stop calls was measured: about
1 ms per call and about 0.07 J per window (negligible).

| Component | Column | Source | Measured? | In the main metric? |
|---|---|---|---|---|
| CPU package | `cpu_J` | RAPL `PKG` counter, Windows Energy Meter Interface (`windows_emi`) | measured | yes |
| RAM | `ram_J_est` | CodeCarbon constant-power model (20 W for 31.8 GB) | **estimate** | no |
| GPU | `gpu_J` | NVML (RTX 2060, drives the display only; 31–51 W) | measured | no |
| Total | `total_J` | sum of the three, informational only | — | no |

- The RAPL `DRAM` channel is visible on this platform but reads 0 W at idle and under memory load;
  memory energy therefore cannot be measured. `ram_J_est` is a workload-independent estimate.
- The CPU energy source is written to the `energy_source` column of every row and to the
  metadata. If the source is not a hardware counter (TDP-based estimation mode), `cpu_J` and every
  metric derived from it are written as **NaN** and the reason is logged.
- The RAPL package counter is machine-wide: the consumption of other processes running during a
  measurement is included. Measurements must be taken with background applications closed.

**Main metric:** `cpu_J_per_token` = CPU package energy of the decode window / number of
generated tokens.

### 5.4 Run start, idle power and net energy

At the start of every energy-measuring run (every start of a session), in this order:

1. **Post-boot wait:** if the system has been up for less than 20 minutes, wait until then
   (`session_start.min_uptime_s`); post-boot Windows tasks distorted the first rounds of session 1.
2. Wait 15 s for the start-up work (model digests, hardware inventory) to finish and for the
   package to cool to 55 °C.
3. **Quiet gate:** wait until the mean CPU utilisation of the last 60 s drops below 2 % (one
   sample per second, at most 15 minutes). If the time runs out the run stops with
   `QUIET_TIMEOUT` and nothing is measured.
4. **60 s idle power** is measured (`idle_cpu_W`; in the metadata and in every row).
5. **Warm-up load:** 5 minutes of unmeasured continuous generation (8B-Q4_K_M, 8 threads;
   `warmup_load`), then the rounds start. Nothing is written to the result files; a summary goes
   to the metadata.

Two energy values are reported together:

- **Raw:** `cpu_J_per_token`, the package energy the device actually drew in that time.
- **Net:** `cpu_J_per_token_net`, where `cpu_J_net = cpu_J − idle_cpu_W × decode_time_s`, the
  energy the workload adds on top of idle consumption.

If CPU utilisation exceeds 5 % during the idle measurement the script warns: background load
inflates the idle value and understates net energy.

Related configuration keys: `session_start.min_uptime_s`, `session_start.quiet_threshold_pct`,
`session_start.quiet_window_s`, `session_start.quiet_max_wait_s`, `idle.settle_s`,
`idle.duration_s`, `warmup_load.duration_s`, `warmup_load.model`, `warmup_load.n_threads`.

### 5.5 Prefill and decode separation

- **Speed:** `tokens_per_sec` = generated tokens / `decode_time_s`; computed from the decode
  window only.
- Prefill is kept in separate columns: `prefill_time_s`, `prefill_cpu_J`,
  `prefill_tokens_per_sec`. Prefill draws clearly more power than decode (about 184 W versus
  138 W in the smoke test).
- **Model loading** is outside both windows; its duration is kept in `model_load_time_s`.
- For comparison with the Phase 1 definition, prefill-inclusive values are also written:
  `tokens_per_sec_incl_prefill`, `cpu_J_per_token_incl_prefill` (idle power not subtracted).

### 5.6 Thermal safety and monitoring

Thermal safety takes precedence over everything else in this project (a previous CPU of the
measurement machine failed from overheating):

- **Turbo Boost off** (maximum processor state 99 %, "Balanced" plan); see Section 2.
- **Temperature measurement:** the CPU package temperature (LibreHardwareMonitor, `CPU Package`
  sensor) is sampled every 0.5 s throughout the run. If it cannot be read, the run **does not
  start**. Every row carries the mean and maximum temperature of its measurement window
  (`temp_mean_C`, `temp_max_C`) and the temperature at the start of the unit (`temp_at_start_C`).
- **Thermal fuse:** if the package stays above 85 °C for 10 s, reaches 95 °C in a single reading,
  or cannot be read for 5 s, the run stops safely between tokens/chunks: the running unit is
  discarded (no partial row), `THERMAL_ABORT` is logged and the session can be resumed with the
  same command.
- **Temperature-gated cool-down:** before every unit the script waits until the package is at or
  below 55 °C (at most 5 minutes; the wait is recorded in `cooldown_wait_s`).
- **Hottest setting:** units with 16 threads have a fuse limit of 80 °C and do not start until
  the package is at 55 °C (`THERMAL_ABORT` if it does not get there within 5 minutes).
- System sleep is prevented with `SetThreadExecutionState` for the duration of the run.
- The CPU frequency is sampled every 0.5 s during each measurement; the time-weighted mean and
  the minimum go to `freq_mean_mhz` and `freq_min_mhz`. Because `psutil.cpu_freq()` always returns
  the nominal frequency (3504 MHz) on Windows, the effective frequency is derived from the Windows
  performance counter `\Processor Information(_Total)\% Processor Performance`.
- **`throttle_flag`** is `True` when the mean frequency of a measurement is more than 5 % below
  the highest mean frequency seen for the same configuration. It indicates a **suspicion** of
  thermal or power-limit throttling, not proof.

Related configuration keys: `thermal.sample_interval_s`, `thermal.abort_c`,
`thermal.abort_hold_s`, `thermal.hard_abort_c`, `thermal.sensor_loss_s`,
`thermal.cooldown_target_c`, `thermal.cooldown_max_s`, `thermal.high_thread_count`,
`thermal.high_thread_abort_c`, `thermal.freq_sample_interval_s`, `thermal.throttle_drop_frac`.

### 5.7 Carbon emissions

CodeCarbon is given the country code `TUR` explicitly (`carbon.country_iso_code` in
`configs/experiment.yaml`); no IP geolocation is used. The grid carbon intensity CodeCarbon applies
for this country (464.586 gCO2eq/kWh in codecarbon 3.3.1) is written to
`results/metadata/<run_id>.json` and to the `grid_intensity_gco2_per_kwh` column of every row.

**gCO2eq/token is not an independent metric.** It is `gco2eq_per_token = cpu_J_per_token /
3.6 × 10⁶ × intensity`, energy multiplied by a constant; it carries the same information as
J/token and changes neither the ranking nor the ratios between models. It is computed from CPU
package energy only (RAM estimate and GPU excluded) and is not shown as a separate panel.

### 5.8 Statistics

- Every configuration is measured in **10 rounds** per session; the **1 warm-up generation** with
  the first prompt after every load is excluded. A round measures each of the three prompts once.
- For every metric the arithmetic mean, the sample standard deviation (n − 1) and the **95 %
  confidence interval** of the mean are computed. The interval uses Student's t distribution:
  `mean ± t(0.975; n−1) × s / √n`.
- The summary file has one row per prompt (n = 10) and an `ALL` row pooled over prompts. In the
  `ALL` row the unit of replication is the round: the mean of the three prompts of a round is one
  observation (n = 10). Because the prompts run in sequence within every round, this is balanced
  against time-dependent drift.
- Summaries are written per session (`session_id` column). Raw rows are kept; summary statistics
  can always be recomputed.
- **Within- and between-session variance** (`scripts/analyze_sessions.py`): a one-way
  random-effects ANOVA per configuration (session = random factor, round = replicate). Reported
  are the within-session SD (`sd_within`), the between-session SD (`sd_between`), the
  coefficients of variation of both and the intraclass correlation (ICC = share of the
  between-session variance in the total). The between-session component is defined only with at
  least two sessions.

### 5.9 Cognitive performance (perplexity) protocol

Model quality is measured with `scripts/benchmark_cognitive.py` on the WikiText-2 (raw) test
set, following the method of llama.cpp's `perplexity` tool:

| Parameter | Value |
|---|---|
| Data | `wiki.test.raw` (`wikitext-2-raw-v1.zip` from the `ggml-org/ci` repository), 1,290,590 bytes, SHA-256 `173c87a5…` |
| Chunk length / stride | `n_ctx` = 512 tokens, non-overlapping (stride = 512) |
| Scoring | The first token of every chunk is replaced by BOS; only the **second half** (256 tokens) is scored, so every scored token has at least 256 tokens of context |
| Subset | First 60 chunks: 30,720 evaluated and 15,360 scored tokens (about 10 % of the test set); `chunks: 0` evaluates the whole set (564 chunks) |
| Reference | `8B-Q8_0` |

Reported metrics:

- **Perplexity** = exp(mean NLL). The 95 % CI is computed **from the chunk means** with the t
  distribution; tokens of the same chunk are correlated and are not treated as independent.
  (The token-level intervals of Phase 1 are narrower for this reason.)
- **Bits per byte** (`bits_per_byte`) = total NLL (bits) / UTF-8 byte count of the scored text.
  It is tokenizer-independent and is the metric for models with different tokenizers.
- **Top-1 accuracy:** share of next tokens that match the most likely prediction.
- **Paired difference** against the reference model: token-level `delta_nll_vs_ref` (with a
  chunk-level 95 % CI), `delta_ppl_pct_vs_ref` and `delta_bits_per_byte_vs_ref`.

Byte-level identity of the scored text is verified, not assumed: the SHA-256 of the scored text
(`scored_text_sha256`) is written for every model, and the paired difference is computed only when
the same text was scored as for the reference (otherwise NaN). The eight models of this study share
the tokenizer and therefore the token sequence (verified); in that case bits/byte and perplexity
give the same ranking. Token-level NLL arrays are kept under `results/nll/`.

### 5.10 Records and traceability

For every run (`run_id`) a `results/metadata/<run_id>.json` file is written, and every CSV row
carries the same `run_id` and `session_id`. The record contains:

- Session id, shuffle seed, planned and realised unit order
- Start/end time and status (`completed`, `interrupted`, `thermal_abort`, `failed`)
- Git commit, branch and whether the working tree was clean (`git_dirty`)
- Power plan, maximum processor state (AC/DC) and `turbo_disabled`
- Temperature sensor, thermal limits, maximum temperature of the run and the abort reason if any
- CPU model, core topology (physical/logical, P/E cores), memory modules and their speed, GPU,
  operating system, motherboard and the active power plan
- Python, llama-cpp-python and CodeCarbon versions, and the llama.cpp build information
  (instruction sets)
- Size and SHA-256 of the model files (cached after the first computation)
- `n_threads`, `n_ctx`, `seed`, decoding method, the full configuration file and its SHA-256
- Energy tracking mode, frequency source, idle measurement result, carbon intensity
- Operator note (`--note`, for example the room temperature)

### 5.11 Known limitations

- **AVX2 wheel.** The pre-built llama-cpp-python wheel is limited to AVX2; the AVX-512 units of
  the CPU are not used. Absolute speed and energy values are specific to this build.
- **XMP off, 2133 MHz.** The RAM runs at its JEDEC speed (theoretical bandwidth 34.1 GB/s).
  Decoding is bandwidth-bound, so speed and J/token are specific to this memory speed; it is the
  reference condition of the study.
- **Pruning may bring no CPU speed-up.** The GGUF format and llama.cpp store 2:4 sparse weights
  densely and multiply densely: the sparse 8B files are byte-for-byte the same size as the dense
  8B files. No speed or energy gain from unstructured or semi-structured pruning is expected in this
  deployment path.
- **Public pruned models were recovered with post-pruning distillation; the pure pruning effect
  cannot be isolated.** Llama-3.2 1B/3B were produced with pruning and distillation together, and
  the sparse model with 2:4 pruning followed by SquareHead distillation. The set contains no model
  that was only pruned or only distilled.
- **Generation gap.** The dense reference is Llama-3-8B-Instruct; the compressed models derive from
  Llama-3.1. Observed quality differences cannot be attributed to compression alone.
- **The sparse model is a base model;** its perplexity is not directly comparable with the
  instruct models.
- **RAM energy is not measured; it is a 20 W constant estimate. The main metric is CPU package
  energy.** The RAPL DRAM counter reads 0 on this platform; `ram_J_est` is CodeCarbon's
  workload-independent constant-power model.
- **Turbo Boost off.** Results are specific to a fixed 3.5 GHz core frequency and cannot be compared
  in absolute terms with the Turbo-on Phase 1 results.
- **Machine-wide measurement.** The package counter includes background processes; the idle
  baseline compensates only partly.
- **Raw completion.** No chat template is used and end-of-generation tokens are suppressed; the
  workload is identical across models but does not represent real chat usage one-to-one.
- **The perplexity subset** of the campaign is about 10 % of the test set (the night runs cover
  the full set).
- **Power limit.** Under load the package power stays above the 125 W TDP (134–184 W); the
  motherboard's power-limit settings affect the results and were not verified in the BIOS.
- **Throttling detection** is based on frequency, not temperature.
- **Single system.** All measurements come from one machine on a Windows Insider build (26300).

### 5.12 Session records and deviations

**session_01** (7 October 2026, 00:16–04:35, `run_id` 20261007T001559-d4dffe; 158/158 units,
no thermal abort, maximum package temperature 59 °C). The deviations below are also stored as
`post_hoc_notes` in the session's metadata file and in `campaign_state.json`; the measured data
were not altered.

- **No reboot.** The system had been up since 6 October 12:41 (about 11.5 hours); the protocol
  requires a reboot before every session. The heat test and the smoke test ran in the same boot.
- **Transient in the first two rounds.** Rounds 1 and 2 (00:17–00:33) gave 3–10 % higher CPU
  J/token and lower speed for almost every model; values settled from round 3. Excluding the first
  two rounds changes the absolute means by at most 2 % and the ratios against 8B-Q8_0 by at most
  0.01 (the interleaved order spread the disturbance over all models). Sensitivity analysis:
  `python scripts/analyze_sessions.py --exclude-first-rounds 2`.
- **The idle baseline was taken in this period:** 16.39 W at 4.1 % CPU utilisation (just below
  the 5 % warning threshold); the net values of this session are less reliable than the raw ones.
- Isolated outliers remain after round 2 (largest: round 10, 8B-Q8_0 `transformer`, 18.67 J/token).
- Because of these deviations, the post-boot wait, the quiet gate and the warm-up load of
  Section 5.4 were added from session_02 on; session_01 was measured without them.

**session_02** (7 October 2026, 16:38–19:45, `run_id` 20261007T163802-aae8e0; 150/150 units).
After a reboot (uptime 79 min), idle 13.81 W / 1.0 %, maximum package temperature 52 °C; no
warnings, no throttling, no cool-down waits.

**session_03** (7–8 October 2026, 22:16–01:23, `run_id` 20261007T221555-885e23; 150/150 units).
After a reboot (uptime 38 min), idle 13.86 W / 0.9 %, maximum package temperature 53 °C; no
warnings, no throttling, no cool-down waits. **Network outage:** according to the Windows event
log the connection dropped at 00:02:13 and fluctuated until 00:03:44
(`results/analysis/session_03_network_events.csv`). The window coincides with unit 94 (thread
scaling, 1 thread, round 2); the run was not interrupted and the unit finished in its normal time.
The `transformer` measurement of that unit is +18 % J/token (z = 2.65); its effect on the 1-thread
session mean is 1.0 %. No data were removed.

**Decision (8 October 2026):** raw metrics (J/token, tokens/s, gCO2eq/token) are reported with
all three sessions; **net** energy is reported with session_02–03 only (the idle baseline of
session_01 was taken under 4.1 % background load). Cross-session analysis: `results/analysis/`
(`python scripts/analyze_campaign.py`).

### 5.13 Memory-bandwidth analysis (no measurement; `scripts/analyze_bandwidth.py`)

During decoding the complete weights are read from memory for every generated token, so the
**effective bandwidth ≈ bytes read per token × decode tokens/s**. Assumptions:

- **Theoretical peak:** channels × bus width × transfer rate. The modules in the run metadata
  (`Win32_PhysicalMemory`: `Controller0-ChannelA-DIMM1` and `ChannelB-DIMM1`, 64 bit, 2133 MT/s)
  give 2 × 8 bytes × 2133 × 10⁶ = **34.13 GB/s** (decimal GB). The achievable peak is lower; a
  STREAM-like read-only test in Phase 1 (Turbo on) measured 30.9 GB/s.
- **Estimate 1 (bytes/token):** the sum of all weight tensors in the GGUF file (header excluded).
- **Estimate 2 (corrected):** the embedding table `token_embd` is not read in full for every
  token, only one row of it; this applies only to models with a separate `output.weight` tensor
  (Llama-3-8B and Sparse-Llama-3.1-8B). In Llama-3.2 1B/3B the embedding and output layers are
  **tied** (no `output.weight`): the same matrix is read in full as the output projection, so no
  correction is applied. The KV cache is added: mean context (prompt + (1 + 100) / 2) × layers ×
  2 × KV heads × head size × 2 bytes (f16); about 8.6 MB/token for 8B, 0.2 % of the weights.
  Activations are ignored.
- Speed is the mean decode tokens/s of the three sessions (8 threads); uncertainty is the
  between-session ±1 SD.
- Tensor sizes are cached in `results/analysis/gguf_tensor_sizes.csv`, so the figures can be
  produced without the models on disk (`--refresh-gguf` re-reads them).

For the roofline sketch (`fig09`), arithmetic intensity = 2 × parameters / bytes per token
(estimate 2), attained performance = 2 × parameters × tokens/s; the compute roof is the AVX2 fp32
FMA bound (8 × 3.5 GHz × 32 = 896 GFLOP/s) and is only a coarse reference for quantized kernels.
Results: T7, fig07–fig09 and [docs/profiling_design.md](docs/profiling_design.md) §7.

## 6. Advanced profiling

The methods considered for tracing the hardware causes of the energy consumption (L3 cache
misses, memory-bandwidth bottleneck) and how they are kept from interfering with the energy
measurement are described in [docs/profiling_design.md](docs/profiling_design.md). An unexecuted
hardware-counter draft and the reason it was not pursued are kept in
`archive/phase3_pmc_draft/`.

## 7. Repository layout

```
.
├── configs/experiment.yaml      All experiment parameters (single source of truth)
├── src/llm_cpu_inference/       Shared package: configuration, model registry, generation and
│                                measurement, energy, thermal and power control, system monitoring,
│                                run metadata, statistics, append-only result files
├── scripts/                     Entry points (run from the repository root)
│   ├── run_campaign.py          Campaign orchestrator: sessions, interleaved rounds, thermal safety, resume
│   ├── night_run.py             Unattended night run: blocks a1/a5/b1/a2, Turbo control with guaranteed restore
│   ├── benchmark_energy.py      Energy and speed measurement (prefill/decode separated)
│   ├── thread_scaling_experiment.py  Speed and energy by thread count
│   ├── benchmark_cognitive.py   WikiText-2 perplexity, bits/byte and top-1 accuracy
│   ├── heat_test.py             Heat test at the highest thread count
│   ├── download_models.py       Downloads the missing GGUF models from Hugging Face
│   ├── fetch_lhm.py             Downloads LibreHardwareMonitor into third_party/ (pinned version and SHA-256)
│   ├── analyze_sessions.py      Within- / between-session variance (ANOVA components)
│   ├── analyze_campaign.py      Cross-session analysis: integrity, deviation, ranking, ratio CIs, TOST
│   ├── analyze_bandwidth.py     Decode memory-bandwidth utilisation (GGUF tensor sizes × tokens/s)
│   ├── analyze_night.py         Night-run analysis: integrity, anomalies, thread optimum, quantization ladder, Turbo, perplexity
│   └── make_figures.py          Thesis/paper figures and tables (en/tr) -> figures/<lang>/, tables/<lang>/
├── tests/                       pytest: statistics, formatting and file-writing unit tests;
│                                -m slow: night-run flow scenarios (fake sensor, dry-run Turbo)
├── docs/
│   ├── findings_report.md       Findings report (research questions, unexpected findings, deviations, limitations)
│   ├── final_campaign_protocol.md  Operating protocol and checklist of the final campaign
│   ├── night_run_01.md, night_run_02.md  Step-by-step checklists of the two night runs
│   └── profiling_design.md      Memory-bottleneck profiling design
├── results/                     Phase 2 measurements (sessions 01–03), metadata, logs, NLL arrays
│   ├── analysis/                Cross-session and night-run analysis tables (see results/analysis/README.md)
│   ├── night/, night_02/        Night-run measurements
│   └── phase1/                  Frozen Phase 1 archive (old schema)
├── figures/en/, figures/tr/     fig01–fig11 (PDF + PNG), English and Turkish labels
├── tables/en/, tables/tr/       T1–T10 (CSV + Markdown + DOCX)
├── archive/
│   ├── phase1_scripts/          Phase 1 scripts (old schema)
│   └── phase3_pmc_draft/        Unexecuted hardware-counter profiling draft and its rationale
├── LICENSE, NOTICE, CITATION.cff, THIRD_PARTY_NOTICES.md
├── pyproject.toml, requirements.txt, pytest.ini
└── models/, data/, third_party/ Downloads (git-ignored)
```

## 8. Reproducing the figures and tables

All figures and tables are generated from the raw measurement files under `results/`; the raw
files are never modified. Language and decimal separator are selected with one option:
`--lang en` (decimal point, CSV separator `,`) or `--lang tr` (decimal comma, CSV separator `;`).
File names are the same in both languages; only the folder differs.

```powershell
python scripts/analyze_campaign.py       # results/analysis/*.csv (integrity, session means, variance, ratio CIs, TOST)
python scripts/analyze_bandwidth.py      # results/analysis/bandwidth_*.csv (effective memory bandwidth; GGUF headers)
python scripts/analyze_night.py          # results/analysis/night_*.csv (night runs; needs results/night/)
python scripts/make_figures.py --lang en # figures/en/ and tables/en/
python scripts/make_figures.py --lang tr # figures/tr/ and tables/tr/   (--lang all: both)
```

| Output | Producer | Content |
|---|---|---|
| `fig01_energy_speed_per_model` | `make_figures.py` | J/token and tokens/s per model (two panels, grouped by family and quantization, with error bars) |
| `fig02_energy_accuracy_pareto` | `make_figures.py` | Energy–quality trade-off: J/token against perplexity and bits/byte; Pareto front of the instruct models |
| `fig03_thread_scaling` | `make_figures.py` | Tokens/s and J/token by thread count; 8-physical-core boundary |
| `fig04_thread_pareto` | `make_figures.py` | Speed–energy Pareto front of the thread settings |
| `fig05_session_consistency` | `make_figures.py` | Deviation of the three session means (appendix) |
| `fig06_prefill_vs_decode` | `make_figures.py` | Prefill and decode speed per model (left) and CPU energy: prefill J per prompt token against decode J/token (right) |
| `fig07_bandwidth_per_model` | `make_figures.py` | Effective memory bandwidth per model with the theoretical peak line (Section 5.13) |
| `fig08_bandwidth_per_thread` | `make_figures.py` | Effective bandwidth by thread count (8B-Q4_K_M) |
| `fig09_roofline` | `make_figures.py` | Coarse roofline: arithmetic intensity against attained performance |
| `fig10_turbo` | `make_figures.py` | Turbo on/off (night-run B1): J/token, tokens/s and package-power panels, on/off ratios |
| `fig11_thread_optimum` | `make_figures.py` | Model × thread heat map (night-run A1): J/token relative to the model's minimum, tokens/s relative to its maximum |
| `T1`–`T10` | `make_figures.py` | Main results, quantization ratios, sparse–dense equivalence, thread scaling, variance decomposition, experimental conditions, sessions, memory bandwidth, prefill/decode energy, Turbo (T9), thread optimum (T10); CSV + Markdown + DOCX. With night-run data, fig02/T1/T2/T7 include the 8B-Q5_K_M/Q6_K rows (†) and the full-set perplexity |
| `results/analysis/bandwidth_*.csv`, `gguf_tensor_sizes.csv` | `analyze_bandwidth.py` | Memory hardware, tensor sizes, effective bandwidth per model and per thread count |
| `results/analysis/night_*.csv` | `analyze_night.py` | Night runs: block integrity, MAD anomalies, thread optimum (8 models), 8B quantization ladder, Turbo ratios, perplexity |
| `results/analysis/*.csv` | `analyze_campaign.py` | Cross-session analysis tables (see `results/analysis/README.md`) |

Unless stated otherwise under a figure, error bars are the ±1 standard deviation of the session
means (n = 3 sessions). For perplexity the vertical bar is the chunk-level 95 % confidence
interval.

## 9. License

This repository (code, documentation and measurement data) is licensed under the
**Apache License 2.0** (`LICENSE`). Third-party components and models are subject to their own licenses (`THIRD_PARTY_NOTICES.md`). 