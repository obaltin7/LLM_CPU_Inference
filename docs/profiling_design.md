# Advanced Profiling: L3 Cache Misses and Memory Bandwidth

This document describes which measurements can be performed on Windows to investigate the
hardware-level root causes of energy consumption, and how they can be carried out without
interfering with the energy measurement (CodeCarbon).

## 1. Framing of the problem

To generate a single token, almost all of the model's weights are read once: 4.92 GB for Q4_K_M
and 8.54 GB for Q8_0. The processor's L3 cache, by contrast, is 16 MB. Since the weights do not
fit in the cache, they are streamed from DRAM again for every token. Consequently, the upper bound
on inference speed is usually set not by the processor's compute capability but by the amount of
data the memory can transfer per second:

```
Required bandwidth (GB/s) ≈ model size (GB) × speed (tokens/s)
```

Because the input embedding table is not read in its entirety, this expression overestimates the
actual traffic by roughly 6%.

Theoretical bandwidth of the reference system: 2 channels × 2133 MT/s × 8 bytes = **34.1 GB/s**.

## 2. Measurement options on Windows

| Method | What it measures | Requirements | Assessment |
|---|---|---|---|
| Bandwidth model and thread scaling (`archive/phase1_scripts/profile_memory.py`) | Peak bandwidth, bandwidth utilisation of inference, scaling efficiency | None (no administrator privileges or driver required) | Implemented and validated |
| ETW hardware counters (`xperf -pmc`) | Per-process L3 misses, L3 references, instruction count | Administrator privileges, Windows Performance Toolkit | Draft ready (`archive/phase3_pmc_draft/profile_pmc.py`), not validated |
| Intel VTune Profiler (memory-access analysis) | L3 misses, DRAM-bound stall ratio, bandwidth over time | VTune installation and sampling driver | Most detailed tool; recommended for cross-validation |
| Intel PCM | System-wide counters and bandwidth | MSR driver that the user must compile and sign | Cumbersome to set up on Windows; not recommended |
| `perf` inside WSL2 | — | — | No reliable access to hardware counters; also alters the workload |

On this machine, the `wpr -pmcsources` command lists `LLCMisses`, `LLCReference`,
`InstructionRetired` and `TotalCycles` among the kernel's counter sources. Since the
virtualisation layer (Hyper-V/VBS) is not enabled, access to the counters is not blocked.

## 3. Two complementary measurements

**Bandwidth (primary evidence).** During sequential weight reads, most misses are served ahead of
time by the hardware prefetcher. The miss count does not indicate whether the core actually waited
for data; the miss rate on its own is therefore not evidence of a bottleneck. The measurement that
directly exposes the bottleneck is the extent to which inference saturates the memory channels.

**L3 miss counters (supporting evidence).** Three metrics are reported:

- Miss rate = `LLCMisses / LLCReference`
- MPKI = `LLCMisses / InstructionRetired × 1000`
- Misses per token = `LLCMisses / generated tokens`

The expected magnitude of misses per token is the number of 64-byte cache lines occupied by the
weights: approximately 77 million for Q4_K_M and approximately 133 million for Q8_0. A measured
value of this order serves as a consistency check on the counter measurement.

## 4. Architecture: separate passes

Profiling is not performed in the same run as the energy measurement. Each type of measurement
runs in its own pass with an identical workload:

```
Pass A  scripts/benchmark_energy.py           -> benchmark_results.csv       CodeCarbon (EMI/RAPL), no trace
Pass B  scripts/thread_scaling_experiment.py  -> thread_scaling_results.csv  speed and energy per thread count
Pass C  archive/phase3_pmc_draft/profile_pmc.py -> profile_results.csv       xperf PMC trace, no energy measurement

Common key: quantization, n_threads, max_tokens
Constants:  seed=42, temperature=0, same prompt, use_mmap=False
```

There are three reasons for this separation:

1. **Observer effect.** Trace collection (reading counters on every context switch, flushing
   buffers to disk) consumes energy. Because the RAPL package counter measures the whole machine,
   this consumption would leak into the Joules/Token value.
2. **Resource sharing.** EMI reads only the RAPL energy counters, whereas the PMC trace programs
   the performance counters. These are different hardware resources and technically do not
   conflict. However, since xperf and VTune program the same performance counters, those two must
   not be run concurrently.
3. **Comparability.** Because the workload is deterministic, the passes produce the same token
   sequence; the results can be joined on the common key.

**Validity check.** If the speed measured in Pass C (`tokens_per_sec`) is noticeably lower
(by more than roughly 3%) than the mean of Pass A, the tracing overhead has perturbed the
workload and that profile result must not be used.

## 5. Findings obtained on the reference system (Phase 1, Turbo on, 4 October 2026)

> The figures in this section belong to the exploratory runs of Phase 1 (Turbo Boost on, PyCharm
> open, Phase 1 measurement window). The bandwidth analysis derived from the controlled campaign
> of Phase 2 is given in §7.

The values below are taken from the run of 4 October 2026 (`run_id: 05b54e9b`). PyCharm and other
applications were open during the run; for the values to be reported in the thesis, the
measurement must be repeated on a quiet system.

**Peak bandwidth.** The measured peak read bandwidth is 30.9 GB/s (91% of the theoretical limit).
A single thread reaches 25.9 GB/s.

**Bandwidth utilisation.**

| Model | Size (GB) | Speed (tokens/s) | Required (GB/s) | Utilisation |
|---|---|---|---|---|
| Q4_K_M | 4.92 | 6.00 | 29.5 | 96% |
| Q8_0 | 8.54 | 3.60 | 30.8 | 100% |

**Thread scaling (speed only; 4 October 2026, 32 tokens/round).**

| Threads | Q4_K_M (tokens/s) | Speed-up | Q8_0 (tokens/s) | Speed-up |
|---|---|---|---|---|
| 1 | 2.94 | 1.00× | 2.08 | 1.00× |
| 2 | 4.60 | 1.56× | 3.07 | 1.48× |
| 4 | 5.09 | 1.73× | 3.43 | 1.65× |
| 6 | 5.58 | 1.90× | 3.54 | 1.71× |
| 8 | 5.93 | 2.02× | 3.46 | 1.66× |

**Thread scaling (speed + energy; 6 October 2026, `run_id: 652d900c`, Q4_K_M, 100 tokens/round,
3 iterations).** Collected with `scripts/thread_scaling_experiment.py`; the Pareto plot is in
`figures/thread_scaling_pareto.pdf`.

| Threads | Tokens/s | Joules/Token | mgCO2eq/Token | Mean power (W) | Pareto |
|---|---|---|---|---|---|
| 1 | 3.13 ± 0.02 | 25.41 ± 0.35 | 3.28 | 79.6 | — |
| 2 | 4.88 ± 0.02 | **20.59 ± 0.11** | 2.66 | 100.5 | optimal |
| 4 | 5.57 ± 0.04 | 21.60 ± 0.18 | 2.79 | 120.3 | optimal |
| 8 | 5.72 ± 0.07 | 24.74 ± 0.68 | 3.19 | 141.6 | optimal |
| 16 | 5.76 ± 0.03 | 26.80 ± 0.11 | 3.46 | 154.4 | optimal |

While speed saturates at 4 threads, package power keeps rising: 2 threads give the lowest energy
cost; 8 threads are 17% faster than that but spend 20% more energy per token; Hyper-Threading (16)
raises speed by only 0.7% while increasing energy by a further 8%. A single thread is both slow and
inefficient: the fixed power of the off-core components (uncore, memory controller) is spread over
a longer duration.

Three findings point to the same conclusion:

- Both models fill the memory channels almost completely.
- Eight cores deliver only a 1.7–2.0× speed-up over a single core; for Q8_0, speed does not
  increase beyond four threads.
- The speed ratio of the two models (6.00 / 3.60 = 1.67) is close to their size ratio
  (8.54 / 4.92 = 1.74).

Hence, on this system, inference speed is limited by memory bandwidth rather than by the processor.
The consequence for energy is as follows: the cores draw power even while waiting for data. The
fact that Q8_0's energy per token is 1.61 times that of Q4_K_M (37.5 / 23.3 J) can largely be
explained by the increase in the number of bytes transferred per token.

## 6. Limitations and recommended next steps

- **`profile_pmc.py` has not been validated.** The script relies on xperf's `-pmc` option and on
  the `Pmc` / `CSwitch` line format of the text dump. It could not be executed during this session
  because administrator privileges and xperf were unavailable; on first use, the command line and
  the parser must be checked against a real trace.
- **Causality test: memory speed (out of scope).** If the RAM speed is raised by enabling XMP in
  the BIOS (e.g. 3200 MHz) and the speed increases by roughly the same proportion, the bandwidth
  bottleneck is demonstrated causally. It was decided to keep XMP disabled for reasons of hardware
  safety (thermal risk); the measurements at 2133 MHz are taken as the baseline condition of the
  study.
- **Counter-test: AVX-512.** llama.cpp can be compiled with AVX-512 and the same test repeated. If
  speed does not increase noticeably, it is additionally confirmed that the bottleneck is memory
  rather than compute.
- **Efficient thread count.** Given that speed saturates at four threads, whether the same speed
  can be reached at lower power with fewer cores can be tested with `scripts/benchmark_energy.py`.
  *(Answered in Phase 2: for 8B-Q4_K_M the lowest J/token is at 2 threads; see T4, fig03–04.)*

## 7. Phase 2 bandwidth analysis and Phase 3 status (8 October 2026)

### 7.1 Method (no new measurement; `scripts/analyze_bandwidth.py`)

The decode speeds from the three sessions of the final campaign (Turbo off, Balanced + 99%,
post-reboot, 8 threads, 100 tokens/generation) were multiplied by the tensor sizes read from the
GGUF headers to compute the effective bandwidth. The assumptions are given in README §5.13; two
estimates are used:

- **Estimate 1:** all weight tensors are read once per token.
- **Estimate 2:** for models with a separate `output.weight` tensor (Llama-3-8B,
  Sparse-Llama-3.1-8B), `token_embd` is excluded because it is read only row-wise
  (8B-Q4: −0.29 GB, 8B-Q8: −0.56 GB); for Llama-3.2 1B/3B it is not excluded because the embedding
  and output layers are tied; the KV cache is added (~8.6 MB/token for 8B, 0.2% of the weights).

**Theoretical peak:** the modules in the metadata sit on two separate channels
(`Controller0-ChannelA-DIMM1`, `ChannelB-DIMM1`), 64-bit data bus, 2133 MT/s →
2 × 8 B × 2133 × 10⁶ = **34.13 GB/s**.

### 7.2 Results (T7, fig07–fig09)

| Model | Bytes/token (estimate 2) | Tokens/s | Effective GB/s (estimate 1 / 2) | Ratio to peak |
|---|---|---|---|---|
| 1B-Q4_K_M | 0.80 GB | 35.80 | 28.6 / 28.7 | 84% |
| 1B-Q8_0 | 1.32 GB | 22.66 | 29.8 / 29.8 | 87% |
| 3B-Q4_K_M | 2.02 GB | 14.24 | 28.7 / 28.8 | 84% |
| 3B-Q8_0 | 3.42 GB | 8.70 | 29.7 / 29.8 | 87% |
| 8B-Q4_K_M | 4.63 GB | 6.40 | 31.4 / 29.6 | 87% |
| 8B-Q8_0 | 7.98 GB | 3.81 | 32.5 / 30.4 | 89% |
| Sparse8B-Q4_K_M | 4.63 GB | 6.39 | 31.4 / 29.5 | 87% |
| Sparse8B-Q8_0 | 7.98 GB | 3.80 | 32.4 / 30.3 | 89% |

By thread count (8B-Q4_K_M, estimate 2): 1 → 13.5 GB/s (40%), 2 → 24.3 (71%),
4 → 29.6 (87%), 6 → 29.9 (88%), 8 → 29.7 (87%), 12 → 29.3 (86%), 16 → 29.1 (85%).

Interpretation:

- **All eight models operate within the 28.7–30.4 GB/s band**, at 84–89% of the theoretical peak.
  While model size varies from 0.8 GB to 8.5 GB and speed from 36 to 3.8 tokens/s, their product
  is constant: this is direct evidence that the decode phase is bounded by memory bandwidth.
- Contrary to expectation, **the 1B models do not show markedly lower utilisation** (84% versus
  87–89%). The larger compute share of the small model leads to a difference of 3–5 percentage
  points; this is explained by the fact that in 1B-Q4 a substantial part of the weights (the
  0.22 GB embedding/output matrix) is read through a dense matrix-vector product, while the
  KV/activation share remains small. No cache effect is expected: even the smallest model is 50
  times the 16 MB L3.
- The Q8_0 models consistently show roughly 3 points higher utilisation than Q4_K_M: Q4_K block
  dequantisation requires more computation per byte; with Q8_0 the kernel is in a "purer"
  streaming regime.
- Not reaching 100% is normal: owing to DRAM refresh, page/bank switches and read-write
  turnarounds, the achievable peak is typically 85–92% of the theoretical value. The STREAM-like
  read-only test of Phase 1 (Turbo on) measured 30.9 GB/s (91%); LLM decode reaches 93–98% of the
  practical peak.
- **Thread saturation coincides with the bandwidth ceiling:** 1 thread is at the single-core limit
  (13.5 GB/s), 2 threads reach 71%, and **the ceiling is reached at 4 threads with 87%**; between
  6 and 16 threads utilisation does not change (±1 point). The saturation of speed at 4–6 threads
  (fig03) is exactly this point. This explains why the lowest J/token occurs at 2 threads (T4):
  2 threads reach 71% of the ceiling at the lowest package power (35 W); 4 threads obtain the last
  16% at 10 W more power; 8+ threads consume power without gaining any bandwidth.
- Roofline sketch (fig09): arithmetic intensity 1.8–3.5 FLOP/byte, roof intersection
  (896 GFLOP/s / 34.1 GB/s) ≈ 26 FLOP/byte. All points lie just below the bandwidth roof and an
  order of magnitude below the compute roof. The compute roof is a coarse fp32 upper bound (the
  actual limit for quantized kernels is lower); the figure should therefore be read only
  qualitatively.

### 7.3 Status of the Phase 3 hardware counter component

- **Intel PCM: will not be implemented.** On Windows, MSR/PMU access requires either PCM's own
  `MSR.sys` driver (must be built and signed with the WDK; an unsigned driver loads only in
  test-signing mode) or WinRing0. WinRing0 is blocked by Windows Defender because of a known
  security vulnerability (CVE-2020-14979), and it was decided not to use it in this study. The
  installed PawnIO is sufficient only for reading temperatures; PCM has no PawnIO back end.
- **ETW PMC (`xperf -pmc`, `archive/phase3_pmc_draft/profile_pmc.py`):** uses the kernel's own PMU
  support and requires no third-party driver; it does require the Windows ADK and administrator
  privileges. The draft script has still not been validated. **Intel VTune** (Intel-signed sampling
  driver) is the second driver-free option. Both require new measurements and have been deferred to
  Phase 3.
- **Why the analytical method is sufficient:** the question asked is "is decode bounded by memory
  bandwidth?", and the three independent pieces of evidence in §7.2 (a constant ~29–30 GB/s across
  eight models, thread saturation coinciding with the ceiling, and the Q4/Q8 speed ratio tracking
  the byte ratio) answer it without counters. The contribution of the counters would be secondary:
  because the L3 miss count also includes lines served by the prefetcher during sequential
  streaming, it is weaker evidence of a bottleneck than bandwidth.
- **Where the analytical method falls short:** (1) it cannot separate whether the 3–5 point drop in
  utilisation for 1B is due to compute, in-core latency or memory-controller efficiency; (2) the
  bandwidth model does not apply to the prefill phase (compute-bound, measured with prompts of
  11–20 tokens); (3) without measuring the achievable practical peak (≈85–92%), the question
  "why not 100%" cannot be answered definitively — a STREAM-like measurement under the Phase 2
  conditions (Turbo off) would close this gap (new measurement).
