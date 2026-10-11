# Phase 3 draft — hardware-counter profiling (never executed)

`profile_pmc.py` is a **draft** that aims to read the L3 miss / reference / instruction counters on a
per-process basis during LLM inference, using the kernel PMU support of Windows (`xperf -pmc`, ETW). It
has never been run and has not been validated against a real trace; the xperf command line and its text
dump format (`Pmc` / `CSwitch` lines) must be checked on first use. It requires administrator privileges
and the Windows Performance Toolkit (Windows ADK).

**Why Intel PCM was not used:** for MSR/PMU access, the Windows build of PCM requires either its own
`MSR.sys` driver (which must be compiled and signed with the WDK; an unsigned driver loads only in
test-signing mode) or the WinRing0 driver. WinRing0 is blocked by Windows Defender because of a known
vulnerability (CVE-2020-14979), and it was decided not to use it in this work.

**Future work:** there are two driver-free options — ETW PMC (`xperf -pmc`, this draft) and Intel VTune
(Intel-signed sampling driver). Both require new measurements. In Phase 2 the same question ("is decode
limited by memory bandwidth?") was answered analytically: `scripts/analyze_bandwidth.py`,
`docs/profiling_design.md` §7.

The script imports the model registry from `src/llm_cpu_inference/models.py`; to run it, add `src/` to
`PYTHONPATH` (e.g. `$env:PYTHONPATH = "src"` in PowerShell) or install the package with `pip install -e .`.
