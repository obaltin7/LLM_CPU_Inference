"""Energy measurement through CodeCarbon, split by component, plus the idle baseline.

Only the CPU package energy is a hardware measurement on the reference machine (RAPL package
counter read through the Windows Energy Meter Interface). RAM energy is a constant-power
estimate made by CodeCarbon (the RAPL DRAM channel reads 0 W on this platform), and the GPU
energy is the NVML reading of a card that only drives the display. The headline metric is
therefore CPU package energy per token; RAM and GPU are stored in separate columns.
"""
import logging
import math
import time
from dataclasses import dataclass

import psutil
from codecarbon import OfflineEmissionsTracker

from llm_cpu_inference.log import get_logger
from llm_cpu_inference.sysmon import FrequencySampler

J_PER_KWH = 3_600_000.0
NAN = float("nan")

# CodeCarbon CPU modes backed by hardware energy counters. Every other mode
# (cpu_load, constant) is a TDP-based estimate and is not reported as a measurement.
MEASURED_CPU_SOURCES = {"windows_emi", "intel_power_gadget", "intel_rapl"}


@dataclass
class EnergyReading:
    duration_s: float
    cpu_J: float
    ram_J_est: float
    gpu_J: float
    total_J: float  # cpu_J + ram_J_est + gpu_J as summed by CodeCarbon


class EnergyMeter:
    """Thin wrapper around a CodeCarbon tracker used in task mode.

    Each start()/stop() pair is one CodeCarbon task: the energy is the difference of the
    cumulative hardware counter between the two calls, so it does not depend on the
    sampling interval. The country is passed explicitly (offline tracker): no IP
    geolocation request is made and the carbon intensity is reproducible.
    """

    def __init__(self, country_iso_code, measure_power_secs=1):
        self.country_iso_code = country_iso_code
        self.measure_power_secs = measure_power_secs
        self.carbon_intensity = NAN  # gCO2eq/kWh actually applied by CodeCarbon
        self._active = False
        self._log = get_logger()
        # CodeCarbon logs a few lines before it applies log_level; silence them up front
        logging.getLogger("codecarbon").setLevel(logging.ERROR)
        self._tracker = OfflineEmissionsTracker(
            country_iso_code=country_iso_code,
            measure_power_secs=measure_power_secs,
            log_level="error",
            save_to_file=False,
        )

        # Hardware is detected lazily by CodeCarbon; a short priming task forces the setup
        self.start()
        time.sleep(1.0)
        primed = self._tracker.stop_task()
        self._active = False
        self._update_intensity(primed)

        self.cpu_source = "unknown"
        self.has_gpu = False
        for hardware in getattr(self._tracker, "_hardware", []):
            kind = type(hardware).__name__
            if kind == "CPU":
                self.cpu_source = getattr(hardware, "_mode", "unknown")
            elif kind == "GPU":
                self.has_gpu = True
        self.cpu_measured = self.cpu_source in MEASURED_CPU_SOURCES

        if not self.cpu_measured:
            self._log.warning("CPU energy source '%s' is not a hardware counter: cpu_J and every metric "
                              "derived from it are written as NaN (estimates are not reported as "
                              "measurements).", self.cpu_source)
        if not self.has_gpu:
            self._log.warning("No GPU is tracked by CodeCarbon: gpu_J is written as NaN.")

    def _update_intensity(self, data):
        if data is not None and data.energy_consumed and data.energy_consumed > 0:
            self.carbon_intensity = data.emissions / data.energy_consumed * 1000.0

    def start(self):
        self._tracker.start_task()
        self._active = True

    def stop(self):
        data = self._tracker.stop_task()
        self._active = False
        self._update_intensity(data)
        cpu_j = data.cpu_energy * J_PER_KWH if self.cpu_measured else NAN
        gpu_j = data.gpu_energy * J_PER_KWH if self.has_gpu else NAN
        ram_j = data.ram_energy * J_PER_KWH
        return EnergyReading(duration_s=data.duration, cpu_J=cpu_j, ram_J_est=ram_j, gpu_J=gpu_j,
                             total_J=cpu_j + ram_j + gpu_j)

    def abort(self):
        """Close a measurement window left open by an exception or Ctrl+C."""
        if self._active:
            try:
                self._tracker.stop_task()
            finally:
                self._active = False

    def close(self):
        self.abort()
        self._tracker.stop()

    def describe(self):
        return {
            "tool": "codecarbon.OfflineEmissionsTracker (task mode)",
            "tracking_mode": getattr(self._tracker, "_tracking_mode", None),
            "measure_power_secs": self.measure_power_secs,
            "cpu_energy_source": self.cpu_source,
            "cpu_energy_is_measured": self.cpu_measured,
            "cpu_energy_scope": "CPU package (RAPL PKG domain), machine-wide" if self.cpu_measured
                                else "TDP-based estimate, written as NaN",
            "ram_energy_source": "CodeCarbon RAM power estimation model (constant power, NOT measured)",
            "gpu_energy_source": "pynvml (NVML power reading)" if self.has_gpu else "none",
            "headline_metric": "cpu_J_per_token (CPU package energy of the decode window / generated tokens); "
                               "RAM and GPU energy are excluded",
        }


def measure_idle(meter, probe, duration_s, freq_interval_s=0.5, thermal=None):
    """Measure idle power for `duration_s` seconds; the result is the baseline for net energy.

    With a thermal monitor the wait honours the thermal fuse and the package temperature
    during the baseline is recorded.
    """
    log = get_logger()
    if not duration_s or duration_s <= 0:
        log.warning("Idle baseline is disabled (duration <= 0): net (idle-subtracted) metrics will be NaN.")
        return {"duration_s": 0.0, "cpu_W": NAN, "ram_W_est": NAN, "gpu_W": NAN, "cpu_util_percent": NAN,
                "freq_mean_mhz": NAN, "freq_min_mhz": NAN, "temp_mean_C": NAN, "temp_max_C": NAN}

    log.info("Measuring idle baseline for %.0f s (keep the machine untouched)...", duration_s)
    psutil.cpu_percent(interval=None)  # Reset the utilisation window
    sampler = FrequencySampler(probe, freq_interval_s).start()
    meter.start()
    start = time.perf_counter()
    try:
        if thermal is not None:
            thermal.sleep(duration_s)
        else:
            time.sleep(duration_s)
    finally:
        end = time.perf_counter()
        elapsed = end - start
        reading = meter.stop()
        freq = sampler.stop()
    cpu_util = psutil.cpu_percent(interval=None)
    temp = thermal.stats(start, end) if thermal is not None else None

    idle = {
        "duration_s": elapsed,
        "cpu_W": reading.cpu_J / elapsed,
        "ram_W_est": reading.ram_J_est / elapsed,
        "gpu_W": reading.gpu_J / elapsed,
        "cpu_util_percent": cpu_util,
        "freq_mean_mhz": freq.mean_mhz,
        "freq_min_mhz": freq.min_mhz,
        "temp_mean_C": temp.mean_c if temp else NAN,
        "temp_max_C": temp.max_c if temp else NAN,
    }
    log.info("Idle baseline: CPU package %.2f W | RAM (est.) %.2f W | GPU %.2f W | CPU utilisation %.1f %% | "
             "%.1f C", idle["cpu_W"], idle["ram_W_est"], idle["gpu_W"], cpu_util, idle["temp_mean_C"])
    if not math.isnan(cpu_util) and cpu_util > 5.0:
        log.warning("CPU utilisation during the idle baseline was %.1f %% (> 5 %%): the baseline is not "
                    "clean and net energy will be underestimated. Close background applications.", cpu_util)
    return idle


def gco2eq(joules, intensity_g_per_kwh):
    """Emissions in gCO2eq for an energy in joules.

    NOTE: this is energy multiplied by a constant grid factor. gCO2eq/token is therefore
    NOT an independent metric; it carries exactly the same information as J/token.
    """
    return joules / J_PER_KWH * intensity_g_per_kwh
