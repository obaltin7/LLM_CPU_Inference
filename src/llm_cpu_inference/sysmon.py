"""CPU frequency sampling, thermal-throttling heuristic and cool-down waits.

psutil.cpu_freq() is static on Windows: it always returns the nominal frequency (3504 MHz on
the reference machine, even while the cores run at ~4800 MHz). The effective frequency is
therefore derived from the PDH counter '% Processor Performance', which reports the delivered
performance relative to the nominal frequency. The English counter name is used so the code
does not depend on the Windows display language.
"""
import ctypes
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import psutil

from llm_cpu_inference.log import get_logger

_COUNTER_PATH = r"\Processor Information(_Total)\% Processor Performance"
_PDH_FMT_DOUBLE = 0x00000200
_PDH_VALID_STATUS = (0, 1)  # PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA

NAN = float("nan")


class _PdhValue(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_uint32), ("doubleValue", ctypes.c_double)]


class FrequencyProbe:
    """Reads the effective CPU frequency (MHz), averaged since the previous read."""

    def __init__(self):
        self.source = "unavailable"
        self.reason = ""
        freq = psutil.cpu_freq()
        self.nominal_mhz = float(freq.max or freq.current) if freq else NAN
        self._pdh = self._query = self._counter = None

        if not sys.platform.startswith("win"):
            self.reason = "PDH counters exist only on Windows"
        elif math.isnan(self.nominal_mhz):
            self.reason = "nominal CPU frequency is unknown"
        else:
            try:
                self._open()
                self.source = "pdh_processor_performance"
            except OSError as exc:
                self.reason = str(exc)
        if self.source == "unavailable":
            get_logger().warning("CPU frequency cannot be measured (%s); frequency columns will be NaN.",
                                 self.reason)

    def _open(self):
        from ctypes import wintypes

        pdh = ctypes.WinDLL("pdh")
        pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(wintypes.HANDLE)]
        pdh.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_void_p,
                                              ctypes.POINTER(wintypes.HANDLE)]
        pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        pdh.PdhGetFormattedCounterValue.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                    ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_PdhValue)]
        query, counter = wintypes.HANDLE(), wintypes.HANDLE()
        status = pdh.PdhOpenQueryW(None, None, ctypes.byref(query))
        if status != 0:
            raise OSError(f"PdhOpenQueryW failed with status {status & 0xFFFFFFFF:#010x}")
        status = pdh.PdhAddEnglishCounterW(query, _COUNTER_PATH, None, ctypes.byref(counter))
        if status != 0:
            raise OSError(f"PdhAddEnglishCounterW failed with status {status & 0xFFFFFFFF:#010x}")
        pdh.PdhCollectQueryData(query)  # Rate counter: the first collection only primes it
        self._pdh, self._query, self._counter = pdh, query, counter

    def read_mhz(self):
        """Effective frequency averaged over the time since the previous call (NaN if unavailable)."""
        if self._pdh is None:
            return NAN
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            return NAN
        value = _PdhValue()
        status = self._pdh.PdhGetFormattedCounterValue(self._counter, _PDH_FMT_DOUBLE, None, ctypes.byref(value))
        if status != 0 or value.CStatus not in _PDH_VALID_STATUS:
            return NAN
        return self.nominal_mhz * value.doubleValue / 100.0

    def describe(self):
        return {"source": self.source, "reason_unavailable": self.reason or None,
                "nominal_mhz": self.nominal_mhz, "counter": _COUNTER_PATH}


@dataclass
class FrequencyStats:
    mean_mhz: float
    min_mhz: float
    max_mhz: float
    n_samples: int


class FrequencySampler:
    """Samples a FrequencyProbe in a background thread between start() and stop()."""

    def __init__(self, probe, interval_s=0.5):
        self._probe = probe
        self._interval = interval_s
        self._samples = []  # (mhz, seconds covered by the sample)
        self._stop_event = threading.Event()
        self._thread = None
        self._last_read = None

    def _take(self):
        now = time.perf_counter()
        self._samples.append((self._probe.read_mhz(), now - self._last_read))
        self._last_read = now

    def _run(self):
        while not self._stop_event.wait(self._interval):
            self._take()

    def start(self):
        self._probe.read_mhz()  # Reset the averaging window to "now"
        self._last_read = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        # Cover the tail since the last periodic sample, unless it is too short to be meaningful
        if not self._samples or time.perf_counter() - self._last_read >= self._interval / 2:
            self._take()
        valid = [(mhz, dt) for mhz, dt in self._samples if not math.isnan(mhz) and dt > 0]
        if not valid:
            return FrequencyStats(NAN, NAN, NAN, 0)
        total = sum(dt for _, dt in valid)
        return FrequencyStats(
            mean_mhz=sum(mhz * dt for mhz, dt in valid) / total,  # Time-weighted mean
            min_mhz=min(mhz for mhz, _ in valid),
            max_mhz=max(mhz for mhz, _ in valid),
            n_samples=len(valid),
        )


def flag_throttling(freq_means, drop_frac, reference_mhz=None):
    """Mark measurements whose mean frequency fell clearly below the best one of the same configuration.

    The reference is the highest mean frequency observed for the configuration (turbo limits depend
    on the number of active cores, so configurations must not be mixed). A measurement is flagged
    when it is more than `drop_frac` below that reference. This is a heuristic: it signals suspected
    thermal or power-limit throttling, it does not prove it. Returns None where the frequency is NaN.
    """
    valid = [f for f in freq_means if f is not None and not math.isnan(f)]
    if reference_mhz is None:
        reference_mhz = max(valid) if valid else NAN
    flags = []
    for freq in freq_means:
        if freq is None or math.isnan(freq) or math.isnan(reference_mhz):
            flags.append(None)
        else:
            flags.append(freq < (1.0 - drop_frac) * reference_mhz)
    return flags


def cooldown(seconds, reason=""):
    """Idle wait that lets the package temperature and turbo budget recover between runs."""
    if seconds and seconds > 0:
        get_logger().info("Cooling down for %.0f s%s", seconds, f" ({reason})" if reason else "")
        time.sleep(seconds)


def system_uptime_s():
    """Seconds since the last boot (GetTickCount64), or NaN if unavailable."""
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetTickCount64.restype = ctypes.c_uint64
        return kernel32.GetTickCount64() / 1000.0
    except (AttributeError, OSError):
        return NAN


def wait_for_uptime(min_uptime_s, thermal):
    """Wait until the system has been up for `min_uptime_s` (post-boot tasks settle). Returns the waited time."""
    uptime = system_uptime_s()
    if math.isnan(uptime) or uptime >= min_uptime_s:
        return 0.0
    wait = min_uptime_s - uptime
    get_logger().info("System booted %.0f s ago; waiting %.0f s until %.0f min after boot before measuring.",
                      uptime, wait, min_uptime_s / 60)
    thermal.sleep(wait)
    return wait


def wait_for_quiet(threshold_pct, window_s, max_wait_s, thermal):
    """Wait until the mean CPU utilisation over the last `window_s` seconds is below `threshold_pct`.

    Sampled once per second (psutil.cpu_percent, whole system). Returns a dict with the result;
    `reached` is False if `max_wait_s` elapsed first - the caller must then stop, not continue.
    """
    log = get_logger()
    log.info("Waiting for a quiet system: mean CPU utilisation < %.1f %% over %d s (at most %.0f min)...",
             threshold_pct, window_s, max_wait_s / 60)
    samples = deque(maxlen=int(window_s))
    psutil.cpu_percent(interval=None)
    start = time.perf_counter()
    last_log = start
    while True:
        thermal.check()
        samples.append(psutil.cpu_percent(interval=1.0))
        waited = time.perf_counter() - start
        mean = sum(samples) / len(samples)
        if len(samples) == samples.maxlen and mean < threshold_pct:
            log.info("System is quiet: mean CPU utilisation %.2f %% over the last %d s (waited %.0f s).",
                     mean, window_s, waited)
            return {"reached": True, "waited_s": waited, "mean_cpu_percent": mean, "max_cpu_percent": max(samples)}
        if waited >= max_wait_s:
            return {"reached": False, "waited_s": waited, "mean_cpu_percent": mean, "max_cpu_percent": max(samples)}
        if time.perf_counter() - last_log >= 60:
            log.info("Not quiet yet: mean CPU utilisation %.2f %% over the last %d s (%.0f s waited).",
                     mean, len(samples), waited)
            last_log = time.perf_counter()
