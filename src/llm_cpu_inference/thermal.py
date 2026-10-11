"""CPU package temperature: sensor access, a background monitor with a thermal fuse, and cool-down waits.

Thermal safety comes before reproducibility in this project. The rules implemented here:

* The temperature must be readable before anything is measured; otherwise the run is refused.
* A background thread samples the CPU package temperature (default every 0.5 s).
* Thermal fuse: if the temperature stays above the abort limit for `abort_hold_s` seconds
  (default 85 C for 10 s), or exceeds `hard_abort_c` once, or cannot be read for
  `sensor_loss_s` seconds, the fuse trips. The measurement code calls `check()` between
  tokens / chunks and gets a ThermalAbort, which stops the unit before any of its rows are
  written. The campaign logs THERMAL_ABORT and can be resumed later.
* Cool-down waits are temperature-driven: wait until the package is at or below a target.

Sensor: LibreHardwareMonitorLib (loaded through pythonnet) reading the 'CPU Package'
temperature sensor. On this platform it needs the PawnIO driver and administrator rights.
"""
import ctypes
import math
import os
import sys
import threading
import time
from bisect import bisect_left
from dataclasses import dataclass

from llm_cpu_inference import PROJECT_ROOT
from llm_cpu_inference.log import get_logger

NAN = float("nan")
LHM_VERSION = "0.9.6"
PACKAGE_SENSOR_NAME = "CPU Package"
# Test hook: path of a text file holding a temperature in C. Only for exercising the fuse logic
# without hardware access; a --final run refuses it.
FAKE_SOURCE_ENV = "LLMCPU_TEST_FAKE_TEMPERATURE_FILE"


class TemperatureUnavailable(RuntimeError):
    """The CPU package temperature cannot be read; measurements must not start."""


class ThermalAbort(Exception):
    """The thermal fuse tripped; the current unit is discarded and the run stops safely."""


def is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return False


class LhmTemperatureSource:
    """CPU package temperature through LibreHardwareMonitorLib (.NET Framework build, via pythonnet)."""

    _runtime_loaded = False

    def __init__(self, lib_dir):
        self.lib_dir = os.path.abspath(lib_dir)
        dll = os.path.join(self.lib_dir, "LibreHardwareMonitorLib.dll")
        if not os.path.exists(dll):
            raise TemperatureUnavailable(
                f"{dll} not found. Run 'python scripts/fetch_lhm.py' to download LibreHardwareMonitor {LHM_VERSION}.")
        if not is_admin():
            raise TemperatureUnavailable(
                "Reading the CPU temperature needs administrator rights (LibreHardwareMonitor + PawnIO driver). "
                "Start the campaign from an elevated PowerShell ('Run as administrator').")
        try:
            if not LhmTemperatureSource._runtime_loaded:
                from pythonnet import load
                load("netfx")
                LhmTemperatureSource._runtime_loaded = True
            import clr

            if self.lib_dir not in sys.path:
                sys.path.append(self.lib_dir)
            clr.AddReference(dll)
            from LibreHardwareMonitor.Hardware import Computer, HardwareType, SensorType
        except Exception as exc:  # pythonnet raises a variety of CLR exception types
            raise TemperatureUnavailable(f"LibreHardwareMonitorLib could not be loaded: {exc}") from exc

        self._computer = Computer()
        self._computer.IsCpuEnabled = True
        self._computer.Open()
        self._hardware = next((h for h in self._computer.Hardware if h.HardwareType == HardwareType.Cpu), None)
        if self._hardware is None:
            self.close()
            raise TemperatureUnavailable("LibreHardwareMonitor found no CPU.")
        self._hardware.Update()
        self._sensor = next((s for s in self._hardware.Sensors
                             if s.SensorType == SensorType.Temperature and s.Name == PACKAGE_SENSOR_NAME), None)
        self.cpu_name = str(self._hardware.Name)
        if self._sensor is None:
            self.close()
            raise TemperatureUnavailable(f"LibreHardwareMonitor exposes no '{PACKAGE_SENSOR_NAME}' sensor.")
        if math.isnan(self.read()):
            self.close()
            raise TemperatureUnavailable(
                "The CPU package temperature reads as empty. The PawnIO driver is probably not installed "
                "(winget install --id namazso.PawnIO) or the process is not elevated.")

    def read(self):
        """Current package temperature in C, or NaN if the sensor returned nothing."""
        try:
            self._hardware.Update()
            value = self._sensor.Value
        except Exception:  # A failing read must not kill the monitor thread; NaN is handled by the fuse
            return NAN
        return float(value) if value is not None else NAN

    def close(self):
        try:
            self._computer.Close()
        except Exception:
            pass

    def describe(self):
        return {"source": f"LibreHardwareMonitorLib {LHM_VERSION} via pythonnet (netfx)",
                "sensor": f"{self.cpu_name} / {PACKAGE_SENSOR_NAME}",
                # Relative to the project root so the metadata carries no machine-specific path
                "lib_dir": os.path.relpath(self.lib_dir, PROJECT_ROOT).replace("\\", "/"),
                "is_test_fake": False}


class FakeFileTemperatureSource:
    """Test-only source: reads the temperature from a text file (see FAKE_SOURCE_ENV)."""

    def __init__(self, path):
        self.path = path
        if math.isnan(self.read()):
            raise TemperatureUnavailable(f"Fake temperature file {path} is unreadable.")

    def read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return float(f.read().strip())
        except (OSError, ValueError):
            return NAN

    def close(self):
        pass

    def describe(self):
        return {"source": "TEST FAKE (temperature read from a file)", "sensor": self.path, "lib_dir": None,
                "is_test_fake": True}


def open_temperature_source(thermal_config, project_root):
    """Open the configured temperature source or raise TemperatureUnavailable."""
    fake = os.environ.get(FAKE_SOURCE_ENV)
    if fake:
        get_logger().warning("TEST MODE: CPU temperature is read from %s, not from the hardware.", fake)
        return FakeFileTemperatureSource(fake)
    if thermal_config["sensor"] != "lhm":
        raise TemperatureUnavailable(f"Unknown temperature sensor '{thermal_config['sensor']}' (supported: lhm)")
    return LhmTemperatureSource(os.path.join(project_root, thermal_config["lhm_dir"]))


@dataclass
class TemperatureStats:
    mean_c: float
    max_c: float
    n_samples: int


class ThermalMonitor:
    """Samples the package temperature in the background and trips a fuse when it is too hot."""

    def __init__(self, source, thermal_config):
        self.source = source
        self.interval_s = thermal_config["sample_interval_s"]
        self.abort_c = thermal_config["abort_c"]
        self.hard_abort_c = thermal_config["hard_abort_c"]
        self.abort_hold_s = thermal_config["abort_hold_s"]
        self.sensor_loss_s = thermal_config["sensor_loss_s"]
        self._limit_c = self.abort_c
        self._times = []  # perf_counter timestamps of the samples
        self._temps = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._abort_reason = None
        self._abort_event = threading.Event()
        self._over_since = None
        self._last_valid = None
        self._thread = None
        self.max_seen_c = NAN
        self._log = get_logger()

    # ---- background sampling -------------------------------------------------------------
    def start(self):
        self._last_valid = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="thermal-monitor", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop_event.wait(self.interval_s):
            self._sample()

    def _sample(self):
        temp = self.source.read()
        now = time.perf_counter()
        with self._lock:
            self._times.append(now)
            self._temps.append(temp)
        if math.isnan(temp):
            if now - self._last_valid >= self.sensor_loss_s:
                self._trip(f"CPU temperature could not be read for {now - self._last_valid:.0f} s")
            return
        self._last_valid = now
        if math.isnan(self.max_seen_c) or temp > self.max_seen_c:
            self.max_seen_c = temp
        if temp >= self.hard_abort_c:
            self._trip(f"CPU package temperature {temp:.1f} C reached the hard limit {self.hard_abort_c} C")
        elif temp > self._limit_c:
            if self._over_since is None:
                self._over_since = now
            elif now - self._over_since >= self.abort_hold_s:
                self._trip(f"CPU package temperature above {self._limit_c} C for "
                           f"{now - self._over_since:.0f} s (now {temp:.1f} C)")
        else:
            self._over_since = None

    def _trip(self, reason):
        if not self._abort_event.is_set():
            self._abort_reason = reason
            self._abort_event.set()
            self._log.error("THERMAL_ABORT: %s", reason)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.source.close()

    # ---- used by the measurement code ----------------------------------------------------
    @property
    def limit_c(self):
        return self._limit_c

    def set_limit(self, limit_c):
        """Abort limit for the current phase (e.g. a stricter one for the highest thread counts)."""
        self._limit_c = limit_c
        self._over_since = None

    @property
    def tripped(self):
        return self._abort_event.is_set()

    @property
    def abort_reason(self):
        return self._abort_reason

    def check(self):
        """Raise ThermalAbort if the fuse has tripped. Called between tokens and chunks."""
        if self._abort_event.is_set():
            raise ThermalAbort(self._abort_reason)

    def current(self):
        """Latest valid temperature (NaN if none yet)."""
        with self._lock:
            for temp in reversed(self._temps):
                if not math.isnan(temp):
                    return temp
        return NAN

    def stats(self, start, end):
        """Mean and maximum of the samples taken between two perf_counter instants."""
        with self._lock:
            lo = bisect_left(self._times, start)
            hi = bisect_left(self._times, end)
            window = [t for t in self._temps[lo:hi] if not math.isnan(t)]
        if not window:  # Window shorter than the sampling interval: use the latest reading
            current = self.current()
            return TemperatureStats(current, current, 0 if math.isnan(current) else 1)
        return TemperatureStats(sum(window) / len(window), max(window), len(window))

    def sleep(self, seconds):
        """Sleep while still honouring the fuse."""
        end = time.perf_counter() + seconds
        while True:
            self.check()
            remaining = end - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(0.5, remaining))

    def wait_until_below(self, target_c, max_wait_s, reason=""):
        """Temperature-driven cool-down. Returns (waited_s, start_c, end_c, reached)."""
        start = time.perf_counter()
        start_c = self.current()
        logged = False
        while True:
            self.check()
            temp = self.current()
            waited = time.perf_counter() - start
            if not math.isnan(temp) and temp <= target_c:
                if logged:
                    self._log.info("Cooled down to %.1f C in %.0f s", temp, waited)
                return waited, start_c, temp, True
            if waited >= max_wait_s:
                self._log.warning("Cool-down did not reach %.0f C within %.0f s (now %.1f C)%s",
                                  target_c, max_wait_s, temp, f" ({reason})" if reason else "")
                return waited, start_c, temp, False
            if not logged:
                self._log.info("Cooling down: %.1f C -> target %.0f C (max %.0f s)%s", temp, target_c, max_wait_s,
                               f" ({reason})" if reason else "")
                logged = True
            time.sleep(1.0)

    def describe(self):
        return {**self.source.describe(), "sample_interval_s": self.interval_s, "abort_c": self.abort_c,
                "abort_hold_s": self.abort_hold_s, "hard_abort_c": self.hard_abort_c,
                "sensor_loss_s": self.sensor_loss_s}
