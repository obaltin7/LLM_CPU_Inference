"""Windows power state: active plan, maximum processor state (Turbo Boost), and sleep prevention.

Turbo Boost is disabled by limiting the "maximum processor state" of the active power plan to
99 %: with the value below 100 % Windows does not request turbo frequencies, so the cores stay
at the nominal clock (3.5 GHz on the i9-11900K, verified under load).
"""
import contextlib
import ctypes
import re
import subprocess
import sys

from llm_cpu_inference.log import get_logger

# Built-in Windows power schemes; the display name is localised, the GUID is not.
POWER_PLANS = {
    "381b4222-f694-41f0-9685-ff5bb260df2e": "balanced",
    "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "high_performance",
    "a1841308-3541-4fab-bc81-f71556f20b4a": "power_saver",
    "e9a42b02-d5df-448d-aa00-03f14749eb61": "ultimate_performance",
}

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


def _powercfg(*args):
    try:
        completed = subprocess.run(["powercfg", *args], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        get_logger().warning("powercfg %s failed: %s", " ".join(args), exc)
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("oem", errors="replace")


def active_power_plan():
    """Active scheme: GUID, localised name and a language-independent label."""
    if not sys.platform.startswith("win"):
        return {"guid": None, "name": None, "canonical": None}
    output = _powercfg("/getactivescheme") or ""
    match = re.search(r"([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})\s*(?:\((.*)\))?", output)
    if not match:
        get_logger().warning("Active power plan could not be determined.")
        return {"guid": None, "name": None, "canonical": None}
    guid = match.group(1).lower()
    return {"guid": guid, "name": match.group(2), "canonical": POWER_PLANS.get(guid, "custom")}


def max_processor_state():
    """Maximum processor state (%) of the active plan on AC and DC power, or None if unknown.

    powercfg prints the current values as the last two hexadecimal indices of the query
    (AC first, then DC); their labels are not relied upon.
    """
    output = _powercfg("/query", "SCHEME_CURRENT", "SUB_PROCESSOR", "PROCTHROTTLEMAX")
    values = re.findall(r":\s*(0x[0-9a-fA-F]+)\s*$", output or "", flags=re.MULTILINE)
    if len(values) < 2:
        get_logger().warning("Maximum processor state could not be read from powercfg.")
        return {"ac": None, "dc": None}
    return {"ac": int(values[-2], 16), "dc": int(values[-1], 16)}


def power_state():
    """Everything the metadata records about the power configuration."""
    plan = active_power_plan()
    state = max_processor_state()
    return {
        **plan,
        "max_processor_state_ac": state["ac"],
        "max_processor_state_dc": state["dc"],
        # Desktop on mains power: the AC value is the one in effect
        "turbo_disabled": None if state["ac"] is None else state["ac"] < 100,
    }


def check_power(state, expected):
    """Differences between the actual and the expected power configuration (empty list = OK)."""
    problems = []
    if state["canonical"] != expected["plan"]:
        problems.append(f"active power plan is '{state['name']}' ({state['canonical']}), "
                        f"expected '{expected['plan']}'")
    for side in ("ac", "dc"):
        value = state[f"max_processor_state_{side}"]
        if value != expected["max_processor_state"]:
            problems.append(f"maximum processor state ({side.upper()}) is {value} %, "
                            f"expected {expected['max_processor_state']} %")
    return problems


def set_max_processor_state(percent):
    """Set the maximum processor state of the active plan on AC and DC (100 = Turbo Boost on, 99 = off)."""
    ok = True
    for side in ("/setacvalueindex", "/setdcvalueindex"):
        ok = _powercfg(side, "SCHEME_CURRENT", "SUB_PROCESSOR", "PROCTHROTTLEMAX", str(int(percent))) is not None and ok
    ok = _powercfg("/setactive", "SCHEME_CURRENT") is not None and ok
    if not ok:
        raise RuntimeError(f"powercfg could not set the maximum processor state to {percent} %")


class TurboController:
    """Switches Turbo Boost through the maximum processor state and remembers every change.

    dry_run=True (smoke tests) never calls powercfg: the state is kept in memory so the control
    flow, including the restore in `finally`, can be tested without touching the machine.
    """

    def __init__(self, off_percent=99, on_percent=100, dry_run=False):
        self.off_percent, self.on_percent, self.dry_run = off_percent, on_percent, dry_run
        self.calls = []
        self._dry_state = off_percent

    def current(self):
        return self._dry_state if self.dry_run else max_processor_state()["ac"]

    def set(self, percent):
        self.calls.append(int(percent))
        if self.dry_run:
            self._dry_state = int(percent)
        else:
            set_max_processor_state(percent)
        get_logger().info("Turbo: maximum processor state set to %s %% (%s)", percent,
                          "Turbo Boost ON" if percent >= 100 else "Turbo Boost OFF")

    def turbo(self, on):
        self.set(self.on_percent if on else self.off_percent)

    def ensure_off(self):
        """Start-of-run guard: a previous run may have left Turbo on; fix it before any check."""
        state = self.current()
        if state is None or state >= 100:
            get_logger().warning("Maximum processor state is %s %% at start (Turbo left on?); setting %s %%.",
                                 state, self.off_percent)
            self.turbo(False)
            return True
        return False


@contextlib.contextmanager
def prevent_sleep():
    """Keep the system and the display awake for the duration of the block (SetThreadExecutionState)."""
    log = get_logger()
    try:
        kernel32 = ctypes.windll.kernel32
    except AttributeError:
        yield False
        return
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
    previous = kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED)
    if previous == 0:
        log.warning("SetThreadExecutionState failed: sleep is not prevented during the run.")
    else:
        log.info("System sleep and display sleep are prevented until the run ends.")
    try:
        yield previous != 0
    finally:
        kernel32.SetThreadExecutionState(_ES_CONTINUOUS)  # Back to the normal idle behaviour
        log.info("Sleep prevention released.")
