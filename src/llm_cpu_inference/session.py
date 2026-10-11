"""A measurement run: one run_id, one metadata file, thermal monitoring and one idle baseline.

A run is one invocation of the campaign. A *session* (session_01, session_02, ...) can span
several runs when it is interrupted and resumed; every run records the session it belongs to.
"""
import math
import os

from llm_cpu_inference.config import PROJECT_ROOT, resolve_thread_count, resolve_thread_list, result_paths
from llm_cpu_inference.energy import NAN, EnergyMeter, measure_idle
from llm_cpu_inference.log import get_logger
from llm_cpu_inference.power import check_power, power_state
from llm_cpu_inference.run_metadata import (collect_run_metadata, detect_topology, git_info, new_run_id, now_iso,
                                write_run_metadata)
from llm_cpu_inference.sysmon import FrequencyProbe, system_uptime_s, wait_for_quiet, wait_for_uptime
from llm_cpu_inference.thermal import (TemperatureUnavailable, ThermalAbort, ThermalMonitor, is_admin,
                           open_temperature_source)


class RunRefused(RuntimeError):
    """A pre-condition of the run is not met; nothing was measured."""


def preflight(config):
    """Checks that do not need the hardware monitor. Returns (problems that refuse a final run, warnings)."""
    log = get_logger()
    refusals, warnings = [], []
    power = power_state()
    for problem in check_power(power, {"plan": config["power"]["expected_plan"],
                                       "max_processor_state": config["power"]["expected_max_processor_state"]}):
        (refusals if config["_meta"]["final"] else warnings).append(f"Power: {problem}")
    git = git_info()  # Logs the dirty-tree warning itself (once)
    if git["dirty"] and config["_meta"]["final"]:
        refusals.append(f"Git working tree is dirty ({', '.join(git['modified_tracked_files'])})")
    elif git["dirty"] is None:
        (refusals if config["_meta"]["final"] else warnings).append("Git state could not be determined")
    for warning in warnings:
        log.warning("%s (allowed outside --final)", warning)
    return refusals, power, git


class MeasurementSession:
    """Owns the shared state of one run.

    On start it checks the pre-conditions, opens the temperature sensor and starts the thermal
    monitor (refusing to run without it), detects the hardware, hashes the model files, sets up
    the energy meter, measures the idle baseline and writes results/metadata/<run_id>.json.
    On exit the metadata is finalised with the end time and the status.
    """

    def __init__(self, config, experiments, model_keys, results_dir=None, measure_energy=True, note=None,
                 session=None):
        self.config = config
        self.experiments = list(experiments)
        self.model_keys = list(model_keys)
        self.measure_energy = measure_energy
        self.note = note  # Free text from the operator (e.g. ambient temperature), stored in the metadata
        self.session = session or {"session_id": None}
        self.session_id = self.session["session_id"]
        self.paths = result_paths(config, results_dir)
        self.run_id = new_run_id()
        self.log = get_logger()
        self.topology = None
        self.probe = None
        self.meter = None
        self.thermal = None
        self.idle = {"cpu_W": NAN}
        self.metadata = None
        self.metadata_path = None

    @property
    def carbon_intensity(self):
        return self.meter.carbon_intensity if self.meter else NAN

    def resolved_threads(self):
        config, topology = self.config, self.topology
        return {
            "energy": resolve_thread_count(config["energy"]["n_threads"], topology),
            "cognitive": resolve_thread_count(config["cognitive"]["n_threads"], topology),
            "thread_scaling": resolve_thread_list(config["thread_scaling"]["threads"], topology),
        }

    def start(self):
        config = self.config
        final = config["_meta"]["final"]
        refusals, power, _ = preflight(config)
        if refusals:
            raise RunRefused("; ".join(refusals))

        for key in ("root", "metadata", "nll", "logs"):
            os.makedirs(self.paths[key], exist_ok=True)
        self.log.info("Run %s | %s | experiments: %s | models: %s", self.run_id, self.session_id,
                      ", ".join(self.experiments), ", ".join(self.model_keys))
        self.log.info("Power: plan %s (%s), max processor state AC %s %% / DC %s %%, turbo disabled: %s",
                      power["name"], power["canonical"], power["max_processor_state_ac"],
                      power["max_processor_state_dc"], power["turbo_disabled"])

        # Thermal safety first: no temperature, no measurement
        try:
            source = open_temperature_source(config["thermal"], PROJECT_ROOT)
        except TemperatureUnavailable as exc:
            raise RunRefused(f"CPU temperature is not readable, refusing to start: {exc}") from exc
        if final and source.describe()["is_test_fake"]:
            source.close()
            raise RunRefused("A --final run cannot use the test temperature source")
        self.thermal = ThermalMonitor(source, config["thermal"]).start()
        self.log.info("Thermal monitor: %s | now %.1f C | fuse %s C for %s s, hard limit %s C (administrator: %s)",
                      source.describe()["sensor"], self.thermal.current(), config["thermal"]["abort_c"],
                      config["thermal"]["abort_hold_s"], config["thermal"]["hard_abort_c"], is_admin())

        self.topology = detect_topology()
        self.log.info("Topology: %s physical cores, %s logical processors, hybrid=%s, P-cores=%s",
                      self.topology["physical_cores"], self.topology["logical_processors"],
                      self.topology["hybrid"], self.topology["p_cores"])
        self.metadata = collect_run_metadata(self.run_id, self.experiments, config, self.model_keys,
                                             self.topology, self.resolved_threads(), session=self.session)
        self.metadata["operator_note"] = self.note
        self.metadata["thermal"] = {**self.thermal.describe(), "temperature_at_start_c": self.thermal.current(),
                                    "cooldown_target_c": config["thermal"]["cooldown_target_c"],
                                    "cooldown_max_s": config["thermal"]["cooldown_max_s"],
                                    "high_thread_count": config["thermal"]["high_thread_count"],
                                    "high_thread_abort_c": config["thermal"]["high_thread_abort_c"]}

        self.probe = FrequencyProbe()
        self.metadata["frequency_tracking"] = self.probe.describe()

        if self.measure_energy:
            self.meter = EnergyMeter(config["carbon"]["country_iso_code"])
            self.log.info("Energy tracking: CPU source = %s (measured: %s), GPU tracked: %s",
                          self.meter.cpu_source, self.meter.cpu_measured, self.meter.has_gpu)
            idle_config = config["idle"]
            guard = config["session_start"]
            start_record = {"uptime_at_start_s": system_uptime_s(), "min_uptime_s": guard["min_uptime_s"]}
            if idle_config["duration_s"] > 0:
                # Post-boot tasks must have finished: wait until the configured time after boot
                start_record["uptime_wait_s"] = wait_for_uptime(guard["min_uptime_s"], self.thermal)
                # Hashing and the hardware inventory load the CPU; let it settle and cool first
                self.thermal.sleep(idle_config.get("settle_s", 0))
                self.thermal.wait_until_below(config["thermal"]["cooldown_target_c"],
                                              config["thermal"]["cooldown_max_s"], "before the idle baseline")
                # The idle baseline is only taken on a quiet system; otherwise stop with the reason
                quiet = wait_for_quiet(guard["quiet_threshold_pct"], guard["quiet_window_s"],
                                       guard["quiet_max_wait_s"], self.thermal)
                start_record["quiet_check"] = {**quiet, "threshold_pct": guard["quiet_threshold_pct"],
                                               "window_s": guard["quiet_window_s"]}
                self.metadata["session_start"] = start_record
                if not quiet["reached"]:
                    write_run_metadata(self.paths["metadata"], self.metadata)
                    raise RunRefused(
                        f"QUIET_TIMEOUT: the mean CPU utilisation did not drop below {guard['quiet_threshold_pct']} % "
                        f"over {guard['quiet_window_s']} s within {guard['quiet_max_wait_s'] / 60:.0f} min "
                        f"(last window: mean {quiet['mean_cpu_percent']:.2f} %, max {quiet['max_cpu_percent']:.1f} %). "
                        f"Something is running in the background; close it (Task Manager) and start again.")
            self.metadata["session_start"] = start_record
            self.idle = measure_idle(self.meter, self.probe, idle_config["duration_s"],
                                     config["thermal"]["freq_sample_interval_s"], thermal=self.thermal)
            self.metadata["energy_tracking"] = self.meter.describe()
            self.metadata["idle_baseline"] = {**self.idle, "measured_at": now_iso()}
            self.metadata["carbon"] = {
                "country_iso_code": self.meter.country_iso_code,
                "grid_intensity_gco2eq_per_kwh": self.meter.carbon_intensity,
                "source": "CodeCarbon built-in energy-mix table for the given country",
                "note": "gCO2eq/token = J/token x constant; it is not an independent metric.",
            }
        else:
            self.metadata["energy_tracking"] = {"tool": None, "note": "energy is not measured in this run"}

        self.metadata_path = write_run_metadata(self.paths["metadata"], self.metadata)
        self.log.info("Metadata written to %s", self.metadata_path)
        return self

    def record_unit(self, record):
        """Append one executed unit (the realised order) to the metadata and save it."""
        self.metadata["executed_units"].append(record)
        write_run_metadata(self.paths["metadata"], self.metadata)

    def finish(self, status):
        if self.meter is not None:
            self.meter.close()
            self.meter = None
        if self.thermal is not None:
            max_seen = self.thermal.max_seen_c
            self.thermal.stop()
            if self.metadata is not None:
                self.metadata["thermal"]["max_temperature_c"] = None if math.isnan(max_seen) else max_seen
                self.metadata["thermal"]["abort_reason"] = self.thermal.abort_reason
        if self.metadata is not None:
            self.metadata["status"] = status
            self.metadata["timestamp_end"] = now_iso()
            write_run_metadata(self.paths["metadata"], self.metadata)
            self.log.info("Run %s finished with status '%s'", self.run_id, status)

    def __enter__(self):
        try:
            return self.start()
        except BaseException as exc:
            # A refused or failed start must not leave the monitor thread or the meter running
            if isinstance(exc, ThermalAbort):
                status = "thermal_abort"
            elif isinstance(exc, KeyboardInterrupt):
                status = "interrupted"
            elif isinstance(exc, RunRefused) or self.metadata is None:
                status = "refused"
            else:
                status = "failed"
            self.finish(status)
            raise

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            status = "completed"
        elif issubclass(exc_type, ThermalAbort):
            status = "thermal_abort"
        elif issubclass(exc_type, KeyboardInterrupt):
            status = "interrupted"
        else:
            status = "failed"
        self.finish(status)
        return False
