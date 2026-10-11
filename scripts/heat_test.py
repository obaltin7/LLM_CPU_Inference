"""Heat test: run the hottest workload (highest thread count) for a fixed time and report temperatures.

Must be run, and reported, before the final campaign. It uses the same sensor, thermal fuse and
power checks as the campaign: it refuses to start if the CPU temperature cannot be read, and it
stops at once (THERMAL_ABORT, exit code 3) if the fuse trips.

    python heat_test.py                 # 2 minutes, 8B-Q4_K_M, 16 threads (configs/experiment.yaml: heat_test)

Result: results/smoke/heat_test_<timestamp>.json and a log in results/smoke/logs/.
"""
import argparse
import math
import os
import sys
import time
from datetime import datetime

import psutil

import _paths  # noqa: F401  (makes src/ importable without installing the package)
from llm_cpu_inference.config import PROJECT_ROOT, load_config, resolve_thread_count
from llm_cpu_inference.energy import EnergyMeter
from llm_cpu_inference.generation import end_of_generation_ids, greedy_decode, load_model
from llm_cpu_inference.log import setup_logging
from llm_cpu_inference.power import power_state, prevent_sleep
from llm_cpu_inference.results_io import write_json_atomic
from llm_cpu_inference.run_metadata import detect_topology, now_iso
from llm_cpu_inference.session import preflight
from llm_cpu_inference.sysmon import FrequencyProbe
from llm_cpu_inference.thermal import TemperatureUnavailable, ThermalAbort, ThermalMonitor, open_temperature_source

EXIT_OK, EXIT_REFUSED, EXIT_THERMAL_ABORT = 0, 2, 3


def main():
    parser = argparse.ArgumentParser(description="Heat test at the highest thread count.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--duration", type=float, default=None, help="Seconds of continuous load (default: config)")
    args = parser.parse_args()

    config = load_config(args.config, smoke=True)
    heat, thermal_config = config["heat_test"], config["thermal"]
    out_dir = os.path.join(PROJECT_ROOT, config["paths"]["results_dir"])
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    stamp = f"{datetime.now():%Y%m%dT%H%M%S}"
    log = setup_logging(os.path.join(out_dir, "logs", f"heat_test_{stamp}.log"))
    duration = args.duration or heat["duration_s"]
    n_threads = resolve_thread_count(heat["n_threads"], detect_topology())

    preflight(config)  # Logs power / git warnings; a heat test is never a --final run
    power = power_state()
    log.info("Power: plan %s, max processor state AC %s %% / DC %s %%, turbo disabled: %s", power["name"],
             power["max_processor_state_ac"], power["max_processor_state_dc"], power["turbo_disabled"])
    try:
        source = open_temperature_source(thermal_config, PROJECT_ROOT)
    except TemperatureUnavailable as exc:
        log.error("REFUSED: CPU temperature is not readable: %s", exc)
        return EXIT_REFUSED

    monitor = ThermalMonitor(source, thermal_config).start()
    high = n_threads >= thermal_config["high_thread_count"]
    limit = thermal_config["high_thread_abort_c"] if high else thermal_config["abort_c"]
    result = {"timestamp": now_iso(), "model": heat["model"], "n_threads": n_threads, "duration_s": duration,
              "power": power, "sensor": source.describe(), "fuse_limit_c": limit,
              "fuse_hold_s": thermal_config["abort_hold_s"], "hard_abort_c": thermal_config["hard_abort_c"],
              "status": "running", "timeline": []}
    path = os.path.join(out_dir, f"heat_test_{stamp}.json")
    probe = FrequencyProbe()
    meter = None
    llm = None
    status = EXIT_OK
    try:
        with prevent_sleep():
            waited, _, start_c, _ = monitor.wait_until_below(thermal_config["cooldown_target_c"],
                                                             thermal_config["cooldown_max_s"], "before the heat test")
            result.update({"temp_at_start_c": start_c, "pre_cooldown_s": waited})
            meter = EnergyMeter(config["carbon"]["country_iso_code"])
            llm, load_time = load_model(heat["model"], n_threads, config["inference"])
            eog_ids = end_of_generation_ids(llm)
            prompt = llm.tokenize(config["prompts"][0]["text"].encode("utf-8"), add_bos=True, special=False)
            monitor.set_limit(limit)
            log.info("Heat test: %s, %d threads, %.0f s | start %.1f C | fuse %s C for %s s, hard %s C",
                     heat["model"], n_threads, duration, start_c, limit, thermal_config["abort_hold_s"],
                     thermal_config["hard_abort_c"])

            tokens = 0
            probe.read_mhz()
            meter.start()
            start = last_report = time.perf_counter()
            while time.perf_counter() - start < duration:
                llm.reset()
                monitor.check()
                llm.eval(prompt)
                generated, _ = greedy_decode(llm, 50, eog_ids, monitor)
                tokens += len(generated)
                now = time.perf_counter()
                if now - last_report >= 10:
                    point = {"t_s": round(now - start, 1), "temp_c": monitor.current(),
                             "freq_mhz": round(probe.read_mhz()), "cpu_percent": psutil.cpu_percent(None)}
                    result["timeline"].append(point)
                    log.info("t=%5.0f s | %.1f C | %d MHz | max so far %.1f C", point["t_s"], point["temp_c"],
                             point["freq_mhz"], monitor.max_seen_c)
                    last_report = now
            load_end = time.perf_counter()
            reading = meter.stop()
            stats = monitor.stats(start, load_end)
            result.update({
                "status": "completed", "load_s": load_end - start, "tokens": tokens,
                "tokens_per_sec": tokens / (load_end - start), "model_load_time_s": load_time,
                "temp_mean_c": stats.mean_c, "temp_max_c": stats.max_c, "temp_end_c": monitor.current(),
                "avg_cpu_package_w": reading.cpu_J / (load_end - start),
            })
            llm.close()
            llm = None
            monitor.set_limit(thermal_config["abort_c"])
            log.info("Load phase done: max %.1f C, mean %.1f C, %.1f W average package power, %.2f tok/s",
                     stats.max_c, stats.mean_c, result["avg_cpu_package_w"], result["tokens_per_sec"])
            cooled_s, _, end_c, reached = monitor.wait_until_below(thermal_config["cooldown_target_c"],
                                                                    thermal_config["cooldown_max_s"], "after the heat test")
            result.update({"cooldown_to_target_s": cooled_s, "cooldown_reached": reached, "temp_after_cooldown_c": end_c})
    except ThermalAbort as exc:
        result.update({"status": "thermal_abort", "abort_reason": str(exc)})
        log.error("THERMAL_ABORT during the heat test: %s", exc)
        status = EXIT_THERMAL_ABORT
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        status = 130
    finally:
        if meter is not None:
            meter.close()
        if llm is not None:
            llm.close()
        result["max_temperature_seen_c"] = None if math.isnan(monitor.max_seen_c) else monitor.max_seen_c
        monitor.stop()
        write_json_atomic(path, result)
    log.info("HEAT TEST RESULT: status %s | highest package temperature %.1f C | written to %s",
             result["status"], monitor.max_seen_c, path)
    return status


if __name__ == "__main__":
    sys.exit(main())
