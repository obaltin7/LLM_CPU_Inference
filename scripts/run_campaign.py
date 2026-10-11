"""Campaign orchestrator: sessions, interleaved rounds, thermal safety and resume.

Sessions
  The campaign is measured in several independent sessions (session_01, session_02, ...),
  ideally on different days after a reboot. The first start creates session_01. Running the
  same command again resumes the current session until it is complete; a new session is
  started explicitly with --new-session. Every CSV row and every metadata file carries the
  session_id, so within- and between-session variance can be separated (analyze_sessions.py).

Interleaved rounds (energy, thread_scaling)
  Instead of measuring one model after another, every round measures every configuration
  (model, or model x thread count) once, in an order shuffled per round. The shuffle seed is
  derived from campaign.shuffle_seed and the session id; the planned and the realised order
  are stored in the state file and in the run metadata. A unit = one configuration in one
  round: load, warm-up round (discarded), every prompt once. Summaries over the rounds of a
  session are written when the session's experiment is complete.

Cognitive
  Perplexity is deterministic and measured only in session_01 (cognitive.first_session_only).

Safety
  The run refuses to start without a readable CPU temperature. Before every unit it waits until
  the package is at or below thermal.cooldown_target_c (at most cooldown_max_s). A thermal fuse
  stops the run safely (THERMAL_ABORT, exit code 3) without writing partial rows; the same
  command resumes later. --final additionally refuses a dirty git tree, the wrong power plan
  and the test temperature source. Sleep is prevented while the campaign runs.

Examples:
  python run_campaign.py --list                     plan of the current/next session (no measurement)
  python run_campaign.py --smoke                    short end-to-end check (results/smoke/)
  python run_campaign.py --final --note "ambient 23 C"
  python run_campaign.py --final --experiments energy
  python run_campaign.py --final --new-session      start the next session (another day, after a reboot)
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass

import _paths  # noqa: F401  (makes src/ importable without installing the package)
from benchmark_cognitive import load_dataset, order_reference_first, run_cognitive_unit
from llm_cpu_inference.models import MODELS
from llm_cpu_inference.config import load_config, resolve_thread_count, resolve_thread_list, result_paths, select_models, \
    unit_fingerprint
from llm_cpu_inference.generation import SUMMARY_FIELDS, run_round_unit, run_warmup_load, session_summary_rows
from llm_cpu_inference.log import setup_logging
from llm_cpu_inference.power import prevent_sleep
from llm_cpu_inference.results_io import append_rows, defer_interrupt, write_json_atomic
from llm_cpu_inference.run_metadata import detect_topology, now_iso
from llm_cpu_inference.session import MeasurementSession, RunRefused, preflight
from llm_cpu_inference.thermal import ThermalAbort

EXPERIMENTS = ("energy", "thread_scaling", "cognitive")
GENERATION_EXPERIMENTS = ("energy", "thread_scaling")
FIRST_SESSION = "session_01"
STATE_VERSION = 2
EXIT_OK, EXIT_FAILED_UNITS, EXIT_REFUSED, EXIT_THERMAL_ABORT, EXIT_INTERRUPTED = 0, 1, 2, 3, 130


@dataclass(frozen=True)
class Unit:
    session_id: str
    experiment: str
    model_key: str
    n_threads: int
    round: int = 0  # 0 for cognitive (not interleaved)
    order_in_round: int = 0

    @property
    def unit_id(self):
        base = f"{self.session_id}|{self.experiment}|{self.model_key}|{self.n_threads}"
        return f"{base}|r{self.round:02d}" if self.round else base


def session_name(number):
    return f"session_{number:02d}"


def session_seed(base_seed, session_id):
    """Shuffle seed of a session, derived from the configured base seed and the session id."""
    return int(hashlib.sha256(f"{base_seed}:{session_id}".encode()).hexdigest()[:8], 16)


def configurations(config, topology, experiment):
    """(model, threads) configurations of an interleaved experiment, in configuration order."""
    if experiment == "energy":
        n_threads = resolve_thread_count(config["energy"]["n_threads"], topology)
        return [(key, n_threads) for key in select_models(config["models"])]
    keys = select_models(config["thread_scaling"]["models"])
    threads = resolve_thread_list(config["thread_scaling"]["threads"], topology)
    return [(key, n) for key in keys for n in threads]


def plan_session(config, topology, session_id, seed):
    """Every unit of a session in execution order, plus the per-round orders for the record."""
    units, orders = [], {}
    for experiment in config["campaign"]["order"]:
        if experiment in GENERATION_EXPERIMENTS:
            configs = configurations(config, topology, experiment)
            for round_index in range(1, config[experiment]["iterations"] + 1):
                order = list(configs)
                random.Random(f"{seed}:{experiment}:{round_index}").shuffle(order)
                orders[f"{experiment}|r{round_index:02d}"] = [f"{key}|{n}" for key, n in order]
                units += [Unit(session_id, experiment, key, n, round_index, position)
                          for position, (key, n) in enumerate(order, start=1)]
        elif experiment == "cognitive":
            if config["cognitive"]["first_session_only"] and session_id != FIRST_SESSION:
                continue
            n_threads = resolve_thread_count(config["cognitive"]["n_threads"], topology)
            keys = order_reference_first(select_models(config["models"]), config["cognitive"]["reference"])
            units += [Unit(session_id, "cognitive", key, n_threads, 0, position)
                      for position, key in enumerate(keys, start=1)]
            orders["cognitive"] = keys
    return units, orders


class CampaignState:
    """Persistent record of sessions, their plans and the units that have been written."""

    def __init__(self, path):
        self.path = path
        self.data = {"version": STATE_VERSION, "sessions": {}, "units": {}}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != STATE_VERSION:
                raise RunRefused(f"{path} was written by an older campaign version (v{data.get('version')}). "
                                 f"Move it aside; it is never rewritten automatically.")
            self.data = data

    def save(self):
        write_json_atomic(self.path, self.data)

    @property
    def sessions(self):
        return self.data["sessions"]

    def latest_session(self):
        return max(self.sessions) if self.sessions else None

    def new_session(self, session_id, seed, base_seed, fingerprints, units, orders):
        self.sessions[session_id] = {
            "session_id": session_id, "created_at": now_iso(), "shuffle_seed": seed, "shuffle_seed_base": base_seed,
            "fingerprints": fingerprints, "planned_units": [u.unit_id for u in units], "orders": orders,
            "summaries_written": [], "completed_at": None,
        }

    def is_done(self, unit):
        return self.data["units"].get(unit.unit_id, {}).get("status") == "completed"

    def mark(self, unit, status, run_id, extra=None):
        self.data["units"][unit.unit_id] = {"status": status, "run_id": run_id, "finished_at": now_iso(),
                                            **(extra or {})}
        self.save()

    def pending(self, session_id):
        return [uid for uid in self.sessions[session_id]["planned_units"]
                if self.data["units"].get(uid, {}).get("status") != "completed"]


def write_summary(session_id, state, experiment, paths, config, log):
    """Append the round summaries of a finished interleaved experiment of the session (once)."""
    session_info = state.sessions[session_id]
    if experiment in session_info["summaries_written"]:
        return
    with open(paths[f"{experiment}_iterations"], newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["session_id"] == session_id]
    summaries = session_summary_rows(rows, config[experiment]["n_predict"], config[experiment]["iterations"])
    with defer_interrupt():
        append_rows(paths[f"{experiment}_summary"], SUMMARY_FIELDS, summaries)
        session_info["summaries_written"].append(experiment)
        state.save()
    log.info("Summary of %s / %s written (%d rows)", session_id, experiment, len(summaries))


def write_missing_summaries(session_id, state, units, paths, config, log):
    """Summaries of interleaved experiments whose units are all done but whose summary is missing
    (e.g. the run stopped right after the last unit)."""
    for experiment in GENERATION_EXPERIMENTS:
        own = [u for u in units if u.experiment == experiment]
        if own and all(state.is_done(u) for u in own):
            write_summary(session_id, state, experiment, paths, config, log)


def cool_down(session, unit):
    """Temperature-driven wait before a unit. The hottest configurations must really reach the target."""
    thermal_config = session.config["thermal"]
    high = unit.n_threads >= thermal_config["high_thread_count"]
    waited, start_c, end_c, reached = session.thermal.wait_until_below(
        thermal_config["cooldown_target_c"], thermal_config["cooldown_max_s"], unit.unit_id)
    if high and not reached:
        raise ThermalAbort(f"{unit.n_threads}-thread unit not started: the package did not cool to "
                           f"{thermal_config['cooldown_target_c']} C within {thermal_config['cooldown_max_s']} s "
                           f"(now {end_c:.1f} C)")
    session.thermal.set_limit(thermal_config["high_thread_abort_c"] if high else thermal_config["abort_c"])
    return {"waited_s": waited, "start_c": start_c, "end_c": end_c, "reached": reached, "high_threads": high}


def parse_args(argv, default_experiments):
    parser = argparse.ArgumentParser(description="Run the measurement campaign defined in the experiment config.")
    parser.add_argument("--config", default=None, help="Experiment config (default: configs/experiment.yaml)")
    parser.add_argument("--smoke", action="store_true",
                        help="Short end-to-end check with the config's smoke overrides (written to results/smoke/)")
    parser.add_argument("--final", action="store_true",
                        help="Thesis run: refuse to start on a dirty git tree, wrong power plan or test sensor")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=default_experiments,
                        help="Parts to run in this invocation (default: all of campaign.order)")
    parser.add_argument("--new-session", action="store_true",
                        help="Start the next session (the previous one must be complete)")
    parser.add_argument("--results-dir", default=None, help="Override paths.results_dir")
    parser.add_argument("--list", action="store_true", help="Print the plan and exit without measuring")
    parser.add_argument("--note", default=None, help="Free text stored in the run metadata, e.g. the ambient temperature")
    return parser.parse_args(argv)


def main(argv=None, default_experiments=None):
    args = parse_args(argv, default_experiments)
    log = setup_logging()
    try:
        config = load_config(args.config, smoke=args.smoke, final=args.final)
        paths = result_paths(config, args.results_dir)
        state = CampaignState(paths["state"])
    except (RunRefused, ValueError) as exc:
        log.error("REFUSED: %s", exc)
        return EXIT_REFUSED
    experiments = [e for e in config["campaign"]["order"] if e in (args.experiments or config["campaign"]["order"])]
    topology = detect_topology()

    # ---- which session? ---------------------------------------------------------------------
    latest = state.latest_session()
    if latest and state.pending(latest) and args.new_session:
        log.error("REFUSED: %s is not complete (%d units pending). Resume it without --new-session first.",
                  latest, len(state.pending(latest)))
        return EXIT_REFUSED
    if latest is None or args.new_session:
        session_id = session_name(1 if latest is None else int(latest.split("_")[1]) + 1)
        seed = session_seed(config["campaign"]["shuffle_seed"], session_id)
        units, orders = plan_session(config, topology, session_id, seed)
        is_new = True
    else:
        session_id = latest
        info = state.sessions[session_id]
        seed = info["shuffle_seed"]
        units, orders = plan_session(config, topology, session_id, seed)
        if [u.unit_id for u in units] != info["planned_units"] or orders != info["orders"]:
            log.error("REFUSED: the configuration no longer produces the plan stored for %s (models, threads, "
                      "rounds or seed changed). Finish it with the original config or start --new-session.",
                      session_id)
            return EXIT_REFUSED
        is_new = False
    fingerprints = {e: unit_fingerprint(config, e) for e in config["campaign"]["order"]}
    if not is_new and fingerprints != state.sessions[session_id]["fingerprints"]:
        log.error("REFUSED: measurement parameters changed since %s was created; one session must not mix "
                  "protocols. Restore the config or start --new-session.", session_id)
        return EXIT_REFUSED

    selected = [u for u in units if u.experiment in experiments]
    pending = [u for u in selected if not state.is_done(u)]

    log.info("Command: %s", " ".join(sys.argv))
    log.info("%s (%s, shuffle seed %d): %d units planned, %d selected for this run, %d pending",
             session_id, "new" if is_new else "resumed", seed, len(units), len(selected), len(pending))
    if args.list:
        for experiment in experiments:
            done = sum(1 for u in selected if u.experiment == experiment and state.is_done(u))
            total = sum(1 for u in selected if u.experiment == experiment)
            log.info("  %-14s %3d / %3d units completed", experiment, done, total)
        for key, order in orders.items():
            if key.split("|")[0] in experiments:
                log.info("  order %-20s %s", key, ", ".join(order))
        return EXIT_OK
    if not pending:
        if not is_new:
            write_missing_summaries(session_id, state, units, paths, config, log)
        log.info("Nothing to do: %s is complete for %s. Start the next one with --new-session.",
                 session_id, ", ".join(experiments))
        return EXIT_OK

    # ---- refuse early, before anything is written ---------------------------------------------
    refusals, _, _ = preflight(config)
    if refusals:
        for problem in refusals:
            log.error("REFUSED (--final): %s", problem)
        return EXIT_REFUSED

    session_record = {"session_id": session_id, "shuffle_seed": seed,
                      "shuffle_seed_base": config["campaign"]["shuffle_seed"], "planned_orders": orders}
    session = MeasurementSession(
        config,
        experiments=[e for e in experiments if any(u.experiment == e for u in pending)],
        model_keys=list(dict.fromkeys(u.model_key for u in pending)),
        results_dir=args.results_dir,
        measure_energy=any(u.experiment in GENERATION_EXPERIMENTS for u in pending),
        note=args.note,
        session=session_record,
    )
    os.makedirs(paths["logs"], exist_ok=True)
    log_file = os.path.join(paths["logs"], f"campaign_{session_id}_{session.run_id}.log")
    setup_logging(log_file)
    log.info("Log file: %s", log_file)

    dataset_cache = []

    def dataset():
        if not dataset_cache:
            dataset_cache.append(load_dataset(config["cognitive"]))
        return dataset_cache[0]

    campaign_start = time.perf_counter()
    completed, failed = [], []
    try:
        with prevent_sleep(), session:
            if is_new:
                state.new_session(session_id, seed, config["campaign"]["shuffle_seed"], fingerprints, units, orders)
                state.save()
            if any(u.experiment in GENERATION_EXPERIMENTS for u in pending):
                # Unmeasured load after the idle baseline: rounds start in the steady working state
                session.metadata["warmup_load"] = run_warmup_load(
                    session, config["warmup_load"], resolve_thread_count(config["warmup_load"]["n_threads"], topology))
                session.record_unit({"unit_id": "warmup_load", "finished_at": now_iso()})
            for index, unit in enumerate(pending, start=1):
                log.info(">>> Unit %d/%d: %s (order %d in round)", index, len(pending), unit.unit_id,
                         unit.order_in_round)
                unit_start = time.perf_counter()
                cooldown = cool_down(session, unit)

                def on_written(unit=unit, cooldown=cooldown):
                    state.mark(unit, "completed", session.run_id,
                               {"cooldown_wait_s": round(cooldown["waited_s"], 1)})

                try:
                    if unit.experiment in GENERATION_EXPERIMENTS:
                        run_round_unit(session, unit, config[unit.experiment],
                                       paths[f"{unit.experiment}_iterations"], cooldown, on_written)
                    else:
                        raw_text, dataset_sha = dataset()
                        run_cognitive_unit(session, unit.model_key, raw_text, dataset_sha, cooldown, on_written)
                except ThermalAbort:
                    raise
                except Exception as exc:  # One failing unit must not stop an unattended campaign
                    log.exception("Unit %s failed and is skipped: %s", unit.unit_id, exc)
                    state.mark(unit, "failed", session.run_id, {"error": repr(exc)})
                    failed.append(unit)
                    continue
                finally:
                    session.thermal.set_limit(config["thermal"]["abort_c"])
                completed.append(unit)
                session.record_unit({**asdict(unit), "unit_id": unit.unit_id, "finished_at": now_iso(),
                                     "duration_s": round(time.perf_counter() - unit_start, 1),
                                     "cooldown_wait_s": round(cooldown["waited_s"], 1),
                                     "temp_at_start_c": cooldown["end_c"]})
                log.info("<<< Unit %d/%d finished in %.1f min (incl. %.0f s cool-down)", index, len(pending),
                         (time.perf_counter() - unit_start) / 60, cooldown["waited_s"])

                # Write the round summaries as soon as an interleaved experiment of the session is complete
                write_missing_summaries(session_id, state, units, paths, config, log)

            if not state.pending(session_id):
                state.sessions[session_id]["completed_at"] = now_iso()
                state.save()
                log.info("%s is complete.", session_id)
    except RunRefused as exc:
        log.error("REFUSED: %s", exc)
        return EXIT_REFUSED
    except ThermalAbort as exc:
        log.error("THERMAL_ABORT: %s. No partial rows were written. Let the machine cool down, check the cooling, "
                  "then run the same command again to resume.", exc)
        return EXIT_THERMAL_ABORT
    except KeyboardInterrupt:
        log.warning("Interrupted by the user after %d of %d pending units. No partial rows were written; "
                    "run the same command again to continue.", len(completed), len(pending))
        return EXIT_INTERRUPTED

    log.info("Run finished in %.1f min: %d completed, %d failed, %d still pending in %s.",
             (time.perf_counter() - campaign_start) / 60, len(completed), len(failed),
             len(state.pending(session_id)), session_id)
    for unit in failed:
        log.error("FAILED: %s (see the traceback above; it will be retried on the next start)", unit.unit_id)
    return EXIT_FAILED_UNITS if failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
