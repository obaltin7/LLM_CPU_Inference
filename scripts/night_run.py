"""Unattended night run: a fixed sequence of measurement blocks in one session (night_01).

Blocks (configs/experiment.yaml -> night):
  a1       all models x {2, 4, 8} threads, interleaved rounds (Turbo off)
  a5       8B-Q4_K_M / Q5_K_M / Q6_K / Q8_0 interleaved (Turbo off, 8 threads)
  b1_pre   heat probe with Turbo on; if the peak stays below night.b1.heat_max_c: 3 models x {Turbo on, off}
           interleaved, the power plan switched per unit; otherwise B1_SKIPPED_THERMAL and the b1 blocks are skipped
  a2       perplexity of all models on night.a2.chunks chunks (reference first)
  b1_post  repeat of b1_pre's measurement (only if night.b1.enabled_post)

Everything the campaign guarantees also holds here: --final pre-conditions, quiet gate, idle baseline, warm-up
load, thermal fuse, sleep prevention, Ctrl+C-safe writes, resume (same command continues where it stopped).
Turbo Boost is restored to night.turbo_off_percent in a `finally` block on every exit path, and checked/fixed at
start. A failing block is marked FAILED and the next block starts; a thermal abort stops the whole run.

    python night_run.py --final --note "oda 23 C"      # the night
    python night_run.py --smoke                        # dry run (fake sensor, powercfg not called, minutes)
    python night_run.py --list                         # plan and time estimate, no measurement
"""
import argparse
import csv
import os
import random
import sys
import time
from dataclasses import asdict

import _paths  # noqa: F401  (makes src/ importable without installing the package)
from benchmark_cognitive import complete_full_set, load_dataset, order_reference_first, run_cognitive_unit
from run_campaign import (EXIT_FAILED_UNITS, EXIT_INTERRUPTED, EXIT_OK, EXIT_REFUSED, EXIT_THERMAL_ABORT,
                          CampaignState, Unit, cool_down, session_seed)
from llm_cpu_inference.config import _deep_merge, load_config, resolve_thread_count, result_paths, select_models
from llm_cpu_inference.generation import (SUMMARY_FIELDS, end_of_generation_ids, greedy_decode, load_model, run_round_unit,
                              run_warmup_load, session_summary_rows)
from llm_cpu_inference.log import setup_logging
from llm_cpu_inference.power import TurboController, prevent_sleep
from llm_cpu_inference.results_io import append_rows, defer_interrupt
from llm_cpu_inference.run_metadata import detect_topology, now_iso
from llm_cpu_inference.session import MeasurementSession, RunRefused, preflight
from llm_cpu_inference.thermal import ThermalAbort

GENERATION_BLOCKS = ("a1", "a5", "b1_pre", "b1_post")
COGNITIVE_BLOCKS = ("a2", "a2_rest")  # a2: chunks [0, n); a2_rest: the remaining chunks merged with an earlier run
TEST_HOOK_ENV = "LLMCPU_NIGHT_TEST"  # smoke-only fault injection: fail:<block> | thermal:<block> | interrupt:<block> | heat:<peak_c>
TEST_SUBDIR_ENV = "LLMCPU_NIGHT_RESULTS_SUBDIR"  # smoke-only: results sub-directory (isolates test scenarios)
TEST_PREVIOUS_ENV = "LLMCPU_NIGHT_PREVIOUS_SUBDIR"  # smoke-only: sub-directory of the earlier run that a2_rest completes
LAST_TURBO = None  # The TurboController of the last main() call (inspected by the tests)


def night_settings(config, smoke=False, key="night"):
    night = config[key]
    subdir = os.environ.get(TEST_SUBDIR_ENV, night["results_subdir"]) if smoke else night["results_subdir"]
    return night, os.path.join(config["paths"]["results_dir"], subdir)


def previous_run(config, night, smoke=False):
    """Location and shape of the earlier perplexity run that the a2_rest block completes."""
    a2 = night["a2_rest"]
    subdir = os.environ.get(TEST_PREVIOUS_ENV, a2["previous_results_subdir"]) if smoke else a2["previous_results_subdir"]
    root = os.path.join(config["paths"]["results_dir"], subdir)
    return {"nll_dir": os.path.join(root, "nll"), "results_csv": os.path.join(root, "cognitive_results.csv"),
            "n_chunks": a2["previous_n_chunks"], "overlap": a2.get("overlap", 0), "root": root}


def interleaved(session_id, block, experiment, configs, rounds, seed):
    """Units of an interleaved block: every round measures every configuration once, shuffled per round."""
    units = []
    for round_index in range(1, rounds + 1):
        order = list(configs)
        random.Random(f"{seed}:{block}:{round_index}").shuffle(order)
        units += [Unit(session_id, experiment(cfg), cfg[0], cfg[1], round_index, pos) for pos, cfg in enumerate(order, 1)]
    return units


def plan_blocks(config, topology, session_id, seed, night=None):
    """{block: [Unit, ...]} in execution order. Turbo state is encoded in the experiment name of the b1 units."""
    night = night or config["night"]
    plan = {}
    n_cog = resolve_thread_count(config["cognitive"]["n_threads"], topology)
    for block in night["order"]:
        if block == "a1":
            a1 = night["a1"]
            plan[block] = interleaved(session_id, "a1", lambda c: "night_a1",
                                      [(k, n) for k in select_models(a1["models"]) for n in a1["threads"]],
                                      a1["iterations"], seed)
        elif block == "a5":
            a5 = night["a5"]
            n_a5 = resolve_thread_count(a5["n_threads"], topology)
            plan[block] = interleaved(session_id, "a5", lambda c: "night_a5",
                                      [(k, n_a5) for k in select_models(a5["models"])], a5["iterations"], seed)
        elif block in ("b1_pre", "b1_post"):
            b1 = night["b1"]
            if block == "b1_post" and not b1.get("enabled_post", False):
                plan[block] = []
                continue
            n_b1 = resolve_thread_count(b1["n_threads"], topology)
            configs = [(k, n_b1, state) for k in select_models(b1["models"]) for state in ("on", "off")]
            plan[block] = interleaved(session_id, block, lambda c, block=block: f"night_{block}_turbo_{c[2]}",
                                      configs, b1["iterations"], seed)
        elif block in COGNITIVE_BLOCKS:
            a2 = night[block]
            keys = order_reference_first(select_models(a2["models"]), a2["reference"])
            plan[block] = [Unit(session_id, f"night_{block}", k, n_cog, 0, pos) for pos, k in enumerate(keys, 1)]
        else:
            raise ValueError(f"unknown night block '{block}'")
    return plan


def turbo_state_of(unit):
    return unit.experiment.endswith("_turbo_on")


def heat_probe(session, model_key, n_threads, duration_s, limit_c):
    """Continuous generation under the current power plan; returns the peak package temperature."""
    thermal = session.thermal
    thermal.set_limit(limit_c)
    llm, _ = load_model(model_key, n_threads, session.config["inference"])
    start = time.perf_counter()
    try:
        eog = end_of_generation_ids(llm)
        prompt = llm.tokenize(session.config["prompts"][0]["text"].encode("utf-8"), add_bos=True, special=False)
        while time.perf_counter() - start < duration_s:
            llm.reset()
            thermal.check()
            llm.eval(prompt)
            greedy_decode(llm, 50, eog, thermal)
    finally:
        llm.close()
    stats = thermal.stats(start, time.perf_counter())
    session.log.info("Heat probe (Turbo on): %s, %d threads, %.0f s -> peak %.1f C, mean %.1f C", model_key,
                     n_threads, duration_s, stats.max_c, stats.mean_c)
    return stats.max_c


def write_block_summary(block, units, state, paths, params, log):
    info = state.sessions[units[0].session_id]
    if block in info["summaries_written"]:
        return
    experiments = {u.experiment for u in units}
    with open(paths["energy_iterations"], newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["session_id"] == units[0].session_id and r["experiment"] in experiments]
    summaries = []
    for experiment in sorted(experiments):
        summaries += session_summary_rows([r for r in rows if r["experiment"] == experiment], params["n_predict"],
                                          params["iterations"])
    with defer_interrupt():
        append_rows(paths["energy_summary"], SUMMARY_FIELDS, summaries)
        info["summaries_written"].append(block)
        state.save()
    log.info("Summary of block %s written (%d rows)", block, len(summaries))


def estimate_minutes(plan, night):
    """Rough duration per block from measured unit times (sessions 02-03 and night_01; seconds per unit at 8 threads)."""
    q4_like = {"1B": 13, "3B": 32, "8B": 71, "Sparse8B": 71}
    q8 = {"1B": 19, "3B": 50, "8B": 115, "Sparse8B": 114}
    mult = {"8B-Q5_K_M": 81 / 71, "8B-Q6_K": 91 / 71}
    thread_mult = {1: 2.2, 2: 1.25, 4: 1.02}
    # Cognitive evaluation: seconds per chunk measured in night_01 (90 chunks, 8 threads), plus ~45 s model load
    chunk_secs = {"1B-Q4_K_M": 2.52, "1B-Q8_0": 3.67, "3B-Q4_K_M": 5.66, "3B-Q8_0": 8.64, "8B-Q4_K_M": 11.8,
                  "8B-Q5_K_M": 13.0, "8B-Q6_K": 13.3, "8B-Q8_0": 18.3, "Sparse8B-Q4_K_M": 11.9, "Sparse8B-Q8_0": 18.4}
    total = {}
    for block, units in plan.items():
        secs = 0.0
        for u in units:
            fam = u.model_key.split("-")[0]
            if block in COGNITIVE_BLOCKS:
                a2 = night[block]
                if block == "a2_rest":
                    chunks = a2["expected_total_chunks"] - a2["previous_n_chunks"] + a2.get("overlap", 0)
                    if a2.get("chunks", 0) > 0:
                        chunks = min(chunks, a2["chunks"])
                else:
                    chunks = a2["chunks"]
                secs += chunks * chunk_secs.get(u.model_key, 12.0) + 45
            else:
                base = q8[fam] if u.model_key.endswith("Q8_0") else q4_like[fam] * mult.get(u.model_key, 1)
                secs += base * thread_mult.get(u.n_threads, 1.0) + (night["b1"]["settle_s"] if "b1" in block else 0)
        if block.startswith("b1") and units:
            secs += night["b1"]["heat_duration_s"] + 2 * night["b1"]["settle_s"] + 60  # heat probe once
        total[block] = secs / 60
    return total


def main(argv=None):
    global LAST_TURBO
    parser = argparse.ArgumentParser(description="Unattended night run: the blocks of a `night*` config section "
                                                 "(a1, a5, b1_pre, a2, a2_rest, b1_post) in its `order`.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--night-config", default="night", help="Config section of the run (night = night_01, night_02, ...)")
    parser.add_argument("--smoke", action="store_true", help="Dry run with the smoke overrides; powercfg is not called")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--note", default=None)
    parser.add_argument("--list", action="store_true", help="Print the plan and the time estimate, no measurement")
    args = parser.parse_args(argv)
    log = setup_logging()
    try:
        config = load_config(args.config, smoke=args.smoke, final=args.final)
        night, results_dir = night_settings(config, args.smoke, args.night_config)
        cognitive_block = next((b for b in night["order"] if b in COGNITIVE_BLOCKS), None)
        if cognitive_block:
            a2 = night[cognitive_block]
            config = _deep_merge(config, {"cognitive": {"chunks": a2.get("chunks", 0), "reference": a2["reference"]}})
        previous = previous_run(config, night, args.smoke) if "a2_rest" in night["order"] else None
        if previous and not os.path.exists(previous["results_csv"]):
            raise RunRefused(f"a2_rest needs the earlier perplexity run in {previous['root']} (cognitive_results.csv missing)")
        paths = result_paths(config, results_dir)
        state = CampaignState(paths["state"])
    except (RunRefused, ValueError, KeyError) as exc:
        log.error("REFUSED: %s", exc)
        return EXIT_REFUSED
    session_id = night["session_id"]
    topology = detect_topology()
    seed = session_seed(config["campaign"]["shuffle_seed"], session_id)
    plan = plan_blocks(config, topology, session_id, seed, night)
    all_units = [u for block in night["order"] for u in plan[block]]
    orders = {block: [u.unit_id for u in units] for block, units in plan.items()}

    if session_id in state.sessions and state.sessions[session_id]["planned_units"] != [u.unit_id for u in all_units]:
        log.error("REFUSED: the night config no longer produces the plan stored for %s; restore it or move the state aside.",
                  session_id)
        return EXIT_REFUSED
    is_new = session_id not in state.sessions
    night_state = state.sessions.get(session_id, {}).get("night") or {"blocks": {}, "b1": None}
    pending_all = [u for u in all_units if not state.is_done(u)]
    estimate = estimate_minutes(plan, night)
    log.info("Command: %s", " ".join(sys.argv))
    log.info("%s (%s): %d units planned, %d pending; estimate per block (min): %s; total %.1f h",
             session_id, "new" if is_new else "resumed", len(all_units), len(pending_all),
             {k: round(v) for k, v in estimate.items()}, sum(estimate.values()) / 60)
    if args.list:
        for block in night["order"]:
            done = sum(1 for u in plan[block] if state.is_done(u))
            log.info("  %-8s %3d / %3d units completed  (~%.0f min)", block, done, len(plan[block]), estimate[block])
        return EXIT_OK
    if not pending_all:
        log.info("Nothing to do: %s is complete.", session_id)
        return EXIT_OK

    turbo = TurboController(night["turbo_off_percent"], night["turbo_on_percent"], dry_run=args.smoke)
    LAST_TURBO = turbo
    hook = os.environ.get(TEST_HOOK_ENV, "") if args.smoke else ""
    exit_code = EXIT_OK
    failed_blocks, completed = [], 0
    try:
        if turbo.ensure_off():
            log.warning("Turbo was ON at start and has been switched OFF before the pre-flight checks.")
        refusals, _, _ = preflight(config)
        if refusals:
            for problem in refusals:
                log.error("REFUSED (--final): %s", problem)
            return EXIT_REFUSED

        session = MeasurementSession(
            config, experiments=[f"night_{b}" for b in night["order"] if any(not state.is_done(u) for u in plan[b])],
            model_keys=list(dict.fromkeys(u.model_key for u in pending_all)), results_dir=results_dir,
            measure_energy=True, note=args.note,
            session={"session_id": session_id, "shuffle_seed": seed, "shuffle_seed_base": config["campaign"]["shuffle_seed"],
                     "planned_orders": orders, "mode": "night"})
        os.makedirs(paths["logs"], exist_ok=True)
        setup_logging(os.path.join(paths["logs"], f"night_{session_id}_{session.run_id}.log"))
        dataset_cache = []

        with prevent_sleep(), session:
            if is_new:
                state.new_session(session_id, seed, config["campaign"]["shuffle_seed"], {}, all_units, orders)
                state.sessions[session_id]["night"] = night_state
                state.save()
            session.metadata["warmup_load"] = run_warmup_load(
                session, config["warmup_load"], resolve_thread_count(config["warmup_load"]["n_threads"], topology))
            if previous:
                session.metadata["cognitive_previous_run"] = {k: v for k, v in previous.items()}
                log.info("a2_rest completes the perplexity run in %s: %d earlier chunks, %d re-evaluated (overlap)",
                         previous["root"], previous["n_chunks"], previous["overlap"])

            for block in night["order"]:
                units = [u for u in plan[block] if not state.is_done(u)]
                if not plan[block]:
                    night_state["blocks"][block] = "disabled"
                    continue
                if not units:
                    continue
                try:
                    # ---- b1: heat probe with Turbo on, once; the decision is persisted for resume
                    if block.startswith("b1"):
                        b1 = night["b1"]
                        if night_state["b1"] is None:
                            turbo.turbo(True)
                            session.thermal.sleep(b1["settle_s"])
                            peak = float(hook.split(":")[1]) if hook.startswith("heat:") else heat_probe(
                                session, b1["heat_model"], resolve_thread_count(b1["n_threads"], topology),
                                b1["heat_duration_s"], b1["turbo_on_abort_c"])
                            turbo.turbo(False)
                            session.thermal.set_limit(config["thermal"]["abort_c"])
                            night_state["b1"] = {"heat_peak_c": peak, "skipped": peak > b1["heat_max_c"],
                                                 "heat_max_c": b1["heat_max_c"], "measured_at": now_iso()}
                            session.metadata["night_b1_heat_probe"] = night_state["b1"]
                            state.sessions[session_id]["night"] = night_state
                            state.save()
                        if night_state["b1"]["skipped"]:
                            log.error("B1_SKIPPED_THERMAL: heat probe peak %.1f C > %s C; block %s skipped.",
                                      night_state["b1"]["heat_peak_c"], b1["heat_max_c"], block)
                            night_state["blocks"][block] = "skipped_thermal"
                            continue
                    if hook == f"fail:{block}":
                        raise RuntimeError(f"test hook: block {block} fails")
                    if hook == f"thermal:{block}":
                        raise ThermalAbort(f"test hook: thermal abort in {block}")
                    if hook == f"interrupt:{block}":
                        raise KeyboardInterrupt

                    log.info("=== Block %s: %d units pending (~%.0f min)", block, len(units), estimate[block])
                    current_turbo = False
                    for index, unit in enumerate(units, 1):
                        want_turbo = turbo_state_of(unit)
                        if want_turbo != current_turbo:
                            turbo.turbo(want_turbo)
                            session.thermal.sleep(night["b1"]["settle_s"])
                            current_turbo = want_turbo
                        log.info(">>> [%s] Unit %d/%d: %s", block, index, len(units), unit.unit_id)
                        unit_start = time.perf_counter()
                        cooldown = cool_down(session, unit)
                        if want_turbo:
                            session.thermal.set_limit(night["b1"]["turbo_on_abort_c"])

                        def on_written(unit=unit, cooldown=cooldown, turbo_on=want_turbo):
                            state.mark(unit, "completed", session.run_id,
                                       {"cooldown_wait_s": round(cooldown["waited_s"], 1), "turbo_on": turbo_on})

                        try:
                            if block in GENERATION_BLOCKS:
                                run_round_unit(session, unit, night[block.split("_")[0]], paths["energy_iterations"],
                                               cooldown, on_written)
                            else:
                                if not dataset_cache:
                                    dataset_cache.append(load_dataset(config["cognitive"]))
                                raw_text, sha = dataset_cache[0]
                                if block == "a2_rest":
                                    complete_full_set(session, unit.model_key, raw_text, sha, cooldown, previous, on_written)
                                else:
                                    run_cognitive_unit(session, unit.model_key, raw_text, sha, cooldown, on_written)
                        except ThermalAbort:
                            raise
                        except Exception as exc:  # One failing unit must not stop the night
                            log.exception("Unit %s failed and is skipped: %s", unit.unit_id, exc)
                            state.mark(unit, "failed", session.run_id, {"error": repr(exc)})
                            continue
                        finally:
                            session.thermal.set_limit(config["thermal"]["abort_c"])
                        completed += 1
                        session.record_unit({**asdict(unit), "unit_id": unit.unit_id, "block": block,
                                             "turbo_on": want_turbo, "max_processor_state": turbo.current(),
                                             "finished_at": now_iso(),
                                             "duration_s": round(time.perf_counter() - unit_start, 1),
                                             "cooldown_wait_s": round(cooldown["waited_s"], 1),
                                             "temp_at_start_c": cooldown["end_c"]})
                        log.info("<<< [%s] Unit %d/%d finished in %.1f min", block, index, len(units),
                                 (time.perf_counter() - unit_start) / 60)
                    if current_turbo:
                        turbo.turbo(False)
                    if block in GENERATION_BLOCKS and all(state.is_done(u) for u in plan[block]):
                        write_block_summary(block, plan[block], state, paths, night[block.split("_")[0]], log)
                    night_state["blocks"][block] = "completed" if all(state.is_done(u) for u in plan[block]) else "partial"
                except (ThermalAbort, KeyboardInterrupt):
                    night_state["blocks"][block] = "interrupted"
                    raise
                except Exception as exc:  # A failing block must not stop the night
                    log.exception("BLOCK FAILED: %s (%s); continuing with the next block", block, exc)
                    night_state["blocks"][block] = "failed"
                    failed_blocks.append(block)
                finally:
                    turbo.turbo(False)  # Never leave Turbo on between blocks
                    state.sessions[session_id]["night"] = night_state
                    state.save()
            if not state.pending(session_id):
                state.sessions[session_id]["completed_at"] = now_iso()
                state.save()
                log.info("%s is complete.", session_id)
    except RunRefused as exc:
        log.error("REFUSED: %s", exc)
        exit_code = EXIT_REFUSED
    except ThermalAbort as exc:
        log.error("THERMAL_ABORT: %s. The run stops; run the same command again to resume after cooling down.", exc)
        exit_code = EXIT_THERMAL_ABORT
    except KeyboardInterrupt:
        log.warning("Interrupted by the user; run the same command again to resume.")
        exit_code = EXIT_INTERRUPTED
    finally:
        # Guaranteed on every exit path: normal end, Ctrl+C, thermal abort, Python error
        try:
            turbo.turbo(False)
            log.info("Turbo restored: maximum processor state %s %%.", turbo.current())
        except Exception as exc:  # pragma: no cover - last resort logging
            log.error("TURBO RESTORE FAILED: %s. Run: powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR "
                      "PROCTHROTTLEMAX 99; powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99; "
                      "powercfg /setactive SCHEME_CURRENT", exc)
    if exit_code == EXIT_OK and failed_blocks:
        exit_code = EXIT_FAILED_UNITS
    log.info("Night run finished: exit %d, %d units completed this run, failed blocks: %s, blocks: %s",
             exit_code, completed, failed_blocks or "none", night_state["blocks"])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
