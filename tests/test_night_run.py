"""Full-flow tests of the night run in smoke mode: fake temperature sensor, powercfg never called (dry-run
controller), fault injection through LLMCPU_NIGHT_TEST. They load the small smoke models, so each scenario
takes 1-3 minutes; run them with `python -m pytest tests/test_night_run.py -m slow`."""
import json
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in ("scripts", "src"):  # also set in pytest.ini; kept so the file runs under any runner
    sys.path.insert(0, os.path.join(ROOT, _dir))

import night_run  # noqa: E402
from llm_cpu_inference.thermal import FAKE_SOURCE_ENV  # noqa: E402

pytestmark = pytest.mark.slow
SMOKE_ROOT = os.path.join(ROOT, "results", "smoke")


def _run(tmp_path, scenario, hook=None, clean=True, night_config="night", previous_scenario=None):
    subdir = f"night_test_{scenario}"
    if clean:
        shutil.rmtree(os.path.join(SMOKE_ROOT, subdir), ignore_errors=True)
    temp_file = tmp_path / "temp.txt"
    temp_file.write_text("40.0", encoding="utf-8")
    env = {FAKE_SOURCE_ENV: str(temp_file), night_run.TEST_SUBDIR_ENV: subdir}
    if hook:
        env[night_run.TEST_HOOK_ENV] = hook
    if previous_scenario:
        env[night_run.TEST_PREVIOUS_ENV] = f"night_test_{previous_scenario}"
    cleared = [night_run.TEST_HOOK_ENV, night_run.TEST_PREVIOUS_ENV]
    previous = {k: os.environ.get(k) for k in list(env) + cleared}
    for k in cleared:
        os.environ.pop(k, None)
    os.environ.update(env)
    try:
        code = night_run.main(["--smoke", "--night-config", night_config])
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    state_path = os.path.join(SMOKE_ROOT, subdir, "campaign_state.json")
    state = json.load(open(state_path, encoding="utf-8")) if os.path.exists(state_path) else {}
    sessions = state.get("sessions", {})
    session = next(iter(sessions.values()), {}) if sessions else {}
    return code, session.get("night", {}), session, night_run.LAST_TURBO


def _turbo_restored(turbo):
    assert turbo is not None and turbo.dry_run and turbo.calls and turbo.calls[-1] == turbo.off_percent
    assert turbo.current() == turbo.off_percent


def test_normal_run_completes_and_restores_turbo(tmp_path):
    code, night, session, turbo = _run(tmp_path, "normal")
    assert code == 0 and session["completed_at"]
    assert night["blocks"]["a1"] == night["blocks"]["a5"] == night["blocks"]["a2"] == "completed"
    assert night["blocks"]["b1_pre"] == "completed" and night["b1"]["skipped"] is False
    assert 100 in turbo.calls  # Turbo was switched on for the heat probe and the turbo-on units
    _turbo_restored(turbo)


def test_block_failure_continues_with_next_block(tmp_path):
    code, night, _, turbo = _run(tmp_path, "fail", hook="fail:a5")
    assert code == 1
    assert night["blocks"]["a5"] == "failed" and night["blocks"]["a2"] == "completed"
    _turbo_restored(turbo)


def test_thermal_abort_stops_everything(tmp_path):
    code, night, _, turbo = _run(tmp_path, "thermal", hook="thermal:a5")
    assert code == 3
    assert night["blocks"]["a5"] == "interrupted" and "a2" not in night["blocks"]
    _turbo_restored(turbo)


def test_interrupt_then_resume(tmp_path):
    code, night, _, turbo = _run(tmp_path, "interrupt", hook="interrupt:a5")
    assert code == 130 and night["blocks"]["a5"] == "interrupted"
    _turbo_restored(turbo)
    code, night, session, turbo = _run(tmp_path, "interrupt", clean=False)  # same command, no hook: resumes
    assert code == 0 and session["completed_at"] and night["blocks"]["a5"] == "completed"
    _turbo_restored(turbo)


def test_night_02_completes_the_full_set_chunk_by_chunk(tmp_path):
    """night_01 (smoke: 1 chunk per model) then night_02: b1 again and a2_rest, which re-evaluates the overlapping
    chunk, proves it bit-identical to night_01 and merges the parts into cognitive_results_full.csv."""
    import csv
    import glob

    import numpy as np

    code, night1, _, _ = _run(tmp_path, "prev")
    assert code == 0 and night1["blocks"]["a2"] == "completed"
    code, night2, session, turbo = _run(tmp_path, "night02", night_config="night_02", previous_scenario="prev")
    assert code == 0 and session["completed_at"]
    assert night2["blocks"]["b1_pre"] == "completed" and night2["blocks"]["a2_rest"] == "completed"
    assert night2["blocks"]["b1_post"] == "disabled" and night2["b1"]["skipped"] is False
    _turbo_restored(turbo)

    root = os.path.join(SMOKE_ROOT, "night_test_night02")
    with open(os.path.join(root, "cognitive_results_full.csv"), newline="", encoding="utf-8") as f:
        full = list(csv.DictReader(f))
    assert [r["model_key"] for r in full] == ["8B-Q4_K_M", "1B-Q4_K_M"] and all(r["n_chunks"] == "2" for r in full)
    assert all(r["paired_with_reference"] == "True" for r in full)
    meta = json.load(open(glob.glob(os.path.join(root, "metadata", "*.json"))[0], encoding="utf-8"))
    checks = meta["cognitive_merge_checks"]
    assert len(checks) == 2 and all(c["overlap_identical"] and c["same_text"] for c in checks)
    assert all(c["overlap_max_abs_diff"] == 0.0 and c["overlap_chunks"] == 1 and c["new_chunks"] == 1 for c in checks)
    # The merged array starts with the earlier run's chunk (bit-identical) and has one new chunk appended
    full_files = sorted(glob.glob(os.path.join(root, "nll", "*_ctx512_c2.npz")))
    assert len(full_files) == 2
    for full_path in full_files:
        prev_path = os.path.join(SMOKE_ROOT, "night_test_prev", "nll", os.path.basename(full_path).replace("_c2", "_c1"))
        merged, earlier = np.load(full_path)["nll"], np.load(prev_path)["nll"]
        assert merged.shape == (2, 256) and earlier.shape == (1, 256) and np.array_equal(merged[:1], earlier)


def test_hot_heat_probe_skips_b1_blocks(tmp_path):
    code, night, _, turbo = _run(tmp_path, "heat", hook="heat:90")
    assert code == 0
    assert night["b1"]["skipped"] is True and night["b1"]["heat_peak_c"] == 90.0
    assert night["blocks"]["b1_pre"] == "skipped_thermal" and night["blocks"]["b1_post"] == "skipped_thermal"
    assert night["blocks"]["a2"] == "completed"
    _turbo_restored(turbo)
