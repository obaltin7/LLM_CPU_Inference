"""Unit tests of the statistics and formatting helpers behind the tables (python -m pytest tests)."""
import csv
import math
import os
import sys

import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in ("scripts", "src"):  # also set in pytest.ini; kept so the file runs under any runner
    sys.path.insert(0, os.path.join(_ROOT, _dir))

from analyze_campaign import t_interval, tost_table  # noqa: E402
from analyze_sessions import variance_components  # noqa: E402
from llm_cpu_inference.results_io import SchemaMismatchError, append_rows  # noqa: E402
from llm_cpu_inference.stats import summarize, t_critical_95, t_critical_95_one_sided  # noqa: E402


def test_t_critical_two_sided_known_values():
    assert t_critical_95(1) == pytest.approx(12.7062)
    assert t_critical_95(2) == pytest.approx(4.3027)
    assert t_critical_95(10) == pytest.approx(2.2281)
    assert t_critical_95(30) == pytest.approx(2.0423)
    assert t_critical_95(60) == pytest.approx(2.0003, abs=1e-3)  # expansion beyond the table
    assert math.isnan(t_critical_95(0))


def test_t_critical_one_sided_known_values():
    assert t_critical_95_one_sided(1) == pytest.approx(6.3138)
    assert t_critical_95_one_sided(2) == pytest.approx(2.9200)
    assert t_critical_95_one_sided(9) == pytest.approx(1.8331)
    assert t_critical_95_one_sided(40) == pytest.approx(1.6839, abs=1e-3)


def test_summarize_mean_std_interval():
    s = summarize([1.0, 2.0, 3.0])
    assert s["n"] == 3 and s["mean"] == 2.0 and s["std"] == pytest.approx(1.0)
    half = 4.3027 * 1.0 / math.sqrt(3)
    assert s["ci95_low"] == pytest.approx(2.0 - half, abs=1e-4)
    assert s["ci95_high"] == pytest.approx(2.0 + half, abs=1e-4)
    assert math.isnan(summarize([5.0])["std"])
    assert summarize([1.0, float("nan"), 3.0])["n"] == 2


def test_variance_components_hand_example():
    # Two sessions of three rounds: within-session MS = 1, between MS = 6, n0 = 3 -> var_between = 5/3
    r = variance_components({"s1": [1, 2, 3], "s2": [3, 4, 5]})
    assert r["grand_mean"] == pytest.approx(3.0)
    assert r["var_within"] == pytest.approx(1.0)
    assert r["var_between"] == pytest.approx(5 / 3)
    assert r["icc"] == pytest.approx((5 / 3) / (5 / 3 + 1))
    assert r["cv_within_pct"] == pytest.approx(100 / 3)
    # A single session has no between component
    assert math.isnan(variance_components({"s1": [1, 2, 3]})["var_between"])


def test_t_interval_two_and_one_sided():
    mean, se, low, high = t_interval([1.0, 2.0, 3.0])
    assert mean == 2.0 and se == pytest.approx(1 / math.sqrt(3))
    assert low == pytest.approx(2.0 - 4.3027 * se, abs=1e-4)
    _, _, low90, high90 = t_interval([1.0, 2.0, 3.0], two_sided=False)
    assert high90 - low90 == pytest.approx(2 * 2.9200 * se, abs=1e-4)
    assert all(math.isnan(v) for v in t_interval([1.0])[1:])


def _energy_frame(ratio):
    """Three sessions x two rounds x one prompt; the sparse models are `ratio` times the dense ones."""
    rows = []
    for s in ("session_01", "session_02", "session_03"):
        for rnd in (1, 2):
            for dense, sparse in (("8B-Q4_K_M", "Sparse8B-Q4_K_M"), ("8B-Q8_0", "Sparse8B-Q8_0")):
                base = 10.0 if "Q4" in dense else 16.0
                for key, j in ((dense, base), (sparse, base * ratio)):
                    rows.append({"session_id": s, "round": rnd, "prompt_id": "p", "model_key": key, "n_threads": 8,
                                 "cpu_J_per_token": j, "tokens_per_sec": 60.0 / j})
    return pd.DataFrame(rows)


def test_tost_equivalent_and_not_equivalent():
    sessions = ["session_01", "session_02", "session_03"]
    same = tost_table(_energy_frame(1.0), {"all": sessions})
    assert set(same["verdict"]) == {"equivalent"}  # zero difference, zero spread: inside +-3 %
    far = tost_table(_energy_frame(1.10), {"all": sessions})
    assert set(far[far["metric"] == "cpu_J_per_token"]["verdict"]) == {"not equivalent"}  # +10 % with zero spread
    assert far["mean_diff_pct"].abs().max() == pytest.approx(10.0, abs=1e-6)


def test_fmt_decimal_separators():
    from make_figures import Fmt
    assert Fmt("tr").num(1234.5, 1) == "1.234,5"
    assert Fmt("en").num(1234.5, 1) == "1,234.5"
    assert Fmt("tr").num(float("nan")) == "—"
    assert Fmt("tr").pm(1.0, 0.25, 2) == "1,00 ± 0,25"


def test_append_rows_is_atomic_and_schema_checked(tmp_path):
    path = tmp_path / "rows.csv"
    fields = ["a", "b"]
    append_rows(path, fields, [{"a": 1, "b": 2.5}, {"a": 3, "b": float("nan")}])
    append_rows(path, fields, [{"a": 4, "b": 5}])
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n") and text.count("\n") == 4  # header + 3 complete lines, no partial line
    assert list(csv.DictReader(open(path, newline="", encoding="utf-8")))[1]["b"] == "NaN"
    with pytest.raises(SchemaMismatchError):
        append_rows(path, ["a", "c"], [{"a": 1, "c": 2}])
    with pytest.raises(KeyError):
        append_rows(path, fields, [{"a": 1}])
    assert path.read_text(encoding="utf-8").count("\n") == 4  # the failed appends wrote nothing
