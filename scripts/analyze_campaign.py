"""Cross-session analysis of the final campaign (Phase 2, sessions 01-03). No plots.

Reads the iteration files in results/ and writes CSV tables to results/analysis/:

    session_integrity.csv            per session: units, rows, NaN cells, throttle flags, max temperature, ...
    session_means_<experiment>.csv   per configuration x session: mean J/token (raw, net), tokens/s
    variance_components.csv          within/between-session variance (random-effects ANOVA) per configuration
    session1_deviation.csv           session_01 against the mean of session_02-03, all rounds and rounds 3-10
    ranking_by_session.csv           model ranks per session (energy experiment) and rank changes
    quantization_ratio_ci.csv        Q8_0 / Q4_K_M ratios with t confidence intervals over sessions
    sparse_vs_dense_tost.csv         equivalence test (TOST, +-3 %) of the sparse against the dense 8B model
    thread_scaling_optimum.csv       most efficient and saturating thread counts per session
    session_03_outage_window.csv     measurements around the network outage of session_03

Statistics on the main comparisons use the SESSION as the unit of replication (n = 3 or n = 2):
a session value is the mean over its rounds (each round = mean of the three prompts).

    python analyze_campaign.py
    python analyze_campaign.py --results-dir results --out-dir results/analysis
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (makes src/ importable without installing the package)
from analyze_sessions import round_values, session_variance_table
from llm_cpu_inference.config import PROJECT_ROOT
from llm_cpu_inference.stats import t_critical_95, t_critical_95_one_sided

EXPERIMENTS = {"energy": 240, "thread_scaling": 210}  # Expected rows per complete session
DEFAULT_EXPECTED_UNITS = "session_01=158,session_02=150,session_03=150"  # Cognitive runs in session_01 only
DEFAULT_OUTAGE_WINDOW = ("2026-10-08T00:00:00", "2026-10-08T00:06:00")  # Network outage of session_03
GROUP = ["model_key", "n_threads"]
METRICS = ["cpu_J_per_token", "cpu_J_per_token_net", "tokens_per_sec"]
FAMILIES = {"1B": ("1B-Q4_K_M", "1B-Q8_0"), "3B": ("3B-Q4_K_M", "3B-Q8_0"), "8B": ("8B-Q4_K_M", "8B-Q8_0")}
SPARSE_PAIRS = {"Q4_K_M": ("Sparse8B-Q4_K_M", "8B-Q4_K_M"), "Q8_0": ("Sparse8B-Q8_0", "8B-Q8_0")}
EQUIVALENCE_MARGIN = 0.03  # +-3 % relative difference
OUTLIER_Z = 2.5



def session_values(df, metric, exclude_first_rounds=0):
    """One value per (configuration, session): the mean over rounds of the round means."""
    rounds = round_values(df, metric, exclude_first_rounds=exclude_first_rounds)
    return rounds.groupby(GROUP + ["session_id"], as_index=False)[metric].mean()


def t_interval(values, two_sided=True):
    """Mean and t confidence interval (95 % two-sided, or 90 % two-sided = TOST's one-sided 95 %)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan")
    se = float(values.std(ddof=1)) / math.sqrt(n)
    t = t_critical_95(n - 1) if two_sided else t_critical_95_one_sided(n - 1)
    return mean, se, mean - t * se, mean + t * se


# ---------------------------------------------------------------- 1. integrity
def parse_log_status(log_dir, session_id):
    """Completion line, warnings and thermal/abort lines of a session's campaign logs."""
    status = {"log_files": 0, "completed": None, "failed": None, "pending": None, "duration_min": None,
              "warnings": 0, "errors": 0, "thermal_abort": 0, "resumed": 0}
    for path in sorted(glob.glob(os.path.join(log_dir, f"campaign_{session_id}_*.log"))):
        status["log_files"] += 1
        for line in open(path, encoding="utf-8"):
            if "| WARNING" in line:
                status["warnings"] += 1
            if "| ERROR" in line or "| CRITICAL" in line:
                status["errors"] += 1
            if "THERMAL_ABORT" in line or "THERMAL" in line.split("|")[-1]:
                status["thermal_abort"] += 1
            if re.search(r"resum", line, re.I):
                status["resumed"] += 1
            m = re.search(r"Run finished in ([\d.]+) min: (\d+) completed, (\d+) failed, (\d+) still pending", line)
            if m:
                status.update(duration_min=float(m.group(1)), completed=int(m.group(2)),
                              failed=int(m.group(3)), pending=int(m.group(4)))
    return status


def within_session_outliers(df):
    """Rows whose metric is more than OUTLIER_Z sample standard deviations from the mean of
    their (configuration, prompt) within the same session."""
    z = df.groupby(GROUP + ["session_id", "prompt_id"])["cpu_J_per_token"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1))
    return df.assign(z_J=z)[z.abs() > OUTLIER_Z]


def deterministic_configs(df):
    """'k/n': configurations (model, threads, prompt) whose generated text (output_sha256) is identical
    in every session, out of all configurations."""
    by_config = df.groupby(GROUP + ["prompt_id"])["output_sha256"].nunique()
    return f"{int((by_config == 1).sum())}/{len(by_config)}"


def integrity_table(frames, results_dir, expected_units=None):
    expected_units = expected_units or {}
    meta_by_session = {}
    for path in glob.glob(os.path.join(results_dir, "metadata", "*.json")):
        meta = json.load(open(path, encoding="utf-8"))
        if not meta.get("smoke"):
            meta_by_session[meta["session_id"]] = meta
    rows = []
    for sid in sorted({s for df in frames.values() for s in df["session_id"].unique()}):
        meta = meta_by_session.get(sid, {})
        log = parse_log_status(os.path.join(results_dir, "logs"), sid)
        row = {"session_id": sid, "run_id": meta.get("run_id"), "status": meta.get("status"),
               "start": meta.get("timestamp_start"), "end": meta.get("timestamp_end"),
               "expected_units": expected_units.get(sid), **{f"log_{k}": v for k, v in log.items()},
               "idle_cpu_W": (meta.get("idle_baseline") or {}).get("cpu_W"),
               "idle_cpu_util_pct": (meta.get("idle_baseline") or {}).get("cpu_util_percent"),
               "thermal_max_C_meta": (meta.get("thermal") or {}).get("max_temperature_c"),
               "thermal_abort_reason": (meta.get("thermal") or {}).get("abort_reason"),
               "operator_note": meta.get("operator_note")}
        for exp, df in frames.items():
            s = df[df["session_id"] == sid]
            row.update({
                f"{exp}_rows": len(s), f"{exp}_rows_expected": EXPERIMENTS[exp],
                f"{exp}_nan_cells": int(s.isna().sum().sum()),
                f"{exp}_throttle_flags": int(s["throttle_flag"].astype(bool).sum()),
                f"{exp}_temp_max_C": float(s["temp_max_C"].max()) if len(s) else None,
                f"{exp}_cooldown_waits_gt1s": int((s["cooldown_wait_s"] > 1).sum()),
                f"{exp}_tokens_not_100": int((s["generated_tokens"] != 100).sum()),
                f"{exp}_outliers_z{OUTLIER_Z}": int(len(within_session_outliers(s))),
                f"{exp}_deterministic_configs": deterministic_configs(df),  # across all sessions
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. session means and variance
def session_means_table(df):
    parts = [session_values(df, m).set_index(GROUP + ["session_id"])[m] for m in METRICS]
    long = pd.concat(parts, axis=1).reset_index()
    wide = long.pivot_table(index=GROUP, columns="session_id", values=METRICS)
    wide.columns = [f"{metric}__{sid}" for metric, sid in wide.columns]
    for m in METRICS:
        cols = [c for c in wide.columns if c.startswith(f"{m}__")]
        wide[f"{m}__cv_between_sessions_pct"] = 100 * wide[cols].std(axis=1, ddof=1) / wide[cols].mean(axis=1)
    return long, wide.reset_index()


def variance_table(frames):
    rows = []
    for exp, df in frames.items():
        for metric in ("cpu_J_per_token", "tokens_per_sec"):
            for label, data, excl in (("all_sessions", df, 0),
                                      ("all_sessions_rounds_3_10", df, 2),
                                      ("sessions_02_03", df[df["session_id"] != "session_01"], 0)):
                t = session_variance_table(data, metric, exclude_first_rounds=excl)
                t.insert(0, "subset", label)
                t.insert(0, "experiment", exp)
                t["between_share_pct"] = 100 * t["icc"]
                t["session_means"] = t["session_means"].apply(lambda v: ";".join(f"{x:.4f}" for x in v))
                rows.append(t)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 3. session_01 deviation
def session1_deviation_table(frames):
    rows = []
    for exp, df in frames.items():
        for metric in ("cpu_J_per_token", "tokens_per_sec"):
            for label, excl in (("all_rounds", 0), ("rounds_3_10", 2)):
                v = session_values(df, metric, exclude_first_rounds=excl)
                w = v.pivot_table(index=GROUP, columns="session_id", values=metric)
                ref = w[["session_02", "session_03"]].mean(axis=1)
                out = pd.DataFrame({
                    "experiment": exp, "metric": metric, "rounds": label,
                    "session_01": w["session_01"], "mean_s02_s03": ref,
                    "dev_s01_pct": 100 * (w["session_01"] - ref) / ref,
                    "dev_s02_vs_s03_pct": 100 * (w["session_02"] - w["session_03"]) / ref,
                }).reset_index()
                rows.append(out)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 4. ranking
def ranking_table(df_energy):
    v = session_values(df_energy, "cpu_J_per_token")
    w = v.pivot_table(index="model_key", columns="session_id", values="cpu_J_per_token")
    ranks = w.rank(axis=0, method="min").astype(int)
    ranks.columns = [f"rank_J_{c}" for c in ranks.columns]
    w.columns = [f"J_{c}" for c in w.columns]
    out = pd.concat([w, ranks], axis=1)
    out["rank_changes"] = (out.filter(like="rank_J_").nunique(axis=1) > 1)
    out = out.reset_index()
    # Explicit pairwise check of the near-identical sparse/dense pairs
    notes = []
    for q, (sparse, dense) in SPARSE_PAIRS.items():
        order = []
        for col in [c for c in w.columns]:
            order.append("sparse<dense" if w.loc[sparse, col] < w.loc[dense, col] else "dense<sparse")
        notes.append({"pair": f"{sparse} vs {dense}", **{c.replace("J_", "order_"): o for c, o in zip(w.columns, order)},
                      "order_changes": len(set(order)) > 1})
    return out, pd.DataFrame(notes)


# ---------------------------------------------------------------- 5. quantization ratios
def ratio_rows(df, metric, numerator, denominator, label, subsets):
    v = session_values(df, metric).pivot_table(index="model_key", columns="session_id", values=metric)
    rows = []
    for subset_label, sessions in subsets.items():
        ratios = (v.loc[numerator, sessions] / v.loc[denominator, sessions]).to_numpy()
        mean, se, low, high = t_interval(ratios)
        lmean, lse, llow, lhigh = t_interval(np.log(ratios))
        rows.append({"comparison": label, "metric": metric, "numerator": numerator, "denominator": denominator,
                     "subset": subset_label, "n_sessions": len(sessions),
                     "ratios_by_session": ";".join(f"{r:.4f}" for r in ratios),
                     "ratio_mean": mean, "ratio_se": se, "ratio_ci95_low": low, "ratio_ci95_high": high,
                     "ratio_geomean": math.exp(lmean), "ratio_ci95_low_log": math.exp(llow) if not math.isnan(llow) else float("nan"),
                     "ratio_ci95_high_log": math.exp(lhigh) if not math.isnan(lhigh) else float("nan")})
    return rows


def quantization_table(df_energy, subsets):
    rows = []
    for family, (q4, q8) in FAMILIES.items():
        rows += ratio_rows(df_energy, "cpu_J_per_token", q8, q4, f"{family}: Q8_0/Q4_K_M", subsets)
        rows += ratio_rows(df_energy, "cpu_J_per_token_net", q8, q4, f"{family}: Q8_0/Q4_K_M", subsets)
        rows += ratio_rows(df_energy, "tokens_per_sec", q4, q8, f"{family}: Q4_K_M/Q8_0 speed", subsets)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 6. TOST sparse vs dense
def tost_table(df_energy, subsets):
    bound = math.log(1 + EQUIVALENCE_MARGIN)
    rows = []
    for q, (sparse, dense) in SPARSE_PAIRS.items():
        for metric in ("cpu_J_per_token", "tokens_per_sec"):
            v = session_values(df_energy, metric).pivot_table(index="model_key", columns="session_id", values=metric)
            for subset_label, sessions in subsets.items():
                log_ratio = np.log(v.loc[sparse, sessions].to_numpy() / v.loc[dense, sessions].to_numpy())
                mean, se, low, high = t_interval(log_ratio, two_sided=False)  # 90 % interval
                if math.isnan(low):
                    verdict = "belirsiz (n<2)"
                elif low > -bound and high < bound:
                    verdict = "equivalent"
                elif low > bound or high < -bound:
                    verdict = "not equivalent"
                else:
                    verdict = "inconclusive"
                rows.append({"pair": f"{sparse} vs {dense}", "metric": metric, "subset": subset_label,
                             "n_sessions": len(sessions), "margin_pct": 100 * EQUIVALENCE_MARGIN,
                             "log_ratio_by_session": ";".join(f"{r:+.5f}" for r in log_ratio),
                             "mean_diff_pct": 100 * (math.exp(mean) - 1),
                             "ci90_low_pct": 100 * (math.exp(low) - 1) if not math.isnan(low) else float("nan"),
                             "ci90_high_pct": 100 * (math.exp(high) - 1) if not math.isnan(high) else float("nan"),
                             "verdict": verdict})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 7. thread scaling optimum
def thread_optimum_table(df_thread, saturation_frac=0.98):
    rows = []
    for metric in ("cpu_J_per_token", "cpu_J_per_token_net"):
        v = session_values(df_thread, metric)
        s = session_values(df_thread, "tokens_per_sec")
        for sid in sorted(v["session_id"].unique()):
            vj = v[v["session_id"] == sid].set_index("n_threads")[metric]
            vs = s[s["session_id"] == sid].set_index("n_threads")["tokens_per_sec"]
            saturating = vs[vs >= saturation_frac * vs.max()].index.min()
            rows.append({"session_id": sid, "energy_metric": metric, "min_energy_threads": int(vj.idxmin()),
                         "min_energy_value": float(vj.min()), "max_speed_threads": int(vs.idxmax()),
                         "max_speed_value": float(vs.max()),
                         f"saturation_threads_{int(saturation_frac * 100)}pct": int(saturating),
                         "speed_by_threads": ";".join(f"{n}:{x:.3f}" for n, x in vs.sort_index().items()),
                         "energy_by_threads": ";".join(f"{n}:{x:.3f}" for n, x in vj.sort_index().items())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 8. robust anomaly scan (median +- 3 MAD)
MAD_K = 3.0


def mad_anomalies(frames, metrics=("cpu_J_per_token", "tokens_per_sec")):
    """Rows farther than MAD_K scaled MADs from the median of their (configuration, prompt, session).

    scaled MAD = 1.4826 x median(|x - median|), i.e. the robust sigma. A row is listed once per metric.
    """
    rows = []
    for exp, df in frames.items():
        for metric in metrics:
            g = df.groupby(GROUP + ["session_id", "prompt_id"])[metric]
            med = g.transform("median")
            mad = 1.4826 * (df[metric] - med).abs().groupby([df[c] for c in GROUP + ["session_id", "prompt_id"]]).transform("median")
            score = (df[metric] - med) / mad.replace(0, np.nan)
            hit = df[score.abs() > MAD_K].copy()
            hit["experiment"] = exp
            hit["metric"] = metric
            hit["median"] = med[hit.index]
            hit["mad_scaled"] = mad[hit.index]
            hit["robust_z"] = score[hit.index]
            hit["deviation_pct"] = 100 * (hit[metric] - hit["median"]) / hit["median"]
            rows.append(hit[["experiment", "metric", "session_id", "round", "order_in_round", "model_key", "n_threads",
                             "prompt_id", "timestamp", metric, "median", "mad_scaled", "robust_z", "deviation_pct",
                             "throttle_flag", "temp_max_C", "freq_mean_mhz"]].rename(columns={metric: "value"}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out.sort_values(["experiment", "metric", "robust_z"], key=lambda s: s.abs() if s.name == "robust_z" else s,
                           ascending=[True, True, False]).reset_index(drop=True) if len(out) else out


def anomaly_sensitivity(frames, anomalies):
    """Session means of cpu_J_per_token with and without the flagged rows (energy metric only)."""
    rows = []
    for exp, df in frames.items():
        flagged = anomalies[(anomalies["experiment"] == exp) & (anomalies["metric"] == "cpu_J_per_token")]
        key_cols = ["session_id", "round", "model_key", "n_threads", "prompt_id"]
        mask = df.set_index(key_cols).index.isin(flagged.set_index(key_cols).index) if len(flagged) else np.zeros(len(df), bool)
        full = session_values(df, "cpu_J_per_token").rename(columns={"cpu_J_per_token": "with_anomalies"})
        trimmed = session_values(df[~mask], "cpu_J_per_token").rename(columns={"cpu_J_per_token": "without_anomalies"})
        m = full.merge(trimmed, on=GROUP + ["session_id"])
        m["change_pct"] = 100 * (m["without_anomalies"] - m["with_anomalies"]) / m["with_anomalies"]
        m.insert(0, "experiment", exp)
        m["n_rows_removed"] = [int(mask[(df["session_id"] == r.session_id) & (df["model_key"] == r.model_key)
                                        & (df["n_threads"] == r.n_threads)].sum()) for r in m.itertuples()]
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 8b. session_03 outage
def outage_table(df_thread, out_dir, start=DEFAULT_OUTAGE_WINDOW[0], end=DEFAULT_OUTAGE_WINDOW[1]):
    """Measurements of session_03 in the outage window, with their z-score inside their configuration."""
    s3 = df_thread[df_thread["session_id"] == "session_03"].copy()
    s3["z_J"] = s3.groupby(GROUP + ["prompt_id"])["cpu_J_per_token"].transform(lambda x: (x - x.mean()) / x.std(ddof=1))
    s3["config_mean_J_other_rounds"] = s3.groupby(GROUP + ["prompt_id"])["cpu_J_per_token"].transform("mean")
    ts = pd.to_datetime(s3["timestamp"].str.slice(0, 19))
    window = s3[(ts >= start) & (ts <= end)]
    cols = ["timestamp", "round", "order_in_round", "model_key", "n_threads", "prompt_id", "cpu_J_per_token",
            "config_mean_J_other_rounds", "z_J", "tokens_per_sec", "avg_cpu_power_W", "freq_mean_mhz", "temp_mean_C"]
    out = window[cols].copy()
    events_path = os.path.join(out_dir, "session_03_network_events.csv")
    if os.path.exists(events_path):
        ev = pd.read_csv(events_path)
        out.attrs["network_events"] = f"{len(ev)} events {ev['time'].min()} - {ev['time'].max()}"
    return out


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Cross-session analysis of the final campaign.")
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--expected-units", default=DEFAULT_EXPECTED_UNITS,
                        help="Planned units per session, e.g. session_01=158,session_02=150")
    parser.add_argument("--outage-window", nargs=2, metavar=("START", "END"), default=list(DEFAULT_OUTAGE_WINDOW),
                        help="ISO timestamps of the window listed in session_03_outage_window.csv")
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "analysis")
    expected_units = {k: int(v) for k, v in (item.split("=") for item in args.expected_units.split(",") if item)}
    os.makedirs(out_dir, exist_ok=True)

    frames = {exp: pd.read_csv(os.path.join(args.results_dir, f"{exp}_iterations.csv")) for exp in EXPERIMENTS}
    sessions = sorted(frames["energy"]["session_id"].unique())
    subsets = {"all_sessions": sessions, "sessions_02_03": [s for s in sessions if s != "session_01"]}

    def save(name, df):
        path = os.path.join(out_dir, name)
        df.to_csv(path, index=False, encoding="utf-8", float_format="%.6g")
        print(f"written: {path} ({len(df)} rows)")

    save("session_integrity.csv", integrity_table(frames, args.results_dir, expected_units))
    for exp, df in frames.items():
        long, wide = session_means_table(df)
        save(f"session_means_{exp}.csv", wide)
        save(f"session_means_{exp}_long.csv", long)
    save("variance_components.csv", variance_table(frames))
    save("session1_deviation.csv", session1_deviation_table(frames))
    ranks, pair_order = ranking_table(frames["energy"])
    save("ranking_by_session.csv", ranks)
    save("ranking_sparse_dense_order.csv", pair_order)
    save("quantization_ratio_ci.csv", quantization_table(frames["energy"], subsets))
    save("sparse_vs_dense_tost.csv", tost_table(frames["energy"], subsets))
    save("thread_scaling_optimum.csv", thread_optimum_table(frames["thread_scaling"]))
    save("session_03_outage_window.csv", outage_table(frames["thread_scaling"], out_dir, *args.outage_window))
    anomalies = mad_anomalies(frames)
    save("anomalies_mad.csv", anomalies)
    save("anomaly_sensitivity.csv", anomaly_sensitivity(frames, anomalies))


if __name__ == "__main__":
    main()
