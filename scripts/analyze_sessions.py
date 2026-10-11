"""Within-session and between-session variance of the interleaved measurements (no plots).

Model: one-way random-effects ANOVA per configuration, with the session as the random factor.
The replicate inside a session is a round; with several prompts the round value is the mean
over the prompts of that round (as in the 'ALL' summary rows).

    y[s, r] = mu + b[s] + e[s, r]     b ~ N(0, var_between), e ~ N(0, var_within)

    MS_within  = sum_s sum_r (y[s,r] - mean_s)^2 / (N - k)
    MS_between = sum_s n_s (mean_s - grand_mean)^2 / (k - 1)
    n0         = (N - sum_s n_s^2 / N) / (k - 1)          (= n for balanced sessions)
    var_within  = MS_within
    var_between = max(0, (MS_between - MS_within) / n0)
    ICC         = var_between / (var_between + var_within)

k = number of sessions, n_s = rounds in session s, N = sum n_s. var_between is the part of the
spread that comes from day-to-day / reboot-to-reboot differences; it is what a single session
cannot reveal. With one session only the within-session part is defined.

    python analyze_sessions.py                                   # energy, cpu_J_per_token
    python analyze_sessions.py --experiment thread_scaling --metric tokens_per_sec
    python analyze_sessions.py --results-dir results/smoke
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (makes src/ importable without installing the package)
from llm_cpu_inference.config import PROJECT_ROOT

GROUP_COLUMNS = ["model_key", "n_threads"]


def round_values(df, metric, prompt=None, exclude_first_rounds=0):
    """One value per (configuration, session, round): the metric averaged over the round's prompts.

    exclude_first_rounds=N drops rounds 1..N of every session (sensitivity analysis for start-up
    transients, e.g. the first two rounds of session_01).
    """
    if prompt is not None:
        df = df[df["prompt_id"] == prompt]
    if exclude_first_rounds:
        df = df[df["round"] > exclude_first_rounds]
    return df.groupby(GROUP_COLUMNS + ["session_id", "round"], as_index=False)[metric].mean()


def variance_components(values_by_session):
    """Random-effects ANOVA for one configuration. `values_by_session`: {session_id: [values]}."""
    groups = [np.asarray(v, dtype=float) for v in values_by_session.values() if len(v) > 0]
    k = len(groups)
    sizes = np.array([len(g) for g in groups])
    total = int(sizes.sum())
    nan = float("nan")
    result = {"n_sessions": k, "n_rounds_total": total, "grand_mean": nan, "var_within": nan, "var_between": nan,
              "sd_within": nan, "sd_between": nan, "icc": nan, "cv_within_pct": nan, "cv_between_pct": nan,
              "session_means": [float(g.mean()) for g in groups]}
    if total == 0:
        return result
    grand_mean = float(np.concatenate(groups).mean())
    result["grand_mean"] = grand_mean
    if total - k > 0:
        ms_within = sum(float(((g - g.mean()) ** 2).sum()) for g in groups) / (total - k)
        result["var_within"] = ms_within
    if k > 1 and not math.isnan(result["var_within"]):
        ms_between = sum(len(g) * (float(g.mean()) - grand_mean) ** 2 for g in groups) / (k - 1)
        n0 = (total - float((sizes ** 2).sum()) / total) / (k - 1)
        result["var_between"] = max(0.0, (ms_between - result["var_within"]) / n0)
        denominator = result["var_between"] + result["var_within"]
        result["icc"] = result["var_between"] / denominator if denominator > 0 else nan
    for part in ("within", "between"):
        var = result[f"var_{part}"]
        if not math.isnan(var):
            result[f"sd_{part}"] = math.sqrt(var)
            result[f"cv_{part}_pct"] = 100 * math.sqrt(var) / grand_mean if grand_mean else nan
    return result


def session_variance_table(df, metric, prompt=None, exclude_first_rounds=0):
    """Variance components of `metric` for every configuration in an iterations DataFrame."""
    values = round_values(df, metric, prompt, exclude_first_rounds)
    rows = []
    for (model_key, n_threads), group in values.groupby(GROUP_COLUMNS):
        by_session = {sid: g[metric].tolist() for sid, g in group.groupby("session_id")}
        rows.append({"model_key": model_key, "n_threads": n_threads, "metric": metric,
                     **variance_components(by_session)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Within- and between-session variance of the measurements.")
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--experiment", choices=["energy", "thread_scaling"], default="energy")
    parser.add_argument("--metric", default="cpu_J_per_token")
    parser.add_argument("--prompt", default=None, help="Restrict to one prompt id (default: mean over prompts)")
    parser.add_argument("--exclude-first-rounds", type=int, default=0, metavar="N",
                        help="Drop rounds 1..N of every session (sensitivity analysis)")
    args = parser.parse_args()

    path = os.path.join(args.results_dir, f"{args.experiment}_iterations.csv")
    df = pd.read_csv(path)
    table = session_variance_table(df, args.metric, args.prompt, args.exclude_first_rounds)
    pd.set_option("display.width", 200)
    columns = ["model_key", "n_threads", "n_sessions", "n_rounds_total", "grand_mean", "sd_within", "sd_between",
               "cv_within_pct", "cv_between_pct", "icc"]
    excluded = f", rounds 1-{args.exclude_first_rounds} excluded" if args.exclude_first_rounds else ""
    print(f"{args.metric} ({args.experiment}{excluded}, {path})")
    print(table[columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
