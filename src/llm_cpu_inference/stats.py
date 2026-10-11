"""Small-sample statistics: mean, sample standard deviation and Student-t confidence intervals."""
import math
import statistics

# Two-sided 95 % critical values of Student's t, t(0.975, df), for df = 1..30.
_T975 = [
    12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281,
    2.2010, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860,
    2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423,
]
_Z975 = 1.959964

# One-sided 95 % critical values, t(0.95, df), for df = 1..30 (used by the TOST equivalence test).
_T95 = [
    6.3138, 2.9200, 2.3534, 2.1318, 2.0150, 1.9432, 1.8946, 1.8595, 1.8331, 1.8125,
    1.7959, 1.7823, 1.7709, 1.7613, 1.7531, 1.7459, 1.7396, 1.7341, 1.7291, 1.7247,
    1.7207, 1.7171, 1.7139, 1.7109, 1.7081, 1.7056, 1.7033, 1.7011, 1.6991, 1.6973,
]
_Z95 = 1.644854

NAN = float("nan")


def t_critical_95(df):
    """t(0.975, df): exact table up to df = 30, Cornish-Fisher expansion beyond (error < 1e-4)."""
    if df < 1:
        return NAN
    if df <= 30:
        return _T975[df - 1]
    z = _Z975
    return (z
            + (z ** 3 + z) / (4 * df)
            + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))


def t_critical_95_one_sided(df):
    """t(0.95, df): exact table up to df = 30, Cornish-Fisher expansion beyond."""
    if df < 1:
        return NAN
    if df <= 30:
        return _T95[df - 1]
    z = _Z95
    return (z
            + (z ** 3 + z) / (4 * df)
            + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))


def summarize(values):
    """Mean, sample std (n - 1) and the 95 % t confidence interval of the mean.

    NaN and None are dropped first. With fewer than two valid values the std and the
    interval are undefined and returned as NaN.
    """
    valid = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(valid)
    if n == 0:
        return {"n": 0, "mean": NAN, "std": NAN, "ci95_low": NAN, "ci95_high": NAN}
    mean = statistics.fmean(valid)
    if n == 1:
        return {"n": 1, "mean": mean, "std": NAN, "ci95_low": NAN, "ci95_high": NAN}
    std = statistics.stdev(valid)
    half_width = t_critical_95(n - 1) * std / math.sqrt(n)
    return {"n": n, "mean": mean, "std": std, "ci95_low": mean - half_width, "ci95_high": mean + half_width}
