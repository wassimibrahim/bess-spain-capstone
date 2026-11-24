"""Statistical tests."""
import numpy as np
from scipy import stats


def dm_test(a: np.ndarray, b: np.ndarray, lags: int = 7):
    """Diebold-Mariano-type test that the mean of the daily difference a - b is zero, with a Newey-West
    (Bartlett kernel) variance so that serial correlation of daily results does not overstate significance.
    Applied to daily profits (a, b = net EUR per day of two strategies) or to daily losses."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    n, m = len(d), d.mean()
    e = d - m
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    t = m / np.sqrt(v / n)
    return float(m), float(t), float(2 * (1 - stats.norm.cdf(abs(t))))
