"""Project valuation of a merchant battery from backtested arbitrage margins (pre-tax, unlevered).

Revenue scenarios are built by bootstrapping calendar months of realised strategy margins (keeps the seasonal
pattern: each simulated January is drawn from the historical Januaries, and so on). Each year's revenue is then
reduced for capacity fade (cell degradation) and for spread compression ('cannibalisation': more storage chasing
the same hours). Assumptions are explicit parameters and every one is varied in the sensitivity tables.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Project:
    capex_eur_kwh: float = 200.0     # all-in installed cost per kWh of usable energy
    energy_mwh: float = 2.0
    power_mw: float = 1.0
    opex_frac: float = 0.02          # fixed O&M per year, share of capex (Ember 2025 assumption)
    fade: float = 0.02               # capacity fade per year (Ember 2025 assumption)
    compression: float = 0.03        # yearly fall in arbitrage margin as storage builds out
    life: int = 15
    wacc: float = 0.10               # merchant cost of capital (Ember: 10-15% merchant, 7% contracted)

    @property
    def capex(self):
        return self.capex_eur_kwh * self.energy_mwh * 1000


def simulate_revenue(monthly: pd.Series, years: int, n_sims: int = 5000, seed: int = 0) -> np.ndarray:
    """(n_sims, years) array of annual revenues before fade/compression, bootstrapped by calendar month."""
    rng = np.random.default_rng(seed)
    by_m = {m: monthly[monthly.index.month == m].values for m in range(1, 13)}
    out = np.zeros((n_sims, years))
    for m in range(1, 13):
        v = by_m[m]
        out += v[rng.integers(0, len(v), size=(n_sims, years))]
    return out


def cash_flows(rev: np.ndarray, p: Project) -> np.ndarray:
    t = np.arange(1, rev.shape[1] + 1)
    adj = rev * (1 - p.fade) ** (t - 1) * (1 - p.compression) ** (t - 1)
    return adj - p.opex_frac * p.capex


def npv(cf: np.ndarray, p: Project) -> np.ndarray:
    t = np.arange(1, cf.shape[-1] + 1)
    return -p.capex + (cf / (1 + p.wacc) ** t).sum(axis=-1)


def irr(cf_row: np.ndarray, capex: float) -> float:
    flows = np.concatenate([[-capex], cf_row])
    lo, hi = -0.99, 1.0
    f = lambda r: (flows / (1 + r) ** np.arange(len(flows))).sum()
    if f(lo + 1e-9) * f(hi) > 0:
        return np.nan
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def breakeven_capex(rev: np.ndarray, p: Project) -> float:
    """Capex per kWh at which the median-scenario NPV is zero (opex scales with capex)."""
    lo, hi = 1.0, 2000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        q = Project(**{**p.__dict__, "capex_eur_kwh": mid})
        v = np.median(npv(cash_flows(rev, q), q))
        lo, hi = (mid, hi) if v > 0 else (lo, mid)
    return (lo + hi) / 2


def value(rev: np.ndarray, p: Project) -> dict:
    cf = cash_flows(rev, p)
    v = npv(cf, p)
    med_row = cf[np.argsort(v)[len(v) // 2]]
    return {"npv_p10": np.percentile(v, 10), "npv_p50": np.percentile(v, 50), "npv_p90": np.percentile(v, 90),
            "prob_npv_pos": float((v > 0).mean()), "irr_p50": irr(med_row, p.capex),
            "breakeven_capex_eur_kwh": breakeven_capex(rev, p)}
