"""Backtest: turn a price forecast into a daily schedule and settle it at realised prices."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .battery import Battery, DailyLP, settle


def clock_profile(M: pd.DataFrame, lookback_days: int = 365) -> pd.DataFrame:
    """'Schedule by the clock': for each day, the average hourly profile of the trailing year (known at D-1).
    Re-estimated monthly, so the schedule only changes at month starts."""
    prof = pd.DataFrame(index=M.index, columns=M.columns, dtype=float)
    for month in pd.period_range(M.index.min(), M.index.max(), freq="M"):
        t0 = month.start_time
        hist = M[(M.index < t0) & (M.index >= t0 - pd.Timedelta(days=lookback_days))]
        if len(hist) < 28:
            continue
        sel = (M.index >= t0) & (M.index <= month.end_time)
        prof.loc[sel] = hist.mean().values
    return prof


def run(M: pd.DataFrame, signal: pd.DataFrame, bat: Battery, start: str, end: str | None = None) -> pd.DataFrame:
    """Schedule each day on `signal` (forecast of D, or realised D for perfect foresight); settle at M."""
    lp = DailyLP(bat)
    days = signal.loc[start:end].dropna().index
    rows = []
    for d in days:
        c, dch = lp.solve(signal.loc[d].values.astype(float))
        r = settle(c, dch, M.loc[d].values.astype(float), bat)
        r["date"] = d
        rows.append(r)
    return pd.DataFrame(rows).set_index("date")


def summarise(res: dict, bat: Battery, freq: str = "YE") -> pd.DataFrame:
    """Annual net margin (EUR per MW-year), capture rate vs perfect foresight, cycles per year."""
    out = {}
    for k, r in res.items():
        g = r.resample(freq)
        out[k] = g["net"].sum() / bat.power_mw
    df = pd.DataFrame(out)
    df.index = df.index.year if freq == "YE" else df.index
    return df
