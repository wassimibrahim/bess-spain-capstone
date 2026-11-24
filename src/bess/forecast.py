"""Day-ahead price forecasts, produced walk-forward with no look-ahead.

Timing. Bids for delivery day D close at noon of D-1. At that moment the prices of D-1 are known (they were
set at noon of D-2), so every feature uses prices up to D-1 and calendar facts about D. Models are refitted at
the start of each month on all data available then (expanding window) and never see the month they forecast.

Models
- naive_d1:  price(D, h) = price(D-1, h)
- naive_d7:  price(D, h) = price(D-7, h)
- lear:      LASSO-estimated autoregressive model, one model per hour, the standard benchmark of the
             electricity-price-forecasting literature (Lago, Marcjasz, De Schutter and Weron 2021).
             Inputs: the 24 prices of D-1, D-2, D-3 and D-7, day-of-week and holiday dummies; prices are
             transformed with asinh (robust to zero and negative prices); 2-year rolling calibration window.
- lgbm:      gradient-boosted trees (LightGBM) on a long table (one row per day and hour) with lags,
             daily statistics of the last week, hour, weekday, month and holiday flags; expanding window.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import holidays
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import StandardScaler

H = [f"h{h:02d}" for h in range(24)]
# Chosen on the 2017-2018 validation years (grid: leaves 15/31/63, 300 trees at 0.05 or 600 at 0.03).
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)


def es_holidays(index):
    hol = holidays.country_holidays("ES", years=range(index.year.min(), index.year.max() + 1))
    return np.array([d in hol for d in index.date], dtype=float)


def naive(M: pd.DataFrame, lag: int) -> pd.DataFrame:
    return M.shift(lag)


# ---------------------------------------------------------------- LEAR
def _lear_X(Z: pd.DataFrame, index) -> pd.DataFrame:
    parts = [Z.shift(l).add_prefix(f"l{l}_") for l in (1, 2, 3, 7)]
    X = pd.concat(parts, axis=1)
    dow = pd.get_dummies(index.dayofweek, prefix="dow", dtype=float).set_index(index)
    X = pd.concat([X, dow], axis=1)
    X["hol"] = es_holidays(index)
    return X


def lear(M: pd.DataFrame, start: str, window_days: int = 728) -> pd.DataFrame:
    """Variance-stabilised LEAR: prices are centred on the median and scaled by the median absolute deviation
    of the calibration window, then transformed with asinh (Uniejewski, Weron and Ziel 2018)."""
    out = pd.DataFrame(index=M.index[M.index >= start], columns=H, dtype=float)
    for month in pd.period_range(out.index.min(), out.index.max(), freq="M"):
        t0 = month.start_time
        win = M[(M.index < t0) & (M.index >= t0 - pd.Timedelta(days=window_days + 7))]
        med = np.median(win.values); mad = np.median(np.abs(win.values - med)) * 1.4826 + 1e-6
        Z = np.arcsinh((M - med) / mad)
        X = _lear_X(Z, M.index)
        tr = X.index[(X.index < t0) & (X.index >= t0 - pd.Timedelta(days=window_days))]
        tr = tr[X.loc[tr].notna().all(axis=1)]
        te = out.index[(out.index >= t0) & (out.index <= month.end_time)]
        sc = StandardScaler().fit(X.loc[tr])
        Xtr, Xte = sc.transform(X.loc[tr]), sc.transform(X.loc[te])
        for h in H:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = LassoLarsIC(criterion="aic").fit(Xtr, Z.loc[tr, h])
            out.loc[te, h] = np.sinh(m.predict(Xte)) * mad + med
    return out


# ---------------------------------------------------------------- LightGBM
def _long_table(M: pd.DataFrame) -> pd.DataFrame:
    d1, d7 = M.shift(1), M.shift(7)
    stats = pd.DataFrame({
        "d1_mean": d1.mean(axis=1), "d1_min": d1.min(axis=1), "d1_max": d1.max(axis=1), "d1_std": d1.std(axis=1),
        "w_mean": M.shift(1).rolling(7).mean().mean(axis=1),
        "d1_solar_dip": d1[["h13", "h14", "h15", "h16"]].mean(axis=1) - d1.mean(axis=1),
        "d1_evening": d1[["h20", "h21", "h22"]].mean(axis=1) - d1.mean(axis=1),
    }, index=M.index)
    hol = es_holidays(M.index)
    rows = []
    for k, h in enumerate(H):
        f = pd.DataFrame(index=M.index)
        f["hour"] = k
        f["p_d1"] = d1[h]; f["p_d2"] = M.shift(2)[h]; f["p_d7"] = d7[h]
        f["p_w"] = M[h].shift(1).rolling(7).mean()
        f["p_d1_rank"] = d1.rank(axis=1, pct=True)[h]
        f = f.join(stats)
        f["dow"] = M.index.dayofweek; f["month"] = M.index.month; f["hol"] = hol
        f["hol_d1"] = np.roll(hol, 1)
        f["y"] = M[h]
        rows.append(f)
    L = pd.concat(rows).sort_index(kind="stable")
    return L


def lgbm(M: pd.DataFrame, start: str, params: dict | None = None, seed: int = 0) -> pd.DataFrame:
    """Walk-forward LightGBM. Default hyperparameters were chosen on the 2017-2018 validation years."""
    import lightgbm as lgb
    params = params or LGBM_PARAMS
    L = _long_table(M)
    feats = [c for c in L.columns if c != "y"]
    out = pd.DataFrame(index=M.index[M.index >= start], columns=H, dtype=float)
    for month in pd.period_range(out.index.min(), out.index.max(), freq="M"):
        t0 = month.start_time
        tr = L[(L.index < t0)].dropna()
        te = L[(L.index >= t0) & (L.index <= month.end_time)]
        m = lgb.LGBMRegressor(objective="l1", random_state=seed, verbose=-1, n_jobs=2, **params)
        m.fit(tr[feats], tr["y"])
        p = m.predict(te[feats])
        for (d, hr), v in zip(zip(te.index, te["hour"]), p):
            out.at[d, H[hr]] = v
    return out


def last_lgbm_model(M: pd.DataFrame, end: str):
    """Model trained on all data before `end` (for interpretation)."""
    import lightgbm as lgb
    L = _long_table(M)
    feats = [c for c in L.columns if c != "y"]
    tr = L[L.index < end].dropna()
    m = lgb.LGBMRegressor(objective="l1", random_state=0, verbose=-1, **LGBM_PARAMS).fit(tr[feats], tr["y"])
    return m, L, feats
