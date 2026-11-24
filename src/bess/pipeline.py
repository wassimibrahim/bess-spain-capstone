"""End-to-end pipeline: data -> walk-forward forecasts -> dispatch backtests -> RL -> sensitivity -> valuation.

Every table and figure of the manuscript is written to results/ by `run_all()`. Set `use_cache=True` to reuse
saved forecasts (results/forecasts.pkl) and skip the slowest step (about 4 minutes on a laptop).
"""
from __future__ import annotations
import os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data import load_hourly, daily_matrix
from .battery import Battery
from . import forecast as F, backtest as B, rl as RL, stats as S, valuation as V
from .plots import style, C, LABEL, INK, INK2, MUTED, eu

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
TEST_START, END = "2019-01-01", "2026-11-25"
VALID = ("2017-01-01", "2018-12-31")
RL_TRAIN, RL_TEST = ("2021-01-01", "2023-12-31"), ("2024-01-01", END)
BASE = Battery()                      # 1 MW / 2 MWh, RTE 87%, wear 15 EUR/MWh, fee 0,5 EUR/MWh
BASE.deg_eur_mwh = 15.0
STRATS = ["perfect", "ensemble", "lear", "lgbm", "naive_d1", "naive_d7", "clock"]


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def build_forecasts(M, use_cache=True):
    path = os.path.join(RES, "forecasts.pkl")
    if use_cache and os.path.exists(path):
        _log("forecasts: loaded from cache")
        return pd.read_pickle(path)
    fc = {"naive_d1": F.naive(M, 1), "naive_d7": F.naive(M, 7)}
    _log("forecasts: LEAR (24 hourly LASSO models, refitted monthly)")
    fc["lear"] = F.lear(M, VALID[0])
    _log("forecasts: LightGBM (refitted monthly)")
    fc["lgbm"] = F.lgbm(M, VALID[0])
    fc["ensemble"] = (fc["lear"] + fc["lgbm"]) / 2
    fc["clock"] = B.clock_profile(M)
    pd.to_pickle(fc, path)
    return fc


def forecast_table(M, fc, start=TEST_START, end=END):
    idx = M.loc[start:end].index
    A = M.loc[idx]
    rows = {}
    base_mae = (fc["naive_d1"].loc[idx] - A).abs().values.mean()
    sp = A.max(axis=1) - A.min(axis=1)
    for k in ["naive_d1", "naive_d7", "lear", "lgbm", "ensemble"]:
        f = fc[k].loc[idx]
        e = (f - A)
        rho = pd.Series([pd.Series(f.loc[d].values).corr(pd.Series(A.loc[d].values), method="spearman")
                         for d in idx]).mean()
        rows[k] = {"MAE": np.abs(e.values).mean(), "RMSE": np.sqrt((e.values ** 2).mean()),
                   "rMAE vs D-1": np.abs(e.values).mean() / base_mae,
                   "Spread MAE": (( f.max(axis=1) - f.min(axis=1)) - sp).abs().mean(),
                   "Rank corr.": rho}
    return pd.DataFrame(rows).T


def backtests(M, fc, bat=BASE, start=TEST_START, end=END, keys=STRATS):
    sig = dict(fc); sig["perfect"] = M
    return {k: B.run(M, sig[k], bat, start, end) for k in keys}


def annual_table(res, bat=BASE):
    S_ = B.summarise(res, bat)
    return S_


def run_all(use_cache=True, n_sims=5000):
    os.makedirs(FIG, exist_ok=True)
    style()
    t0 = time.time()
    M = daily_matrix(load_hourly())
    _log(f"data: {len(M)} days, {M.index.min().date()} to {M.index.max().date()}")
    key = {}

    # ---------- data facts
    yr = M.index.year
    facts = pd.DataFrame({
        "Mean price": M.mean(axis=1).groupby(yr).mean(),
        "Mean daily spread (max-min)": (M.max(axis=1) - M.min(axis=1)).groupby(yr).mean(),
        "Hours at or below 0 (%)": (M <= 0).mean(axis=1).groupby(yr).mean() * 100,
        "Cheapest hour (mode)": M.idxmin(axis=1).str[1:].astype(int).groupby(yr).agg(lambda s: s.mode().iloc[0]),
    })
    facts.to_csv(os.path.join(RES, "t1_market_facts.csv"), float_format="%.2f")

    # ---------- forecasts
    fc = build_forecasts(M, use_cache)
    ft = forecast_table(M, fc); ft.to_csv(os.path.join(RES, "t2_forecast_accuracy_test.csv"), float_format="%.3f")
    fv = forecast_table(M, fc, *VALID); fv.to_csv(os.path.join(RES, "t2b_forecast_accuracy_validation.csv"), float_format="%.3f")
    _log("forecast accuracy computed")

    # ---------- base-case backtests
    res = backtests(M, fc)
    ann = annual_table(res)
    ann.to_csv(os.path.join(RES, "t3_annual_net_margin_eur_per_mw.csv"), float_format="%.0f")
    cap = ann.div(ann["perfect"], axis=0) * 100
    cap.to_csv(os.path.join(RES, "t3b_capture_rate_pct.csv"), float_format="%.1f")
    tot = {k: r["net"].sum() for k, r in res.items()}
    yrs = (pd.Timestamp(END) - pd.Timestamp(TEST_START)).days / 365.25 + 1 / 365.25
    summ = pd.DataFrame({
        "Net margin, test total (EUR/MW)": tot,
        "Capture of perfect foresight (%)": {k: 100 * v / tot["perfect"] for k, v in tot.items()},
        "Cycles per year": {k: r["cycles"].sum() / yrs for k, r in res.items()},
        "Days with a loss (%)": {k: 100 * (r["net"] < -1e-6).mean() for k, r in res.items()},
    })
    summ.to_csv(os.path.join(RES, "t3c_test_summary.csv"), float_format="%.2f")
    # regimes
    reg = {"2019-2020 pre-crisis": ("2019", "2020"), "2021-2022 gas crisis": ("2021", "2022"),
           "2023-2026 solar era": ("2023", "2026")}
    regt = pd.DataFrame({n: {k: 100 * r.loc[a:b, "net"].sum() / res["perfect"].loc[a:b, "net"].sum()
                             for k, r in res.items()} for n, (a, b) in reg.items()})
    regt.to_csv(os.path.join(RES, "t4_capture_by_regime_pct.csv"), float_format="%.1f")
    # DM tests on daily profit: ensemble vs each
    dm = {k: S.dm_test(res["ensemble"]["net"].values, res[k]["net"].values) for k in STRATS if k not in ("ensemble", "perfect")}
    dmt = pd.DataFrame(dm, index=["Mean daily gain (EUR)", "DM statistic", "p-value"]).T
    dmt.to_csv(os.path.join(RES, "t5_dm_tests_ensemble_vs.csv"), float_format="%.4f")
    _log("backtests done")

    # ---------- reinforcement learning (same forecast in the state)
    tr, te = slice(*RL_TRAIN), slice(*RL_TEST)
    Q = RL.train_q(M.loc[tr], fc["ensemble"].loc[tr], BASE, passes=300, gamma=1.0)
    rlres = RL.run_q(Q, M.loc[te], fc["ensemble"].loc[te], BASE)
    rlt = pd.DataFrame({
        "Net margin (EUR/MW)": {"perfect": res["perfect"].loc[te, "net"].sum(), "ensemble": res["ensemble"].loc[te, "net"].sum(),
                                "rl": rlres["net"].sum(), "clock": res["clock"].loc[te, "net"].sum()},
        "Cycles per year": {"perfect": res["perfect"].loc[te, "cycles"].sum(), "ensemble": res["ensemble"].loc[te, "cycles"].sum(),
                            "rl": rlres["cycles"].sum(), "clock": res["clock"].loc[te, "cycles"].sum()}})
    ny = len(rlres) / 365.25
    rlt["Cycles per year"] /= ny
    rlt["Capture (%)"] = 100 * rlt["Net margin (EUR/MW)"] / rlt.loc["perfect", "Net margin (EUR/MW)"]
    rlt.to_csv(os.path.join(RES, "t6_rl_vs_optimisation.csv"), float_format="%.1f")
    _log("RL done")

    # ---------- sensitivity (solar era 2023-2026, perfect vs ensemble)
    sens = []
    for name, kw in [("Base: 2 h, RTE 87%, wear 15", {}), ("Duration 1 h", {"energy_mwh": 1.0}),
                     ("Duration 4 h", {"energy_mwh": 4.0}), ("RTE 80%", {"rte": 0.80}), ("RTE 92%", {"rte": 0.92}),
                     ("Wear 0 EUR/MWh", {"deg_eur_mwh": 0.0}), ("Wear 30 EUR/MWh", {"deg_eur_mwh": 30.0})]:
        b = Battery(**{**BASE.__dict__, **kw})
        r = backtests(M, fc, b, "2023-01-01", END, keys=["perfect", "ensemble"])
        yrs_s = (pd.Timestamp(END) - pd.Timestamp("2023-01-01")).days / 365.25
        sens.append({"Case": name, "Perfect (EUR/MW-yr)": r["perfect"]["net"].sum() / yrs_s,
                     "Ensemble (EUR/MW-yr)": r["ensemble"]["net"].sum() / yrs_s,
                     "Capture (%)": 100 * r["ensemble"]["net"].sum() / r["perfect"]["net"].sum(),
                     "Ensemble cycles/yr": r["ensemble"]["cycles"].sum() / yrs_s,
                     "Ensemble per MWh of capacity (EUR/yr)": r["ensemble"]["net"].sum() / yrs_s / b.energy_mwh})
    sens = pd.DataFrame(sens).set_index("Case"); sens.to_csv(os.path.join(RES, "t7_sensitivity_2023_2026.csv"), float_format="%.1f")
    _log("sensitivity done")

    # ---------- valuation (gross margin: wear is a dispatch shadow price; in the valuation degradation is
    # captured by the capex, the 15-year life and the yearly capacity fade, so it must not be charged twice)
    vrows, revs = [], {}
    for win, (a, b) in {"2023-26": ("2023-01-01", "2026-10-31"), "2025-26": ("2025-01-01", "2026-10-31")}.items():
        monthly = res["ensemble"]["gross"].loc[a:b].resample("ME").sum()
        revs[win] = V.simulate_revenue(monthly, 15, n_sims=n_sims)
        for capex in (120, 160, 200, 280):
            for comp in (0.0, 0.03, 0.06):
                p = V.Project(capex_eur_kwh=capex, compression=comp)
                v = V.value(revs[win], p)
                v.update({"Revenue window": win, "Capex (EUR/kWh)": capex, "Margin compression (%/yr)": comp * 100,
                          "Year-1 revenue P50 (EUR/MW)": float(np.median(revs[win][:, 0]))})
                vrows.append(v)
    vt = pd.DataFrame(vrows); vt.to_csv(os.path.join(RES, "t8_valuation.csv"), float_format="%.4f", index=False)
    rev = revs["2023-26"]
    base_p = V.Project()
    npv_base = V.npv(V.cash_flows(rev, base_p), base_p)
    _log("valuation done")

    # ---------- interpretation: SHAP on the final LightGBM model
    shap_imp = None
    try:
        import shap
        m, L, feats = F.last_lgbm_model(M, "2026-01-01")
        Xs = L.loc["2026-01-01":END, feats].dropna().sample(3000, random_state=0)
        sv = shap.TreeExplainer(m).shap_values(Xs)
        shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
        shap_imp.to_csv(os.path.join(RES, "t9_shap_importance_lgbm.csv"), float_format="%.3f")
    except Exception as e:
        _log(f"SHAP skipped: {e}")

    # ---------- figures
    _figures(M, fc, res, ann, cap, rlres, rev, npv_base, vt, shap_imp)

    # ---------- key numbers
    e, p = res["ensemble"], res["perfect"]
    key.update({
        "test_period": f"{TEST_START} to {END}", "days_test": int(len(e)),
        "ann_perfect": ann["perfect"].round(0).to_dict(), "ann_ensemble": ann["ensemble"].round(0).to_dict(),
        "capture_total": summ["Capture of perfect foresight (%)"].round(1).to_dict(),
        "forecast_mae": ft["MAE"].round(2).to_dict(), "rank_corr": ft["Rank corr."].round(3).to_dict(),
        "regime_capture": regt.round(1).to_dict(), "dm": dmt.round(4).to_dict(),
        "rl": rlt.round(1).to_dict(), "sens": sens.round(1).to_dict(),
        "valuation_base": V.value(rev, base_p), "valuation_2025_26": V.value(revs["2025-26"], base_p),
        "npv_base_mean": float(npv_base.mean()),
        "revenue_year1_p50": float(np.median(rev[:, 0])), "revenue_year1_p10": float(np.percentile(rev[:, 0], 10)),
        "revenue_year1_p90": float(np.percentile(rev[:, 0], 90)),
        "runtime_s": round(time.time() - t0),
    })
    with open(os.path.join(RES, "key_numbers.json"), "w") as f:
        json.dump(key, f, indent=1, default=float)
    _demo_data(M, fc, res)
    _log(f"all done in {time.time() - t0:.0f} s")
    return key


def _figures(M, fc, res, ann, cap, rlres, rev, npv_base, vt, shap_imp):
    # F1 average daily price shape by year
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    years = [2016, 2019, 2022, 2024, 2026]
    shades = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
    for y, c in zip(years, shades):
        prof = M[M.index.year == y].mean()
        ax.plot(range(24), prof.values, color=c, lw=2, label=str(y))
    ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("hour of the day"); ax.set_ylabel("EUR/MWh")
    ax.set_title("Average day-ahead price by hour: the solar 'duck' deepens")
    ax.legend(ncol=5, loc="upper left", fontsize=8.5)
    fig.savefig(os.path.join(FIG, "f1_price_shape.png")); plt.close(fig)

    # F2 annual net margin by strategy
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ks = ["clock", "naive_d1", "lear", "lgbm", "ensemble"]
    yrs = ann.index.values; w = 0.16
    for i, k in enumerate(ks):
        ax.bar(yrs + (i - 2) * w, ann[k].values / 1000, width=w - 0.02, color=C[k], label=LABEL[k])
    ax.plot(yrs, ann["perfect"].values / 1000, color=INK, marker="o", ms=4, lw=1.5, label=LABEL["perfect"])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("thousand EUR per MW-year"); ax.set_title("Net arbitrage margin of a 1 MW / 2 MWh battery (2026 to 25 Nov)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.savefig(os.path.join(FIG, "f2_annual_margin.png")); plt.close(fig)

    # F3 capture rate by year
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for k in ["clock", "naive_d1", "ensemble"]:
        ax.plot(cap.index, cap[k], color=C[k], marker="o", ms=4, label=LABEL[k])
    ax.axhline(100, color=INK2, lw=1, ls="--"); ax.text(cap.index.min(), 101.5, "perfect foresight = 100%", color=INK2, fontsize=8)
    ax.set_ylim(min(-60, cap[["clock", "naive_d1", "ensemble"]].min().min() - 5), 110)
    ax.set_ylabel("% of perfect-foresight margin"); ax.set_title("Capture rate: forecasting matters most when spreads are thin")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(os.path.join(FIG, "f3_capture.png")); plt.close(fig)

    # F4 dispatch heat map (ensemble): average net discharge by hour and year
    from .battery import DailyLP
    lp = DailyLP(BASE)
    rows = []
    for d in M.loc[TEST_START:END].index[::3]:
        c, dch = lp.solve(fc["ensemble"].loc[d].values.astype(float))
        rows.append(pd.Series(dch - c, index=range(24), name=d))
    D = pd.DataFrame(rows); D = D.groupby(D.index.year).mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    im = ax.imshow(D.values, aspect="auto", cmap="RdBu", vmin=-0.6, vmax=0.6)
    ax.set_yticks(range(len(D))); ax.set_yticklabels(D.index); ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("hour of the day")
    ax.grid(False); ax.set_title("Where the battery buys (red) and sells (blue): average MW by hour")
    cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label("net discharge, MW", color=INK2)
    fig.savefig(os.path.join(FIG, "f4_dispatch_heatmap.png")); plt.close(fig)

    # F5 example week: forecast vs actual and state of charge
    wk = M.loc["2026-04-13":"2026-04-19"]
    fwk = fc["ensemble"].loc[wk.index]
    lp = DailyLP(BASE)
    soc, flow = [], []
    for d in wk.index:
        c, dch = lp.solve(fwk.loc[d].values.astype(float))
        s = np.concatenate([[0], np.cumsum(BASE.eta * c - dch / BASE.eta)])[:-1]
        soc += list(s); flow += list(dch - c)
    x = np.arange(len(soc))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    a1.plot(x, wk.values.ravel(), color=INK2, lw=1.6, label="realised price")
    a1.plot(x, fwk.values.ravel(), color=C["ensemble"], lw=1.6, ls="--", label="forecast at D-1 noon")
    a1.set_ylabel("EUR/MWh"); a1.legend(fontsize=8, loc="upper left", ncol=2)
    a1.set_title("One week of April 2026: forecast, prices and the battery's state of charge")
    a2.fill_between(x, soc, step="post", color=C["ensemble"], alpha=0.35, lw=0)
    a2.step(x, soc, where="post", color=C["ensemble"], lw=1.2)
    a2.set_ylabel("MWh stored"); a2.set_xticks(range(0, len(x), 24)); a2.set_xticklabels([d.strftime("%a %d") for d in wk.index])
    fig.savefig(os.path.join(FIG, "f5_example_week.png")); plt.close(fig)

    # F6 valuation: NPV distribution (base) and NPV vs capex
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a1.hist(npv_base / 1000, bins=50, color=C["ensemble"], edgecolor="#fcfcfb", linewidth=0.5)
    a1.axvline(0, color=INK, lw=1); a1.set_xlabel("NPV, thousand EUR per MW"); a1.set_ylabel("scenarios")
    a1.set_title("Base case NPV (capex 200 EUR/kWh)")
    caps = np.arange(80, 321, 20)
    for comp, ls in ((0.0, "-"), (0.03, "--"), (0.06, ":")):
        med = [np.median(V.npv(V.cash_flows(rev, q := V.Project(capex_eur_kwh=c_, compression=comp)), q)) / 1000 for c_ in caps]
        a2.plot(caps, med, color=C["ensemble"], ls=ls, lw=1.8, label=f"compression {int(comp*100)}%/yr")
    a2.axhline(0, color=INK, lw=1); a2.set_xlabel("capex, EUR per kWh"); a2.set_ylabel("median NPV, k EUR/MW")
    a2.set_title("Value falls with capex and cannibalisation"); a2.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f6_valuation.png")); plt.close(fig)

    # F7 SHAP importance
    if shap_imp is not None:
        names = {"p_d1": "price, same hour, D-1", "d1_evening": "evening premium, D-1", "p_w": "7-day mean, same hour",
                 "p_d7": "price, same hour, D-7", "d1_mean": "daily mean, D-1", "d1_max": "daily max, D-1",
                 "d1_solar_dip": "solar dip, D-1", "dow": "day of week", "d1_min": "daily min, D-1",
                 "d1_std": "daily st. dev., D-1", "w_mean": "7-day mean, all hours", "hour": "hour of day",
                 "month": "month", "p_d2": "price, same hour, D-2", "p_d1_rank": "rank of hour, D-1",
                 "hol": "holiday", "hol_d1": "holiday D-1"}
        top = shap_imp.head(10)[::-1].rename(index=names)
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        ax.barh(top.index, top.values, color=C["lgbm"], height=0.6)
        ax.set_xlabel("mean |SHAP value|, EUR/MWh"); ax.set_title("What the LightGBM forecast relies on (2026 test days)")
        fig.savefig(os.path.join(FIG, "f7_shap.png")); plt.close(fig)

    # F8 error analysis (solar era): forecasts shrink spreads, and a few days carry most of the lost value
    e, p = res["ensemble"].loc["2023":], res["perfect"].loc["2023":]
    A = M.loc[e.index]; Fc = fc["ensemble"].loc[e.index]
    sp_err = (Fc.max(axis=1) - Fc.min(axis=1)) - (A.max(axis=1) - A.min(axis=1))
    lost = (p["net"] - e["net"]).sort_values(ascending=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a1.hist(sp_err.clip(-150, 150), bins=60, color=C["ensemble"], edgecolor="#fcfcfb", linewidth=0.4)
    a1.axvline(0, color=INK, lw=1); a1.axvline(sp_err.mean(), color=C["lear"], lw=1.5, ls="--")
    a1.set_xlabel("forecast minus realised daily spread, EUR/MWh"); a1.set_ylabel("days")
    a1.set_title("Forecasts shrink the spread")
    a1.text(0.02, 0.95, f"mean {eu(sp_err.mean(), 1)}\nunder-forecast on {eu(100 * (sp_err < 0).mean(), 0)}% of days",
            transform=a1.transAxes, va="top", fontsize=8, color=INK2)
    share_days = np.arange(1, len(lost) + 1) / len(lost) * 100
    share_lost = lost.cumsum().values / lost.sum() * 100
    a2.plot(share_days, share_lost, color=C["ensemble"])
    a2.plot([0, 100], [0, 100], color=MUTED, lw=1, ls=":")
    a2.set_xlabel("% of days, worst first"); a2.set_ylabel("% of value lost vs perfect")
    a2.set_title("A few days carry the loss")
    i10 = int(len(lost) * 0.1)
    a2.annotate(f"worst 10% of days:\n{eu(share_lost[i10 - 1], 0)}% of the loss", (10, share_lost[i10 - 1]),
                (30, 25), fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f8_error_analysis.png")); plt.close(fig)
    pd.Series({"mean_spread_error": sp_err.mean(), "share_days_underforecast": (sp_err < 0).mean(),
               "worst10pct_share_of_loss": share_lost[i10 - 1] / 100}).to_csv(os.path.join(RES, "t10_error_analysis.csv"))


def _demo_data(M, fc, res):
    """Compact JSON for the interactive demo: daily prices + forecast (2023-2026) and annual results."""
    sub = M.loc["2023-01-01":END]
    out = {"dates": [d.strftime("%Y-%m-%d") for d in sub.index],
           "price": np.round(sub.values, 1).tolist(),
           "fcst": np.round(fc["ensemble"].loc[sub.index].values, 1).tolist()}
    with open(os.path.join(RES, "demo_prices.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
