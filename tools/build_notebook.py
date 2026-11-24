"""Builds notebooks/capstone_bess_spain.ipynb from the package sources, so the notebook and the repo never drift."""
import os, lzma, base64, uuid
import nbformat as nbf
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# What is a grid battery worth in Spain?
### Forecast-driven arbitrage in the day-ahead market, a reinforcement-learning comparison and an investment valuation

**AI for Finance — Final Capstone (Type A: free application)** · IE Business School, Master in Finance · Wassim Ibrahim

This notebook reproduces **every table and figure** of the manuscript from public data. Run it top to bottom
(*Runtime → Run all*). It needs no file upload and no API key: the hourly prices are stored in the notebook
(and can be re-downloaded from Red Eléctrica's public API with one function call).

| Step | What it does | Runtime |
|---|---|---|
| 0 | install and import | ~30 s |
| 1 | write the `bess` package (data, battery LP, forecasts, RL, valuation) | 1 s |
| 2 | load 2015–2026 hourly Spanish day-ahead prices | 5 s |
| 3 | the decision problem: one day, one battery, one linear programme | 1 s |
| 4 | walk-forward price forecasts (naive, LEAR, LightGBM, ensemble) — no look-ahead | ~4 min |
| 5 | backtests, reinforcement learning, robustness, valuation; all tables and figures | ~3 min |
| 6 | results, one section per manuscript table and figure | — |
""")
md("## 0. Setup")
code("""
# Colab ships pandas, numpy, scipy, scikit-learn, matplotlib and lightgbm. These three may be missing elsewhere.
!pip -q install lightgbm holidays shap
import os, sys, json, lzma, base64, io
os.makedirs("src/bess", exist_ok=True); os.makedirs("data", exist_ok=True); os.makedirs("results/figures", exist_ok=True)
sys.path.insert(0, "src")
""")
md("""
## 1. The code
The project is a small Python package, `bess`, identical to the `src/bess` folder of the GitHub repository.
Each cell below writes one module. Read them in this order: `battery` (the physics and the optimiser),
`forecast` (the price models), `backtest`, `rl`, `valuation`, `pipeline` (which runs everything).
""")
for mod in ["__init__", "data", "battery", "forecast", "backtest", "rl", "stats", "valuation", "plots", "pipeline"]:
    src = open(os.path.join(ROOT, "src", "bess", f"{mod}.py")).read()
    code(f"%%writefile src/bess/{mod}.py\n{src}")
md("""
## 2. Data
Hourly prices of the Spanish day-ahead market (Red Eléctrica de España, REData API, series *Precio mercado spot*),
1 January 2015 to 25 November 2026, downloaded on 24 November 2026. From October 2025 the market clears in
quarter-hours; those are averaged to hours. A compressed copy is stored in the next cell so the notebook runs
offline; `bess.data.download_ree()` rebuilds the file from the API.
""")
d = pd.read_csv(os.path.join(ROOT, "data", "ree_spot_prices_2015_2026.csv"))
blob = base64.b64encode(lzma.compress(d.to_csv(index=False).encode(), preset=9)).decode()
chunks = [blob[i:i + 20000] for i in range(0, len(blob), 20000)]
code("#@title Data copy (run it; nothing to read)\nPRICES_XZ = (\n" + "\n".join(f'    "{c}"' for c in chunks) + "\n)\n"
     "if not os.path.exists('data/ree_spot_prices_2015_2026.csv'):\n"
     "    open('data/ree_spot_prices_2015_2026.csv', 'wb').write(lzma.decompress(base64.b64decode(PRICES_XZ)))\n"
     "print('data file ready')")
code("""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
from bess.data import load_hourly, daily_matrix
from bess.plots import style; style()
M = daily_matrix(load_hourly())
print(f"{len(M):,} delivery days, {M.index.min().date()} to {M.index.max().date()}, {M.size:,} hourly prices")
M.tail(3).round(1)
""")
md("""
## 3. The decision problem on one day
At noon of day D−1 the battery submits its 24 hourly buy/sell quantities for day D. It knows the prices of D−1
(set the day before) but not those of D. Given *some* price vector, the best schedule solves a linear programme:
maximise revenue minus fees minus wear, subject to the power limit (1 MW), the energy limit (2 MWh), round-trip
efficiency (87%) and an empty battery at midnight. Below: the schedule computed with the realised prices of one
day (perfect foresight) and with the forecast available at D−1 noon.
""")
code("""
from bess.battery import Battery, DailyLP, settle
bat = Battery(deg_eur_mwh=15.0); lp = DailyLP(bat)
day = pd.Timestamp("2026-04-15")
c, d = lp.solve(M.loc[day].values)
pd.DataFrame({"price": M.loc[day].values.round(1), "charge MW": c.round(2), "discharge MW": d.round(2)},
             index=range(24)).T
""")
code("""
print("Perfect foresight on", day.date(), settle(c, d, M.loc[day].values, bat))
""")
md("""
## 4. Walk-forward forecasts (no look-ahead)
Four forecasts of the 24 prices of day D, each made with information available at noon of D−1:
yesterday's prices, last week's prices, **LEAR** (24 LASSO regressions on asinh-transformed lags — the standard
benchmark of the electricity-price-forecasting literature) and **LightGBM** (gradient-boosted trees on lags,
daily statistics and calendar). Every model is refitted at the start of each month on data before that month
only. Hyperparameters were chosen on 2017–2018 (validation); 2019–2026 is the untouched test period.
Set `USE_CACHE = True` to reuse forecasts already computed in this runtime.
""")
code("""
from bess.pipeline import build_forecasts, forecast_table, VALID
USE_CACHE = False
fc = build_forecasts(M, use_cache=USE_CACHE)
print("Validation 2017-2018 (used to choose hyperparameters):")
display(forecast_table(M, fc, *VALID).round(3))
print("Test 2019 - 25 Nov 2026:")
forecast_table(M, fc).round(3)
""")
md("## 5. Backtests, reinforcement learning, robustness and valuation\nThis cell writes every table to `results/` and every figure to `results/figures/`.")
code("""
from bess.pipeline import run_all
key = run_all(use_cache=True)
""")
md("## 6. Results\n### 6.1 The market: the solar 'duck' and the spread it creates")
code("""display(pd.read_csv("results/t1_market_facts.csv", index_col=0).round(1)); display(Image("results/figures/f1_price_shape.png", width=720))""")
md("### 6.2 Economic value of each strategy (EUR per MW per year, net of fees and wear)")
code("""
display(pd.read_csv("results/t3_annual_net_margin_eur_per_mw.csv", index_col=0))
display(pd.read_csv("results/t3c_test_summary.csv", index_col=0).round(1))
display(Image("results/figures/f2_annual_margin.png", width=720)); display(Image("results/figures/f3_capture.png", width=720))
""")
md("### 6.3 Is the gain significant? Diebold–Mariano tests on daily profits (ensemble minus alternative, Newey–West variance)")
code("""pd.read_csv("results/t5_dm_tests_ensemble_vs.csv", index_col=0).round(4)""")
md("### 6.4 Robustness across market regimes (capture of perfect foresight, %)")
code("""pd.read_csv("results/t4_capture_by_regime_pct.csv", index_col=0).round(1)""")
md("### 6.5 Reinforcement learning versus optimisation (train 2021–2023, test 2024–Nov 2026, same forecast in both)")
code("""pd.read_csv("results/t6_rl_vs_optimisation.csv", index_col=0).round(1)""")
md("### 6.6 Sensitivity to the battery's design (2023–2026)")
code("""pd.read_csv("results/t7_sensitivity_2023_2026.csv", index_col=0).round(1)""")
md("### 6.7 Interpretation: what the forecast uses and how the battery trades")
code("""
display(Image("results/figures/f7_shap.png", width=720)); display(Image("results/figures/f4_dispatch_heatmap.png", width=720))
display(Image("results/figures/f5_example_week.png", width=720))
""")
md("### 6.8 Error analysis: where the value is lost")
code("""display(pd.read_csv("results/t10_error_analysis.csv", index_col=0)); display(Image("results/figures/f8_error_analysis.png", width=720))""")
md("### 6.9 Valuation of a 1 MW / 2 MWh merchant battery (15 years, 10% WACC, 2% fade, opex 2% of capex)")
code("""
v = pd.read_csv("results/t8_valuation.csv")
cols = ["Revenue window", "Capex (EUR/kWh)", "Margin compression (%/yr)", "Year-1 revenue P50 (EUR/MW)",
        "npv_p10", "npv_p50", "npv_p90", "prob_npv_pos", "irr_p50", "breakeven_capex_eur_kwh"]
display(v[cols].round(3)); display(Image("results/figures/f6_valuation.png", width=720))
""")
md("""
### 6.10 Reproducibility map
| Manuscript | File |
|---|---|
| Table 1 market facts | `results/t1_market_facts.csv` |
| Table 2 forecast accuracy | `results/t2_forecast_accuracy_test.csv`, `t2b_…validation.csv` |
| Table 3 economic results | `results/t3_…`, `t3b_…`, `t3c_test_summary.csv`, `t5_dm_tests_ensemble_vs.csv` |
| Table 4 regimes, RL, sensitivity | `results/t4_…`, `t6_…`, `t7_…` |
| Table 5 valuation | `results/t8_valuation.csv` |
| Figures 1–8 | `results/figures/f1…f8` |
| Every number quoted in the text | `results/key_numbers.json` |

**Use of generative AI.** Claude (Anthropic) was used as a coding and writing assistant; all modelling choices,
results and interpretations were checked by the author, who is responsible for them.
""")
nb["cells"] = cells
for c in nb["cells"]:
    c["id"] = uuid.uuid4().hex[:8]
nb["metadata"] = {"colab": {"provenance": [], "toc_visible": True}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
nbf.write(nb, os.path.join(ROOT, "notebooks", "capstone_bess_spain.ipynb"))
print("written")
