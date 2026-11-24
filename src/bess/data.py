"""Data loading: hourly Spanish day-ahead prices (Red Electrica, REData API, series 'Precio mercado spot').

The repository ships a cached CSV (data/ree_spot_prices_2015_2026.csv). If it is missing, `download_ree`
rebuilds it from the public REE API (no key needed). Quarter-hour prices (from late 2025) are averaged to hours.
"""
from __future__ import annotations
import os, json, time, urllib.request
import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ree_spot_prices_2015_2026.csv")
API = ("https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"
       "?start_date={s}T00:00&end_date={e}T23:59&time_trunc=hour")


def download_ree(start="2015-01-01", end="2026-11-25", path=CACHE):
    """Download hourly spot prices month by month from the public REE API and save them as CSV."""
    rows = {}
    for m in pd.period_range(start[:7], end[:7], freq="M"):
        s, e = m.start_time.strftime("%Y-%m-%d"), min(m.end_time, pd.Timestamp(end)).strftime("%Y-%m-%d")
        with urllib.request.urlopen(API.format(s=s, e=e), timeout=60) as r:
            j = json.load(r)
        ser = next((x for x in j["included"] if x["attributes"]["title"] == "Precio mercado spot"), j["included"][0])
        for v in ser["attributes"]["values"]:
            rows[v["datetime"]] = v["value"]
        time.sleep(0.2)
    s = pd.Series(rows)
    t = pd.to_datetime(s.index, utc=True)
    df = pd.DataFrame({"utc": t.floor("h"), "local": [k[:13] + ":00" + k[23:] for k in s.index], "p": s.values})
    out = df.groupby("utc").agg(local_datetime=("local", "first"), price_eur_mwh=("p", "mean"), n_obs=("p", "size")).reset_index()
    out["utc_hour"] = out["utc"].dt.strftime("%Y-%m-%dT%H:%MZ")
    out[["utc_hour", "local_datetime", "price_eur_mwh", "n_obs"]].to_csv(path, index=False, float_format="%.3f")
    return path


def load_hourly(path=CACHE) -> pd.DataFrame:
    if not os.path.exists(path):
        download_ree(path=path)
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["local_datetime"].str[:10])
    d["hour"] = d["local_datetime"].str[11:13].astype(int)
    return d[["date", "hour", "price_eur_mwh"]].rename(columns={"price_eur_mwh": "price"})


def daily_matrix(hourly: pd.DataFrame) -> pd.DataFrame:
    """One row per delivery day, 24 columns (local hours 0-23).

    Daylight-saving days: the missing spring hour is filled by linear interpolation; the repeated autumn
    hour is averaged (standard practice in electricity-price forecasting, Lago et al. 2021)."""
    m = hourly.groupby(["date", "hour"])["price"].mean().unstack("hour")
    m = m.reindex(columns=range(24)).interpolate(axis=1, limit_direction="both")
    m.columns = [f"h{h:02d}" for h in range(24)]
    return m.sort_index()
