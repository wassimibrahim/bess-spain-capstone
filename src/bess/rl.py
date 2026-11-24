"""Tabular Q-learning dispatch agent (the reinforcement-learning alternative to predict-then-optimise).

Each hour of the delivery day the agent observes a state and picks charge / idle / discharge at full power.
State (1.800 states): hour of day (24) x state-of-charge bucket (5) x quintile of the hour's *forecast* price
within the forecast day (5) x where that forecast stands among the hours still to come (3). Reward: realised cash flow of the hour (price x energy, minus fee and wear).
It learns on the training days with epsilon-greedy exploration, then trades the test days greedily.
The forecast used in the state is the same walk-forward forecast given to the optimiser, so the comparison
isolates the decision method: learned policy vs. explicit optimisation of the battery model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .battery import Battery

CH, IDLE, DIS = 0, 1, 2


def _quintile(f: np.ndarray) -> np.ndarray:
    r = f.argsort(axis=1).argsort(axis=1)            # rank 0..23 within the day
    return np.minimum(4, (r * 5) // 24)


def _ahead(f: np.ndarray) -> np.ndarray:
    """Where this hour's forecast stands among the hours still to come today (incl. itself):
    0 = among the cheapest third, 1 = middle, 2 = among the dearest third."""
    out = np.zeros(f.shape, dtype=int)
    for h in range(24):
        rest = f[:, h:]
        pct = (rest < f[:, [h]]).sum(axis=1) / max(1, rest.shape[1] - 1) if rest.shape[1] > 1 else np.full(len(f), 0.5)
        out[:, h] = np.minimum(2, (pct * 3).astype(int))
    return out


def _state_feats(fcst: np.ndarray):
    return _quintile(fcst), _ahead(fcst)


def _step(soc, a, price, bat: Battery):
    e = bat.eta
    if a == CH:
        q = min(bat.power_mw, (bat.energy_mwh - soc) / e)        # grid MWh bought
        return soc + e * q, -price * q - bat.fee_eur_mwh * q
    if a == DIS:
        q = min(bat.power_mw, soc * e)                            # grid MWh sold
        return soc - q / e, price * q - (bat.fee_eur_mwh + bat.deg_eur_mwh) * q
    return soc, 0.0


def _soc_bin(soc, bat):
    return min(4, int(soc / bat.energy_mwh * 5 + 1e-9))


def train_q(actual: pd.DataFrame, fcst: pd.DataFrame, bat: Battery, passes=150, gamma=0.995,
            eps_end=0.02, seed=0):
    rng = np.random.default_rng(seed)
    A = actual.values
    F, G = _state_feats(fcst.values)
    Q = np.zeros((24, 5, 5, 3, 3)); N = np.zeros_like(Q)
    nd = len(A)
    for p in range(passes):
        eps = max(eps_end, 1.0 - p / (0.7 * passes))
        for d in rng.permutation(nd):
            soc = 0.0
            scale = 1.0 / (np.abs(A[d]).mean() + 10.0)   # rewards in units of the day's price level: stabler learning
            for h in range(24):
                s = (h, _soc_bin(soc, bat), F[d, h], G[d, h])
                a = rng.integers(3) if rng.random() < eps else int(Q[s].argmax())
                soc2, r = _step(soc, a, A[d, h], bat)
                r *= scale
                if h < 23:
                    tgt = r + gamma * Q[(h + 1, _soc_bin(soc2, bat), F[d, h + 1], G[d, h + 1])].max()
                else:
                    tgt = r
                N[s + (a,)] += 1
                Q[s + (a,)] += (1.0 / N[s + (a,)] ** 0.6) * (tgt - Q[s + (a,)])
                soc = soc2
    return Q


def run_q(Q, actual: pd.DataFrame, fcst: pd.DataFrame, bat: Battery) -> pd.DataFrame:
    A = actual.values
    F, G = _state_feats(fcst.values)
    rows = []
    for d in range(len(A)):
        soc, cash, out = 0.0, 0.0, 0.0
        for h in range(24):
            a = int(Q[(h, _soc_bin(soc, bat), F[d, h], G[d, h])].argmax())
            soc2, r = _step(soc, a, A[d, h], bat)
            if a == DIS:
                out += (soc - soc2) * bat.eta
            soc, cash = soc2, cash + r
        rows.append({"date": actual.index[d], "net": cash, "mwh_out": out,
                     "cycles": out / (bat.energy_mwh * bat.eta)})
    return pd.DataFrame(rows).set_index("date")
