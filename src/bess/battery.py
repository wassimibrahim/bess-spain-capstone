"""Battery model and optimal daily dispatch (linear programme, solved with HiGHS via scipy).

Decision variables per hour h: grid energy charged c_h >= 0 and discharged d_h >= 0 (MWh).
State of charge: s_{h+1} = s_h + eta_c * c_h - d_h / eta_d, 0 <= s_h <= E, s_0 = s_24 = s_init.
Objective: maximise sum_h p_h (d_h - c_h) - fee * (c_h + d_h) - deg * d_h.
`deg` is the degradation (wear) cost per MWh discharged: replacement cost per MWh of capacity divided by
the cycle life. It stops the optimiser from cycling for spreads that do not pay for the wear they cause.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


@dataclass
class Battery:
    power_mw: float = 1.0          # charge/discharge limit, MW
    energy_mwh: float = 2.0        # usable energy, MWh (2-hour battery)
    rte: float = 0.87              # round-trip efficiency (AC-AC)
    deg_eur_mwh: float = 20.0      # wear cost per MWh discharged
    fee_eur_mwh: float = 0.5       # market/grid fee per MWh traded, each direction
    soc_init_frac: float = 0.0     # state of charge at midnight (start = end)

    @property
    def eta(self):
        return np.sqrt(self.rte)


class DailyLP:
    """Pre-built LP for a 24-hour horizon; only the price vector changes between days."""

    def __init__(self, bat: Battery, T: int = 24):
        self.b, self.T = bat, T
        n = 3 * T + 1                    # c (T), d (T), s (T+1)
        self.n = n
        A = lil_matrix((T + 2, n))
        b = np.zeros(T + 2)
        e = bat.eta
        for h in range(T):               # s_{h+1} - s_h - e*c_h + d_h/e = 0
            A[h, 2 * T + h + 1] = 1
            A[h, 2 * T + h] = -1
            A[h, h] = -e
            A[h, T + h] = 1 / e
        A[T, 2 * T] = 1; b[T] = bat.soc_init_frac * bat.energy_mwh
        A[T + 1, 3 * T] = 1; b[T + 1] = bat.soc_init_frac * bat.energy_mwh
        self.A_eq, self.b_eq = A.tocsr(), b
        self.bounds = [(0, bat.power_mw)] * (2 * T) + [(0, bat.energy_mwh)] * (T + 1)

    def solve(self, price: np.ndarray):
        T, b = self.T, self.b
        cost = np.zeros(self.n)
        cost[:T] = price + b.fee_eur_mwh                    # charging: pay price + fee
        cost[T:2 * T] = -price + b.fee_eur_mwh + b.deg_eur_mwh  # discharging: earn price - fee - wear
        r = linprog(cost, A_eq=self.A_eq, b_eq=self.b_eq, bounds=self.bounds, method="highs")
        x = r.x
        return x[:T], x[T:2 * T]


def settle(c, d, price, bat: Battery):
    """Cash flows of a schedule (c, d) settled at realised prices."""
    revenue = float(np.dot(price, d - c))
    fees = bat.fee_eur_mwh * float(c.sum() + d.sum())
    wear = bat.deg_eur_mwh * float(d.sum())
    return {"gross": revenue - fees, "wear": wear, "net": revenue - fees - wear,
            "mwh_out": float(d.sum()), "cycles": float(d.sum()) / (bat.energy_mwh * bat.eta)}
