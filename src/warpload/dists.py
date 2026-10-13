"""Target distributions: scipy.stats laws plus a smooth density fitted to Estonian day-ahead prices.

Every target exposes cdf, ppf and pdf on scalars or arrays. The loaders only ever use the cdf
(exact bin masses) and the ppf (grid ranges and quantile checks).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats

ROOT = Path(__file__).resolve().parents[2]
PRICES = ROOT / "data" / "ee_prices_hourly.csv"


@dataclass
class Target:
    name: str
    label: str
    law: object          # anything with cdf/ppf/pdf
    heavy: bool          # power-law (or empirical heavy) tail?

    def cdf(self, x):
        return self.law.cdf(x)

    def sf(self, x):
        return self.law.sf(x)

    def ppf(self, p):
        return self.law.ppf(p)

    def pdf(self, x):
        return self.law.pdf(x)


class PriceKDE:
    """Adaptive-bandwidth Gaussian KDE (Abramson square-root law) of hourly EE prices.

    It is built on 1024 empirical quantile atoms so the cdf is cheap. It is fitted in raw
    EUR/MWh, NOT in a warped space, so the asinh grid gets no built-in advantage.
    """

    def __init__(self, prices: np.ndarray, n_atoms: int = 1024):
        x = np.sort(np.asarray(prices, float))
        levels = (np.arange(n_atoms) + 0.5) / n_atoms
        self.atoms = np.quantile(x, levels)
        iqr = np.subtract(*np.quantile(x, [0.75, 0.25]))
        sigma = min(x.std(), iqr / 1.349)
        h0 = 0.9 * sigma * len(self.atoms) ** (-0.2)                    # Silverman
        pilot = stats.norm.pdf((self.atoms[:, None] - self.atoms[None, :]) / h0).mean(1) / h0
        g = np.exp(np.log(pilot).mean())
        self.h = h0 * np.sqrt(g / pilot)                                 # Abramson
        self._lo = self.atoms[0] - 12 * self.h.max()
        self._hi = self.atoms[-1] + 12 * self.h.max()

    def _apply(self, x, fn):
        x = np.asarray(x, float)
        flat = x.ravel()
        out = np.empty_like(flat)
        step = max(1, 2_000_000 // len(self.atoms))
        for i in range(0, len(flat), step):
            z = (flat[i:i + step, None] - self.atoms[None, :]) / self.h[None, :]
            out[i:i + step] = fn(z)
        return out.reshape(x.shape)

    def cdf(self, x):
        return self._apply(x, lambda z: special.ndtr(z).mean(1))

    def sf(self, x):
        return self._apply(x, lambda z: special.ndtr(-z).mean(1))

    def pdf(self, x):
        return self._apply(x, lambda z: (stats.norm.pdf(z) / self.h[None, :]).mean(1))

    def ppf(self, p):
        p = np.atleast_1d(np.asarray(p, float))
        out = np.array([optimize.brentq(lambda t, q=q: float(self.cdf(t)) - q, self._lo, self._hi,
                                        xtol=1e-12, rtol=1e-14) for q in p])
        return out if out.size > 1 else float(out[0])


def ee_prices() -> np.ndarray:
    import pandas as pd
    return pd.read_csv(PRICES, index_col=0)["ee"].to_numpy()


def get(name: str) -> Target:
    """All targets used anywhere in the study, by name."""
    if name == "price_ee":
        return Target(name, "EE day-ahead price (KDE, 2023–2026)", PriceKDE(ee_prices()), True)
    table = {
        # development set (used to freeze the procedure)
        "t3":        ("Student-t, ν=3",          stats.t(df=3),              True),
        "lognorm1":  ("log-normal, σ=1",         stats.lognorm(s=1.0),       True),
        "normal_dev": ("normal (dev control)",   stats.norm(loc=0.5, scale=0.125), False),
        # test set (run once, after PREREGISTRATION.md is frozen)
        "levy":      ("Lévy, c=1",               stats.levy(scale=1.0),      True),
        "cauchy":    ("Cauchy",                  stats.cauchy(),             True),
        "t1_5":      ("Student-t, ν=1.5",        stats.t(df=1.5),            True),
        "lomax":     ("Lomax (Pareto II), α=1.5", stats.lomax(c=1.5),        True),
        "lognorm2":  ("log-normal, σ=2",         stats.lognorm(s=2.0),       True),
        "gamma05":   ("Gamma, a=0.5",            stats.gamma(a=0.5),         False),
        "normal":    ("normal (control)",        stats.norm(),               False),
    }
    label, law, heavy = table[name]
    return Target(name, label, law, heavy)


DEV = ["t3", "lognorm1", "normal_dev"]
TEST = ["levy", "cauchy", "t1_5", "lomax", "lognorm2", "price_ee", "gamma05", "normal"]
