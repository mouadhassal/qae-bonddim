"""Grids: how the 2^N computational basis states map to bins on the real line.

A grid is a monotone map x = phi(u) with u on a uniform N-bit lattice over [u_lo, u_hi].
- UniformGrid: phi(u) = u. This is what Haiqu's distribution_loading() does today.
- AsinhGrid:   phi(u) = m + s*sinh(u). Linear near the centre m, logarithmic in both tails.

Bin i covers [phi(u_i), phi(u_{i+1})] and gets probability mass exactly P(X in bin), so the
encoded amplitude is sqrt(mass). In u-coordinates that is sqrt(p_U(u) du): the loader sees the
density of U = phi^{-1}(X), which for power-law tails decays exponentially instead of polynomially.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    kind: str
    u_lo: float
    u_hi: float
    m: float = 0.0
    s: float = 1.0

    law: object = None       # quantile grid only: the target law (phi = its ppf)

    def phi(self, u):
        u = np.asarray(u, float)
        if self.kind == "uniform":
            return u
        if self.kind == "quantile":
            return self.law.ppf(u)
        return self.m + self.s * np.sinh(u)

    def phi_inv(self, x):
        x = np.asarray(x, float)
        if self.kind == "uniform":
            return x
        if self.kind == "quantile":
            return self.law.cdf(x)
        return np.arcsinh((x - self.m) / self.s)

    @property
    def x_lo(self):
        return float(self.phi(self.u_lo))

    @property
    def x_hi(self):
        return float(self.phi(self.u_hi))

    def u_edges(self, n: int) -> np.ndarray:
        return np.linspace(self.u_lo, self.u_hi, 2 ** n + 1)

    def edges(self, n: int) -> np.ndarray:
        return self.phi(self.u_edges(n))

    def describe(self) -> dict:
        d = {"kind": self.kind, "x_lo": self.x_lo, "x_hi": self.x_hi}
        if self.kind == "asinh":
            d.update(m=self.m, s=self.s)
        return d


def tail_range(target, delta: float) -> tuple[float, float]:
    """Interval that leaves out probability delta, split equally between the two tails."""
    lo = float(target.ppf(delta / 2))
    hi = float(target.ppf(1 - delta / 2))
    return lo, hi


def uniform(target, delta: float) -> Grid:
    lo, hi = tail_range(target, delta)
    return Grid("uniform", lo, hi)


def asinh(target, delta: float, kappa: float, centre: str = "median") -> Grid:
    """x = m + s*sinh(u), with s = kappa * IQR / 2.

    centre="median": m is the median (two-sided warp).
    centre="support": m is the finite lower end of the support, e.g. 0. Then the grid is
    logarithmic over the whole bulk, which suits positive laws (Lévy, log-normal, Gamma, Lomax).
    """
    lo, hi = tail_range(target, delta)
    if centre == "support":
        m = support_lo(target)
        if m is None:
            raise ValueError("no finite lower support bound")
    else:
        m = float(target.ppf(0.5))
    s = kappa * float(target.ppf(0.75) - target.ppf(0.25)) / 2
    return Grid("asinh", float(np.arcsinh((lo - m) / s)), float(np.arcsinh((hi - m) / s)), m, s)


def quantile(target, delta: float) -> Grid:
    """The extreme warp x = F^{-1}(u): every bin holds equal mass, so the state is |+>^n."""
    return Grid("quantile", delta / 2, 1 - delta / 2, law=target.law)


def support_lo(target):
    law = getattr(target, "law", None)
    if law is not None and hasattr(law, "support"):
        a = float(law.support()[0])
        return a if np.isfinite(a) else None
    return None
