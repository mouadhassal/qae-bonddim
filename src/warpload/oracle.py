"""Where does the difficulty go? Payoff ranks on each grid, and the Fisher width of the state.

A warp moves complexity out of the state and into the observable. For quantum Monte Carlo the
payoff f(x) must also be loaded (as rotation angles), as f(phi(u)) on the u-lattice. We score
that cost by the quantics (MPS) rank of the payoff vector, the same currency as the state.

- uniform grid: x is linear in u, rank 2.
- asinh grid: x = m + s*sinh(u) is spanned by {1, e^u, e^-u}, so its rank is <= 3 EXACTLY (each
  exponential is a product state on the binary lattice).
- quantile grid: x = F^{-1}(u) is singular at u -> 0, 1 for heavy tails; no low-rank structure.
"""
from __future__ import annotations

import numpy as np


def qtt_rank(v: np.ndarray, n: int, rtol: float = 1e-10) -> int:
    """Max bond dimension of the MPS of v after dropping a tail of relative Frobenius norm <= rtol."""
    v = np.asarray(v, float)
    nrm = np.linalg.norm(v)
    if nrm == 0:
        return 1
    m = (v / nrm).reshape(1, -1)
    budget = rtol ** 2 / max(n - 1, 1)
    best = 1
    for _ in range(1, n):
        m = m.reshape(m.shape[0] * 2, -1)
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        tail = np.cumsum((s ** 2)[::-1])[::-1]               # tail[i] = sum_{j>=i} s_j^2
        keep = max(1, int(np.sum(tail > budget)))
        best = max(best, keep)
        m = s[:keep, None] * vh[:keep]
    return best


def payoffs(target, grid, n: int) -> dict:
    u = grid.u_edges(n)
    um = 0.5 * (u[1:] + u[:-1])
    x = grid.phi(um)
    k = float(target.ppf(0.99))
    return {"identity": x, "call_q99": np.maximum(x - k, 0.0), "tail_indicator": (x > k).astype(float)}


def fisher_width(probs: np.ndarray, n: int) -> float:
    """g1 of Bohun et al. Theorem 1 for f = sqrt(density) on [0,1]: integral of f'^2 (real f).

    The first-layer loss at bond k is ~ g1 / (6 * 4^k); log2(sqrt(g1)) estimates how many leading
    qubits carry entanglement before the smooth (universal) regime starts.
    """
    f = np.sqrt(np.asarray(probs, float) * 2 ** n)
    df = np.diff(f) * 2 ** n
    return float(np.sum(df ** 2) / 2 ** n)
