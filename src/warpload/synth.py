"""Exact synthesis of real one- and two-qubit gates from RY rotations and CNOTs.

A two-qubit gate on the ordered pair (a, b) with k CNOTs is
    [RY ⊗ RY] CX(a->b) [RY ⊗ RY] ... CX(a->b) [RY ⊗ RY]
which is real, as all our amplitudes are. For an isometry (only some input columns fixed) we
search k = 0, 1, 2, 3 and keep the smallest k that reproduces it to 1e-9.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], float)
Z = np.diag([1.0, -1.0])


def ry(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def two_qubit(theta: np.ndarray, ncx: int) -> np.ndarray:
    u = np.kron(ry(theta[0]), ry(theta[1]))
    for i in range(ncx):
        u = np.kron(ry(theta[2 * i + 2]), ry(theta[2 * i + 3])) @ CX @ u
    return u


@dataclass
class Gate:
    qubits: tuple            # (q,) or (a, b), 1-based, a is the more significant qubit of the pair
    theta: np.ndarray
    ncx: int = 0
    z: bool = False          # one-qubit only: RY(theta) Z (a reflection)
    tag: str = ""
    _m: np.ndarray | None = field(default=None, repr=False)

    def matrix(self) -> np.ndarray:
        if len(self.qubits) == 1:
            m = ry(self.theta[0])
            return m @ Z if self.z else m
        return two_qubit(self.theta, self.ncx)


def one_qubit(w: np.ndarray, q: int, tag="") -> Gate:
    """Exact real 2x2 orthogonal matrix (columns given) as RY or RY·Z."""
    w = np.asarray(w, float)
    if w.shape[1] == 1:                                   # only |0> -> v is fixed
        th = 2 * np.arctan2(w[1, 0], w[0, 0])
        return Gate((q,), np.array([th]), tag=tag)
    det = np.linalg.det(w)
    th = 2 * np.arctan2(w[1, 0], w[0, 0])
    return Gate((q,), np.array([th]), z=det < 0, tag=tag)


def fit_isometry(w: np.ndarray, cols: list[int], pair: tuple, tag="", seed=0) -> Gate:
    """Two-qubit gate G with G[:, cols] = ±w, using the fewest CNOTs (0..3)."""
    w = np.asarray(w, float)
    rng = np.random.default_rng(seed)
    for ncx in range(4):
        npar = 2 * (ncx + 1)
        best = None
        for attempt in range(4 if ncx < 2 else 24):
            x0 = rng.uniform(-np.pi, np.pi, npar)
            for sign in (1.0, -1.0):
                f = lambda th: (two_qubit(th, ncx)[:, cols] - sign * w).ravel()
                r = optimize.least_squares(f, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
                err = np.abs(r.fun).max()
                if best is None or err < best[0]:
                    best = (err, r.x)
                if err < 1e-10:
                    return Gate(pair, r.x, ncx, tag=tag)
    raise RuntimeError(f"could not synthesise isometry {tag}: best residual {best[0]:.2e}")
