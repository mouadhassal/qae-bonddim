"""Matrix product state utilities for real state vectors (qubit 1 = most significant bit)."""
from __future__ import annotations

import numpy as np


def schmidt_spectra(psi: np.ndarray, n: int, tol: float = 1e-14) -> list[np.ndarray]:
    """Normalised Schmidt coefficients at every bond k = 1..n-1 (between qubit k and k+1)."""
    psi = np.asarray(psi, float)
    psi = psi / np.linalg.norm(psi)
    out = []
    m = psi.reshape(1, -1)
    for _ in range(1, n):
        chi = m.shape[0]
        m = m.reshape(chi * 2, -1)
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        keep = max(1, int(np.sum(s > tol * s[0])))
        out.append(s[:keep] / np.linalg.norm(s[:keep]))
        m = s[:keep, None] * vh[:keep]
    return out


def entropy(lam: np.ndarray) -> float:
    p = lam ** 2
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def tail_weight(lam: np.ndarray) -> float:
    """Sum_{i>=1} Lambda_i^2: what a bond-dimension-1 cut throws away (Haiqu's truncation test)."""
    return float(np.sum(lam[1:] ** 2))


def left_canonical(psi: np.ndarray, n: int, chis: list[int]) -> list[np.ndarray]:
    """Sequential-SVD MPS with bond dims capped at chis[k-1]; tensors have shape (l, 2, r)."""
    m = np.asarray(psi, float).reshape(1, -1)
    tensors = []
    for k in range(1, n):
        chi = m.shape[0]
        m = m.reshape(chi * 2, -1)
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        keep = min(chis[k - 1], int(np.sum(s > 1e-14 * s[0])) or 1)
        tensors.append(u[:, :keep].reshape(chi, 2, keep))
        m = s[:keep, None] * vh[:keep]
    tensors.append(m.reshape(m.shape[0], 2, 1))
    return tensors


def mixed_canonical(tensors: list[np.ndarray], c: int):
    """Move the orthogonality centre to bond c (between qubit c and c+1, 1-based).

    Returns (A, lam, B): A[0..c-1] left-canonical, lam the normalised bond-c Schmidt values,
    B[c..n-1] right-canonical, with psi = A1..Ac diag(lam) B_{c+1}..Bn.
    """
    t = [x.copy() for x in tensors]
    n = len(t)
    for k in range(n - 1, c - 1, -1):                     # right-canonicalise qubits n..c+1
        l, d, r = t[k].shape
        u, s, vh = np.linalg.svd(t[k].reshape(l, d * r), full_matrices=False)
        t[k] = vh.reshape(-1, d, r)
        t[k - 1] = np.einsum("adb,bc->adc", t[k - 1], u * s)
    l, d, r = t[c - 1].shape                               # split qubit c's tensor at bond c
    u, s, vh = np.linalg.svd(t[c - 1].reshape(l * d, r), full_matrices=False)
    keep = max(1, int(np.sum(s > 1e-14 * s[0])))
    t[c - 1] = u[:, :keep].reshape(l, d, keep)
    lam = s[:keep] / np.linalg.norm(s[:keep])
    t[c] = np.einsum("ab,bdc->adc", vh[:keep], t[c])
    return t[:c], lam, t[c:]
