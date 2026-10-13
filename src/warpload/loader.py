"""Shallow-circuit state loader: a reimplementation of the layered MPS method in Bohun et al.
(Haiqu, arXiv:2412.05202 / PRR 2026), Sec. IV.

One layer: take the current state, cut every bond to chi <= 2 (chi = 1 where the discarded
Schmidt weight sum_{i>=1} Lambda_i^2 is below eps_trunc), put the orthogonality centre on the most
entangled bond c, and write the chi=2 MPS as a "V" of gates. A 1-CNOT gate on (c, c+1) prepares
sum_i Lambda_i |ii>, then two chains of 2-CNOT isometries run outward to both ends in parallel
(depth ~ n/2). The next layer disentangles what is left: psi <- P^dagger psi. With L layers the
circuit is P_1 P_2 ... P_L |0...0>, and its fidelity is |<0|psi_L>|^2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import mps
from .synth import Gate, fit_isometry, one_qubit


def apply(psi: np.ndarray, gate: Gate, n: int, adjoint: bool = False) -> np.ndarray:
    """Apply a gate to a state of shape (2,)*n (axis k-1 is qubit k)."""
    m = gate.matrix()
    if adjoint:
        m = m.T                                       # real orthogonal
    q = [x - 1 for x in gate.qubits]
    k = len(q)
    t = np.tensordot(m.reshape((2,) * 2 * k), psi, axes=(list(range(k, 2 * k)), q))
    return np.moveaxis(t, list(range(k)), q)


def build_layer(psi: np.ndarray, n: int, eps: float, seed: int = 0) -> tuple[list[Gate], int]:
    flat = psi.reshape(-1)
    spectra = mps.schmidt_spectra(flat, n)
    chis = [1 if mps.tail_weight(l) < eps else 2 for l in spectra]
    ent = [mps.entropy(l) if c == 2 else -1.0 for l, c in zip(spectra, chis)]
    c = int(np.argmax(ent)) + 1                        # bond between qubit c and c+1
    tensors = mps.left_canonical(flat, n, chis)
    A, lam, B = mps.mixed_canonical(tensors, c)
    gates: list[Gate] = []
    if len(lam) == 2:                                  # centre: |00> -> l0|00> + l1|11>
        w = np.zeros((4, 1))
        w[0, 0], w[3, 0] = lam
        gates.append(fit_isometry(w, [0], (c, c + 1), tag="centre", seed=seed))
    for i, k in enumerate(range(c + 1, n + 1)):        # right chain, outward
        t = B[i]
        l, _, r = t.shape
        if k == n:
            gates.append(one_qubit(t[:, :, 0].T, k, tag="right-end"))
            break
        w = np.zeros((4, l))
        for a in range(l):
            for s in range(2):
                for b in range(r):
                    w[2 * s + b, a] = t[a, s, b]
        gates.append(fit_isometry(w, [0, 2][:l], (k, k + 1), tag="right", seed=seed + k))
    for k in range(c, 0, -1):                           # left chain, outward
        t = A[k - 1]
        l, _, r = t.shape
        if k == 1:
            gates.append(one_qubit(t[0, :, :], 1, tag="left-end"))
            break
        w = np.zeros((4, r))
        for a in range(l):
            for s in range(2):
                for b in range(r):
                    w[2 * a + s, b] = t[a, s, b]
        gates.append(fit_isometry(w, [0, 1][:r], (k - 1, k), tag="left", seed=seed + 100 + k))
    return gates, c


@dataclass
class Build:
    n: int
    layers: list            # layers[j] = gates of P_{j+1} in application order
    centres: list
    fidelity: list          # fidelity after 1..L layers

    def circuit(self, L: int) -> list[Gate]:
        """Gate list in application order for the L-layer circuit P_1 ... P_L |0>."""
        out = []
        for j in range(L - 1, -1, -1):
            out.extend(self.layers[j])
        return out


def build(target_amps: np.ndarray, n: int, max_layers: int, eps: float = 1e-6) -> Build:
    psi = np.asarray(target_amps, float).reshape((2,) * n).copy()
    psi /= np.linalg.norm(psi)
    layers, centres, fids = [], [], []
    for j in range(max_layers):
        gates, c = build_layer(psi, n, eps, seed=1000 * j)
        for g in reversed(gates):
            psi = apply(psi, g, n, adjoint=True)
        layers.append(gates)
        centres.append(c)
        fids.append(float(psi.reshape(-1)[0] ** 2))
    return Build(n, layers, centres, fids)


def run(gates: list[Gate], n: int) -> np.ndarray:
    psi = np.zeros((2,) * n)
    psi[(0,) * n] = 1.0
    for g in gates:
        psi = apply(psi, g, n)
    return psi.reshape(-1)


def cnots(gates: list[Gate]) -> int:
    return int(sum(g.ncx for g in gates))


def cx_depth(gates: list[Gate], n: int) -> int:
    free = np.zeros(n + 1, int)
    for g in gates:
        if g.ncx == 0:
            continue
        start = max(free[q] for q in g.qubits)
        for q in g.qubits:
            free[q] = start + g.ncx
    return int(free.max())
