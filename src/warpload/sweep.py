"""Global fidelity sweeps over a fixed gate structure (same CNOTs, better angles).

The layered loader is greedy: each layer is fixed once built, and nothing optimises the whole
circuit. Here every gate is re-optimised against its environment, as in the tensor-network
sweeps of Rudolph et al. (arXiv:2209.00595), but inside the RY+CNOT ansatz of each gate so
the CNOT count and layout do not change. For gate i the overlap is
    o = <chi_i| G_i(theta) |phi_i> = sum_ab G_ab(theta) M_ab,   M_ab = sum_rest chi_i[a,rest] phi_i[b,rest]
with phi_i = G_{i-1}..G_1|0> and chi_i = G_{i+1}^T..G_M^T |target>. Each update cannot lower |o|,
so the fidelity is monotone non-decreasing.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy import optimize

from .loader import apply
from .synth import Gate


def _env(chi, phi, qubits, n):
    q = [x - 1 for x in qubits]
    rest = [i for i in range(n) if i not in q]
    k = len(q)
    c = np.transpose(chi, q + rest).reshape(2 ** k, -1)
    p = np.transpose(phi, q + rest).reshape(2 ** k, -1)
    return c @ p.T


def _optimise_gate(g: Gate, M: np.ndarray, sign: float) -> Gate:
    if len(g.qubits) == 1:                               # closed form for RY(·)Z^z
        zf = np.diag([1.0, -1.0]) if g.z else np.eye(2)
        N = M @ zf.T                                     # o = tr(RY^T N) ... maximise over angle
        a = N[0, 0] + N[1, 1]
        b = N[1, 0] - N[0, 1]
        th = 2 * np.arctan2(sign * b, sign * a)
        return Gate(g.qubits, np.array([th]), 0, g.z, g.tag)
    f = lambda th: -sign * float(np.sum(Gate(g.qubits, th, g.ncx).matrix() * M))
    r = optimize.minimize(f, g.theta, method="BFGS", options={"gtol": 1e-12})
    th = r.x if r.fun <= f(g.theta) else g.theta
    return Gate(g.qubits, th, g.ncx, g.z, g.tag)


def sweep(gates: list[Gate], target: np.ndarray, n: int, max_sweeps: int = 10, tol: float = 1e-10):
    """Return (optimised gates, fidelity history). Gates are copied, not modified in place."""
    gates = copy.deepcopy(gates)
    t = np.asarray(target, float).reshape((2,) * n)
    t = t / np.linalg.norm(t)
    zero = np.zeros((2,) * n)
    zero[(0,) * n] = 1.0

    def overlap(gs):
        psi = zero
        for g in gs:
            psi = apply(psi, g, n)
        return float(np.sum(psi * t))

    o = overlap(gates)
    hist = [o * o]
    for _ in range(max_sweeps):
        sign = 1.0 if o >= 0 else -1.0
        # forward pass
        chi = t
        for g in reversed(gates[1:]):
            chi = apply(chi, g, n, adjoint=True)
        phi = zero
        for i in range(len(gates)):
            gates[i] = _optimise_gate(gates[i], _env(chi, phi, gates[i].qubits, n), sign)
            phi = apply(phi, gates[i], n)
            if i + 1 < len(gates):
                chi = apply(chi, gates[i + 1], n)
        # backward pass (phi is now the full prepared state)
        chi = t
        for i in range(len(gates) - 1, -1, -1):
            phi = apply(phi, gates[i], n, adjoint=True)
            gates[i] = _optimise_gate(gates[i], _env(chi, phi, gates[i].qubits, n), sign)
            chi = apply(chi, gates[i], n, adjoint=True)
        o = overlap(gates)
        hist.append(o * o)
        if hist[-1] - hist[-2] < tol:
            break
    return gates, hist
