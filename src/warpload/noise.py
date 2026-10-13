"""Qiskit export and noisy simulation.

Noise model (per gate, Pauli-depolarising): p2 on every CNOT, p1 = p2/10 on every RY/Z, plus a
symmetric readout flip r on every qubit. The defaults are typical of IBM Heron devices (two-qubit
error ~3e-3, readout ~1e-2). Simulated exactly with Aer's density-matrix method; readout error is
applied to the exact probabilities as a tensor product of 2x2 confusion matrices.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, qasm2

from .synth import Gate


def to_qiskit(gates: list[Gate], n: int) -> QuantumCircuit:
    """Our qubit k (1 = most significant bit of the bin index) is Qiskit qubit n-k (little-endian)."""
    qc = QuantumCircuit(n)
    qb = lambda k: n - k
    for g in gates:
        if len(g.qubits) == 1:
            q = qb(g.qubits[0])
            if g.z:
                qc.z(q)
            qc.ry(float(g.theta[0]), q)
            continue
        a, b = (qb(k) for k in g.qubits)
        qc.ry(float(g.theta[0]), a)
        qc.ry(float(g.theta[1]), b)
        for i in range(g.ncx):
            qc.cx(a, b)
            qc.ry(float(g.theta[2 * i + 2]), a)
            qc.ry(float(g.theta[2 * i + 3]), b)
    return qc


def to_qasm(gates: list[Gate], n: int) -> str:
    return qasm2.dumps(to_qiskit(gates, n))


def noise_model(p2: float, p1: float | None = None):
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    p1 = p2 / 10 if p1 is None else p1
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry", "z"])
    return nm


def readout(probs: np.ndarray, n: int, r: float) -> np.ndarray:
    if r <= 0:
        return probs
    c = np.array([[1 - r, r], [r, 1 - r]])
    t = probs.reshape((2,) * n)
    for ax in range(n):
        t = np.moveaxis(np.tensordot(c, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def mitigate_readout(probs: np.ndarray, n: int, r: float) -> np.ndarray:
    """Invert the (known) per-qubit confusion matrices, clip negatives, renormalise."""
    if r <= 0:
        return probs
    inv = np.linalg.inv(np.array([[1 - r, r], [r, 1 - r]]))
    t = probs.reshape((2,) * n)
    for ax in range(n):
        t = np.moveaxis(np.tensordot(inv, t, axes=([1], [ax])), 0, ax)
    t = np.clip(t.reshape(-1), 0, None)
    return t / t.sum()


def noisy_probs(gates: list[Gate], n: int, p2: float, r: float = 1e-2) -> np.ndarray:
    """Exact output distribution (bin order: index = sum_k s_k 2^(n-k)) under the noise model."""
    from qiskit_aer import AerSimulator
    qc = to_qiskit(gates, n)
    qc.save_probabilities()
    sim = AerSimulator(method="density_matrix", noise_model=noise_model(p2))
    res = sim.run(qc, shots=1).result()                  # no transpile: our gates are native
    probs = np.asarray(res.data()["probabilities"], float)
    return readout(np.clip(probs, 0, None) / probs.sum(), n, r)


def predicted_fidelity(f_ideal: float, gates: list[Gate], p2: float, r: float = 1e-2, n: int = 0) -> float:
    """Noise-aware estimate: ideal fidelity x survival of every gate and readout.

    A depolarising channel with parameter p leaves the state untouched with probability 1 - p*(d^2-1)/d^2
    (d = 4 for CNOT, 2 for one-qubit gates); readout survives with (1-r)^n.
    """
    n2 = sum(g.ncx for g in gates)
    n1 = sum((2 * (g.ncx + 1) if len(g.qubits) == 2 else 1 + int(g.z)) for g in gates)
    return f_ideal * (1 - p2 * 15 / 16) ** n2 * (1 - p2 / 10 * 3 / 4) ** n1 * (1 - r) ** n
