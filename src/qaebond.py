"""Amplitude-estimation circuits on top of MPS-loaded distributions, and how hard they are to emulate.

A  = loader (warped-loading, L layers) on n data qubits, then a payoff rotation on one ancilla:
     P(anc = 1 | bin i) = sin^2(pi/4 + c (i / 2^n - 1/2)),  c = PAYOFF_C   (Woerner & Egger-style linear payoff)
Q  = A S0 A^dagger S_good,   S_good = Z on the ancilla,   S0 = reflection about |0...0> on all n + 1 qubits
After k Grover iterations P(anc = 1) = sin^2((2k + 1) theta) with sin^2(theta) = a = sum_i p_i sin^2(...).

The reflection S0 needs an (n+1)-controlled Z. Three compilations are compared:
  "auto"   : Qiskit's default synthesis of the multi-controlled gate (no helper qubits)
  "vchain" : a Toffoli V-chain with n-1 clean helper qubits, placed between the data qubits so that
             every Toffoli acts on neighbouring qubits of the chain (MPS-friendly)
  "vchain_far": the same V-chain with the helper qubits appended at the end (the naive layout)
"""
from __future__ import annotations

import math
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from warpload import dists, loader, metrics, select
from warpload.noise import to_qiskit

PAYOFF_C = 0.25
BUDGET = 0.01
EPS = 1e-6


def distribution(name: str, kind: str, n: int) -> np.ndarray:
    tg = dists.get(name)
    g, _ = select.choose(tg, n, kind, BUDGET)
    p = metrics.bin_masses(tg, g, n)
    return p / p.sum()


def loader_gates(p: np.ndarray, n: int, L: int):
    return loader.build(np.sqrt(p), n, L, eps=EPS).circuit(L)


def payoff_probs(n: int) -> np.ndarray:
    i = np.arange(2 ** n)
    return np.sin(math.pi / 4 + PAYOFF_C * (i / 2 ** n - 0.5)) ** 2


class Layout:
    """Positions of data qubits, payoff ancilla and helper qubits on the simulator's qubit line."""

    def __init__(self, n: int, mode: str):
        self.n, self.mode = n, mode
        if mode == "auto":
            self.data = list(range(n))            # data[j] = Qiskit index of loader qubit j (little-endian)
            self.anc, self.helpers, self.size = n, [], n + 1
            return
        # chain of controls for the (n+1)-controlled Z: data qubits 0..n-1, then the payoff ancilla is the target
        m = n                                       # number of controls (data); target = payoff ancilla
        if mode == "vchain_far":
            self.data = list(range(n))
            self.anc = n
            self.helpers = list(range(n + 1, n + 1 + (m - 1)))
            self.size = n + 1 + (m - 1)
            return
        # interleaved: d0 d1 h0 d2 h1 d3 h2 ... d_{n-1} h_{n-2} anc
        pos, data, helpers = 0, [None] * n, []
        data[0], data[1] = 0, 1
        pos = 2
        for j in range(2, n):
            helpers.append(pos)
            data[j] = pos + 1
            pos += 2
        helpers.append(pos)
        self.anc = pos + 1
        self.data, self.helpers, self.size = data, helpers, pos + 2


def build_A(gates, n: int, lay: Layout) -> QuantumCircuit:
    qc = QuantumCircuit(lay.size)
    qc.compose(to_qiskit(gates, n), qubits=lay.data, inplace=True)
    # payoff: angle(i) = pi/4 + c (i/2^n - 1/2); RY(2*angle) on the ancilla
    qc.ry(2 * (math.pi / 4 - PAYOFF_C / 2), lay.anc)
    for j in range(n):                       # Qiskit qubit j carries weight 2^j
        qc.cry(2 * PAYOFF_C * 2 ** j / 2 ** n, lay.data[j], lay.anc)
    return qc


def mcz(qc: QuantumCircuit, lay: Layout):
    """Z on the payoff ancilla controlled on all data qubits being 1."""
    ctrls = [lay.data[j] for j in range(lay.n)]
    if lay.mode == "auto":
        qc.h(lay.anc)
        qc.mcx(ctrls, lay.anc)
        qc.h(lay.anc)
        return
    h = lay.helpers
    seq = [(ctrls[0], ctrls[1], h[0])] + [(h[j - 2], ctrls[j], h[j - 1]) for j in range(2, lay.n)]
    for a, b, t in seq:
        qc.ccx(a, b, t)
    qc.cz(h[lay.n - 2], lay.anc)
    for a, b, t in reversed(seq):
        qc.ccx(a, b, t)


def s0(qc: QuantumCircuit, lay: Layout):
    reg = [lay.data[j] for j in range(lay.n)] + [lay.anc]
    qc.x(reg)
    mcz(qc, lay)
    qc.x(reg)


def circuit(gates, n: int, k: int, mode: str) -> tuple[QuantumCircuit, Layout]:
    lay = Layout(n, mode)
    A = build_A(gates, n, lay)
    qc = QuantumCircuit(lay.size)
    qc.compose(A, inplace=True)
    Ainv = A.inverse()
    for _ in range(k):
        qc.z(lay.anc)                        # S_good
        qc.compose(Ainv, inplace=True)
        s0(qc, lay)
        qc.compose(A, inplace=True)
    return qc, lay


def exact_a(p: np.ndarray, n: int) -> float:
    return float(np.dot(p, payoff_probs(n)))


def exact_after(a: float, k: int) -> float:
    th = math.asin(math.sqrt(a))
    return math.sin((2 * k + 1) * th) ** 2


TWO_QUBIT_BASIS = ["cx", "u"]


def two_qubit(qc: QuantumCircuit) -> QuantumCircuit:
    """Compile to CNOT + single-qubit gates, as a hardware backend or a 2-qubit-gate emulator sees it."""
    return transpile(qc, basis_gates=TWO_QUBIT_BASIS, optimization_level=1)


def max_bond(log: str) -> int:
    """Largest bond dimension in Aer's MPS log ('BD=[1 2 4 ...]' entries)."""
    import re
    best = 1
    for grp in re.findall(r"BD=\[([^\]]*)\]", log):
        vals = [int(v) for v in grp.split()]
        if vals:
            best = max(best, max(vals))
    return best


def simulate(qc: QuantumCircuit, anc: int, method: str, chi: int | None = None, log: bool = False) -> dict:
    qc = qc.copy()
    qc.save_probabilities([anc])
    opts = {"method": method}
    if method == "matrix_product_state":
        opts["matrix_product_state_truncation_threshold"] = 1e-16
        opts["mps_log_data"] = log
        if chi is not None:
            opts["matrix_product_state_max_bond_dimension"] = chi
    sim = AerSimulator(**opts)
    tqc = transpile(qc, sim, optimization_level=0)
    t0 = time.perf_counter()
    res = sim.run(tqc, shots=1).result()
    dt = time.perf_counter() - t0
    out = {"p1": float(res.data(0)["probabilities"][1]), "seconds": dt, "gates": int(tqc.size()),
           "cx": int(tqc.count_ops().get("cx", 0))}
    if log:
        out["max_bond"] = max_bond(str(res.results[0].metadata.get("MPS_log_data", "")))
    return out
