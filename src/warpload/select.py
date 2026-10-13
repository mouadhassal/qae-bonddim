"""Automatic grid choice: the most loadable grid that still discretises accurately.

Loadability is the leading-order one-layer infidelity of the bond-2 truncation (Bohun et al.,
Eq. 20):  1 - F_1 ~ sum_k sum_{i>=2} Lambda_{k,i}^2.  It costs one sweep of SVDs and needs no circuit.
Among candidate grids whose exact discretisation already has RTE <= budget, pick the one with
the smallest proxy. If none qualifies, pick the most accurate one. The uniform grid gets the
same rule over its only free parameter, the truncated tail mass delta.
"""
from __future__ import annotations

import numpy as np

from . import grids, metrics, mps

DELTAS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
KAPPAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def proxy(probs: np.ndarray, n: int) -> float:
    sp = mps.schmidt_spectra(np.sqrt(probs), n)
    return float(sum(np.sum(l[2:] ** 2) for l in sp))


def candidates(target, kind: str):
    if kind == "uniform":
        for d in DELTAS:
            yield {"delta": d}, grids.uniform(target, d)
        return
    centres = ["median"] + (["support"] if grids.support_lo(target) is not None else [])
    for c in centres:
        for d in DELTAS:
            for k in KAPPAS:
                yield {"delta": d, "kappa": k, "centre": c}, grids.asinh(target, d, k, c)


def choose(target, n: int, kind: str, budget: float = 0.01):
    rows = []
    for params, g in candidates(target, kind):
        r = metrics.rte_ideal(target, g, n)
        rows.append((params, g, r))
    feasible = [x for x in rows if x[2] <= budget]
    if not feasible:
        params, g, r = min(rows, key=lambda x: x[2])
        return g, {**params, "rte_ideal": r, "proxy": proxy(metrics.bin_masses(target, g, n), n), "feasible": False}
    best = None
    for params, g, r in feasible:
        px = proxy(metrics.bin_masses(target, g, n), n)
        if best is None or px < best[3]:
            best = (params, g, r, px)
    params, g, r, px = best
    return g, {**params, "rte_ideal": r, "proxy": px, "feasible": True}
