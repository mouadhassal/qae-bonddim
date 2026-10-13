"""Experiments: how much bond dimension does amplitude estimation on MPS-loaded distributions need?

    python run.py chi       # E1: smallest MPS bond-dimension cap that reproduces P(anc=1) to 1e-6
    python run.py scale     # E2: emulation time vs qubits, MPS vs statevector
    python run.py figures
"""
from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from qaebond import circuit, distribution, exact_a, exact_after, loader_gates, simulate  # noqa: E402
from warpload import loader as ld  # noqa: E402

RES = ROOT / "results"
LAWS = ["levy", "lognorm2", "price_ee", "normal"]
NS = [8, 12, 16]
LS = [1, 4]
KS = [0, 1, 4, 16]
MODES = ["auto", "vchain", "vchain_far"]
CHIS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
TOL = 1e-6
WORKERS = 6


def _chi_job(args):
    law, n, L, k, mode = args
    p = distribution(law, "asinh", n)
    g = loader_gates(p, n, L)
    q = np.abs(ld.run(g, n)) ** 2                       # the distribution the circuit actually loads
    target = exact_after(exact_a(q, n), k)
    qc, lay = circuit(g, n, k, mode)
    rows, need = [], None
    for chi in CHIS:
        r = simulate(qc, lay.anc, "matrix_product_state", chi=chi)
        err = abs(r["p1"] - target)
        rows.append({"chi": chi, "err": err, "seconds": r["seconds"]})
        if err < TOL:
            need = chi
            break
    full = simulate(qc, lay.anc, "matrix_product_state")   # no cap
    return {"law": law, "n": n, "L": L, "k": k, "mode": mode, "qubits": lay.size, "cx": full["cx"],
            "gates": full["gates"], "chi_needed": need, "sweep": rows, "uncapped_seconds": full["seconds"],
            "uncapped_err": abs(full["p1"] - target), "p1_exact": target}


def stage_chi():
    jobs = [(law, n, L, k, m) for law in LAWS for n in NS for L in LS for k in KS for m in MODES]
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(_chi_job, jobs), 1):
            out.append(r)
            if i % 24 == 0:
                print(f"chi: {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    RES.mkdir(exist_ok=True)
    (RES / "chi.json").write_text(json.dumps(out, indent=1))
    print("wrote results/chi.json")


SCALE_NS = [8, 10, 12, 14, 16, 18, 20]
SCALE_K = 4


def _scale_job(args):
    n, mode, method = args
    p = distribution("lognorm2", "asinh", n)
    g = loader_gates(p, n, 1)
    q = np.abs(ld.run(g, n)) ** 2
    target = exact_after(exact_a(q, n), SCALE_K)
    qc, lay = circuit(g, n, SCALE_K, mode)
    if method == "statevector" and lay.size > 24:
        return {"n": n, "mode": mode, "method": method, "qubits": lay.size, "seconds": None, "err": None,
                "note": f"{lay.size} qubits: {2 ** lay.size * 16 / 2 ** 30:.0f} GiB statevector, not run"}
    r = simulate(qc, lay.anc, method)
    return {"n": n, "mode": mode, "method": method, "qubits": lay.size, "seconds": r["seconds"],
            "err": abs(r["p1"] - target), "cx": r["cx"]}


def stage_scale():
    jobs = [(n, m, meth) for n in SCALE_NS for m in ["auto", "vchain"] for meth in ["matrix_product_state", "statevector"]]
    out = []
    for j in jobs:                                       # sequential: timings must not compete
        r = _scale_job(j)
        print(r, flush=True)
        out.append(r)
    (RES / "scale.json").write_text(json.dumps(out, indent=1))
    print("wrote results/scale.json")


if __name__ == "__main__":
    {"chi": stage_chi, "scale": stage_scale}[sys.argv[1]]()
