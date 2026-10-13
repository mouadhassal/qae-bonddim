"""Figures for the README: python figures.py"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def fig_chi():
    d = json.loads((RES / "chi.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, L in zip(axes, [1, 4]):
        for mode, color, label in [("auto", "#c0392b", "default compilation of S0"),
                                   ("vchain", "#1f77b4", "Toffoli V-chain S0 (+ n-1 helper qubits)")]:
            for n, ls in [(8, ":"), (12, "--"), (16, "-")]:
                ks = sorted({r["k"] for r in d})
                med = [np.max([r["chi_needed"] for r in d if r["mode"] == mode and r["n"] == n and r["L"] == L
                               and r["k"] == k]) for k in ks]
                ax.plot(ks, med, ls, color=color, marker="o", ms=3,
                        label=f"{label}, n={n}" if L == 1 else None)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log", base=2)
        ax.set_xticks([0, 1, 4, 16])
        ax.set_xticklabels(["0", "1", "4", "16"])
        ax.set_xlabel("Grover iterations k")
        ax.set_title(f"loader with {L} layer{'s' if L > 1 else ''}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("bond dimension needed\n(max over 4 laws, error < 1e-6)")
    fig.legend(loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(RES / "fig_chi.png", dpi=160, bbox_inches="tight")


def fig_scale():
    d = json.loads((RES / "scale.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for mode, method, color, ls, label in [
            ("auto", "statevector", "#7f7f7f", "-", "statevector, native multi-controlled gate (n+1 qubits)"),
            ("auto", "matrix_product_state", "#c0392b", "-", "MPS, default compilation (n+1 qubits)"),
            ("vchain", "statevector", "#7f7f7f", "--", "statevector, V-chain (2n qubits)"),
            ("vchain", "matrix_product_state", "#1f77b4", "-", "MPS, V-chain (2n qubits)")]:
        rows = [r for r in d if r["mode"] == mode and r["method"] == method and r["seconds"] is not None]
        ax.plot([r["n"] for r in rows], [r["seconds"] for r in rows], ls, color=color, marker="o", ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("data qubits n (k = 4 Grover iterations)")
    ax.set_ylabel("emulation time (s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "fig_scale.png", dpi=160)


if __name__ == "__main__":
    fig_chi()
    if (RES / "scale.json").exists():
        fig_scale()
    print("figures in results/")
