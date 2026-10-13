"""Discretisation and end-to-end accuracy metrics.

The headline metric is the relative tail-probability error (RTE). For each level p in LEVELS we
read the p-quantile off the loaded distribution (inverting its CDF linearly in u inside a bin,
i.e. a sample is a bin followed by a uniform draw in u), then ask the TRUE distribution how much
probability actually lies beyond that point:

    e_p = |F(q_hat_p) - p| / min(p, 1 - p)
    RTE = max_p e_p

RTE = 0.05 means every tail probability from 0.1% to 99.9% is right to within 5% of itself.
Unlike state fidelity to the binned target (Haiqu's `job.quality`), it sees truncation and tails.
"""
from __future__ import annotations

import numpy as np

LEVELS = np.array([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])


def _cdf_sf(target, x):
    """(cdf, sf) evaluated without cancellation on either side of the median."""
    x = np.asarray(x, float)
    med = float(target.ppf(0.5))
    cdf = np.empty_like(x)
    sf = np.empty_like(x)
    left = x <= med
    if left.any():
        cdf[left] = target.cdf(x[left])
        sf[left] = 1 - cdf[left]
    if (~left).any():
        sf[~left] = target.sf(x[~left])
        cdf[~left] = 1 - sf[~left]
    return cdf, sf


def bin_masses(target, grid, n: int) -> np.ndarray:
    """Exact P(X in bin) for every bin, normalised over the grid range (truncation renormalises)."""
    x = grid.edges(n)
    cdf, sf = _cdf_sf(target, x)
    med = float(target.ppf(0.5))
    m = np.where(x[1:] <= med, cdf[1:] - cdf[:-1], sf[:-1] - sf[1:])
    m = np.clip(m, 0.0, None)
    return m / m.sum()


def quantiles(probs: np.ndarray, grid, p=LEVELS) -> np.ndarray:
    """Quantiles of a loaded distribution: bin by cumulative mass, linear in u inside the bin."""
    probs = np.asarray(probs, float)
    probs = probs / probs.sum()
    cum = np.concatenate([[0.0], np.cumsum(probs)])
    j = np.clip(np.searchsorted(cum, p, side="right") - 1, 0, len(probs) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(probs[j] > 0, (p - cum[j]) / probs[j], 0.5)
    frac = np.clip(frac, 0.0, 1.0)
    du = (grid.u_hi - grid.u_lo) / len(probs)
    return grid.phi(grid.u_lo + (j + frac) * du)


def tail_errors(target, qhat, p=LEVELS) -> np.ndarray:
    cdf, sf = _cdf_sf(target, qhat)
    return np.where(p <= 0.5, np.abs(cdf - p) / p, np.abs(sf - (1 - p)) / (1 - p))


def rte(target, probs, grid) -> float:
    return float(tail_errors(target, quantiles(probs, grid)).max())


def rte_ideal(target, grid, n: int, p=LEVELS) -> float:
    """RTE of the exact discretisation at any n (up to ~60 qubits) without building 2^n bins."""
    u_lo, u_hi = grid.u_lo, grid.u_hi
    du = (u_hi - u_lo) / 2 ** n
    F_lo = float(target.cdf(grid.x_lo))
    S_hi = float(target.sf(grid.x_hi))
    Z = 1.0 - F_lo - S_hi
    out = []
    for q in p:
        xs = float(target.ppf(F_lo + q * Z))              # exact quantile of the truncated law
        j = int(np.clip(np.floor((float(grid.phi_inv(xs)) - u_lo) / du), 0, 2 ** n - 1))
        a, b = grid.phi(u_lo + j * du), grid.phi(u_lo + (j + 1) * du)
        ca, sa = _cdf_sf(target, np.array([a, b]))
        Ga = (ca[0] - F_lo) / Z
        Gb = (ca[1] - F_lo) / Z
        if q > 0.5:                                        # complement form for precision
            Ga = 1 - (sa[0] - S_hi) / Z
            Gb = 1 - (sa[1] - S_hi) / Z
        frac = 0.5 if Gb <= Ga else np.clip((q - Ga) / (Gb - Ga), 0, 1)
        qhat = grid.phi(u_lo + (j + frac) * du)
        out.append(tail_errors(target, np.array([qhat]), np.array([q]))[0])
    return float(np.max(out))


def ks_continuous(target, probs, grid) -> float:
    """sup_x |F_loaded(x) - F(x)| over bin edges and bin midpoints (in u)."""
    probs = np.asarray(probs, float) / np.sum(probs)
    n = int(np.log2(len(probs)))
    u = grid.u_edges(n)
    cum = np.concatenate([[0.0], np.cumsum(probs)])
    mid_u = 0.5 * (u[1:] + u[:-1])
    x = np.concatenate([grid.phi(u), grid.phi(mid_u)])
    fhat = np.concatenate([cum, cum[:-1] + probs / 2])
    cdf, _ = _cdf_sf(target, x)
    return float(np.max(np.abs(fhat - cdf)))


def fidelity(p, q) -> float:
    """Classical (Bhattacharyya) fidelity; equals state fidelity for non-negative real amplitudes."""
    return float(np.sum(np.sqrt(np.asarray(p) * np.asarray(q))) ** 2)


def kl(p, q, floor: float = 1e-300) -> float:
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], floor))))


def ks_test_pass_rate(target, probs, grid, n_samples=200, reps=200, seed=0, alpha=0.05) -> float:
    """Haiqu's acceptance test: share of 200-sample KS tests (vs the true law) that pass."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs, float) / np.sum(probs)
    n = int(np.log2(len(probs)))
    du = (grid.u_hi - grid.u_lo) / 2 ** n
    passed = 0
    for _ in range(reps):
        j = rng.choice(len(probs), size=n_samples, p=probs)
        x = grid.phi(grid.u_lo + (j + rng.random(n_samples)) * du)
        passed += stats.kstest(x, target.cdf).pvalue > alpha
    return passed / reps
