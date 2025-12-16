"""
circular_kde_utils.py

Circular kernel density estimation (KDE) for azimuth samples using von Mises kernels.

This module implements the circular KDE approach described in the section
"Directional analysis and circular KDE":

2) Compute empirical mean resultant length:
       R̂ = | (1/n) Σ exp(i φ_w) |
3) Determine concentration κ by solving:
       I1(κ)/I0(κ) = R̂
4) Circular KDE:
       p̂_κ(φ) = (1/n) Σ C_κ(φ - φ_w),
       C_κ(Δφ) = [2π I0(κ)]^{-1} exp(κ cos Δφ)
5) Mode: argmax p̂_κ(φ)
6) Circular HDI_α: shortest arc containing probability mass α under p̂_κ

Notes
-----
- Angles are in radians.
- HDI and mode are evaluated on a uniform grid on [0, 2π).
  Increase n_grid to improve precision.
- Requires SciPy for modified Bessel functions I0 and I1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.special import i0, i1

from kde_utils import kde_1d_mode_and_hdi

TAU = 2.0 * np.pi


def wrap_0_2pi(phi: np.ndarray) -> np.ndarray:
    """Wrap angles (radians) to [0, 2π)."""
    return np.mod(np.asarray(phi, dtype=float), TAU)


def mean_resultant_length(phi: np.ndarray) -> Tuple[float, float]:
    """
    Compute empirical mean resultant length R̂ and mean direction φ̄.

    Returns
    -------
    R_hat : float
        Mean resultant length in [0, 1].
    phi_bar : float
        Mean direction in [0, 2π).
    """
    phi = wrap_0_2pi(np.asarray(phi, dtype=float))
    phi = phi[np.isfinite(phi)]
    if phi.size == 0:
        raise ValueError("mean_resultant_length: no finite angles.")
    m = np.exp(1j * phi).mean()
    return float(np.abs(m)), float(np.mod(np.angle(m), TAU))


def A1(kappa: float) -> float:
    """A1(κ) = I1(κ) / I0(κ)."""
    if kappa <= 0.0:
        return 0.0
    return float(i1(kappa) / i0(kappa))


def kappa_from_R(R: float, tol: float = 1e-10, max_iter: int = 80) -> float:
    """
    Solve A1(κ) = R for κ ≥ 0 using a bracketed Newton method.
    Parameters
    ----------
    R : float
        Target mean resultant length in [0, 1).
    """
    R = float(R)
    if not np.isfinite(R):
        raise ValueError("kappa_from_R: R must be finite.")
    if R <= 0.0:
        return 0.0
    if R >= 0.999999999:
        return 1e6

    # Initial approximation (common in circular statistics)
    if R < 0.53:
        k = 2.0 * R + R**3 + (5.0 * R**5) / 6.0
    elif R < 0.85:
        k = -0.4 + 1.39 * R + 0.43 / (1.0 - R)
    else:
        k = 1.0 / (R**3 - 4.0 * R**2 + 3.0 * R)
    k = max(k, 1e-12)

    def f(x: float) -> float:
        return A1(x) - R

    lo = 0.0
    hi = max(1.0, k)
    while f(hi) < 0.0 and hi < 1e8:
        hi *= 2.0
    if f(hi) < 0.0:
        return float(hi)

    x = min(max(k, 0.0), hi)
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return float(x)

        a = A1(x)
        deriv = 1.0 - a*a - (a / x if x > 0.0 else 0.0)
        x_new = None
        if np.isfinite(deriv) and deriv > 0.0:
            x_new = x - fx / deriv

        if fx < 0.0:
            lo = x
        else:
            hi = x

        if x_new is None or x_new <= lo or x_new >= hi or not np.isfinite(x_new):
            x = 0.5 * (lo + hi)
        else:
            x = x_new

    return float(x)

def compute_directional_stats_vonmises(freq, Sxx, Syy, Szz, Cxy, alpha: float = 0.68, n_grid: int = 720):
    """Directional diagnostics using von Mises circular KDE for azimuth statistics."""
    freq = np.asarray(freq, dtype=float)
    Sxx = np.asarray(Sxx, dtype=float)
    Syy = np.asarray(Syy, dtype=float)
    Szz = np.asarray(Szz, dtype=float)
    Cxy = np.asarray(Cxy, dtype=float)

    n_win, n_freq = Sxx.shape

    Hmin_all = np.empty_like(Sxx)
    Hmax_all = np.empty_like(Sxx)
    phi_all = np.empty_like(Sxx)

    for w in range(n_win):
        diff = 0.5 * (Sxx[w] - Syy[w])
        term = np.sqrt(diff**2 + Cxy[w]**2)

        Hmin_all[w] = 0.5 * (Sxx[w] + Syy[w]) - term
        Hmax_all[w] = 0.5 * (Sxx[w] + Syy[w]) + term
        phi_all[w] = 0.5 * np.arctan2(2.0 * Cxy[w], Sxx[w] - Syy[w])

    phi_mode = np.full(n_freq, np.nan)
    dphi = np.full(n_freq, np.nan)

    for k in range(n_freq):
        phi_k = phi_all[:, k]
        phi_k = phi_k[np.isfinite(phi_k)]
        if phi_k.size < 2:
            continue
        res = circular_kde_von_mises(phi_k, alpha=alpha, n_grid=n_grid)
        phi_mode[k] = res.phi_mode
        dphi[k] = res.dphi

    Rmin_all = np.full_like(Hmin_all, np.nan)
    Rmax_all = np.full_like(Hmax_all, np.nan)

    for k in range(n_freq):
        Hmin_k = Hmin_all[:, k]
        Hmax_k = Hmax_all[:, k]
        V_k = Szz[:, k]

        valid = (
            np.isfinite(Hmin_k) & np.isfinite(Hmax_k) & np.isfinite(V_k) &
            (Hmin_k > 0.0) & (Hmax_k > 0.0) & (V_k > 0.0)
        )
        Rmin_all[valid, k] = np.sqrt(2.0 * Hmin_k[valid] / V_k[valid])
        Rmax_all[valid, k] = np.sqrt(2.0 * Hmax_k[valid] / V_k[valid])

    Rmin = np.full(n_freq, np.nan)
    Rmax = np.full(n_freq, np.nan)

    for k in range(n_freq):
        vmin = Rmin_all[:, k]
        vmax = Rmax_all[:, k]

        vmin = vmin[np.isfinite(vmin) & (vmin > 0.0)]
        vmax = vmax[np.isfinite(vmax) & (vmax > 0.0)]

        if vmin.size >= 3:
            y_mode, _, _ = kde_1d_mode_and_hdi(np.log(vmin), alpha=alpha, c=1.0, grid_size=512, grid_margin=3.0)
            Rmin[k] = np.exp(y_mode)
        if vmax.size >= 3:
            y_mode, _, _ = kde_1d_mode_and_hdi(np.log(vmax), alpha=alpha, c=1.0, grid_size=512, grid_margin=3.0)
            Rmax[k] = np.exp(y_mode)

    return dict(freq=freq, phi_mode=phi_mode, dphi=dphi, Rmin=Rmin, Rmax=Rmax, Rmin_all=Rmin_all, Rmax_all=Rmax_all)



@dataclass(frozen=True)
class CircularKDEResult:
    """Outputs of circular KDE at a single frequency."""
    phi_grid: np.ndarray
    pdf: np.ndarray
    phi_mode: float
    phi_lo: float
    phi_hi: float
    dphi: float
    R_hat: float
    kappa: float


def von_mises_kernel(delta_phi: np.ndarray, kappa: float) -> np.ndarray:
    """von Mises kernel C_κ(Δφ)."""
    if kappa <= 0.0:
        return np.full_like(delta_phi, 1.0 / TAU, dtype=float)
    return np.exp(kappa * np.cos(delta_phi)) / (TAU * i0(kappa))


def circular_kde_pdf(phi_samples: np.ndarray, kappa: float, n_grid: int = 720) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate circular KDE on a uniform grid on [0, 2π).
    Returns
    -------
    phi_grid, pdf
    """
    phi = wrap_0_2pi(np.asarray(phi_samples, dtype=float))
    phi = phi[np.isfinite(phi)]
    if phi.size == 0:
        raise ValueError("circular_kde_pdf: no finite angles.")

    n_grid = int(n_grid)
    phi_grid = np.linspace(0.0, TAU, n_grid, endpoint=False)
    dphi = TAU / float(n_grid)

    if kappa <= 0.0:
        pdf = np.full_like(phi_grid, 1.0 / TAU, dtype=float)
        return phi_grid, pdf

    diff = phi_grid[:, None] - phi[None, :]
    pdf = von_mises_kernel(diff, float(kappa)).mean(axis=1)

    area = float(np.sum(pdf) * dphi)
    if np.isfinite(area) and area > 0.0:
        pdf /= area
    else:
        pdf[:] = 1.0 / TAU

    return phi_grid, pdf


def circular_hdi_from_pdf(phi_grid: np.ndarray, pdf: np.ndarray, alpha: float = 0.68) -> Tuple[float, float, float]:
    """
    Shortest arc on the circle containing probability mass α under a circular pdf.

    Returns
    -------
    phi_lo, phi_hi, dphi
    """
    phi_grid = np.asarray(phi_grid, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    if phi_grid.ndim != 1 or pdf.ndim != 1 or phi_grid.size != pdf.size:
        raise ValueError("circular_hdi_from_pdf: phi_grid and pdf must be 1D of equal length.")
    n = phi_grid.size
    if n < 8:
        raise ValueError("circular_hdi_from_pdf: grid too small (increase n_grid).")

    dphi = TAU / float(n)
    mass = np.clip(pdf, 0.0, np.inf) * dphi
    s = float(mass.sum())
    if not np.isfinite(s) or s <= 0.0:
        mass[:] = 1.0 / float(n)
    else:
        mass /= s

    # Two-pointer scan on the doubled array
    mass2 = np.concatenate([mass, mass])
    best_len = n + 1
    best_start = 0
    best_end = 0

    j = 0
    acc = 0.0
    for i in range(n):
        if j < i:
            j = i
            acc = 0.0
        while acc < alpha and j < i + n:
            acc += mass2[j]
            j += 1
        win_len = j - i
        if acc >= alpha and win_len < best_len:
            best_len = win_len
            best_start = i
            best_end = (j - 1) % n
        acc -= mass2[i]

    phi_lo = float(phi_grid[best_start])
    phi_hi = float((phi_grid[best_end] + dphi) % TAU)
    dphi_out = float((phi_hi - phi_lo) % TAU)
    return phi_lo, phi_hi, dphi_out


def circular_kde_von_mises(
    phi_samples: np.ndarray,
    alpha: float = 0.68,
    n_grid: int = 720,
    kappa: Optional[float] = None,
) -> CircularKDEResult:
    """
    Circular KDE using von Mises kernels.

    Parameters
    ----------
    phi_samples : array_like
        Azimuth samples in radians (φ_w).
    alpha : float
        Target HDI mass (e.g. 0.68).
    n_grid : int
        Grid size for KDE evaluation on [0, 2π).
    kappa : float or None
        If None, κ is estimated from empirical mean resultant length R̂.

    Returns
    -------
    CircularKDEResult
        KDE pdf, mode, HDI bounds, HDI width, and κ diagnostics.
    """
    phi = wrap_0_2pi(np.asarray(phi_samples, dtype=float))
    phi = phi[np.isfinite(phi)]
    if phi.size == 0:
        raise ValueError("circular_kde_von_mises: no finite angles.")

    R_hat, _ = mean_resultant_length(phi)
    kappa_est = float(kappa) if kappa is not None else kappa_from_R(R_hat)
    kappa_est = max(kappa_est, 0.0)

    phi_grid, pdf = circular_kde_pdf(phi, kappa_est, n_grid=int(n_grid))
    phi_mode = float(phi_grid[int(np.argmax(pdf))])
    phi_lo, phi_hi, dphi = circular_hdi_from_pdf(phi_grid, pdf, alpha=float(alpha))

    return CircularKDEResult(
        phi_grid=phi_grid,
        pdf=pdf,
        phi_mode=phi_mode,
        phi_lo=phi_lo,
        phi_hi=phi_hi,
        dphi=dphi,
        R_hat=R_hat,
        kappa=kappa_est,
    )


__all__ = [
    "TAU",
    "wrap_0_2pi",
    "mean_resultant_length",
    "A1",
    "kappa_from_R",
    "CircularKDEResult",
    "von_mises_kernel",
    "circular_kde_pdf",
    "circular_hdi_from_pdf",
    "circular_kde_von_mises",
]
