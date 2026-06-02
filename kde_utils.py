"""
kde_utils.py

Core utilities for:
- 1D KDE statistics in log-amplitude domain (mode + HDI)
- 2D KDE density maps (frequency × amplitude) for HVSR-like curves
- Konno–Ohmachi smoothing for curves and spectra
- Directional HVSR diagnostics (Hmin/Hmax, azimuth, Rmin/Rmax)

Design goals
------------
- Use a consistent API and naming across helpers.
- Provide small, testable primitives and a few high-level convenience functions.

Conventions
-----------
- Amplitude KDE is performed in log-domain: y = ln(R), where R > 0.
  Returned *modes* and *HDI bounds* from kde_1d_mode_and_hdi are in log-domain.
  Exponentiate to obtain linear amplitudes.
- Circular statistics work in radians, wrapped to [0, 2π).
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

from obspy import read as obspy_read
from scipy.signal import detrend
from scipy.signal.windows import tukey

import numpy as np

TAU = 2.0 * np.pi


# =============================================================================
# Small helpers
# =============================================================================

def _as_finite_1d(x: np.ndarray) -> np.ndarray:
    """Return a 1D float array with only finite entries."""
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def wrap_0_2pi(phi: np.ndarray) -> np.ndarray:
    """Wrap angles (radians) to [0, 2π)."""
    return np.mod(np.asarray(phi, dtype=float), TAU)


# =============================================================================
# 1D KDE in log-domain: mode + HDI
# =============================================================================

def kde_1d_mode_and_hdi(
    log_values: np.ndarray,
    alpha: float = 0.68,
    c: float = 1.5,
    grid_size: int = 512,
    grid_margin: float = 3.0,
) -> Tuple[float, float, float]:
    """
    One-dimensional Gaussian KDE in the log-domain (Y = ln R).

    Parameters
    ----------
    log_values : array_like
        Sample values in log-domain (Y = ln R). Non-finite values are ignored.
    alpha : float
        Credible level for highest-density interval (e.g. 0.68 for "1σ-like").
    c : float
        Factor multiplying Silverman's bandwidth (for flexibility).
    grid_size : int
        Number of points in the evaluation grid.
    grid_margin : float
        Half-width of the grid in units of sigma around the mean
        (i.e. grid covers [mu - grid_margin * sigma, mu + grid_margin * sigma]).

    Returns
    -------
    y_mode : float
        Mode of the distribution in log-domain.
    y_lo : float
        Lower bound of the highest-density interval in log-domain.
    y_hi : float
        Upper bound of the highest-density interval in log-domain.
    """
    y = _as_finite_1d(log_values)
    if y.size == 0:
        raise ValueError("kde_1d_mode_and_hdi: no finite samples.")
    if y.size == 1:
        v = float(y[0])
        return v, v, v

    n = y.size
    mu = float(np.mean(y))
    sigma = float(np.std(y, ddof=1))

    if sigma <= 0.0:
        return mu, mu, mu

    h = float(c * 1.06 * sigma * n ** (-1.0 / 5.0))

    y_min = mu - grid_margin * sigma
    y_max = mu + grid_margin * sigma
    y_grid = np.linspace(y_min, y_max, int(grid_size))

    diff = (y_grid[:, None] - y[None, :]) / h
    pdf_y = np.exp(-0.5 * diff**2).sum(axis=1) / (n * h * np.sqrt(2.0 * np.pi))

    r_grid = np.exp(y_grid)
    pdf_r = pdf_y / r_grid

    area = float(np.trapz(pdf_r, r_grid))
    if not np.isfinite(area) or area <= 0.0:
        return mu, mu, mu
    pdf_r /= area

    y_mode = float(y_grid[int(np.argmax(pdf_r))])

    dr = np.gradient(r_grid)
    pmass = pdf_r * dr
    pmass = pmass / float(pmass.sum())

    order = np.argsort(pdf_r)[::-1]
    cum = np.cumsum(pmass[order])
    inside = cum <= alpha
    if not np.any(inside):
        inside[:] = True

    sel = order[inside]
    y_sel = y_grid[sel]
    y_lo = float(np.min(y_sel))
    y_hi = float(np.max(y_sel))
    return y_mode, y_lo, y_hi


# --------------------------------------------------------------------- #
# Optional: bootstrap wrapper for mode (log-domain)
# --------------------------------------------------------------------- #

def kde_1d_mode_and_hdi_bootstrap(
    log_values: np.ndarray,
    alpha: float = 0.68,
    c: float = 1.5,
    grid_size: int = 512,
    grid_margin: float = 3.0,
    n_bootstrap: int = 200,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Bootstrap-stabilized KDE mode (log-domain), with HDI from the original sample."""
    y = _as_finite_1d(log_values)
    if y.size == 0:
        raise ValueError("kde_1d_mode_and_hdi_bootstrap: no finite samples.")

    y_mode0, y_lo, y_hi = kde_1d_mode_and_hdi(y, alpha=alpha, c=c, grid_size=grid_size, grid_margin=grid_margin)
    if y.size < 2 or n_bootstrap <= 1:
        return y_mode0, y_lo, y_hi

    rng = np.random.default_rng(seed)
    modes = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, y.size, size=y.size)
        modes[i], _, _ = kde_1d_mode_and_hdi(y[idx], alpha=alpha, c=c, grid_size=grid_size, grid_margin=grid_margin)

    return float(np.median(modes)), y_lo, y_hi


# =============================================================================
# Konno–Ohmachi smoothing
# =============================================================================

def konno_ohmachi_weights(freq: np.ndarray, fc: float, B: float) -> np.ndarray:
    """Konno–Ohmachi weights for smoothing around central frequency fc."""
    freq = np.asarray(freq, dtype=float)
    w = np.zeros_like(freq)

    if fc <= 0.0:
        return w

    valid = (freq > 0.0) & np.isfinite(freq)
    if not np.any(valid):
        return w

    x = np.zeros_like(freq)
    x[valid] = B * np.log10(freq[valid] / fc)

    w[valid] = 1.0
    mask = valid & (x != 0.0)
    xm = x[mask]
    w[mask] = (np.sin(xm) / xm) ** 4

    s = float(w.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.zeros_like(freq)
    return w / s


def konno_ohmachi_smooth_curve(freq: np.ndarray, y: np.ndarray, B: float = 40.0) -> np.ndarray:
    """Konno–Ohmachi smoothing of an arbitrary curve y(f) defined on freq."""
    freq = np.asarray(freq, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)

    valid = np.isfinite(y) & np.isfinite(freq) & (freq > 0.0)
    if valid.sum() < 3:
        return y.copy()

    for k, fc in enumerate(freq):
        if not valid[k] or fc <= 0.0:
            continue
        w = konno_ohmachi_weights(freq, float(fc), float(B))
        w[~valid] = 0.0
        s = float(w.sum())
        if s <= 0.0:
            out[k] = y[k]
        else:
            out[k] = float(np.sum((w / s) * y))
    return out


def konno_ohmachi_smooth_spectrum(
    freq_in: np.ndarray,
    spec_in: np.ndarray,
    fc_array: np.ndarray,
    B: float = 40.0,
) -> np.ndarray:
    """Konno–Ohmachi smoothing of a spectrum onto a desired center-frequency grid."""
    freq_in = np.asarray(freq_in, dtype=float)
    spec_in = np.asarray(spec_in, dtype=float)
    fc_array = np.asarray(fc_array, dtype=float)

    valid = np.isfinite(spec_in) & (freq_in > 0.0)
    if valid.sum() < 3:
        return np.zeros_like(fc_array, dtype=float)

    out = np.zeros_like(fc_array, dtype=float)
    for i, fc in enumerate(fc_array):
        if fc <= 0.0:
            continue
        w = konno_ohmachi_weights(freq_in, float(fc), float(B))
        w[~valid] = 0.0
        s = float(w.sum())
        if s > 0.0:
            out[i] = float(np.sum((w / s) * spec_in))
    return out

def compute_windowwise_hvsr_from_spectra(
    Sxx: np.ndarray,
    Syy: np.ndarray,
    Szz: np.ndarray,
    horizontal_divisor: float = 1.0,
) -> np.ndarray:
    """
    Compute window-wise H/V ratios from component power spectra.

    R_i(f) = sqrt( H_i(f) / V_i(f) )

    with

        H_i = (Sxx_i + Syy_i) / horizontal_divisor
        V_i = Szz_i

    Use horizontal_divisor=1.0 if H is defined as total horizontal power.
    Use horizontal_divisor=2.0 if H is defined as average horizontal power.
    """
    Sxx = np.asarray(Sxx, dtype=float)
    Syy = np.asarray(Syy, dtype=float)
    Szz = np.asarray(Szz, dtype=float)

    if Sxx.shape != Syy.shape or Sxx.shape != Szz.shape:
        raise ValueError("Sxx, Syy and Szz must have the same shape.")

    H = (Sxx + Syy) / float(horizontal_divisor)
    V = Szz

    R = np.full_like(H, np.nan, dtype=float)
    valid = (
        np.isfinite(H) & np.isfinite(V) &
        (H > 0.0) & (V > 0.0)
    )
    R[valid] = np.sqrt(H[valid] / V[valid])
    return R


def compute_energy_ratio_estimator(
    Sxx: np.ndarray,
    Syy: np.ndarray,
    Szz: np.ndarray,
    horizontal_divisor: float = 1.0,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the energy-ratio estimator

        R_E(f) = sqrt( <H(f)> / <V(f)> )

    from the same window-wise spectra used to form R_i(f).
    """
    Sxx = np.asarray(Sxx, dtype=float)
    Syy = np.asarray(Syy, dtype=float)
    Szz = np.asarray(Szz, dtype=float)

    if Sxx.shape != Syy.shape or Sxx.shape != Szz.shape:
        raise ValueError("Sxx, Syy and Szz must have the same shape.")

    H = (Sxx + Syy) / float(horizontal_divisor)
    V = Szz

    valid = (
        np.isfinite(H) & np.isfinite(V) &
        (H > 0.0) & (V > 0.0)
    )

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.ndim != 1 or valid_mask.size != H.shape[0]:
            raise ValueError("valid_mask must have shape (n_win,).")
        valid &= valid_mask[:, None]

    H_use = np.where(valid, H, np.nan)
    V_use = np.where(valid, V, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        H_mean = np.nanmean(H_use, axis=0)
        V_mean = np.nanmean(V_use, axis=0)
        R_E = np.sqrt(H_mean / V_mean)

    R_E[~np.isfinite(R_E)] = np.nan
    return R_E
    
# --------------------------------------------------------------------------- #
# Directional spectra from 3C time series (KO B=40)
# --------------------------------------------------------------------------- #

def compute_spectra_B40(reftek_file: Path, window_length: float, ko_center_freqs: np.ndarray ):
    """
    Compute windowed/smoothed spectra Sxx, Syy, Szz, Cxy (KO B=40)
    directly from the 3C time series.

    - Window length = WINDOW_LENGTH seconds
    - Tukey(0.1) taper
    - FFT per window, PSD and cross-PSD
    - Konno–Ohmachi smoothing onto ko_center_freqs

    Returns
    -------
    freq : (n_freq,) array  (ko_center_freqs)
    Sxx, Syy, Szz, Cxy : (n_win, n_freq) arrays
    """
    st = obspy_read(str(reftek_file))
    trZ = st.select(channel="*Z")[0]
    trN = st.select(channel="*N")[0]
    trE = st.select(channel="*E")[0]

    fs = trZ.stats.sampling_rate
    dt = trZ.stats.delta
    nperwin = int(window_length * fs)

    dataZ = trZ.data.astype(float)
    dataN = trN.data.astype(float)
    dataE = trE.data.astype(float)

    n_samples = len(dataZ)
    n_win = n_samples // nperwin

    Sxx_list, Syy_list, Szz_list, Cxy_list = [], [], [], []

    for w in range(n_win):
        i0 = w * nperwin
        i1 = i0 + nperwin

        x = detrend(dataN[i0:i1].astype(float), type="linear")
        y = detrend(dataE[i0:i1].astype(float), type="linear")
        z = detrend(dataZ[i0:i1].astype(float), type="linear")

        taper = tukey(nperwin, alpha=0.1)
        x *= taper; y *= taper; z *= taper

        X = np.fft.rfft(x)
        Y = np.fft.rfft(y)
        Z = np.fft.rfft(z)
        f_fft = np.fft.rfftfreq(nperwin, dt)

        Sxx_raw = np.abs(X) ** 2
        Syy_raw = np.abs(Y) ** 2
        Szz_raw = np.abs(Z) ** 2
        Cxy_raw = np.real(X * np.conj(Y))

        Sxx_sm = konno_ohmachi_smooth_spectrum(f_fft, Sxx_raw, ko_center_freqs, B=40.0)
        Syy_sm = konno_ohmachi_smooth_spectrum(f_fft, Syy_raw, ko_center_freqs, B=40.0)
        Szz_sm = konno_ohmachi_smooth_spectrum(f_fft, Szz_raw, ko_center_freqs, B=40.0)
        Cxy_sm = konno_ohmachi_smooth_spectrum(f_fft, Cxy_raw, ko_center_freqs, B=40.0)

        Sxx_list.append(Sxx_sm)
        Syy_list.append(Syy_sm)
        Szz_list.append(Szz_sm)
        Cxy_list.append(Cxy_sm)

    return ko_center_freqs, np.vstack(Sxx_list), np.vstack(Syy_list), np.vstack(Szz_list), np.vstack(Cxy_list)

    
# ===================================================================== 
# 2D KDE density from HVSR (for Fig.1 / Fig.2 style plots)
# ===================================================================== 

def compute_hvsr_kde_density(
    hvsr,
    alpha: float = 0.68,
    c: float = 1.0,
    n_amp: int = 200,
    p_min: float = 0.0001,
    p_max: float = 99.9,
):
    """
    2D KDE density (frequency, H/V) for an HvsrTraditional object.

    - KDE is carried out in the log(H/V) domain, then transformed back to
      linear amplitudes.
    - Mode + HDI_α are returned in log-domain by kde_1d_mode_and_hdi and
      then exponentiated.

    Returns
    -------
    freq, amp_grid, density, kde_mode, kde_lo, kde_hi
    """
    amplitudes = np.asarray(hvsr.amplitude, dtype=float)
    valid_mask = getattr(hvsr, "valid_curve_boolean_mask", None)
    if valid_mask is not None:
        amps = amplitudes[np.asarray(valid_mask, dtype=bool), :]
    else:
        amps = amplitudes

    freq = np.asarray(hvsr.frequency, dtype=float)
    log_amps = np.log(amps)
    log_all = log_amps[np.isfinite(log_amps)]
    if log_all.size == 0:
        raise ValueError("compute_hvsr_kde_density: no finite ln(H/V) samples.")

    y_min = float(np.percentile(log_all, p_min))
    y_max = float(np.percentile(log_all, p_max))
    y_grid = np.linspace(y_min, y_max, int(n_amp))
    amp_grid = np.exp(y_grid)

    n_freq = freq.size
    density = np.zeros((amp_grid.size, n_freq), dtype=float)
    mode = np.full(n_freq, np.nan, dtype=float)
    lo = np.full(n_freq, np.nan, dtype=float)
    hi = np.full(n_freq, np.nan, dtype=float)

    sqrt_2pi = np.sqrt(2.0 * np.pi)

    for j in range(n_freq):
        yj = log_amps[:, j]
        yj = yj[np.isfinite(yj)]
        if yj.size < 2:
            continue

        n = yj.size
        sigma = float(np.std(yj, ddof=1))
        if sigma <= 0.0:
            r0 = float(np.exp(np.mean(yj)))
            mode[j] = lo[j] = hi[j] = r0
            continue

        h = float(c * 1.06 * sigma * n ** (-1.0 / 5.0))

        diffs = (y_grid[:, None] - yj[None, :]) / h
        pdf_y = np.exp(-0.5 * diffs**2).sum(axis=1) / (n * h * sqrt_2pi)

        r_grid = np.exp(y_grid)
        pdf_r = pdf_y / r_grid
        area = float(np.trapz(pdf_r, r_grid))
        if not np.isfinite(area) or area <= 0.0:
            continue
        pdf_r /= area

        m = float(np.max(pdf_r))
        if m > 0.0:
            density[:, j] = pdf_r / m

        y_mode, y_lo, y_hi = kde_1d_mode_and_hdi(yj, alpha=alpha, c=c, grid_size=512, grid_margin=3.0)
        mode[j] = float(np.exp(y_mode))
        lo[j] = float(np.exp(y_lo))
        hi[j] = float(np.exp(y_hi))

    return freq, amp_grid, density, mode, lo, hi

# ===================================================================== #
# 2D KDE for Rmin/Rmax (used for Fig.3c)
# ===================================================================== #

def compute_r_kde_density(
    freq: np.ndarray,
    R_all: np.ndarray,
    alpha: float = 0.68,
    n_amp: int = 200,
    p_min: float = 0.01,
    p_max: float = 99.99,
):
    """
    2D KDE density for a given matrix of amplitudes R_all (n_win x n_freq).

    Input
    -----
    freq  : (n_freq,) frequency vector
    R_all : (n_win, n_freq) amplitudes (e.g. sqrt(2 H_min/V))
    """

    freq = np.asarray(freq, dtype=float)
    R_all = np.asarray(R_all, dtype=float)

    R_all = np.where((R_all > 0.0) & np.isfinite(R_all), R_all, np.nan)
    log_R = np.log(R_all)
    log_all = log_R[np.isfinite(log_R)]
    if log_all.size == 0:
        raise ValueError("compute_r_kde_density: no finite ln(R) samples.")

    y_min = float(np.percentile(log_all, p_min))
    y_max = float(np.percentile(log_all, p_max))
    y_grid = np.linspace(y_min, y_max, int(n_amp))
    amp_grid = np.exp(y_grid)

    n_freq = freq.size
    density = np.zeros((amp_grid.size, n_freq), dtype=float)
    mode = np.full(n_freq, np.nan, dtype=float)
    lo = np.full(n_freq, np.nan, dtype=float)
    hi = np.full(n_freq, np.nan, dtype=float)

    sqrt_2pi = np.sqrt(2.0 * np.pi)

    for k in range(n_freq):
        yk = log_R[:, k]
        yk = yk[np.isfinite(yk)]
        if yk.size < 3:
            continue

        n = yk.size
        sigma = float(np.std(yk, ddof=1))
        if sigma <= 0.0:
            r0 = float(np.exp(np.mean(yk)))
            mode[k] = lo[k] = hi[k] = r0
            continue

        h = float(1.06 * sigma * n ** (-1.0 / 5.0))

        diffs = (y_grid[:, None] - yk[None, :]) / h
        pdf_y = np.exp(-0.5 * diffs**2).sum(axis=1) / (n * h * sqrt_2pi)

        r_grid = np.exp(y_grid)
        pdf_r = pdf_y / r_grid
        area = float(np.trapz(pdf_r, r_grid))
        if not np.isfinite(area) or area <= 0.0:
            continue
        pdf_r /= area

        m = float(np.max(pdf_r))
        if m > 0.0:
            density[:, k] = pdf_r / m

        y_mode, y_lo, y_hi = kde_1d_mode_and_hdi(yk, alpha=alpha, c=1.0, grid_size=512, grid_margin=3.0)
        mode[k] = float(np.exp(y_mode))
        lo[k] = float(np.exp(y_lo))
        hi[k] = float(np.exp(y_hi))

    return freq, amp_grid, density, mode, lo, hi


# =============================================================================
# Directional HVSR diagnostics
# =============================================================================

def compute_directional_stats(
    freq: np.ndarray,
    Sxx: np.ndarray,
    Syy: np.ndarray,
    Szz: np.ndarray,
    Cxy: np.ndarray,
    alpha: float = 0.68,
) -> Dict[str, np.ndarray]:
    """Directional HVSR diagnostics from windowed spectra."""
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
        phi_k = _as_finite_1d(phi_all[:, k])
        if phi_k.size < 2:
            continue
        phi_m, phi_lo, phi_hi = circular_mode_hdi(phi_k, alpha=alpha)
        phi_mode[k] = phi_m
        dphi[k] = float((phi_hi - phi_lo) % TAU)

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
            Rmin[k] = float(np.exp(y_mode))
        if vmax.size >= 3:
            y_mode, _, _ = kde_1d_mode_and_hdi(np.log(vmax), alpha=alpha, c=1.0, grid_size=512, grid_margin=3.0)
            Rmax[k] = float(np.exp(y_mode))

    return dict(
        freq=freq,
        phi_mode=phi_mode,
        dphi=dphi,
        Rmin=Rmin,
        Rmax=Rmax,
        Rmin_all=Rmin_all,
        Rmax_all=Rmax_all,
    )


__all__ = [
    "TAU",
    "kde_1d_mode_and_hdi",
    "kde_1d_mode_and_hdi_bootstrap",
    "konno_ohmachi_weights",
    "konno_ohmachi_smooth_curve",
    "konno_ohmachi_smooth_spectrum",
    "compute_windowwise_hvsr_from_spectra",
    "compute_energy_ratio_estimator",
    "compute_spectra_B40",
    "compute_hvsr_kde_density",
    "compute_r_kde_density",
    "wrap_0_2pi",
    "compute_directional_stats",
]
