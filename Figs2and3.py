"""Figs2and3.py

Rotmoos example (P1 475): KDE vs lognormal HVSR and directional HVSR diagnostics,
where the principal azimuth distribution is modeled with a circular KDE
using von Mises kernels (as described in manuscript Section 2.2).

This is an example/driver script:
- Amplitude KDE uses kde_utils.py.
- Circular KDE for azimuths uses circular_kde_utils.py.

Outputs
-------
- fig2_rotmoos_kde_lognormal.png
- fig3a_dphi_rotmoos.png
- fig3b_envelope_rotmoos.png
- fig3c_Rmin_Rmax_kde.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hvsrpy import read, preprocess, process
from hvsrpy.settings import (
    HvsrPreProcessingSettings,
    HvsrTraditionalProcessingSettings,
)

from kde_utils import (
    compute_hvsr_kde_density,
    compute_r_kde_density,
    kde_1d_mode_and_hdi,
    konno_ohmachi_smooth_curve,
    konno_ohmachi_smooth_spectrum,
    compute_spectra_B40,
)

from circular_kde_utils import (
    circular_kde_von_mises,
    compute_directional_stats_vonmises,
)

# --------------------------------------------------------------------------- #
# Paths and settings
# --------------------------------------------------------------------------- #

EXAMPLE_FILE = Path("d:/hvsr-kde/examples") / "reftek_3C.mseed"

KO_CENTER_FREQS = np.geomspace(5.0, 150.0, 256)
WINDOW_LENGTH = 3.0


def load_and_preprocess_hvsr(fname: Path):
    """
    Read 3C record and perform HVSR pre-processing for hvsrpy.

    Returns
    -------
    records_pp : list of SeismicRecording3C
        Preprocessed and windowed records.
    """
    if not fname.exists():
        raise FileNotFoundError(
            f"Cannot find example file: {fname}\n"
            "Adjust EXAMPLE_FILE to match your local path."
        )

    records = read([str(fname)])
    pre_settings = HvsrPreProcessingSettings(
        window_length_in_seconds=WINDOW_LENGTH,
        orient_to_degrees_from_north=0.0,
        filter_corner_frequencies_in_hz=[None, None],
        detrend="linear",
        ignore_dissimilar_time_step_warning=False,
    )
    return preprocess(records, pre_settings)


def compute_hvsr_traditional(records_pp, bandwidth: float = 40.0):
    """Traditional multi-window HVSR using hvsrpy (Konno–Ohmachi smoothing)."""
    proc_settings = HvsrTraditionalProcessingSettings(
        method_to_combine_horizontals="total_horizontal_energy",
        window_type_and_width=("tukey", 0.1),
        handle_dissimilar_time_steps_by="frequency_domain_resampling",
    )
    proc_settings.smoothing = dict(
        operator="konno_and_ohmachi",
        bandwidth=float(bandwidth),
        center_frequencies_in_hz=KO_CENTER_FREQS,
    )
    return process(records_pp, proc_settings)



def plot_fig2_kde_vs_lognormal(hvsr, alpha: float = 0.68, kde_c: float = 1.0):
    freq = np.asarray(hvsr.frequency)
    freq2, amp_grid, density, kde_mode, kde_lo, kde_hi = compute_hvsr_kde_density(hvsr, alpha=alpha, c=kde_c, n_amp=200)
    if not np.allclose(freq, freq2):
        raise ValueError("Frequency grid mismatch between HVSR and KDE output.")

    kde_mode_s = konno_ohmachi_smooth_curve(freq, kde_mode, B=80.0)
    kde_lo_s = konno_ohmachi_smooth_curve(freq, kde_lo, B=80.0)
    kde_hi_s = konno_ohmachi_smooth_curve(freq, kde_hi, B=80.0)

    lognorm_median = hvsr.mean_curve(distribution="lognormal")
    try:
        lognorm_plus1 = hvsr.nth_std_curve(+1.0, distribution="lognormal")
        lognorm_minus1 = hvsr.nth_std_curve(-1.0, distribution="lognormal")
    except Exception:
        lognorm_plus1 = None
        lognorm_minus1 = None

    amps = np.asarray(hvsr.amplitude)
    valid_mask = getattr(hvsr, "valid_curve_boolean_mask", None)
    if valid_mask is not None:
        amps = amps[np.asarray(valid_mask, dtype=bool), :]
    log_amps = np.log(amps)
    mu = np.mean(log_amps, axis=0)
    sigma = np.std(log_amps, axis=0, ddof=1)
    lognorm_mode = np.exp(mu - sigma**2)

    fig, ax = plt.subplots(figsize=(10, 5))
    F, A = np.meshgrid(freq, amp_grid)
    pcm = ax.pcolormesh(F, A, density, cmap="viridis", shading="auto")
    ax.set_xscale("log")
    fig.colorbar(pcm, ax=ax, label="relative KDE density")

    ax.plot(freq, kde_mode, color="white", linewidth=1.5)

    ax.plot(freq, lognorm_median, color="tab:red", linestyle="--", linewidth=2.0, label="lognormal median")
    ax.plot(freq, lognorm_mode, color="tab:red", linestyle="-", linewidth=2.0, label="lognormal mode")
    if lognorm_plus1 is not None and lognorm_minus1 is not None:
        ax.plot(freq, lognorm_plus1, color="tab:red", linestyle=":", linewidth=1.8, label="lognormal median ±1σ")
        ax.plot(freq, lognorm_minus1, color="tab:red", linestyle=":", linewidth=1.8)

    ax.plot(freq, kde_mode_s, color="b", linewidth=2.5, label="KDE mode (smoothed, B=80)")
    ax.plot(freq, kde_hi_s, color="b", linestyle=":", linewidth=1.8, label=f"KDE HDI ({int(alpha*100)}%) (smoothed, B=80)")
    ax.plot(freq, kde_lo_s, color="b", linestyle=":", linewidth=1.8)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("HVSR amplitude [-]")
    ax.set_title("Rotmoos example: HVSR KDE density vs. lognormal stats")
    ax.grid(True, which="both", ls=":")
    ax.set_ylim(0.0, 8.0)

    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig("fig2_rotmoos_kde_lognormal.png", dpi=300, bbox_inches="tight")
    plt.show()

    return freq, kde_mode_s


def plot_fig3a_dphi(freq: np.ndarray, dphi: np.ndarray, threshold_deg: float = 80.0):
    """Figure 3a: angular spread Δφ_α(f) in degrees."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freq, np.rad2deg(dphi), label=r"$\Delta \varphi_{0.68}(f)$")
    ax.axhline(threshold_deg, color="r", linestyle="--", label=f"threshold {threshold_deg:.0f}°")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Angular spread $\Delta \varphi_{0.68}$ [deg]")
    ax.set_xlim(freq.min(), freq.max())
    ax.set_ylim(0, 180)
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig3a_dphi_rotmoos.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_fig3b_envelope(freq: np.ndarray, kde_hvsr_mode_s: np.ndarray, Rmin: np.ndarray, Rmax: np.ndarray, dphi: np.ndarray, threshold_deg: float = 80.0):
    """Figure 3b: directional envelopes and directionally constrained band."""
    Rmin_s = konno_ohmachi_smooth_curve(freq, Rmin, B=80.0)
    Rmax_s = konno_ohmachi_smooth_curve(freq, Rmax, B=80.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("HVSR amplitude [-]")

    ax.plot(freq, kde_hvsr_mode_s, color="tab:blue", linewidth=2.0, label="KDE HVSR mode (smoothed, B=80)")
    ax.plot(freq, Rmin_s, color="tab:orange", linestyle="--", label=r"KDE $\sqrt{2H_{\min}/V}$ mode (smoothed, B=80)")
    ax.plot(freq, Rmax_s, color="tab:orange", linestyle="-", label=r"KDE $\sqrt{2H_{\max}/V}$ mode (smoothed, B=80)")

    mask = np.isfinite(dphi) & (np.rad2deg(dphi) < threshold_deg)
    ax.fill_between(freq, Rmin_s, Rmax_s, where=mask, color="tab:orange", alpha=0.2, label="directionally constrained band")

    ax.grid(True, which="both", ls=":")
    ax.set_xlim(freq.min(), freq.max())
    ax.set_ylim(0, 8)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig("fig3b_envelope_rotmoos.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_fig3c_rmin_rmax_kde(freq: np.ndarray, Rmin_all: np.ndarray, Rmax_all: np.ndarray, alpha: float = 0.68):
    """Figure 3c: 2D KDE density maps for directional envelopes plus modal curves."""
    f1, amp1, dens1, mode1, _, _ = compute_r_kde_density(freq, Rmin_all, alpha=alpha, n_amp=200)
    f2, amp2, dens2, mode2, _, _ = compute_r_kde_density(freq, Rmax_all, alpha=alpha, n_amp=200)

    mode1_s = konno_ohmachi_smooth_curve(f1, mode1, B=80.0)
    mode2_s = konno_ohmachi_smooth_curve(f2, mode2, B=80.0)

    fig, ax = plt.subplots(figsize=(10, 5))

    F2, A2 = np.meshgrid(f2, amp2)
    ax.pcolormesh(F2, A2, dens2, cmap="Blues", alpha=0.6, shading="auto")

    F1, A1 = np.meshgrid(f1, amp1)
    ax.pcolormesh(F1, A1, dens1, cmap="Oranges", alpha=0.4, shading="auto")

    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("HVSR amplitude [-]")
    ax.set_ylim(0, 8.0)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    m1 = np.isfinite(mode1_s)
    m2 = np.isfinite(mode2_s)
    if np.any(m1):
        ax.plot(f1[m1], mode1_s[m1], color="darkorange", linewidth=3.0, label=r"KDE $\sqrt{2H_{\min}/V}$ mode (smoothed, B=80)")
    if np.any(m2):
        ax.plot(f2[m2], mode2_s[m2], color="darkblue", linewidth=3.0, label=r"KDE $\sqrt{2H_{\max}/V}$ mode (smoothed, B=80)")

    ax.set_title(r"Directional HVSR: KDE density of $\sqrt{2H_{\min}/V}$ and $\sqrt{2H_{\max}/V}$")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig("fig3c_Rmin_Rmax_kde.png", dpi=300, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    records_pp = load_and_preprocess_hvsr(EXAMPLE_FILE)
    hvsr = compute_hvsr_traditional(records_pp, bandwidth=40.0)

    freq_hvsr, kde_hvsr_mode_s = plot_fig2_kde_vs_lognormal(hvsr, alpha=0.68, kde_c=1.0)

    freq_spec, Sxx, Syy, Szz, Cxy = compute_spectra_B40(EXAMPLE_FILE, WINDOW_LENGTH, KO_CENTER_FREQS)
    if not np.allclose(freq_spec, freq_hvsr):
        print("Warning: frequency grid of directional spectra differs from HVSR grid.")
    freq = freq_spec

    stats = compute_directional_stats_vonmises(freq, Sxx, Syy, Szz, Cxy, alpha=0.68, n_grid=720)

    plot_fig3a_dphi(freq, stats["dphi"], threshold_deg=80.0)

    if not np.allclose(freq, freq_hvsr):
        kde_mode_interp = np.interp(freq, freq_hvsr, kde_hvsr_mode_s)
    else:
        kde_mode_interp = kde_hvsr_mode_s

    plot_fig3b_envelope(freq, kde_mode_interp, stats["Rmin"], stats["Rmax"], stats["dphi"], threshold_deg=80.0)

    plot_fig3c_rmin_rmax_kde(freq, stats["Rmin_all"], stats["Rmax_all"], alpha=0.68)


if __name__ == "__main__":
    main()
