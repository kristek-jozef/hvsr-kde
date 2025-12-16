"""
Fig1.py

Compare hvsrpy's lognormal statistics (Cox et al. 2020) with
a KDE-mode + HDI representation for one of the example records,
e.g. UT.STN11.A2_C300.miniseed.

- Both lognormal and KDE statistics are computed from HVSR processed
  with Konno–Ohmachi smoothing bandwidth B = 40.
- KDE curves (mode and HDI) are then additionally smoothed in frequency
  with Konno–Ohmachi bandwidth B = 80 for visualization.

This script produces a single figure (Fig. 1):
    - 2D KDE density (frequency, H/V),
    - lognormal median, lognormal mode, and ±1σ curves (B = 40),
    - KDE mode and HDI_0.68 (computed from the same B = 40 HVSR,
      then smoothed in frequency with B = 80).
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
    konno_ohmachi_smooth_curve,
)


# --------------------------------------------------------------------------- #
# Paths and HVSR settings
# --------------------------------------------------------------------------- #

# Adjust this path to your local setup.
EXAMPLE_FILE = Path("d:/hvsr-kde/examples") / "UT.STN11.A2_C300.miniseed"

# KO center frequencies used for HVSR processing.
KO_CENTER_FREQS = np.geomspace(0.3, 2.0, 1024)


# --------------------------------------------------------------------------- #
# HVSR via hvsrpy
# --------------------------------------------------------------------------- #

def load_and_preprocess(fname: Path):
    """
    Read 3C record from file and perform HVSR pre-processing.

    Returns
    -------
    records_pp : list of SeismicRecording3C
        Preprocessed and windowed records as expected by hvsrpy.process().
    """
    if not fname.exists():
        raise FileNotFoundError(
            f"Cannot find example file: {fname}\n"
            "Adjust EXAMPLE_FILE to match your local path."
        )

    records = read([str(fname)])

    pre_settings = HvsrPreProcessingSettings(
        window_length_in_seconds=30.0,
        orient_to_degrees_from_north=0.0,
        filter_corner_frequencies_in_hz=[None, None],
        detrend="linear",
        ignore_dissimilar_time_step_warning=False,
    )

    return preprocess(records, pre_settings)


# --------------------------------------------------------------------------- #
# Traditional HVSR (lognormal statistics)
# --------------------------------------------------------------------------- #

def compute_hvsr_traditional(records_pp, bandwidth: float = 40.0):
    """
    Compute traditional multi-window HVSR using hvsrpy
    for a given Konno–Ohmachi bandwidth.

    Parameters
    ----------
    records_pp : list of SeismicRecording3C
        Preprocessed recordings.
    bandwidth : float
        Konno–Ohmachi bandwidth B (e.g. 40).

    Returns
    -------
    hvsr : HvsrTraditional
        HVSR object containing per-window curves and statistics.
    """
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


# --------------------------------------------------------------------------- #
# Plot: Fig. 1 – KDE vs lognormal
# --------------------------------------------------------------------------- #

def plot_fig1(hvsr, alpha: float = 0.68, kde_c: float = 1.0):
    """Create Fig. 1: KDE density vs lognormal curves."""
    freq = np.asarray(hvsr.frequency)

    # 2D KDE density and unsmoothed KDE mode/HDI
    freq2, amp_grid, density, kde_mode, kde_lo, kde_hi = compute_hvsr_kde_density(
        hvsr, alpha=alpha, c=kde_c, n_amp=200
    )
    if not np.allclose(freq, freq2):
        raise ValueError("Frequency grid mismatch between HVSR and KDE output.")

    # Smooth KDE curves in frequency (KO B=80)
    kde_mode_s = konno_ohmachi_smooth_curve(freq, kde_mode, B=80.0)
    kde_lo_s = konno_ohmachi_smooth_curve(freq, kde_lo, B=80.0)
    kde_hi_s = konno_ohmachi_smooth_curve(freq, kde_hi, B=80.0)

    # Lognormal curves from hvsrpy (computed from the same windows)
    lognorm_median = hvsr.mean_curve(distribution="lognormal")
    try:
        lognorm_plus1 = hvsr.nth_std_curve(+1.0, distribution="lognormal")
        lognorm_minus1 = hvsr.nth_std_curve(-1.0, distribution="lognormal")
    except Exception:
        lognorm_plus1 = None
        lognorm_minus1 = None

    # Lognormal mode from log-amplitudes: exp(mu - sigma^2)
    amps = np.asarray(hvsr.amplitude)
    valid_mask = getattr(hvsr, "valid_curve_boolean_mask", None)
    if valid_mask is not None:
        amps = amps[np.asarray(valid_mask, dtype=bool), :]
    log_amps = np.log(amps)
    mu = np.mean(log_amps, axis=0)
    sigma = np.std(log_amps, axis=0, ddof=1)
    lognorm_mode = np.exp(mu - sigma**2)

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5))

    F, A = np.meshgrid(freq, amp_grid)
    pcm = ax.pcolormesh(F, A, density, cmap="viridis", shading="auto")
    ax.set_xscale("log")
    fig.colorbar(pcm, ax=ax, label="relative KDE density")

    # KDE mode ridge (unsmoothed)
    ax.plot(freq, kde_mode, color="white", linewidth=1.5, label="KDE mode (unsmoothed)")

    # Lognormal stats
    ax.plot(freq, lognorm_median, color="tab:red", linestyle="--", linewidth=2.0, label="lognormal median")
    ax.plot(freq, lognorm_mode, color="tab:red", linestyle="-", linewidth=2.0, label="lognormal mode")
    if lognorm_plus1 is not None and lognorm_minus1 is not None:
        ax.plot(freq, lognorm_plus1, color="tab:red", linestyle=":", linewidth=1.8, label="lognormal median ±1σ")
        ax.plot(freq, lognorm_minus1, color="tab:red", linestyle=":", linewidth=1.8)

    # Smoothed KDE curves
    ax.plot(freq, kde_mode_s, color="b", linewidth=2.5, label="KDE mode (smoothed, B=80)")
    ax.plot(freq, kde_hi_s, color="b", linestyle=":", linewidth=1.8,
            label=f"KDE HDI ({int(alpha*100)}%) (smoothed, B=80)")
    ax.plot(freq, kde_lo_s, color="b", linestyle=":", linewidth=1.8)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("HVSR amplitude [-]")
    ax.set_title("UT STN11 example: HVSR KDE density vs. lognormal stats")
    ax.grid(True, which="both", ls=":")
    ax.set_ylim(0.0, 12.0)

    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig("Fig1_kde_lognormal_ut.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    records_pp = load_and_preprocess(EXAMPLE_FILE)
    hvsr = compute_hvsr_traditional(records_pp, bandwidth=40.0)
    plot_fig1(hvsr, alpha=0.68, kde_c=1.0)


if __name__ == "__main__":
    main()
