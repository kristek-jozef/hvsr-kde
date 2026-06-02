"""Create Figs. 3 and 5 for the Rotmoos HVSR KDE/directional example.

The script generates:

- Fig. 3: KDE HVSR density versus lognormal statistics and the energy-ratio
  estimator R_E.
- Fig. 5a: frequency-dependent principal-azimuth spread.
- Fig. 5b: directional HVSR envelopes and directionally constrained band.
- Fig. 5c: KDE density maps for the directional-envelope ratios.

By default, the script looks for ``reftek_3C.mseed`` in the current working
directory, next to this script, or in an ``examples/`` subdirectory. Use
``--input`` to specify another file.

Examples
--------
Run with repository-default paths::

    python Figs3and5.py

Run with explicit input and output directory::

    python Figs3and5.py --input examples/reftek_3C.mseed --output-dir figures
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from hvsrpy import preprocess, process, read
from hvsrpy.settings import (
    HvsrPreProcessingSettings,
    HvsrTraditionalProcessingSettings,
)

from circular_kde_utils import compute_directional_stats_vonmises
from kde_utils import (
    compute_hvsr_kde_density,
    compute_r_kde_density,
    compute_spectra_B40,
    konno_ohmachi_smooth_curve,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_FILE = "reftek_3C.mseed"
DEFAULT_WINDOW_LENGTH = 3.0
DEFAULT_FMIN = 5.0
DEFAULT_FMAX = 150.0
DEFAULT_N_FREQ = 256
DEFAULT_HVSR_BANDWIDTH = 40.0
DEFAULT_DISPLAY_BANDWIDTH = 80.0
DEFAULT_ALPHA = 0.68
DEFAULT_KDE_C = 1.0
DEFAULT_DPHI_THRESHOLD_DEG = 80.0
DEFAULT_HORIZONTAL_DIVISOR = 1.0

DEFAULT_RE_SEARCH_RANGE_HZ = (5.0, 150.0)
DEFAULT_RE_N_SIGMA = 2.1
DEFAULT_RE_MAX_ITERATIONS = 20
DEFAULT_RE_SCORE_PERCENTILE = 95.0

FIG3_NAME = "fig3_rotmoos_kde_lognormal.png"
FIG5A_NAME = "fig5a_dphi_rotmoos.png"
FIG5B_NAME = "fig5b_envelope_rotmoos.png"
FIG5C_NAME = "fig5c_Rmin_Rmax_kde.png"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create Rotmoos KDE/lognormal HVSR and directional HVSR diagnostics figures."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DEFAULT_INPUT_FILE),
        help=(
            "Input 3-component MiniSEED file. Relative paths are resolved against "
            "the current directory, the script directory, and ./examples."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where the figures will be written.",
    )
    parser.add_argument(
        "--window-length",
        type=float,
        default=DEFAULT_WINDOW_LENGTH,
        help="Window length in seconds used for HVSR processing.",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=DEFAULT_FMIN,
        help="Minimum KO center frequency [Hz].",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=DEFAULT_FMAX,
        help="Maximum KO center frequency [Hz].",
    )
    parser.add_argument(
        "--n-freq",
        type=int,
        default=DEFAULT_N_FREQ,
        help="Number of logarithmically spaced KO center frequencies.",
    )
    parser.add_argument(
        "--hvsr-bandwidth",
        type=float,
        default=DEFAULT_HVSR_BANDWIDTH,
        help="Konno-Ohmachi bandwidth used by hvsrpy processing.",
    )
    parser.add_argument(
        "--display-bandwidth",
        type=float,
        default=DEFAULT_DISPLAY_BANDWIDTH,
        help="Konno-Ohmachi bandwidth used only for smoothing displayed KDE curves.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="HDI probability mass.")
    parser.add_argument(
        "--kde-c",
        type=float,
        default=DEFAULT_KDE_C,
        help="Multiplier for Silverman's KDE bandwidth in log-amplitude space.",
    )
    parser.add_argument(
        "--dphi-threshold-deg",
        type=float,
        default=DEFAULT_DPHI_THRESHOLD_DEG,
        help="Azimuth-spread threshold used for the constrained directional band.",
    )
    parser.add_argument(
        "--horizontal-divisor",
        type=float,
        default=DEFAULT_HORIZONTAL_DIVISOR,
        help="Horizontal energy divisor for R_E: 1 gives H=Sxx+Syy, 2 gives H=(Sxx+Syy)/2.",
    )
    parser.add_argument(
        "--disable-re-rejection",
        action="store_true",
        help="Disable conservative window rejection for R_E only.",
    )
    parser.add_argument(
        "--re-fmin",
        type=float,
        default=DEFAULT_RE_SEARCH_RANGE_HZ[0],
        help="Lower frequency bound for R_E-only rejection [Hz].",
    )
    parser.add_argument(
        "--re-fmax",
        type=float,
        default=DEFAULT_RE_SEARCH_RANGE_HZ[1],
        help="Upper frequency bound for R_E-only rejection [Hz].",
    )
    parser.add_argument(
        "--re-n-sigma",
        type=float,
        default=DEFAULT_RE_N_SIGMA,
        help="Robust z-score threshold for R_E-only rejection.",
    )
    parser.add_argument(
        "--re-max-iterations",
        type=int,
        default=DEFAULT_RE_MAX_ITERATIONS,
        help="Maximum number of R_E-only rejection iterations.",
    )
    parser.add_argument(
        "--re-score-percentile",
        type=float,
        default=DEFAULT_RE_SCORE_PERCENTILE,
        help="Percentile of absolute z-scores used as the per-window rejection score.",
    )
    parser.add_argument(
        "--no-plot-re-all",
        action="store_true",
        help="Do not plot R_E computed from all windows in Fig. 3.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution in dots per inch.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more progress information.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    """Configure command-line logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )


def resolve_input_path(path: Path) -> Path:
    """Resolve an input file path against common repository locations."""
    candidates = [path]
    if not path.is_absolute():
        script_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                Path.cwd() / path,
                script_dir / path,
                Path.cwd() / "examples" / path,
                script_dir / "examples" / path,
            ]
        )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {candidate.expanduser()}" for candidate in candidates)
    raise FileNotFoundError(
        "Cannot find the input MiniSEED file. Searched:\n"
        f"{searched}\n"
        "Use --input to specify the file explicitly."
    )


def validate_positive_float(value: float, name: str) -> None:
    """Raise ValueError if *value* is not finite and positive."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}.")


def build_center_frequencies(fmin: float, fmax: float, n_freq: int) -> np.ndarray:
    """Create a logarithmically spaced frequency grid."""
    validate_positive_float(fmin, "fmin")
    validate_positive_float(fmax, "fmax")
    if fmax <= fmin:
        raise ValueError(f"fmax must be larger than fmin, got fmin={fmin}, fmax={fmax}.")
    if n_freq < 2:
        raise ValueError(f"n_freq must be at least 2, got {n_freq}.")
    return np.geomspace(fmin, fmax, int(n_freq))


def load_and_preprocess_hvsr(filename: Path, window_length: float) -> Any:
    """Read a 3C record and perform hvsrpy preprocessing."""
    records = read([str(filename)])
    pre_settings = HvsrPreProcessingSettings(
        window_length_in_seconds=float(window_length),
        orient_to_degrees_from_north=0.0,
        filter_corner_frequencies_in_hz=[None, None],
        detrend="linear",
        ignore_dissimilar_time_step_warning=False,
    )
    return preprocess(records, pre_settings)


def compute_hvsr_traditional(
    records_pp: Any,
    center_frequencies: np.ndarray,
    bandwidth: float,
) -> Any:
    """Compute traditional multi-window HVSR using hvsrpy and KO smoothing."""
    proc_settings = HvsrTraditionalProcessingSettings(
        method_to_combine_horizontals="total_horizontal_energy",
        window_type_and_width=("tukey", 0.1),
        handle_dissimilar_time_steps_by="frequency_domain_resampling",
    )
    proc_settings.smoothing = {
        "operator": "konno_and_ohmachi",
        "bandwidth": float(bandwidth),
        "center_frequencies_in_hz": np.asarray(center_frequencies, dtype=float),
    }
    return process(records_pp, proc_settings)


def compute_windowwise_ratio_from_spectra(
    Sxx: np.ndarray,
    Syy: np.ndarray,
    Szz: np.ndarray,
    horizontal_divisor: float = DEFAULT_HORIZONTAL_DIVISOR,
) -> np.ndarray:
    """Compute window-wise H/V ratios R_i(f) = sqrt(H_i(f) / V_i(f))."""
    Sxx = np.asarray(Sxx, dtype=float)
    Syy = np.asarray(Syy, dtype=float)
    Szz = np.asarray(Szz, dtype=float)

    if Sxx.shape != Syy.shape or Sxx.shape != Szz.shape:
        raise ValueError("Sxx, Syy, and Szz must have the same shape.")

    H = (Sxx + Syy) / float(horizontal_divisor)
    V = Szz
    ratio = np.full_like(H, np.nan, dtype=float)
    valid = np.isfinite(H) & np.isfinite(V) & (H > 0.0) & (V > 0.0)
    ratio[valid] = np.sqrt(H[valid] / V[valid])
    return ratio


def compute_energy_ratio_estimator_from_spectra(
    Sxx: np.ndarray,
    Syy: np.ndarray,
    Szz: np.ndarray,
    valid_mask: np.ndarray | None = None,
    horizontal_divisor: float = DEFAULT_HORIZONTAL_DIVISOR,
) -> np.ndarray:
    """Compute R_E(f) = sqrt(<H(f)> / <V(f)>) from component spectra."""
    Sxx = np.asarray(Sxx, dtype=float)
    Syy = np.asarray(Syy, dtype=float)
    Szz = np.asarray(Szz, dtype=float)

    if Sxx.shape != Syy.shape or Sxx.shape != Szz.shape:
        raise ValueError("Sxx, Syy, and Szz must have the same shape.")

    H = (Sxx + Syy) / float(horizontal_divisor)
    V = Szz
    valid = np.isfinite(H) & np.isfinite(V) & (H > 0.0) & (V > 0.0)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.ndim != 1 or valid_mask.size != H.shape[0]:
            raise ValueError(
                "valid_mask must have shape (n_windows,), "
                f"got {valid_mask.shape}, expected ({H.shape[0]},)."
            )
        valid &= valid_mask[:, None]

    H_use = np.where(valid, H, np.nan)
    V_use = np.where(valid, V, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        R_E = np.sqrt(np.nanmean(H_use, axis=0) / np.nanmean(V_use, axis=0))

    R_E[~np.isfinite(R_E)] = np.nan
    return R_E


def rejection_mask_for_RE_from_ratios(
    freq: np.ndarray,
    R_all: np.ndarray,
    search_range_hz: tuple[float, float] = DEFAULT_RE_SEARCH_RANGE_HZ,
    n_sigma: float = DEFAULT_RE_N_SIGMA,
    max_iterations: int = DEFAULT_RE_MAX_ITERATIONS,
    score_percentile: float = DEFAULT_RE_SCORE_PERCENTILE,
    min_scale: float = 0.05,
) -> np.ndarray:
    """Construct a conservative window-rejection mask used only for R_E."""
    freq = np.asarray(freq, dtype=float)
    R_all = np.asarray(R_all, dtype=float)

    if R_all.ndim != 2 or R_all.shape[1] != freq.size:
        raise ValueError("R_all must have shape (n_windows, n_frequencies).")

    fmin, fmax = search_range_hz
    fmask = np.isfinite(freq)
    if fmin is not None:
        fmask &= freq >= fmin
    if fmax is not None:
        fmask &= freq <= fmax
    if not np.any(fmask):
        raise ValueError("No frequencies inside the R_E rejection search range.")

    X = np.full_like(R_all, np.nan, dtype=float)
    good = np.isfinite(R_all) & (R_all > 0.0)
    X[good] = np.log(R_all[good])

    finite_fraction = np.mean(np.isfinite(X[:, fmask]), axis=1)
    valid_mask = finite_fraction > 0.8

    for iteration in range(int(max_iterations)):
        old_mask = valid_mask.copy()
        X_use = X[valid_mask][:, fmask]
        if X_use.shape[0] < 5:
            LOGGER.warning("Too few windows left for R_E rejection; stopping.")
            break

        center = np.nanmedian(X_use, axis=0)
        mad = np.nanmedian(np.abs(X_use - center[None, :]), axis=0)
        robust_scale = 1.4826 * mad
        std_scale = np.nanstd(X_use, axis=0, ddof=1)

        scale = np.where(
            np.isfinite(robust_scale) & (robust_scale > min_scale),
            robust_scale,
            std_scale,
        )
        scale = np.where(np.isfinite(scale) & (scale > min_scale), scale, min_scale)

        Z = np.abs((X[:, fmask] - center[None, :]) / scale[None, :])
        curve_score = np.nanpercentile(Z, score_percentile, axis=1)

        valid_mask &= np.isfinite(curve_score)
        valid_mask &= curve_score <= n_sigma

        if np.array_equal(valid_mask, old_mask):
            LOGGER.info("R_E-only rejection converged after %d iterations.", iteration + 1)
            break

    LOGGER.info(
        "R_E-only rejection kept %d / %d windows (%.1f%%).",
        int(valid_mask.sum()),
        int(valid_mask.size),
        100.0 * float(valid_mask.mean()),
    )
    return valid_mask


def log_RE_diagnostics(
    R_all_for_mask: np.ndarray,
    R_E_all: np.ndarray,
    R_E_acc: np.ndarray,
) -> None:
    """Log simple diagnostics comparing R_E to the window-wise median."""
    ratio_matrix = np.where(
        np.isfinite(R_all_for_mask) & (R_all_for_mask > 0.0),
        R_all_for_mask,
        np.nan,
    )
    with np.errstate(invalid="ignore"):
        median_ratio = np.exp(np.nanmedian(np.log(ratio_matrix), axis=0))

    ratio_all = R_E_all / median_ratio
    ratio_acc = R_E_acc / median_ratio

    LOGGER.info(
        "R_E_all / median: median=%.3f, 5-95%%=%.3f-%.3f",
        np.nanmedian(ratio_all),
        np.nanpercentile(ratio_all, 5),
        np.nanpercentile(ratio_all, 95),
    )
    LOGGER.info(
        "R_E_acc / median: median=%.3f, 5-95%%=%.3f-%.3f",
        np.nanmedian(ratio_acc),
        np.nanpercentile(ratio_acc, 5),
        np.nanpercentile(ratio_acc, 95),
    )


def lognormal_mode_from_hvsr(hvsr: Any) -> np.ndarray:
    """Compute the lognormal mode from hvsrpy's window-wise amplitudes."""
    amplitudes = np.asarray(hvsr.amplitude, dtype=float)
    amplitudes = np.where(np.isfinite(amplitudes) & (amplitudes > 0.0), amplitudes, np.nan)
    log_amplitudes = np.log(amplitudes)
    mu = np.nanmean(log_amplitudes, axis=0)
    sigma = np.nanstd(log_amplitudes, axis=0, ddof=1)
    return np.exp(mu - sigma**2)


def save_or_show(fig: plt.Figure, output_path: Path, dpi: int, show: bool) -> Path:
    """Save a Matplotlib figure and optionally show it interactively."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    LOGGER.info("Wrote %s", output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_fig3_kde_vs_lognormal(
    hvsr: Any,
    output_path: Path,
    alpha: float,
    kde_c: float,
    display_bandwidth: float,
    dpi: int,
    show: bool,
    R_E_acc: np.ndarray | None = None,
    R_E_all: np.ndarray | None = None,
    plot_re_all: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create Fig. 3 and return the frequency grid and smoothed KDE mode."""
    freq = np.asarray(hvsr.frequency, dtype=float)
    freq2, amp_grid, density, kde_mode, kde_lo, kde_hi = compute_hvsr_kde_density(
        hvsr,
        alpha=alpha,
        c=kde_c,
        n_amp=200,
    )
    if not np.allclose(freq, freq2):
        raise ValueError("Frequency grid mismatch between HVSR and KDE output.")

    kde_mode_s = konno_ohmachi_smooth_curve(freq, kde_mode, B=display_bandwidth)
    kde_lo_s = konno_ohmachi_smooth_curve(freq, kde_lo, B=display_bandwidth)
    kde_hi_s = konno_ohmachi_smooth_curve(freq, kde_hi, B=display_bandwidth)

    lognorm_median = hvsr.mean_curve(distribution="lognormal")
    lognorm_mode = lognormal_mode_from_hvsr(hvsr)
    try:
        lognorm_plus1 = hvsr.nth_std_curve(+1.0, distribution="lognormal")
        lognorm_minus1 = hvsr.nth_std_curve(-1.0, distribution="lognormal")
    except Exception as exc:  # hvsrpy versions may differ here.
        LOGGER.warning("Could not compute lognormal ±1σ curves: %s", exc)
        lognorm_plus1 = None
        lognorm_minus1 = None

    fig, ax = plt.subplots(figsize=(10, 5))
    F, A = np.meshgrid(freq, amp_grid)
    pcm = ax.pcolormesh(F, A, density, cmap="viridis", shading="auto")
    ax.set_xscale("log")
    fig.colorbar(pcm, ax=ax, label="relative KDE density")

    ax.plot(
        freq, kde_mode, color="white", linewidth=1.5, label="KDE mode (unsmoothed)"
    )
    ax.plot(
        freq,
        lognorm_median,
        color="tab:red",
        linestyle="--",
        linewidth=2.0,
        label="lognormal median",
    )
    ax.plot(
        freq,
        lognorm_mode,
        color="tab:red",
        linestyle="-",
        linewidth=2.0,
        label="lognormal mode",
    )

    if lognorm_plus1 is not None and lognorm_minus1 is not None:
        ax.plot(
            freq,
            lognorm_plus1,
            color="tab:red",
            linestyle=":",
            linewidth=1.8,
            label="lognormal median ±1σ",
        )
        ax.plot(freq, lognorm_minus1, color="tab:red", linestyle=":", linewidth=1.8)

    ax.plot(
        freq,
        kde_mode_s,
        color="b",
        linewidth=2.5,
        label=rf"KDE mode (smoothed, B={display_bandwidth:g})",
    )
    ax.plot(
        freq,
        kde_hi_s,
        color="b",
        linestyle=":",
        linewidth=1.8,
        label=rf"KDE HDI ({int(alpha * 100)}%) (smoothed, B={display_bandwidth:g})",
    )
    ax.plot(freq, kde_lo_s, color="b", linestyle=":", linewidth=1.8)

    if R_E_all is not None and plot_re_all:
        R_E_all = np.asarray(R_E_all, dtype=float)
        if R_E_all.shape != freq.shape:
            raise ValueError("R_E_all must have the same shape as hvsr.frequency.")
        ax.plot(
            freq,
            R_E_all,
            color="0.55",
            linestyle="-",
            linewidth=2.2,
            alpha=0.8,
            label=(r"$R_E^{\rm all}=\sqrt{\langle H\rangle_{\rm all}/"
                   r"\langle V\rangle_{\rm all}}$"),
        )

    if R_E_acc is not None:
        R_E_acc = np.asarray(R_E_acc, dtype=float)
        if R_E_acc.shape != freq.shape:
            raise ValueError("R_E_acc must have the same shape as hvsr.frequency.")
        ax.plot(
            freq,
            R_E_acc,
            color="k",
            linestyle="-",
            linewidth=2.2,
            alpha=0.85,
            label=(r"$R_E^{\rm acc}=\sqrt{\langle H\rangle_{\rm acc}/"
                   r"\langle V\rangle_{\rm acc}}$"),
        )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("HVSR amplitude [-]")
    ax.set_title("Rotmoos example: HVSR KDE density vs. lognormal stats")
    ax.grid(True, which="both", ls=":")
    ax.set_ylim(top=12.0)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    save_or_show(fig, output_path, dpi=dpi, show=show)
    return freq, kde_mode_s


def plot_fig5a_dphi(
    freq: np.ndarray,
    dphi: np.ndarray,
    output_path: Path,
    threshold_deg: float,
    alpha: float,
    dpi: int,
    show: bool,
) -> None:
    """Create Fig. 5a: angular spread Δφ_α(f) in degrees."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freq, np.rad2deg(dphi), label=rf"$\Delta \varphi_{{{alpha:.2f}}}(f)$")
    ax.axhline(threshold_deg, color="r", linestyle="--", label=f"threshold {threshold_deg:.0f}°")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(rf"Angular spread $\Delta \varphi_{{{alpha:.2f}}}$ [deg]")
    ax.set_xlim(float(np.nanmin(freq)), float(np.nanmax(freq)))
    ax.set_ylim(0, 180)
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    save_or_show(fig, output_path, dpi=dpi, show=show)


def plot_fig5b_envelope(
    freq: np.ndarray,
    kde_hvsr_mode_s: np.ndarray,
    Rmin: np.ndarray,
    Rmax: np.ndarray,
    dphi: np.ndarray,
    output_path: Path,
    threshold_deg: float,
    display_bandwidth: float,
    dpi: int,
    show: bool,
) -> None:
    """Create Fig. 5b: directional envelopes and directionally constrained band."""
    Rmin_s = konno_ohmachi_smooth_curve(freq, Rmin, B=display_bandwidth)
    Rmax_s = konno_ohmachi_smooth_curve(freq, Rmax, B=display_bandwidth)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("HVSR amplitude [-]")

    ax.plot(
        freq,
        kde_hvsr_mode_s,
        color="tab:blue",
        linewidth=2.0,
        label=rf"KDE HVSR mode (smoothed, B={display_bandwidth:g})",
    )
    ax.plot(
        freq,
        Rmin_s,
        color="tab:orange",
        linestyle="--",
        label=rf"KDE $\sqrt{{2H_{{\min}}/V}}$ mode (smoothed, B={display_bandwidth:g})",
    )
    ax.plot(
        freq,
        Rmax_s,
        color="tab:orange",
        linestyle="-",
        label=rf"KDE $\sqrt{{2H_{{\max}}/V}}$ mode (smoothed, B={display_bandwidth:g})",
    )

    mask = np.isfinite(dphi) & (np.rad2deg(dphi) < threshold_deg)
    ax.fill_between(
        freq,
        Rmin_s,
        Rmax_s,
        where=mask,
        color="tab:orange",
        alpha=0.2,
        label="directionally constrained band",
    )

    ax.grid(True, which="both", ls=":")
    ax.set_xlim(float(np.nanmin(freq)), float(np.nanmax(freq)))
    ax.set_ylim(0, 8)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    save_or_show(fig, output_path, dpi=dpi, show=show)


def plot_fig5c_rmin_rmax_kde(
    freq: np.ndarray,
    Rmin_all: np.ndarray,
    Rmax_all: np.ndarray,
    output_path: Path,
    alpha: float,
    display_bandwidth: float,
    dpi: int,
    show: bool,
) -> None:
    """Create Fig. 5c: 2D KDE density maps for directional envelopes."""
    f1, amp1, dens1, mode1, _, _ = compute_r_kde_density(freq, Rmin_all, alpha=alpha, n_amp=200)
    f2, amp2, dens2, mode2, _, _ = compute_r_kde_density(freq, Rmax_all, alpha=alpha, n_amp=200)

    mode1_s = konno_ohmachi_smooth_curve(f1, mode1, B=display_bandwidth)
    mode2_s = konno_ohmachi_smooth_curve(f2, mode2, B=display_bandwidth)

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
        ax.plot(
            f1[m1],
            mode1_s[m1],
            color="darkorange",
            linewidth=3.0,
            label=rf"KDE $\sqrt{{2H_{{\min}}/V}}$ mode (smoothed, B={display_bandwidth:g})",
        )
    if np.any(m2):
        ax.plot(
            f2[m2],
            mode2_s[m2],
            color="darkblue",
            linewidth=3.0,
            label=rf"KDE $\sqrt{{2H_{{\max}}/V}}$ mode (smoothed, B={display_bandwidth:g})",
        )

    ax.set_title(
        r"Directional HVSR: KDE density of "
        r"$\sqrt{2H_{\min}/V}$ and $\sqrt{2H_{\max}/V}$"
    )
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    save_or_show(fig, output_path, dpi=dpi, show=show)


def interpolate_to_grid(
    freq_source: np.ndarray,
    values: np.ndarray,
    freq_target: np.ndarray,
    name: str,
) -> np.ndarray:
    """Interpolate a curve between frequency grids when needed."""
    if np.array_equal(freq_source, freq_target) or np.allclose(freq_source, freq_target):
        return values
    LOGGER.warning("Frequency grid mismatch for %s; interpolating.", name)
    return np.interp(freq_target, freq_source, values)


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    configure_logging(args.verbose)

    input_file = resolve_input_path(args.input)
    output_dir = args.output_dir
    center_frequencies = build_center_frequencies(args.fmin, args.fmax, args.n_freq)

    LOGGER.info("Reading %s", input_file)
    records_pp = load_and_preprocess_hvsr(input_file, window_length=args.window_length)
    hvsr = compute_hvsr_traditional(
        records_pp,
        center_frequencies=center_frequencies,
        bandwidth=args.hvsr_bandwidth,
    )

    freq_spec, Sxx, Syy, Szz, Cxy = compute_spectra_B40(
        input_file,
        args.window_length,
        center_frequencies,
    )

    R_all_for_RE_mask = compute_windowwise_ratio_from_spectra(
        Sxx,
        Syy,
        Szz,
        horizontal_divisor=args.horizontal_divisor,
    )
    R_E_all = compute_energy_ratio_estimator_from_spectra(
        Sxx,
        Syy,
        Szz,
        valid_mask=None,
        horizontal_divisor=args.horizontal_divisor,
    )

    if args.disable_re_rejection:
        re_mask = None
        LOGGER.info("R_E-only rejection disabled; R_E_acc equals R_E_all.")
    else:
        re_mask = rejection_mask_for_RE_from_ratios(
            freq_spec,
            R_all_for_RE_mask,
            search_range_hz=(args.re_fmin, args.re_fmax),
            n_sigma=args.re_n_sigma,
            max_iterations=args.re_max_iterations,
            score_percentile=args.re_score_percentile,
        )

    R_E_acc = compute_energy_ratio_estimator_from_spectra(
        Sxx,
        Syy,
        Szz,
        valid_mask=re_mask,
        horizontal_divisor=args.horizontal_divisor,
    )
    log_RE_diagnostics(R_all_for_RE_mask, R_E_all, R_E_acc)

    freq_hvsr = np.asarray(hvsr.frequency, dtype=float)
    R_E_acc_for_hvsr = interpolate_to_grid(freq_spec, R_E_acc, freq_hvsr, "R_E_acc")
    R_E_all_for_hvsr = interpolate_to_grid(freq_spec, R_E_all, freq_hvsr, "R_E_all")

    freq_hvsr, kde_hvsr_mode_s = plot_fig3_kde_vs_lognormal(
        hvsr,
        output_path=output_dir / FIG3_NAME,
        alpha=args.alpha,
        kde_c=args.kde_c,
        display_bandwidth=args.display_bandwidth,
        dpi=args.dpi,
        show=args.show,
        R_E_acc=R_E_acc_for_hvsr,
        R_E_all=R_E_all_for_hvsr,
        plot_re_all=not args.no_plot_re_all,
    )

    stats = compute_directional_stats_vonmises(
        freq_spec,
        Sxx,
        Syy,
        Szz,
        Cxy,
        alpha=args.alpha,
        n_grid=720,
    )
    plot_fig5a_dphi(
        freq_spec,
        stats["dphi"],
        output_path=output_dir / FIG5A_NAME,
        threshold_deg=args.dphi_threshold_deg,
        alpha=args.alpha,
        dpi=args.dpi,
        show=args.show,
    )

    kde_mode_on_spec_grid = interpolate_to_grid(
        freq_hvsr, kde_hvsr_mode_s, freq_spec, "KDE HVSR mode"
    )
    plot_fig5b_envelope(
        freq_spec,
        kde_mode_on_spec_grid,
        stats["Rmin"],
        stats["Rmax"],
        stats["dphi"],
        output_path=output_dir / FIG5B_NAME,
        threshold_deg=args.dphi_threshold_deg,
        display_bandwidth=args.display_bandwidth,
        dpi=args.dpi,
        show=args.show,
    )
    plot_fig5c_rmin_rmax_kde(
        freq_spec,
        stats["Rmin_all"],
        stats["Rmax_all"],
        output_path=output_dir / FIG5C_NAME,
        alpha=args.alpha,
        display_bandwidth=args.display_bandwidth,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
