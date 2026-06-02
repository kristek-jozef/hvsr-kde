"""Create Fig. 2: KDE-mode/HDI HVSR statistics versus lognormal statistics.

The script compares hvsrpy's lognormal summary curves with a KDE-mode and
highest-density interval (HDI) representation for the UT.STN11.A2_C300 example
record.

By default, the script looks for ``UT.STN11.A2_C300.miniseed`` in the current
working directory, next to this script, or in an ``examples/`` subdirectory. Use
``--input`` to specify another file.

Examples
--------
Run with repository-default paths::

    python Fig2.py

Run with explicit input and output directory::

    python Fig2.py --input examples/UT.STN11.A2_C300.miniseed --output-dir figures
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

from kde_utils import (
    compute_hvsr_kde_density,
    compute_spectra_B40,
    konno_ohmachi_smooth_curve,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_FILE = "UT.STN11.A2_C300.miniseed"
DEFAULT_OUTPUT_NAME = "Fig2_kde_lognormal_ut.png"
DEFAULT_WINDOW_LENGTH = 30.0
DEFAULT_FMIN = 0.3
DEFAULT_FMAX = 2.0
DEFAULT_N_FREQ = 1024
DEFAULT_HVSR_BANDWIDTH = 40.0
DEFAULT_DISPLAY_BANDWIDTH = 80.0
DEFAULT_HORIZONTAL_DIVISOR = 1.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create Fig. 2: HVSR KDE density versus lognormal statistics."
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
        help="Directory where the figure will be written.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output figure filename. Default: {DEFAULT_OUTPUT_NAME}",
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
    parser.add_argument("--alpha", type=float, default=0.68, help="HDI probability mass.")
    parser.add_argument(
        "--kde-c",
        type=float,
        default=1.0,
        help="Multiplier for Silverman's KDE bandwidth in log-amplitude space.",
    )
    parser.add_argument(
        "--horizontal-divisor",
        type=float,
        default=DEFAULT_HORIZONTAL_DIVISOR,
        help=(
            "Horizontal energy divisor for R_E: 1 gives H=Sxx+Syy, "
            "2 gives H=(Sxx+Syy)/2."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution in dots per inch.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
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


def load_and_preprocess(filename: Path, window_length: float) -> Any:
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


def compute_energy_ratio_estimator(
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


def hvsr_valid_mask(hvsr: Any, n_windows: int) -> np.ndarray | None:
    """Return hvsrpy's valid-window mask if it matches the spectra window count."""
    valid_mask = getattr(hvsr, "valid_curve_boolean_mask", None)
    if valid_mask is None:
        return None

    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.size != n_windows:
        LOGGER.warning(
            "valid_curve_boolean_mask size (%s) differs from spectra windows (%s); "
            "using all windows for R_E.",
            valid_mask.size,
            n_windows,
        )
        return None
    return valid_mask


def lognormal_mode_from_hvsr(hvsr: Any) -> np.ndarray:
    """Compute the lognormal mode from hvsrpy's window-wise amplitudes."""
    amplitudes = np.asarray(hvsr.amplitude, dtype=float)
    valid_mask = getattr(hvsr, "valid_curve_boolean_mask", None)
    if valid_mask is not None:
        amplitudes = amplitudes[np.asarray(valid_mask, dtype=bool), :]

    amplitudes = np.where(
        np.isfinite(amplitudes) & (amplitudes > 0.0),
        amplitudes,
        np.nan,
    )
    log_amplitudes = np.log(amplitudes)
    mu = np.nanmean(log_amplitudes, axis=0)
    sigma = np.nanstd(log_amplitudes, axis=0, ddof=1)
    return np.exp(mu - sigma**2)


def interpolate_to_hvsr_grid(
    freq_source: np.ndarray,
    values: np.ndarray,
    freq_target: np.ndarray,
) -> np.ndarray:
    """Interpolate values to the hvsrpy frequency grid when grids differ."""
    if np.array_equal(freq_source, freq_target) or np.allclose(freq_source, freq_target):
        return values
    LOGGER.warning(
        "Spectra and HVSR frequency grids differ; interpolating R_E to the HVSR grid."
    )
    return np.interp(freq_target, freq_source, values)


def plot_fig2(
    hvsr: Any,
    output_path: Path,
    alpha: float,
    kde_c: float,
    display_bandwidth: float,
    R_E: np.ndarray | None = None,
    dpi: int = 300,
    show: bool = False,
) -> Path:
    """Create and save the Fig. 2 KDE-versus-lognormal plot."""
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

    if R_E is not None:
        R_E = np.asarray(R_E, dtype=float)
        if R_E.shape != freq.shape:
            raise ValueError("R_E must have the same shape as hvsr.frequency.")
        ax.plot(
            freq,
            R_E,
            color="k",
            linestyle="-",
            linewidth=2.2,
            label=r"$R_E=\sqrt{\langle H\rangle/\langle V\rangle}$",
        )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("HVSR amplitude [-]")
    ax.set_title("UT STN11 example: HVSR KDE density vs. lognormal stats")
    ax.grid(True, which="both", ls=":")
    ax.set_ylim(top=12.0)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    LOGGER.info("Wrote %s", output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    configure_logging(args.verbose)

    input_file = resolve_input_path(args.input)
    output_path = args.output_dir / args.output_name
    center_frequencies = build_center_frequencies(args.fmin, args.fmax, args.n_freq)

    LOGGER.info("Reading %s", input_file)
    records_pp = load_and_preprocess(input_file, window_length=args.window_length)
    hvsr = compute_hvsr_traditional(
        records_pp,
        center_frequencies=center_frequencies,
        bandwidth=args.hvsr_bandwidth,
    )

    freq_spec, Sxx, Syy, Szz, _ = compute_spectra_B40(
        input_file,
        args.window_length,
        center_frequencies,
    )
    valid_mask = hvsr_valid_mask(hvsr, n_windows=Sxx.shape[0])
    R_E = compute_energy_ratio_estimator(
        Sxx,
        Syy,
        Szz,
        valid_mask=valid_mask,
        horizontal_divisor=args.horizontal_divisor,
    )
    R_E_for_hvsr = interpolate_to_hvsr_grid(
        np.asarray(freq_spec, dtype=float),
        R_E,
        np.asarray(hvsr.frequency, dtype=float),
    )

    plot_fig2(
        hvsr,
        output_path=output_path,
        alpha=args.alpha,
        kde_c=args.kde_c,
        display_bandwidth=args.display_bandwidth,
        R_E=R_E_for_hvsr,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
