"""Create Fig. 4: bandwidth sensitivity of KDE-mode and HDI descriptors.

The KDE bandwidth is scaled as

    h_c(f) = c h_S(f),

where h_S is Silverman's normal-reference bandwidth in the log-amplitude domain.
The baseline c=1.0 corresponds to h_S.

By default, the script looks for the example MiniSEED files in the current
working directory, next to this script, or in an ``examples/`` subdirectory. Use
``--ut-input`` and ``--rotmoos-input`` to specify other files.

Examples
--------
Run with repository-default paths::

    python Fig4_bandwidth_sensitivity.py

Run with explicit data files and output directory::

    python Fig4_bandwidth_sensitivity.py \
        --ut-input examples/UT.STN11.A2_C300.miniseed \
        --rotmoos-input examples/reftek_3C.mseed \
        --output-dir figures

Show detailed progress messages::

    python Fig4_bandwidth_sensitivity.py --verbose
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
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
    konno_ohmachi_smooth_curve,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_UT_FILE = "UT.STN11.A2_C300.miniseed"
DEFAULT_ROTM_FILE = "reftek_3C.mseed"
DEFAULT_ALPHA = 0.68
DEFAULT_KDE_C_VALUES = (0.5, 1.0, 2.0)
DEFAULT_DISPLAY_BANDWIDTH = 80.0
DEFAULT_HVSR_BANDWIDTH = 40.0
DEFAULT_OUTPUT_STEM = "fig4_bandwidth_sensitivity"

# --------------------------------------------------------------------------- #
# Figure typography
# --------------------------------------------------------------------------- #

FIGURE_SIZE = (13.5, 6.0)
FONT_SIZE = 13
AXIS_LABEL_SIZE = 14
TITLE_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 11
PANEL_LABEL_SIZE = 15


@dataclass(frozen=True)
class CaseConfig:
    """Configuration for one bandwidth-sensitivity example case."""

    name: str
    filename: Path
    window_length: float
    center_frequencies: np.ndarray
    ymax: float
    xlim: tuple[float, float]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create bandwidth-sensitivity plots for KDE-mode and HDI descriptors."
    )
    parser.add_argument(
        "--ut-input",
        type=Path,
        default=Path(DEFAULT_UT_FILE),
        help="Input MiniSEED file for the UT.STN11.A2_C300 example.",
    )
    parser.add_argument(
        "--rotmoos-input",
        type=Path,
        default=Path(DEFAULT_ROTM_FILE),
        help="Input MiniSEED file for the Rotmoos example.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where the figure files will be written.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help=f"Output filename without extension. Default: {DEFAULT_OUTPUT_STEM}",
    )
    parser.add_argument(
        "--kde-c-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_KDE_C_VALUES),
        help="KDE bandwidth multipliers to compare. Include 1.0 as the reference.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="HDI probability mass.",
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
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG figure resolution in dots per inch.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    """Configure command-line logging.

    By default only warnings and errors are printed. This suppresses the many
    INFO messages produced by hvsrpy during window creation. Use --verbose to
    show progress messages from this script and imported packages.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
        force=True,
    )

    if not verbose:
        logging.getLogger("hvsrpy").setLevel(logging.WARNING)


def configure_matplotlib() -> None:
    """Set global Matplotlib font sizes for this figure."""
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
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
        "Specify the file explicitly with --ut-input or --rotmoos-input."
    )


def validate_kde_c_values(c_values: list[float]) -> tuple[float, ...]:
    """Validate KDE bandwidth multipliers and return them as a sorted tuple."""
    clean = tuple(sorted(float(c) for c in c_values))
    if not clean:
        raise ValueError("At least one KDE bandwidth multiplier must be provided.")
    if any((not np.isfinite(c) or c <= 0.0) for c in clean):
        raise ValueError(f"All KDE bandwidth multipliers must be finite and positive: {clean!r}.")
    if 1.0 not in clean:
        raise ValueError("The reference bandwidth multiplier 1.0 must be included.")
    return clean


def make_cases(ut_file: Path, rotmoos_file: Path) -> tuple[CaseConfig, ...]:
    """Create the case configuration objects."""
    return (
        CaseConfig(
            name="UT.STN11.A2_C300",
            filename=ut_file,
            window_length=30.0,
            center_frequencies=np.geomspace(0.3, 2.0, 1024),
            ymax=12.0,
            xlim=(0.3, 2.0),
        ),
        CaseConfig(
            name="Rotmoos P1 475",
            filename=rotmoos_file,
            window_length=3.0,
            center_frequencies=np.geomspace(5.0, 150.0, 256),
            ymax=9.0,
            xlim=(5.0, 150.0),
        ),
    )


def load_and_preprocess_hvsr(case: CaseConfig) -> Any:
    """Read and preprocess a 3C record for hvsrpy."""
    records = read([str(case.filename)])
    pre_settings = HvsrPreProcessingSettings(
        window_length_in_seconds=float(case.window_length),
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


def compute_bandwidth_sensitivity(
    hvsr: Any,
    c_values: tuple[float, ...],
    alpha: float,
    display_bandwidth: float,
) -> dict[float, dict[str, np.ndarray]]:
    """Compute KDE-mode and HDI curves for multiple Silverman bandwidth factors."""
    results: dict[float, dict[str, np.ndarray]] = {}

    for c in c_values:
        freq, _amp_grid, _density, mode, lo, hi = compute_hvsr_kde_density(
            hvsr,
            alpha=alpha,
            c=float(c),
            n_amp=200,
        )
        results[float(c)] = {
            "freq": np.asarray(freq, dtype=float),
            "mode": np.asarray(mode, dtype=float),
            "lo": np.asarray(lo, dtype=float),
            "hi": np.asarray(hi, dtype=float),
            "mode_s": np.asarray(
                konno_ohmachi_smooth_curve(freq, mode, B=display_bandwidth),
                dtype=float,
            ),
            "lo_s": np.asarray(
                konno_ohmachi_smooth_curve(freq, lo, B=display_bandwidth),
                dtype=float,
            ),
            "hi_s": np.asarray(
                konno_ohmachi_smooth_curve(freq, hi, B=display_bandwidth),
                dtype=float,
            ),
        }

    return results


def label_for_c(c: float) -> str:
    """Return a concise Matplotlib label for a KDE bandwidth multiplier."""
    if np.isclose(c, 1.0):
        return r"$h_S$"
    return rf"${c:g}h_S$"


def plot_case_sensitivity(
    ax_curve: plt.Axes,
    ax_diff: plt.Axes,
    case: CaseConfig,
    results: dict[float, dict[str, np.ndarray]],
    alpha: float,
) -> None:
    """Plot one case: KDE-mode sensitivity and relative difference."""
    c_ref = 1.0
    if c_ref not in results:
        raise ValueError("The reference bandwidth c=1.0 must be included.")

    ref = results[c_ref]
    freq = ref["freq"]
    mode_ref = ref["mode_s"]

    ax_curve.fill_between(
        freq,
        ref["lo_s"],
        ref["hi_s"],
        color="0.85",
        alpha=0.8,
        label=rf"{int(alpha * 100)}% HDI for $h_S$",
        zorder=1,
    )

    linestyles = {0.5: "-", 1.0: "-", 2.0: "-"}
    colors = {0.5: "r", 1.0: "k", 2.0: "b"}
    linewidths = {0.5: 2.0, 1.0: 2.8, 2.0: 2.0}

    for c in sorted(results):
        r = results[c]
        ax_curve.plot(
            r["freq"],
            r["mode_s"],
            linestyle=linestyles.get(c, "-"),
            color=colors.get(c, "k"),
            linewidth=linewidths.get(c, 2.0),
            label=rf"KDE mode, {label_for_c(c)}",
            zorder=3,
        )

    ax_curve.set_xscale("log")
    ax_curve.set_xlim(case.xlim)
    ax_curve.set_ylim(top=case.ymax)
    ax_curve.set_ylabel("HVSR amplitude [-]", fontsize=AXIS_LABEL_SIZE)
    ax_curve.set_title(case.name, fontsize=TITLE_SIZE)
    ax_curve.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax_curve.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE - 1)
    ax_curve.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_curve.legend(loc="upper right", framealpha=0.9, fontsize=LEGEND_SIZE)

    valid_ref = np.isfinite(mode_ref) & (mode_ref > 0.0)
    for c in sorted(results):
        if np.isclose(c, c_ref):
            continue

        r = results[c]
        if not np.allclose(r["freq"], freq):
            mode_c = np.interp(freq, r["freq"], r["mode_s"])
        else:
            mode_c = r["mode_s"]

        valid = valid_ref & np.isfinite(mode_c)
        rel_diff = np.full_like(freq, np.nan, dtype=float)
        rel_diff[valid] = 100.0 * (mode_c[valid] / mode_ref[valid] - 1.0)

        ax_diff.plot(
            freq,
            rel_diff,
            linestyle=linestyles.get(c, "-"),
            color=colors.get(c, "k"),
            linewidth=2.0,
            label=rf"{label_for_c(c)} relative to $h_S$",
        )

    ax_diff.axhline(0.0, color="k", linewidth=0.8)
    ax_diff.set_xscale("log")
    ax_diff.set_xlim(case.xlim)
    ax_diff.set_ylim(-100, 100)
    ax_diff.set_xlabel("Frequency [Hz]", fontsize=AXIS_LABEL_SIZE)
    ax_diff.set_ylabel("Relative difference [%]", fontsize=AXIS_LABEL_SIZE)
    ax_diff.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax_diff.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE - 1)
    ax_diff.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_diff.legend(loc="upper right", framealpha=0.9, fontsize=LEGEND_SIZE)


def make_figure(
    all_results: dict[str, tuple[CaseConfig, dict[float, dict[str, np.ndarray]]]],
    alpha: float,
) -> plt.Figure:
    """Create Fig. 4 as a 2-by-2 panel."""
    configure_matplotlib()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=FIGURE_SIZE,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    for col, case_name in enumerate(all_results):
        case, results = all_results[case_name]
        plot_case_sensitivity(
            ax_curve=axes[0, col],
            ax_diff=axes[1, col],
            case=case,
            results=results,
            alpha=alpha,
        )

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.ravel()):
        ax.text(
            0.02,
            0.95,
            label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            fontsize=PANEL_LABEL_SIZE,
        )

    return fig


def save_outputs(
    fig: plt.Figure,
    output_dir: Path,
    output_stem: str,
    dpi: int,
    show: bool,
) -> list[Path]:
    """Save Fig. 4 as PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_stem}.png"
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

    c_values = validate_kde_c_values(args.kde_c_values)
    cases = make_cases(
        ut_file=resolve_input_path(args.ut_input),
        rotmoos_file=resolve_input_path(args.rotmoos_input),
    )

    all_results: dict[str, tuple[CaseConfig, dict[float, dict[str, np.ndarray]]]] = {}
    for case in cases:
        LOGGER.info("Processing %s from %s", case.name, case.filename)
        records_pp = load_and_preprocess_hvsr(case)
        hvsr = compute_hvsr_traditional(
            records_pp,
            center_frequencies=case.center_frequencies,
            bandwidth=args.hvsr_bandwidth,
        )
        results = compute_bandwidth_sensitivity(
            hvsr,
            c_values=c_values,
            alpha=args.alpha,
            display_bandwidth=args.display_bandwidth,
        )
        all_results[case.name] = (case, results)

    fig = make_figure(all_results, alpha=args.alpha)
    save_outputs(fig, args.output_dir, args.output_stem, dpi=args.dpi, show=args.show)


if __name__ == "__main__":
    main()
