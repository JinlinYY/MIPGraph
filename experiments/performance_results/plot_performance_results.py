"""Create the audited nine-panel MIPGraph performance figure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

PROPERTY_ORDER = [
    "Density",
    "Viscosity",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
]
PROPERTY_NAMES = {
    "Density": "Density",
    "Viscosity": "Viscosity",
    "ElectricalConductivity": "Electrical conductivity",
    "HeatCapacity": "Heat capacity",
    "SurfaceTension": "Surface tension",
    "ThermalConductivity": "Thermal conductivity",
}
PROPERTY_SHORT = {
    "Density": "Density",
    "Viscosity": "Visc.",
    "ElectricalConductivity": "EC",
    "HeatCapacity": "HC",
    "SurfaceTension": "ST",
    "ThermalConductivity": "TC",
}
PROPERTY_UNITS = {
    "Density": r"kg m$^{-3}$",
    "Viscosity": "Pa s",
    "ElectricalConductivity": r"S m$^{-1}$",
    "HeatCapacity": r"J K$^{-1}$ mol$^{-1}$",
    "SurfaceTension": r"N m$^{-1}$",
    "ThermalConductivity": r"W m$^{-1}$ K$^{-1}$",
}
LOG_AXIS_PROPERTIES = {"ElectricalConductivity", "Viscosity"}
Function = TypeVar("Function", bound=Callable[..., object])


def close_new_figures_on_error(function: Function) -> Function:
    """Close only figures created by ``function`` when plotting fails."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        existing_figures = set(plt.get_fignums())
        try:
            return function(*args, **kwargs)
        except Exception:
            new_figures = set(plt.get_fignums()) - existing_figures
            for figure_number in new_figures:
                plt.close(figure_number)
            raise

    return wrapped


def configure_style() -> None:
    """Apply journal-safe typography and vector-font settings."""

    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    font_family = "Arial" if "Arial" in available else "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": [font_family, "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.1,
            "legend.frameon": False,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.1,
            "ytick.major.size": 2.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def style_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    """Apply the shared lightweight axis treatment."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#3F3F3F")
    ax.spines["bottom"].set_color("#3F3F3F")
    ax.tick_params(direction="out", pad=1.4)
    ax.grid(True, axis=grid_axis, color="#E5E8EA", linewidth=0.35)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    """Place a manuscript panel label in axes coordinates."""

    ax.text(
        -0.15,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        fontweight="bold",
    )


def nice_linear_limits(
    values: np.ndarray,
    pad: float = 0.06,
) -> tuple[float, float]:
    """Return padded finite limits for a linear parity plot."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot determine linear limits from empty finite data.")
    low, high = float(finite.min()), float(finite.max())
    span = high - low
    if span <= 0:
        span = max(abs(high), 1.0)
    return low - pad * span, high + pad * span


def nice_log_limits(values: np.ndarray) -> tuple[float, float]:
    """Return decade limits for positive finite values."""

    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        raise ValueError("Cannot determine log limits without positive finite data.")
    low = 10 ** np.floor(np.log10(float(finite.min())))
    high = 10 ** np.ceil(np.log10(float(finite.max())))
    return low, high


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = (
    REPO_ROOT
    / "experiments"
    / "result_analysis"
    / "figures"
    / "performance_results"
)

CURRENT_BEST_INPUT_ROOT = (
    REPO_ROOT
    / "il_property_prediction"
    / "outputs"
    / "mps_weak_merged_validation"
    / "figure_inputs"
)

SAVED_SOURCE_DATA_DIR = (
    REPO_ROOT
    / "experiments"
    / "manuscript_figure_source_data"
    / "performance_results"
)

SPLIT_SPECS = (
    ("random_point", "Random point", "Point", "Pt", "#3F5D7D"),
    ("random_il_level", "Random IL", "IL", "IL", "#7095BD"),
    (
        "property_balanced_il_level",
        "Balanced IL",
        "Balanced",
        "Bal",
        "#78A89C",
    ),
    ("ion_family", "Ion-family", "Family", "Fam", "#C18578"),
)


@dataclass(frozen=True)
class SplitCase:
    """Display metadata and staged paths for one split protocol."""

    key: str
    name: str
    short: str
    abbreviation: str
    color: str
    metrics_path: Path
    predictions_path: Path


def make_split_cases(input_root: Path) -> tuple[SplitCase, ...]:
    return tuple(
        SplitCase(
            key=key,
            name=name,
            short=short,
            abbreviation=abbr,
            color=color,
            metrics_path=input_root / key / "test_metrics_log.csv",
            predictions_path=input_root / key / "test_predictions.csv",
        )
        for key, name, short, abbr, color in SPLIT_SPECS
    )


def _split_names(split_cases: tuple[SplitCase, ...]) -> list[str]:
    return [case.name for case in split_cases]


def _split_shorts(split_cases: tuple[SplitCase, ...]) -> list[str]:
    return [case.short for case in split_cases]


def _split_colors(split_cases: tuple[SplitCase, ...]) -> list[str]:
    return [case.color for case in split_cases]


def load_split_metrics(split_cases: tuple[SplitCase, ...]) -> pd.DataFrame:
    frames = []
    missing = []
    for case in split_cases:
        path = case.metrics_path
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        frame["split_strategy"] = case.name
        frame["split_short"] = case.short
        frames.append(frame)
    if missing:
        raise FileNotFoundError("Missing split metrics:\n" + "\n".join(missing))
    return pd.concat(frames, ignore_index=True)


def load_split_predictions(split_cases: tuple[SplitCase, ...]) -> pd.DataFrame:
    frames = []
    missing = []
    for case in split_cases:
        path = case.predictions_path
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        frame["split_strategy"] = case.name
        frame["split_short"] = case.short
        frames.append(frame)
    if missing:
        raise FileNotFoundError("Missing split predictions:\n" + "\n".join(missing))
    return pd.concat(frames, ignore_index=True)


def plot_multi_split_parity(
    ax: plt.Axes,
    predictions: pd.DataFrame,
    split_metrics: pd.DataFrame,
    split_cases: tuple[SplitCase, ...],
    prop: str,
    label: str,
    *,
    compact: bool = False,
) -> pd.DataFrame:
    data = predictions[predictions["property"] == prop].copy()
    plotted_all = data.copy()
    if prop in LOG_AXIS_PROPERTIES:
        plotted_all = plotted_all[(plotted_all["y_true"] > 0) & (plotted_all["y_pred"] > 0)].copy()
    values = np.concatenate([plotted_all["y_true"].to_numpy(), plotted_all["y_pred"].to_numpy()])
    if prop in LOG_AXIS_PROPERTIES:
        lo, hi = nice_log_limits(values)
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        lo, hi = nice_linear_limits(values)

    for case in split_cases:
        split_data = data[data["split_strategy"] == case.name].copy()
        plotted = split_data.copy()
        if prop in LOG_AXIS_PROPERTIES:
            plotted = plotted[(plotted["y_true"] > 0) & (plotted["y_pred"] > 0)].copy()
        ax.scatter(
            plotted["y_true"],
            plotted["y_pred"],
            s=4.6 if compact else 6.4,
            color=case.color,
            alpha=0.25,
            edgecolors="none",
            rasterized=True,
            label=case.short,
        )

    ax.plot([lo, hi], [lo, hi], color="#2F3437", linewidth=0.65, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if compact:
        compact_titles = {
            "Density": "Density",
            "Viscosity": "Viscosity",
            "ElectricalConductivity": "Electrical\nconductivity",
            "HeatCapacity": "Heat capacity",
            "SurfaceTension": "Surface tension",
            "ThermalConductivity": "Thermal\nconductivity",
        }
        ax.set_title(
            compact_titles[prop] + f"\n({PROPERTY_UNITS[prop]})",
            fontsize=7.0,
            linespacing=1.02,
            pad=2.2,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_title(PROPERTY_NAMES[prop])
        ax.set_xlabel(f"Experimental ({PROPERTY_UNITS[prop]})")
        ax.set_ylabel(f"Predicted ({PROPERTY_UNITS[prop]})")

    metric_lines = [r"$R^2$/NMAE"]
    for case in split_cases:
        row = split_metrics[
            (split_metrics["property"] == prop)
            & (split_metrics["split_strategy"] == case.name)
        ].iloc[0]
        metric_lines.append(
            f"{case.abbreviation} {row['log_R2']:.2f}/{row['log_NMAE']:.2f}"
        )
    ax.text(
        0.045,
        0.955,
        "\n".join(metric_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.0 if compact else 5.5,
        family="monospace",
        color="#20252A",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.84),
    )
    style_axes(ax)
    if compact:
        ax.tick_params(labelsize=5.2, pad=1.0, length=1.8)
        ax.set_box_aspect(1)
        ax.text(
            -0.11,
            1.08,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    else:
        panel_label(ax, label)
    data["panel"] = label.upper()
    data["panel_title"] = f"{PROPERTY_NAMES[prop]} multi-split parity plot"
    data["plot_included"] = False
    data.loc[plotted_all.index, "plot_included"] = True
    return data


def plot_macro_split_summary(
    ax: plt.Axes,
    split_metrics: pd.DataFrame,
    split_cases: tuple[SplitCase, ...],
) -> pd.DataFrame:
    avg = split_metrics[split_metrics["property"] == "Average"].copy()
    avg["split_strategy"] = pd.Categorical(
        avg["split_strategy"],
        _split_names(split_cases),
        ordered=True,
    )
    avg = avg.sort_values("split_strategy")
    x = np.arange(len(avg))
    bars = ax.bar(
        x,
        avg["log_R2"],
        width=0.66,
        color=_split_colors(split_cases),
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_ylim(0.55, 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels(_split_shorts(split_cases), rotation=20, ha="right")
    ax.set_ylabel(r"macro log-space $R^2$")
    ax.set_title("Generalization across split protocols")
    for bar, (_, row) in zip(bars, avg.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(row["log_R2"]) + 0.012,
            f"{row['log_R2']:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.565,
            f"NMAE\n{row['log_NMAE']:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color="#30363D",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.82),
        )
    ax.axhline(float(avg.iloc[0]["log_R2"]), color="#30363D", linewidth=0.65, linestyle="--")
    ax.text(
        0.98,
        float(avg.iloc[0]["log_R2"]) + 0.006,
        "random point",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6.1,
        color="#30363D",
    )
    style_axes(ax, "y")
    panel_label(ax, "g")
    source = avg[["split_strategy", "log_MAE", "log_RMSE", "log_R2", "log_NMAE"]].copy()
    source["panel"] = "G"
    source["panel_title"] = "Generalization across split protocols"
    return source


def _heatmap_matrix(
    split_metrics: pd.DataFrame,
    split_cases: tuple[SplitCase, ...],
    metric: str,
) -> pd.DataFrame:
    data = split_metrics[split_metrics["property"].isin(PROPERTY_ORDER)].copy()
    data["property"] = pd.Categorical(data["property"], PROPERTY_ORDER, ordered=True)
    data["split_strategy"] = pd.Categorical(
        data["split_strategy"],
        _split_names(split_cases),
        ordered=True,
    )
    return data.pivot(index="split_strategy", columns="property", values=metric).loc[
        _split_names(split_cases),
        PROPERTY_ORDER,
    ]


def plot_split_heatmap(
    ax: plt.Axes,
    fig: plt.Figure,
    split_metrics: pd.DataFrame,
    split_cases: tuple[SplitCase, ...],
    metric: str,
    title: str,
    label: str,
    cmap,
    vmin: float,
    vmax: float,
    cbar_label: str,
    value_format: str = ".2f",
) -> pd.DataFrame:
    matrix = _heatmap_matrix(split_metrics, split_cases, metric)
    norm = Normalize(vmin=vmin, vmax=vmax)
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(PROPERTY_ORDER)))
    ax.set_xticklabels([PROPERTY_SHORT[prop] for prop in PROPERTY_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(_split_shorts(split_cases))
    ax.set_title(title)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(PROPERTY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, split_name in enumerate(matrix.index):
        for j, prop in enumerate(matrix.columns):
            value = float(matrix.loc[split_name, prop])
            red, green, blue, _ = cmap(norm(value))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(
                j,
                i,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=6.1,
                color="white" if luminance < 0.56 else "#26313B",
            )
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.046,
        pad=0.025,
    )
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=6.6)
    cbar.ax.tick_params(labelsize=6.1, length=2)
    panel_label(ax, label)
    source = matrix.reset_index().melt(
        id_vars="split_strategy",
        var_name="property",
        value_name=metric,
    )
    source["panel"] = label.upper()
    source["panel_title"] = title
    return source


def write_source_data(output_dir: Path, sources: dict[str, pd.DataFrame]) -> None:
    for suffix, frame in sources.items():
        frame.to_csv(output_dir / f"performance_results_source_data_{suffix}.csv", index=False)


def load_saved_source_data(
    source_data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the plotting inputs from the source tables of an existing figure."""

    parity_frames = [
        pd.read_csv(
            source_data_dir / f"performance_results_source_data_{panel}.csv"
        )
        for panel in "ABCDEF"
    ]
    predictions = pd.concat(parity_frames, ignore_index=True)

    r2 = pd.read_csv(
        source_data_dir / "performance_results_source_data_H.csv"
    )[["split_strategy", "property", "log_R2"]]
    nmae = pd.read_csv(
        source_data_dir / "performance_results_source_data_I.csv"
    )[["split_strategy", "property", "log_NMAE"]]
    property_metrics = r2.merge(
        nmae,
        on=["split_strategy", "property"],
        validate="one_to_one",
    )
    property_metrics["log_MAE"] = np.nan
    property_metrics["log_RMSE"] = np.nan

    average_metrics = pd.read_csv(
        source_data_dir / "performance_results_source_data_G.csv"
    )
    average_metrics["property"] = "Average"
    split_metrics = pd.concat(
        [
            property_metrics[
                [
                    "property",
                    "log_MAE",
                    "log_RMSE",
                    "log_R2",
                    "log_NMAE",
                    "split_strategy",
                ]
            ],
            average_metrics[
                [
                    "property",
                    "log_MAE",
                    "log_RMSE",
                    "log_R2",
                    "log_NMAE",
                    "split_strategy",
                ]
            ],
        ],
        ignore_index=True,
    )
    return predictions, split_metrics


@close_new_figures_on_error
def _create_figure_from_frames(
    pred_df: pd.DataFrame,
    split_metrics: pd.DataFrame,
    split_cases: tuple[SplitCase, ...],
    output_dir: Path,
    name: str,
    dpi: int,
    *,
    write_sources: bool,
) -> None:
    """Draw the 1-by-6 parity row and the three protocol-summary panels."""

    configure_style()
    mpl.rcParams.update({
        "font.size": 6.6,
        "axes.titlesize": 8.0,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 6.6,
    })

    # Double-column journal canvas: 183 mm wide.
    fig = plt.figure(figsize=(183.0 / 25.4, 96.0 / 25.4))
    gs = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.0, 1.28],
        hspace=0.40,
        wspace=0.62,
        left=0.055,
        right=0.960,
        top=0.900,
        bottom=0.110,
    )

    sources: dict[str, pd.DataFrame] = {}
    axes = [fig.add_subplot(gs[0, index]) for index in range(6)]
    for ax, prop, label in zip(axes, PROPERTY_ORDER, list("abcdef")):
        parity = plot_multi_split_parity(
            ax,
            pred_df,
            split_metrics,
            split_cases,
            prop,
            label,
            compact=True,
        )
        sources[label.upper()] = parity

    legend_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.6,
            color=case.color,
            label=case.short,
        )
        for case in split_cases
    ]
    row_top = max(axis.get_position().y1 for axis in axes)
    row_bottom = min(axis.get_position().y0 for axis in axes)
    shared_row_y = row_bottom - 0.048
    fig.text(
        0.385,
        shared_row_y,
        "Experimental value",
        ha="center",
        va="center",
        fontsize=6.8,
    )
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.495, shared_row_y),
        ncol=4,
        handletextpad=0.35,
        columnspacing=1.05,
        frameon=False,
        fontsize=6.8,
    )
    fig.text(
        0.018,
        0.5 * (row_top + row_bottom),
        "Predicted value",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.8,
    )

    ax_g = fig.add_subplot(gs[1, 0:2])
    ax_h = fig.add_subplot(gs[1, 2:4])
    ax_i = fig.add_subplot(gs[1, 4:6])

    sources["G"] = plot_macro_split_summary(ax_g, split_metrics, split_cases)
    r2_cmap = LinearSegmentedColormap.from_list(
        "r2_cmap",
        ["#F2F6F5", "#A8CCC4", "#356F68"],
    )
    nmae_cmap = LinearSegmentedColormap.from_list(
        "nmae_cmap",
        ["#FAF5F3", "#E8C3BC", "#B45F5F"],
    )
    sources["H"] = plot_split_heatmap(
        ax_h,
        fig,
        split_metrics,
        split_cases,
        "log_R2",
        r"Property-wise log-space $R^2$",
        "h",
        r2_cmap,
        0.25,
        1.0,
        "",
        ".2f",
    )
    sources["I"] = plot_split_heatmap(
        ax_i,
        fig,
        split_metrics,
        split_cases,
        "log_NMAE",
        "Property-wise log-space NMAE",
        "i",
        nmae_cmap,
        0.0,
        0.60,
        "",
        ".2f",
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "svg", "png", "tiff"):
            fig.savefig(
                output_dir / f"{name}.{extension}",
                bbox_inches=fig.bbox_inches,
                pad_inches=0,
                facecolor="white",
                dpi=dpi,
            )
        if write_sources:
            write_source_data(output_dir, sources)
    finally:
        plt.close(fig)


def create_figure(
    output_dir: Path,
    name: str,
    dpi: int,
    input_root: Path = CURRENT_BEST_INPUT_ROOT,
) -> None:
    split_cases = make_split_cases(input_root)
    pred_df = load_split_predictions(split_cases)
    split_metrics = load_split_metrics(split_cases)
    _create_figure_from_frames(
        pred_df,
        split_metrics,
        split_cases,
        output_dir,
        name,
        dpi,
        write_sources=True,
    )


def create_figure_from_saved_source(
    source_data_dir: Path,
    output_dir: Path,
    name: str,
    dpi: int,
) -> None:
    """Redraw an existing figure without refreshing any numerical results."""

    split_cases = make_split_cases(CURRENT_BEST_INPUT_ROOT)
    pred_df, split_metrics = load_saved_source_data(source_data_dir)
    _create_figure_from_frames(
        pred_df,
        split_metrics,
        split_cases,
        output_dir,
        name,
        dpi,
        write_sources=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the audited nine-panel performance figure."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Recompute the plotted source tables from staged split outputs. "
            "When omitted, the audited A--I source-data CSVs are used."
        ),
    )
    source_group.add_argument(
        "--source-data-dir",
        type=Path,
        default=None,
        help=(
            "Redraw from existing A--I source-data CSVs without refreshing "
            "any model predictions or metrics."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--name", default="performance_results")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_root is not None:
        create_figure(args.output_dir, args.name, args.dpi, args.input_root)
    else:
        source_data_dir = args.source_data_dir or SAVED_SOURCE_DATA_DIR
        create_figure_from_saved_source(
            source_data_dir,
            args.output_dir,
            args.name,
            args.dpi,
        )
    print(f"Wrote {args.output_dir / (args.name + '.png')}")


if __name__ == "__main__":
    main()
