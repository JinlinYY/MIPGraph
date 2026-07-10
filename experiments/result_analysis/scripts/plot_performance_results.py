"""Create final performance figure with test-set parity plots and metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

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
PROPERTY_COLORS = {
    "Density": "#4C78A8",
    "Viscosity": "#E15759",
    "ElectricalConductivity": "#59A14F",
    "HeatCapacity": "#F2BE5C",
    "SurfaceTension": "#B07AA1",
    "ThermalConductivity": "#76B7B2",
}
LOG_AXIS_PROPERTIES = {"ElectricalConductivity", "Viscosity"}
PANEL_LABELS = list("abcdefgh")


def configure_style() -> None:
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    font_family = "Arial" if "Arial" in available else "DejaVu Sans"
    mpl.rcParams.update({
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
    })


def style_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#3F3F3F")
    ax.spines["bottom"].set_color("#3F3F3F")
    ax.tick_params(direction="out", pad=1.4)
    ax.grid(True, axis=grid_axis, color="#E5E8EA", linewidth=0.35)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.15, 1.08, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.7, fontweight="bold")


def nice_linear_limits(values: np.ndarray, pad: float = 0.06) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    return lo - pad * span, hi + pad * span


def nice_log_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values) & (values > 0)]
    lo = 10 ** np.floor(np.log10(float(finite.min())))
    hi = 10 ** np.ceil(np.log10(float(finite.max())))
    return lo, hi


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def plot_parity(ax: plt.Axes,
                pred_df: pd.DataFrame,
                metrics: pd.DataFrame,
                prop: str,
                label: str) -> pd.DataFrame:
    color = PROPERTY_COLORS[prop]
    data = pred_df[pred_df["property"] == prop].copy()
    plotted = data.copy()
    if prop in LOG_AXIS_PROPERTIES:
        plotted = plotted[(plotted["y_true"] > 0) & (plotted["y_pred"] > 0)].copy()
    values = np.concatenate([plotted["y_true"].to_numpy(),
                             plotted["y_pred"].to_numpy()])
    if prop in LOG_AXIS_PROPERTIES:
        lo, hi = nice_log_limits(values)
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        lo, hi = nice_linear_limits(values)
    ax.scatter(plotted["y_true"], plotted["y_pred"], s=7.5,
               color=color, alpha=0.45, edgecolors="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="#2F3437", linewidth=0.65,
            linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(PROPERTY_NAMES[prop])
    ax.set_xlabel(f"Experimental ({PROPERTY_UNITS[prop]})")
    ax.set_ylabel(f"Predicted ({PROPERTY_UNITS[prop]})")
    metric_row = metrics[metrics["property"] == prop].iloc[0]
    dropped = len(data) - len(plotted)
    note = f"n={len(data):,}\nlog $R^2$={metric_row['log_R2']:.3f}"
    if dropped:
        note += f"\n{dropped} nonpositive hidden"
    ax.text(0.05, 0.95, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=5.8, color="#20252A")
    if prop not in LOG_AXIS_PROPERTIES:
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    style_axes(ax)
    panel_label(ax, label)
    data["panel"] = label.upper()
    data["panel_title"] = f"{PROPERTY_NAMES[prop]} parity plot"
    data["plot_included"] = False
    data.loc[plotted.index, "plot_included"] = True
    return data


def plot_metric_bars(ax: plt.Axes,
                     metrics: pd.DataFrame,
                     metric_col: str,
                     avg_value: float,
                     ylabel: str,
                     title: str,
                     label: str) -> pd.DataFrame:
    data = metrics[metrics["property"].isin(PROPERTY_ORDER)].copy()
    data["property"] = pd.Categorical(data["property"], PROPERTY_ORDER,
                                      ordered=True)
    data = data.sort_values("property")
    x = np.arange(len(data))
    colors = [PROPERTY_COLORS[str(p)] for p in data["property"]]
    bars = ax.bar(x, data[metric_col], color=colors, edgecolor="white",
                  linewidth=0.35, width=0.68)
    ax.axhline(avg_value, color="#30363D", linewidth=0.7, linestyle="--")
    for bar, value in zip(bars, data[metric_col]):
        offset = 0.003 if metric_col == "log_R2" else 0.004
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset,
                format_metric(float(value)), ha="center", va="bottom",
                fontsize=5.8)
    ax.text(0.98, avg_value + (0.004 if metric_col == "log_R2" else 0.004),
            f"macro avg. {avg_value:.3f}",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=5.8, color="#30363D")
    ax.set_xticks(x)
    ax.set_xticklabels([PROPERTY_SHORT[str(p)] for p in data["property"]])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if metric_col == "log_R2":
        ax.set_ylim(0.92, 1.005)
    else:
        ax.set_ylim(0, max(float(data[metric_col].max()) * 1.28,
                           avg_value * 1.28))
    style_axes(ax, "y")
    panel_label(ax, label)
    out = data[["property", metric_col]].copy()
    out["panel"] = label.upper()
    out["panel_title"] = title
    out["macro_average"] = avg_value
    return out


def write_panel_source(output_dir: Path,
                       parity_sources: list[pd.DataFrame],
                       r2_source: pd.DataFrame,
                       nmae_source: pd.DataFrame) -> None:
    for i, data in enumerate(parity_sources):
        panel = PANEL_LABELS[i].upper()
        cols = ["panel", "panel_title", "IL_Name", "IL_SMILES",
                "Temperature_K", "Pressure_kPa", "property", "y_true",
                "y_pred", "absolute_error", "split", "plot_included"]
        data[cols].to_csv(output_dir / f"performance_results_source_data_{panel}.csv",
                          index=False)
    r2_source.to_csv(output_dir / "performance_results_source_data_G.csv",
                     index=False)
    nmae_source.to_csv(output_dir / "performance_results_source_data_H.csv",
                       index=False)


def create_figure(predictions_path: Path,
                  metrics_path: Path,
                  output_dir: Path,
                  name: str,
                  dpi: int) -> None:
    pred_df = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    avg = metrics[metrics["property"] == "Average"].iloc[0]
    metrics_no_avg = metrics[metrics["property"] != "Average"].copy()

    configure_style()
    fig = plt.figure(figsize=(7.2, 7.05))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.9],
                          hspace=0.62, wspace=0.52,
                          left=0.075, right=0.985,
                          top=0.965, bottom=0.075)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    parity_sources = []
    for ax, prop, label in zip(axes, PROPERTY_ORDER, PANEL_LABELS[:6]):
        parity_sources.append(plot_parity(ax, pred_df, metrics_no_avg,
                                          prop, label))

    ax_g = fig.add_subplot(gs[2, 0:2])
    ax_h = fig.add_subplot(gs[2, 2])
    r2_source = plot_metric_bars(ax_g, metrics_no_avg, "log_R2",
                                 float(avg["log_R2"]),
                                 r"log-space $R^2$",
                                 "Final explained variance", "g")
    nmae_source = plot_metric_bars(ax_h, metrics_no_avg, "log_NMAE",
                                   float(avg["log_NMAE"]),
                                   "log-space NMAE",
                                   "Final NMAE", "h")

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": dpi},
        "tiff": {"dpi": dpi},
    }.items():
        fig.savefig(output_dir / f"{name}.{ext}", **kwargs)
    plt.close(fig)
    write_panel_source(output_dir, parity_sources, r2_source, nmae_source)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final performance results with test-set parity plots.")
    root = Path("il_property_prediction/outputs/"
                "fg_transformer_random_point_seed42_noamp")
    run = "unimol2_fg_transformer_random_point_seed42_noamp_resume56"
    parser.add_argument("--predictions", type=Path,
                        default=root / "predictions" / run / "test_predictions.csv")
    parser.add_argument("--metrics", type=Path,
                        default=root / "metrics" / run / "test_metrics_log.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("LaTex-MIPGraph/Fig"))
    parser.add_argument("--name", default="performance_results")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    create_figure(args.predictions, args.metrics,
                  args.output_dir, args.name, args.dpi)
    print(f"Wrote {args.output_dir / (args.name + '.png')}")


if __name__ == "__main__":
    main()
