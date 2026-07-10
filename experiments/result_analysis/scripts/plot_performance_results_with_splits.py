"""Create the main performance figure with parity plots and split-strategy comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

from plot_performance_results import (
    LOG_AXIS_PROPERTIES,
    PROPERTY_NAMES,
    PROPERTY_ORDER,
    PROPERTY_SHORT,
    PROPERTY_UNITS,
    configure_style,
    nice_linear_limits,
    nice_log_limits,
    panel_label,
    style_axes,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = REPO_ROOT / "LaTex-MIPGraph" / "Fig"

CURRENT_BEST_INPUT_ROOT = (
    REPO_ROOT
    / "il_property_prediction"
    / "outputs"
    / "mps_weak_merged_validation"
    / "figure_inputs"
)

SPLIT_SPECS = [
    ("random_point", "Random point", "Point", "Pt", "#334E68"),
    ("random_il_level", "Random IL", "IL", "IL", "#6C8EBF"),
    ("property_balanced_il_level", "Balanced IL", "Balanced", "Bal", "#A8B95A"),
    ("ion_family", "Ion-family", "Family", "Fam", "#C07A58"),
]


def make_split_cases(input_root: Path) -> list[dict[str, object]]:
    return [
        {
            "key": key,
            "name": name,
            "short": short,
            "abbr": abbr,
            "color": color,
            "metrics": input_root / key / "test_metrics_log.csv",
            "predictions": input_root / key / "test_predictions.csv",
        }
        for key, name, short, abbr, color in SPLIT_SPECS
    ]


SPLIT_CASES = make_split_cases(CURRENT_BEST_INPUT_ROOT)


def _split_names() -> list[str]:
    return [case["name"] for case in SPLIT_CASES]


def _split_shorts() -> list[str]:
    return [case["short"] for case in SPLIT_CASES]


def _split_colors() -> list[str]:
    return [case["color"] for case in SPLIT_CASES]


def load_split_metrics() -> pd.DataFrame:
    frames = []
    missing = []
    for case in SPLIT_CASES:
        path = Path(case["metrics"])
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        frame["split_strategy"] = case["name"]
        frame["split_short"] = case["short"]
        frames.append(frame)
    if missing:
        raise FileNotFoundError("Missing split metrics:\n" + "\n".join(missing))
    return pd.concat(frames, ignore_index=True)


def load_split_predictions() -> pd.DataFrame:
    frames = []
    missing = []
    for case in SPLIT_CASES:
        path = Path(case["predictions"])
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        frame["split_strategy"] = case["name"]
        frame["split_short"] = case["short"]
        frames.append(frame)
    if missing:
        raise FileNotFoundError("Missing split predictions:\n" + "\n".join(missing))
    return pd.concat(frames, ignore_index=True)


def plot_multi_split_parity(
    ax: plt.Axes,
    predictions: pd.DataFrame,
    split_metrics: pd.DataFrame,
    prop: str,
    label: str,
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

    for case in SPLIT_CASES:
        split_data = data[data["split_strategy"] == case["name"]].copy()
        plotted = split_data.copy()
        if prop in LOG_AXIS_PROPERTIES:
            plotted = plotted[(plotted["y_true"] > 0) & (plotted["y_pred"] > 0)].copy()
        ax.scatter(
            plotted["y_true"],
            plotted["y_pred"],
            s=6.4,
            color=case["color"],
            alpha=0.25,
            edgecolors="none",
            rasterized=True,
            label=case["short"],
        )

    ax.plot([lo, hi], [lo, hi], color="#2F3437", linewidth=0.65, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(PROPERTY_NAMES[prop])
    ax.set_xlabel(f"Experimental ({PROPERTY_UNITS[prop]})")
    ax.set_ylabel(f"Predicted ({PROPERTY_UNITS[prop]})")

    metric_lines = [r"$R^2$/NMAE"]
    for case in SPLIT_CASES:
        row = split_metrics[
            (split_metrics["property"] == prop) & (split_metrics["split_strategy"] == case["name"])
        ].iloc[0]
        metric_lines.append(f"{case['abbr']} {row['log_R2']:.2f}/{row['log_NMAE']:.2f}")
    ax.text(
        0.045,
        0.955,
        "\n".join(metric_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        family="monospace",
        color="#20252A",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.84),
    )
    style_axes(ax)
    panel_label(ax, label)
    data["panel"] = label.upper()
    data["panel_title"] = f"{PROPERTY_NAMES[prop]} multi-split parity plot"
    data["plot_included"] = False
    data.loc[plotted_all.index, "plot_included"] = True
    return data


def plot_macro_split_summary(ax: plt.Axes, split_metrics: pd.DataFrame) -> pd.DataFrame:
    avg = split_metrics[split_metrics["property"] == "Average"].copy()
    avg["split_strategy"] = pd.Categorical(
        avg["split_strategy"],
        _split_names(),
        ordered=True,
    )
    avg = avg.sort_values("split_strategy")
    x = np.arange(len(avg))
    bars = ax.bar(x, avg["log_R2"], width=0.66, color=_split_colors(), edgecolor="white", linewidth=0.5)
    ax.set_ylim(0.55, 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels(_split_shorts(), rotation=20, ha="right")
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


def _heatmap_matrix(split_metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = split_metrics[split_metrics["property"].isin(PROPERTY_ORDER)].copy()
    data["property"] = pd.Categorical(data["property"], PROPERTY_ORDER, ordered=True)
    data["split_strategy"] = pd.Categorical(
        data["split_strategy"],
        _split_names(),
        ordered=True,
    )
    return data.pivot(index="split_strategy", columns="property", values=metric).loc[
        _split_names(),
        PROPERTY_ORDER,
    ]


def plot_split_heatmap(
    ax: plt.Axes,
    fig: plt.Figure,
    split_metrics: pd.DataFrame,
    metric: str,
    title: str,
    label: str,
    cmap,
    vmin: float,
    vmax: float,
    cbar_label: str,
    value_format: str = ".2f",
) -> pd.DataFrame:
    matrix = _heatmap_matrix(split_metrics, metric)
    norm = Normalize(vmin=vmin, vmax=vmax)
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(PROPERTY_ORDER)))
    ax.set_xticklabels([PROPERTY_SHORT[prop] for prop in PROPERTY_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(_split_shorts())
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
            ax.text(
                j,
                i,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=6.1,
                color="#1F2933",
                path_effects=[pe.withStroke(linewidth=1.4, foreground="white", alpha=0.82)],
            )
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label(cbar_label, fontsize=6.6)
    cbar.ax.tick_params(labelsize=6.1, length=2)
    panel_label(ax, label)
    source = matrix.reset_index().melt(id_vars="split_strategy", var_name="property", value_name=metric)
    source["panel"] = label.upper()
    source["panel_title"] = title
    return source


def write_source_data(output_dir: Path, sources: dict[str, pd.DataFrame]) -> None:
    for suffix, frame in sources.items():
        frame.to_csv(output_dir / f"performance_results_source_data_{suffix}.csv", index=False)


def create_figure(output_dir: Path, name: str, dpi: int, input_root: Path = CURRENT_BEST_INPUT_ROOT) -> None:
    global SPLIT_CASES
    SPLIT_CASES = make_split_cases(input_root)
    pred_df = load_split_predictions()
    split_metrics = load_split_metrics()

    configure_style()
    mpl.rcParams.update({
        "font.size": 8.0,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.3,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
    })

    fig = plt.figure(figsize=(7.25, 7.55))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.0, 0.98],
        hspace=0.44,
        wspace=0.56,
        left=0.070,
        right=0.988,
        top=0.975,
        bottom=0.070,
    )

    sources: dict[str, pd.DataFrame] = {}
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for idx, (ax, prop, label) in enumerate(zip(axes, PROPERTY_ORDER, list("abcdef"))):
        parity = plot_multi_split_parity(ax, pred_df, split_metrics, prop, label)
        sources[label.upper()] = parity

    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="none", markersize=5.2, color=case["color"], label=case["short"])
        for case in SPLIT_CASES
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.035),
        ncol=1,
        handletextpad=0.35,
        labelspacing=0.28,
        borderpad=0.25,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.84,
        fontsize=6.6,
    )

    ax_g = fig.add_subplot(gs[2, 0])
    ax_h = fig.add_subplot(gs[2, 1])
    ax_i = fig.add_subplot(gs[2, 2])

    sources["G"] = plot_macro_split_summary(ax_g, split_metrics)
    r2_cmap = LinearSegmentedColormap.from_list("r2_cmap", ["#F4F1EA", "#C5D7E6", "#2F6F9F"])
    nmae_cmap = LinearSegmentedColormap.from_list("nmae_cmap", ["#F7F7F7", "#F0C987", "#B75D4A"])
    sources["H"] = plot_split_heatmap(
        ax_h,
        fig,
        split_metrics,
        "log_R2",
        r"Property-wise log-space $R^2$",
        "h",
        r2_cmap,
        0.25,
        1.0,
        r"log-space $R^2$",
        ".2f",
    )
    sources["I"] = plot_split_heatmap(
        ax_i,
        fig,
        split_metrics,
        "log_NMAE",
        "Property-wise log-space NMAE",
        "i",
        nmae_cmap,
        0.0,
        0.60,
        "log-space NMAE",
        ".2f",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": dpi},
        "tiff": {"dpi": dpi},
    }.items():
        fig.savefig(output_dir / f"{name}.{ext}", **kwargs)
    plt.close(fig)
    write_source_data(output_dir, sources)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot performance results with split-strategy panels.")
    parser.add_argument("--input-root", type=Path, default=CURRENT_BEST_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--name", default="performance_results")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_figure(args.output_dir, args.name, args.dpi, args.input_root)
    print(f"Wrote {args.output_dir / (args.name + '.png')}")


if __name__ == "__main__":
    main()
