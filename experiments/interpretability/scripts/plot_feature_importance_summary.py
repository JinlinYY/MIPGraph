"""Plot feature attribution source data as ranked dot-bar summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = REPO_ROOT / "LaTex-MIPGraph" / "Fig"
DEFAULT_PREFIX = FIG_DIR / "feature_importance_heatmap"

PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]
PROPERTY_SHORT = {
    "Density": "Density",
    "ElectricalConductivity": "EC",
    "HeatCapacity": "HC",
    "SurfaceTension": "ST",
    "ThermalConductivity": "TC",
    "Viscosity": "Visc.",
}
PROPERTY_COLORS = {
    "Density": "#4C78A8",
    "ElectricalConductivity": "#59A14F",
    "HeatCapacity": "#E3AE3D",
    "SurfaceTension": "#A879B2",
    "ThermalConductivity": "#62A8A8",
    "Viscosity": "#D96B5F",
}
PROPERTY_MARKERS = {
    "Density": "o",
    "ElectricalConductivity": "s",
    "HeatCapacity": "^",
    "SurfaceTension": "D",
    "ThermalConductivity": "P",
    "Viscosity": "X",
}
PANEL_COLOR_SCHEMES = [
    {
        "bar": "#D7E6F5",
        "points": {
            "Density": "#1F4E79",
            "ElectricalConductivity": "#2C7FB8",
            "HeatCapacity": "#41B6C4",
            "SurfaceTension": "#7FCDBB",
            "ThermalConductivity": "#A6BDDB",
            "Viscosity": "#08519C",
        },
    },
    {
        "bar": "#DDEEDB",
        "points": {
            "Density": "#1B7837",
            "ElectricalConductivity": "#4DAF4A",
            "HeatCapacity": "#80C67A",
            "SurfaceTension": "#A6D96A",
            "ThermalConductivity": "#35978F",
            "Viscosity": "#006D2C",
        },
    },
    {
        "bar": "#E8DDF0",
        "points": {
            "Density": "#542788",
            "ElectricalConductivity": "#8073AC",
            "HeatCapacity": "#B2ABD2",
            "SurfaceTension": "#C51B7D",
            "ThermalConductivity": "#E78AC3",
            "Viscosity": "#A6611A",
        },
    },
]
NODE_ORDER = [
    "charged atoms",
    "H-bond atoms",
    "aromatic atoms",
    "hetero atoms",
    "halogen/F atoms",
    "carbon atoms",
]
EDGE_ORDER = [
    "charge-complementary pairs",
    "H-bond compatible pairs",
    "aromatic-contact pairs",
    "charge-magnitude pairs",
    "other attended pairs",
]


def configure_style(title_size: float = 11.6) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": title_size,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.1,
            "legend.fontsize": 9.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
        }
    )


def ranked_frame(
    frame: pd.DataFrame,
    feature_order: list[str] | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    data = frame.copy()
    if "feature_pretty" not in data.columns:
        data["feature_pretty"] = data["feature"]
    if feature_order is None:
        order = (
            data.groupby("feature_pretty")["normalized_importance"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        pretty_lookup = (
            data.drop_duplicates("feature")
            .set_index("feature")["feature_pretty"]
            .to_dict()
        )
        order = [pretty_lookup.get(feature, feature) for feature in feature_order]
    if top_n is not None:
        order = order[:top_n]
    data = data[data["feature_pretty"].isin(order)].copy()
    data["feature_pretty"] = pd.Categorical(
        data["feature_pretty"],
        order[::-1],
        ordered=True,
    )
    data["property"] = pd.Categorical(data["property"], PROPERTY_NAMES, ordered=True)
    return data.sort_values(["feature_pretty", "property"])


def draw_rank_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    panel: str,
    wrap_width: int,
    point_colors: dict[str, str] | None = None,
    mean_bar_color: str = "#D8DEE6",
) -> pd.DataFrame:
    data = frame.copy()
    point_colors = point_colors or PROPERTY_COLORS
    order = data["feature_pretty"].cat.categories.tolist()
    y_lookup = {feature: idx for idx, feature in enumerate(order)}
    summary = (
        data.groupby("feature_pretty", observed=False)["normalized_importance"]
        .mean()
        .reindex(order)
        .reset_index(name="mean_normalized_importance")
    )
    y = np.arange(len(order))
    ax.barh(
        y,
        summary["mean_normalized_importance"].to_numpy(),
        height=0.74,
        color=mean_bar_color,
        edgecolor="white",
        linewidth=0.65,
        zorder=1,
    )
    offsets = np.linspace(-0.22, 0.22, len(PROPERTY_NAMES))
    for prop_idx, prop in enumerate(PROPERTY_NAMES):
        sub = data[data["property"] == prop]
        ys = [y_lookup[str(feature)] + offsets[prop_idx] for feature in sub["feature_pretty"]]
        ax.scatter(
            sub["normalized_importance"],
            ys,
            s=44,
            marker=PROPERTY_MARKERS[prop],
            color=point_colors[prop],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.95,
            zorder=3,
            label=PROPERTY_SHORT[prop],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([fill(str(feature), wrap_width) for feature in order])
    ax.set_xlabel("normalized attribution")
    ax.set_title(title)
    ax.grid(axis="x", color="#E6E8EB", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#3F3F3F")
        ax.spines[side].set_linewidth(0.55)
    ax.tick_params(direction="out", pad=1.4)
    ax.text(
        -0.16,
        1.06,
        panel,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    return summary


def panel_plot_data(
    frame: pd.DataFrame,
    panel: str,
    group: str,
    point_colors: dict[str, str] | None = None,
    mean_bar_color: str = "#D8DEE6",
) -> pd.DataFrame:
    data = frame.copy()
    point_colors = point_colors or PROPERTY_COLORS
    data["feature_pretty"] = data["feature_pretty"].astype(str)
    data["property"] = data["property"].astype(str)
    order = frame["feature_pretty"].cat.categories.tolist()
    y_lookup = {str(feature): idx for idx, feature in enumerate(order)}
    offsets = {
        prop: offset
        for prop, offset in zip(PROPERTY_NAMES, np.linspace(-0.22, 0.22, len(PROPERTY_NAMES)))
    }
    summary = (
        data.groupby("feature_pretty", observed=False)["normalized_importance"]
        .mean()
        .reset_index(name="mean_normalized_importance")
    )
    out = data.merge(summary, on="feature_pretty", how="left")
    out["panel"] = panel
    out["feature_group"] = group
    out["property_short"] = out["property"].map(PROPERTY_SHORT)
    out["marker"] = out["property"].map(PROPERTY_MARKERS)
    out["color"] = out["property"].map(point_colors)
    out["mean_bar_color"] = mean_bar_color
    out["plot_y_base"] = out["feature_pretty"].map(y_lookup)
    out["plot_y_offset"] = out["property"].map(offsets)
    out["plot_y"] = out["plot_y_base"] + out["plot_y_offset"]
    out["plot_order_bottom_to_top"] = out["plot_y_base"] + 1
    out["plot_order_top_to_bottom"] = len(order) - out["plot_y_base"]
    preferred = [
        "panel",
        "feature_group",
        "feature_type",
        "feature",
        "feature_pretty",
        "plot_order_top_to_bottom",
        "plot_order_bottom_to_top",
        "plot_y_base",
        "plot_y_offset",
        "plot_y",
        "property",
        "property_short",
        "importance",
        "normalized_importance",
        "mean_normalized_importance",
        "n_labels",
        "marker",
        "color",
        "mean_bar_color",
    ]
    return out[[col for col in preferred if col in out.columns]]


def plot_summary(
    out_prefix: Path,
    top_k: int,
    dpi: int,
    panel_labels: tuple[str, str, str],
    source_prefix: Path | None = None,
    title_size: float = 11.6,
    color_mode: str = "property",
) -> None:
    source_prefix = source_prefix or out_prefix
    node = pd.read_csv(source_prefix.with_name(source_prefix.name + "_source_data_nodes.csv"))
    edge = pd.read_csv(source_prefix.with_name(source_prefix.name + "_source_data_edges.csv"))
    fg = pd.read_csv(source_prefix.with_name(source_prefix.name + "_source_data_functional_groups.csv"))
    if "feature_pretty" not in fg.columns:
        fg["feature_pretty"] = fg["feature"]

    node_rank = ranked_frame(node)
    edge_rank = ranked_frame(edge)
    top_features = (
        fg.groupby("feature_pretty")["normalized_importance"]
        .mean()
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )
    fg_rank = ranked_frame(fg[fg["feature_pretty"].isin(top_features)], top_n=top_k)
    if color_mode == "panel":
        styles = [(scheme["points"], scheme["bar"]) for scheme in PANEL_COLOR_SCHEMES]
    elif color_mode == "property":
        styles = [(PROPERTY_COLORS, "#D8DEE6")] * 3
    else:
        raise ValueError("--color-mode must be 'property' or 'panel'")

    configure_style(title_size)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.1, 5.15),
        gridspec_kw={"width_ratios": [1.04, 1.12, 1.84], "wspace": 0.50},
    )
    node_summary = draw_rank_panel(axes[0], node_rank, "Atom-node attribution", panel_labels[0], 16, *styles[0])
    edge_summary = draw_rank_panel(axes[1], edge_rank, "Cross-ion interaction attribution", panel_labels[1], 19, *styles[1])
    fg_summary = draw_rank_panel(axes[2], fg_rank, "Top functional-group descriptors", panel_labels[2], 25, *styles[2])
    axes[0].set_xlim(0, max(0.34, float(node_rank["normalized_importance"].max()) * 1.12))
    axes[1].set_xlim(0, 1.0)
    axes[2].set_xlim(0, max(0.22, float(fg_rank["normalized_importance"].max()) * 1.14))

    if color_mode == "panel":
        handles = [
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=PROPERTY_MARKERS[prop],
                markersize=7.0,
                markerfacecolor="#4A4F55",
                markeredgecolor="#4A4F55",
                color="#4A4F55",
                label=PROPERTY_SHORT[prop],
            )
            for prop in PROPERTY_NAMES
        ]
        labels = [PROPERTY_SHORT[prop] for prop in PROPERTY_NAMES]
    else:
        handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.018),
        ncol=6,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.15,
        markerscale=1.0,
    )
    fig.subplots_adjust(left=0.073, right=0.992, top=0.920, bottom=0.205)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        "png": {"dpi": dpi},
        "pdf": {},
        "svg": {},
        "tiff": {"dpi": dpi},
    }.items():
        fig.savefig(out_prefix.with_suffix(f".{ext}"), bbox_inches="tight", **kwargs)
    plt.close(fig)

    source = pd.concat(
        [
            node_summary.assign(panel=panel_labels[0].upper(), group="atom-node"),
            edge_summary.assign(panel=panel_labels[1].upper(), group="cross-ion"),
            fg_summary.assign(panel=panel_labels[2].upper(), group="functional-group"),
        ],
        ignore_index=True,
    )
    source.to_csv(out_prefix.with_name(out_prefix.name + "_ranked_summary.csv"), index=False)
    plot_data = pd.concat(
        [
            panel_plot_data(node_rank, panel_labels[0].upper(), "atom-node"),
            panel_plot_data(edge_rank, panel_labels[1].upper(), "cross-ion"),
            panel_plot_data(fg_rank, panel_labels[2].upper(), "functional-group"),
        ],
        ignore_index=True,
    )
    if color_mode == "panel":
        plot_data = pd.concat(
            [
                panel_plot_data(node_rank, panel_labels[0].upper(), "atom-node", *styles[0]),
                panel_plot_data(edge_rank, panel_labels[1].upper(), "cross-ion", *styles[1]),
                panel_plot_data(fg_rank, panel_labels[2].upper(), "functional-group", *styles[2]),
            ],
            ignore_index=True,
        )
    plot_data.to_csv(out_prefix.with_name(out_prefix.name + "_plot_data.csv"), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot feature attribution as ranked summaries.")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--top-functional-groups", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--panel-labels", default="a,b,c")
    parser.add_argument("--source-prefix", type=Path, default=None)
    parser.add_argument("--title-size", type=float, default=11.6)
    parser.add_argument("--color-mode", choices=["property", "panel"], default="property")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel_labels = tuple(item.strip() for item in args.panel_labels.split(","))
    if len(panel_labels) != 3:
        raise ValueError("--panel-labels must contain exactly three comma-separated labels")
    plot_summary(
        args.out_prefix,
        args.top_functional_groups,
        args.dpi,
        panel_labels,
        args.source_prefix,
        args.title_size,
        args.color_mode,
    )
    print(f"Wrote {args.out_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
