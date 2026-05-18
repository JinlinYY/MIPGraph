

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from _common import (
    LATENT_DISPLAY,
    LATENT_NAMES,
    PROPERTY_DISPLAY,
    PROPERTY_NAMES,
    collapse_minor_families,
    family_palette,
    output_dirs,
    safe_log10,
    setup_style,
)


FONT_SIZE = 50
EXPORT_PAD_INCHES = 50 / 72

CMAP_CORRELATION = LinearSegmentedColormap.from_list(
    "nature_blue_white_red",
    ["#3d6f9f", "#f7f7f4", "#b4544a"],
)
CMAP_FACTOR = LinearSegmentedColormap.from_list(
    "nature_factor_disentanglement",
    ["#fbfcf7", "#e6f2e6", "#c7e4dd", "#8fc9c0", "#4f9ab1", "#2f6f9e"],
)
CMAP_UMAP = LinearSegmentedColormap.from_list(
    "nature_umap_viscosity",
    ["#2b1a67", "#7b3294", "#c85d6a", "#f0c75e"],
)
CMAP_EDGE = LinearSegmentedColormap.from_list(
    "nature_edge_attribution",
    ["#f7fbf7", "#cfe5d2", "#8fc6bb", "#4f92b3", "#344b73"],
)

FAMILY_DISPLAY_SHORT = {
    "Amino acid": "Amino\nacid",
    "Carboxylate": "Carbox.",
    "Cholinium": "Chol.",
    "Halide": "Halide",
    "Hydrogen sulfate": "HSO4",
    "Imidazolium": "Imid.",
    "NTf2": "NTf2",
    "Other": "Other",
    "Phosphonium": "Phosph.",
    "Protic ammonium": "Protic amm.",
    "Pyridinium": "Pyrid.",
    "Quaternary ammonium": "Quat. amm.",
    "Sulfate": "SO4",
    "Thiocyanate": "SCN",
    "Triflate": "OTf",
}
LATENT_DISPLAY_SHORT = {
    "packing": "Packing",
    "cohesion": "Cohesion",
    "transport": "Transport",
    "thermal": "Thermal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="Viscosity", help="Edge-attribution target.")
    parser.add_argument("--feature", default="h_structure", help="Feature used for UMAP.")
    parser.add_argument("--per-il-suffix", default="_il", help="Suffix written by visualize_umap.py.")
    return parser.parse_args()


def load_correlation_matrix(out: dict[str, Path]) -> pd.DataFrame:
    path = out["tables"] / "latent_pearson.csv"
    df = pd.read_csv(path, index_col=0)
    df = df.reindex(LATENT_NAMES)[PROPERTY_NAMES]
    return df


def load_disentanglement_matrices(out: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson = pd.read_csv(out["tables"] / "disentanglement_pearson_pc1.csv", index_col=0)
    cka = pd.read_csv(out["tables"] / "disentanglement_linear_cka.csv", index_col=0)
    return pearson.reindex(LATENT_NAMES)[LATENT_NAMES], cka.reindex(LATENT_NAMES)[LATENT_NAMES]


def load_umap_coords(out: dict[str, Path], feature: str, suffix: str) -> pd.DataFrame:
    path = out["tables"] / f"latent_2d_coords_{feature}{suffix}.csv"
    df = pd.read_csv(path)
    return df


def load_family_attribution(out: dict[str, Path], target: str) -> pd.DataFrame:
    path = out["tables"] / f"cross_ion_family_attribution_{target}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_top_cross_edges(out: dict[str, Path], target: str) -> pd.DataFrame:
    path = out["tables"] / f"top_cross_ion_edges_{target}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def style_heatmap_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=FONT_SIZE)
    ax.set_xlabel(ax.get_xlabel(), fontsize=FONT_SIZE)
    ax.set_ylabel(ax.get_ylabel(), fontsize=FONT_SIZE)
    ax.grid(False)


def style_scatter_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=FONT_SIZE)
    ax.set_xlabel(ax.get_xlabel(), fontsize=FONT_SIZE)
    ax.set_ylabel(ax.get_ylabel(), fontsize=FONT_SIZE)
    ax.grid(False)


def style_colorbar(cbar) -> None:
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    cbar.ax.yaxis.label.set_size(FONT_SIZE)
    cbar.ax.yaxis.labelpad = 14
    cbar.outline.set_visible(True)
    cbar.outline.set_linewidth(1.0)
    cbar.outline.set_edgecolor("#222222")


def save_main_figure(fig: plt.Figure, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(
        out_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=EXPORT_PAD_INCHES,
    )
    try:
        fig.savefig(
            out_stem.with_suffix(".pdf"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=EXPORT_PAD_INCHES,
        )
    except PermissionError as exc:
        print(f"[plot_main_figure] skipped PDF export: {exc}")
    plt.close(fig)


def panel_a(ax: plt.Axes, matrix: pd.DataFrame, fig: plt.Figure) -> None:
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=CMAP_CORRELATION,
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        cbar_kws={"shrink": 0.7, "label": r"$r$", "pad": 0.02},
        linewidths=0.4,
        linecolor="white",
        annot_kws={"fontsize": FONT_SIZE},
        ax=ax,
        xticklabels=[PROPERTY_DISPLAY[p] for p in matrix.columns],
        yticklabels=[LATENT_DISPLAY[n] for n in matrix.index],
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    style_heatmap_axes(ax)
    style_colorbar(ax.collections[0].colorbar)


def panel_b(ax: plt.Axes, df: pd.DataFrame, fig: plt.Figure) -> None:
    coords = df[["umap_1", "umap_2"]].to_numpy()
    values = safe_log10(df["Viscosity"].to_numpy(dtype=np.float64)) if "Viscosity" in df.columns else np.zeros(len(df))
    valid = np.isfinite(values)
    ax.scatter(
        coords[~valid, 0],
        coords[~valid, 1],
        s=6,
        c="lightgrey",
        edgecolors="none",
        alpha=0.4,
    )
    sc = ax.scatter(
        coords[valid, 0],
        coords[valid, 1],
        s=11,
        c=values[valid],
        cmap=CMAP_UMAP,
        edgecolors="white",
        linewidths=0.2,
        alpha=0.9,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.tick_params(axis="both", which="both", labelsize=FONT_SIZE)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(r"$\log_{10}\eta$", fontsize=FONT_SIZE)
    style_colorbar(cb)
    style_scatter_axes(ax)


def panel_c(
    ax: plt.Axes,
    cka: pd.DataFrame,
    pearson: pd.DataFrame,
    fig: plt.Figure,
) -> None:
    sns.heatmap(
        cka,
        annot=True,
        fmt=".2f",
        cmap=CMAP_FACTOR,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"shrink": 0.7, "label": "", "pad": 0.02},
        linewidths=0.4,
        linecolor="white",
        annot_kws={"fontsize": FONT_SIZE},
        ax=ax,
        xticklabels=[LATENT_DISPLAY_SHORT[n] for n in cka.columns],
        yticklabels=[LATENT_DISPLAY_SHORT[n] for n in cka.index],
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    style_heatmap_axes(ax)
    style_colorbar(ax.collections[0].colorbar)


def panel_d(
    ax: plt.Axes,
    family_pivot: pd.DataFrame,
    top_edges: pd.DataFrame,
    fig: plt.Figure,
    target: str,
    top_families: int = 8,
) -> None:
    if family_pivot.empty:
        ax.set_axis_off()
        return
    df = family_pivot.copy()
    df["Cation_Family"] = collapse_minor_families(
        df["Cation_Family"].fillna("Unknown"), top_k=top_families
    )
    df["Anion_Family"] = collapse_minor_families(
        df["Anion_Family"].fillna("Unknown"), top_k=top_families
    )
    heat = df.pivot_table(index="Cation_Family", columns="Anion_Family", values="mean_abs_attr", aggfunc="mean")
    sns.heatmap(
        heat,
        annot=False,
        fmt=".2f",
        cmap=CMAP_EDGE,
        cbar_kws={"shrink": 0.7, "label": "", "pad": 0.02},
        linewidths=0.4,
        linecolor="white",
        annot_kws={"fontsize": FONT_SIZE},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=FONT_SIZE, rotation=55)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.set_yticklabels([FAMILY_DISPLAY_SHORT.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()])
    ax.set_xticklabels([FAMILY_DISPLAY_SHORT.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
    style_heatmap_axes(ax)
    style_colorbar(ax.collections[0].colorbar)


def main() -> None:
    args = parse_args()
    setup_style()
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    out = output_dirs()

    correlation = load_correlation_matrix(out)
    pearson_dis, cka_dis = load_disentanglement_matrices(out)
    umap_df = load_umap_coords(out, args.feature, args.per_il_suffix)
    family_df = load_family_attribution(out, args.target)
    top_edges = load_top_cross_edges(out, args.target)

    fig = plt.figure(figsize=(38.0, 28.0), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.24], wspace=0.36, hspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, correlation, fig)
    panel_b(ax_b, umap_df, fig)
    panel_c(ax_c, cka_dis, pearson_dis, fig)
    panel_d(ax_d, family_df, top_edges, fig, args.target)

    save_main_figure(fig, out["figures"] / "fig_main_interpretability")
    print(f"[plot_main_figure] saved {out['figures'] / 'fig_main_interpretability'}.png/.pdf")


if __name__ == "__main__":
    main()
