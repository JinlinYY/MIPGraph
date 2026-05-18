"""Publication-quality figures for the IL multi-property dataset.

The script generates:

* Six individual panels saved as PDF + PNG under ``outputs/figures/``::

    figA_property_counts.{pdf,png}
    figB_label_missing_heatmap.{pdf,png}
    figC_il_label_coverage.{pdf,png}
    figD_temperature_distribution.{pdf,png}
    figE_cation_family.{pdf,png}
    figF_property_value_distributions.{pdf,png}

* A composite 3 × 2 figure ``Figure_dataset_overview.{pdf,png}`` matching
  the layout requested in the task brief: (A) property sample counts,
  (B) label-missing heatmap, (C) IL label coverage histogram,
  (D) temperature distribution, (E) cation family distribution,
  (F) property value distributions.

The plotting script consumes the CSV tables produced by
``analyze_dataset.py``.  If the tables are missing, the analysis is
re-run automatically.  All paths are configurable via the CLI.

Run::

    python plot_statistics.py \
        --tables-dir exp1_dataset_analysis/outputs/tables \
        --figures-dir exp1_dataset_analysis/outputs/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_dataset import (  # noqa: E402  (import after sys.path tweak)
    LOG_SCALE_PROPERTIES,
    PROPERTIES,
    PROPERTY_DISPLAY,
    PROPERTY_UNITS,
    Paths,
    run_analysis,
)

logger = logging.getLogger("plot_statistics")

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
DEFAULT_FONT_FAMILY = "Arial"

# Colour-blind safe palette (Paul Tol "muted") for property bars.
PROPERTY_PALETTE = sns.color_palette("Set2", n_colors=len(PROPERTIES))
PROPERTY_COLORS: Dict[str, tuple] = dict(zip(PROPERTIES, PROPERTY_PALETTE))

# Colour for "missing" cells in the heatmap (light grey).
HEATMAP_MISSING = "#EAECEE"
HEATMAP_PRESENT = "#1F77B4"


def configure_style(font_family: str = DEFAULT_FONT_FAMILY,
                    base_size: float = 18.0) -> None:
    """Configure matplotlib for publication-quality, 2-column figures."""

    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    if font_family not in available:
        logger.warning("Font %r not available; falling back to DejaVu Sans.",
                       font_family)
        font_family = "DejaVu Sans"

    sns.set_context("paper")

    mpl.rcParams.update({
        "font.family":        font_family,
        "font.sans-serif":    [font_family, "DejaVu Sans"],
        "font.size":          base_size,
        "axes.titlesize":     base_size + 3.0,
        "axes.labelsize":     base_size + 1.0,
        "xtick.labelsize":    base_size,
        "ytick.labelsize":    base_size,
        "legend.fontsize":    base_size,
        "legend.frameon":     False,
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.7,
        "ytick.major.width":  0.7,
        "xtick.major.size":   4,
        "ytick.major.size":   4,
        "lines.linewidth":    1.2,
        "patch.linewidth":    0.5,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype":       42,   # Editable text in vector output.
        "ps.fonttype":        42,
        "mathtext.default":   "regular",
    })


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@dataclass
class Tables:
    property_sample_counts: pd.DataFrame
    property_value_summary: pd.DataFrame
    il_label_matrix: pd.DataFrame
    il_label_coverage: pd.DataFrame
    il_coverage_histogram: pd.DataFrame
    temperature_pressure_summary: pd.DataFrame
    per_il_family: pd.DataFrame
    cation_family_counts: pd.DataFrame
    anion_family_counts: pd.DataFrame
    summary_statistics: pd.DataFrame
    dataset_with_family: pd.DataFrame


def _load_dataset_with_family(tables_dir: Path) -> pd.DataFrame:
    parquet = tables_dir / "dataset_with_family.parquet"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:  # pragma: no cover  (depends on optional deps)
            logger.warning("Could not read parquet %s; falling back to CSV.",
                           parquet)
    return pd.read_csv(tables_dir / "dataset_with_family.csv")


def load_tables(tables_dir: Path) -> Tables:
    return Tables(
        property_sample_counts       = pd.read_csv(tables_dir / "property_sample_counts.csv"),
        property_value_summary       = pd.read_csv(tables_dir / "property_value_summary.csv"),
        il_label_matrix              = pd.read_csv(tables_dir / "il_label_matrix.csv"),
        il_label_coverage            = pd.read_csv(tables_dir / "il_label_coverage.csv"),
        il_coverage_histogram        = pd.read_csv(tables_dir / "il_coverage_histogram.csv"),
        temperature_pressure_summary = pd.read_csv(tables_dir / "temperature_pressure_summary.csv"),
        per_il_family                = pd.read_csv(tables_dir / "per_il_family_assignment.csv"),
        cation_family_counts         = pd.read_csv(tables_dir / "cation_family_counts.csv"),
        anion_family_counts          = pd.read_csv(tables_dir / "anion_family_counts.csv"),
        summary_statistics           = pd.read_csv(tables_dir / "summary_statistics.csv"),
        dataset_with_family          = _load_dataset_with_family(tables_dir),
    )


def _required_table_files() -> Sequence[str]:
    return (
        "property_sample_counts.csv",
        "property_value_summary.csv",
        "il_label_matrix.csv",
        "il_label_coverage.csv",
        "il_coverage_histogram.csv",
        "temperature_pressure_summary.csv",
        "per_il_family_assignment.csv",
        "cation_family_counts.csv",
        "anion_family_counts.csv",
        "summary_statistics.csv",
    )


def ensure_tables(tables_dir: Path,
                  input_xlsx: Optional[Path],
                  output_dir: Optional[Path],
                  sheet_name: str = "Merged",
                  rerun: bool = False) -> None:
    """Make sure the analysis tables exist, otherwise run the pipeline."""

    missing = [f for f in _required_table_files()
               if not (tables_dir / f).exists()]
    has_dataset = ((tables_dir / "dataset_with_family.parquet").exists()
                   or (tables_dir / "dataset_with_family.csv").exists())
    if not rerun and not missing and has_dataset:
        return

    if input_xlsx is None or output_dir is None:
        raise FileNotFoundError(
            f"Required table files are missing in {tables_dir}: {missing}. "
            "Re-run analyze_dataset.py first or pass --input / --output-dir.")

    logger.info("Tables missing or rerun requested; running analysis now.")
    run_analysis(Paths(input_xlsx=input_xlsx,
                       output_dir=output_dir,
                       sheet_name=sheet_name))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def save_figure(fig: plt.Figure, fig_dir: Path, name: str, dpi: int) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / f"{name}.pdf"
    png_path = fig_dir / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    logger.info("Saved %s and %s", pdf_path, png_path)


def annotate_bars(ax: plt.Axes, fmt: str = "{:.0f}", offset: float = 0.01,
                  fontsize: float = 18.0) -> None:
    """Write the numeric value on top of each bar in *ax*."""

    ymax = max((p.get_height() for p in ax.patches), default=1.0)
    for patch in ax.patches:
        h = patch.get_height()
        if not np.isfinite(h) or h == 0:
            continue
        ax.text(patch.get_x() + patch.get_width() / 2,
                h + offset * ymax,
                fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize)


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="in")


def short_property_labels() -> Dict[str, str]:
    """Two-line short labels for tick marks on small panels."""

    return {
        "Density":                "Dens.",
        "ElectricalConductivity": "EC",
        "HeatCapacity":           "HC",
        "SurfaceTension":         "ST",
        "ThermalConductivity":    "TC",
        "Viscosity":              "Visc.",
    }


def compact_property_value_labels() -> Dict[str, str]:
    """Compact property + unit labels for the value-distribution panel."""

    return {
        "Density":                "Dens.",
        "ElectricalConductivity": "EC\n(log)",
        "HeatCapacity":           "HC",
        "SurfaceTension":         "ST",
        "ThermalConductivity":    "TC",
        "Viscosity":              "Visc.\n(log)",
    }


def compact_property_legend_labels() -> Dict[str, str]:
    """Compact legend labels for crowded multi-property panels."""

    return {
        "Density":                "Density",
        "ElectricalConductivity": "EC",
        "HeatCapacity":           "HC",
        "SurfaceTension":         "ST",
        "ThermalConductivity":    "TC",
        "Viscosity":              "Visc.",
    }


def cation_family_short_labels() -> Dict[str, str]:
    """Compact labels for cation families in the composite figure."""

    return {
        "Imidazolium": "Im",
        "Pyridinium": "Py",
        "Phosphonium": "Phos",
        "Quaternary ammonium": "QAm",
        "Protic ammonium": "PAm",
        "Cholinium": "Chol",
        "Other": "Other",
    }


def _format_log10(_x, pos) -> str:
    if _x == 0:
        return "0"
    exp = int(np.log10(_x)) if _x > 0 else 0
    return f"$10^{{{exp}}}$"


# --------------------------------------------------------------------------- #
# Individual panels
# --------------------------------------------------------------------------- #
def plot_property_counts(ax: plt.Axes,
                         counts: pd.DataFrame,
                         show_il: bool = True) -> None:
    counts = counts.set_index("Property").loc[list(PROPERTIES)].reset_index()
    short = short_property_labels()
    x = np.arange(len(PROPERTIES))
    width = 0.38 if show_il else 0.65

    bars1 = ax.bar(x - (width / 2 if show_il else 0),
                   counts["N_Measurements"],
                   width=width,
                   color=[PROPERTY_COLORS[p] for p in counts["Property"]],
                   edgecolor="black", linewidth=0.4,
                   label="Measurements")

    if show_il:
        bars2 = ax.bar(x + width / 2,
                       counts["N_Unique_IL"],
                       width=width,
                       color=[PROPERTY_COLORS[p] for p in counts["Property"]],
                       edgecolor="black", linewidth=0.4,
                       hatch="////", alpha=0.55,
                       label="Unique ILs")

    ax.set_xticks(x)
    ax.set_xticklabels([short[p] for p in counts["Property"]],
                       rotation=25, ha="right", fontsize=18.0)
    ax.set_ylabel("Count")
    ax.set_title("Sample count per property")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.22)
    if show_il:
        legend_handles = [
            Patch(facecolor="grey", edgecolor="black", label="Measurements"),
            Patch(facecolor="grey", edgecolor="black", hatch="////",
                  alpha=0.55, label="Unique ILs"),
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=18.0, ncol=1)


def plot_label_heatmap(ax: plt.Axes, label_matrix: pd.DataFrame) -> None:
    """IL × property presence heatmap; rows sorted by coverage."""

    mat = label_matrix.copy()
    if "N_Labels" not in mat.columns:
        mat["N_Labels"] = mat[list(PROPERTIES)].sum(axis=1)

    # Sort rows by total coverage (most-covered ILs at the top).
    mat = mat.sort_values(
        by=["N_Labels"] + list(PROPERTIES),
        ascending=[False] + [False] * len(PROPERTIES),
    )

    data = mat[list(PROPERTIES)].astype(int).to_numpy()
    cmap = mpl.colors.ListedColormap([HEATMAP_MISSING, HEATMAP_PRESENT])
    ax.imshow(data, aspect="auto", cmap=cmap,
              interpolation="nearest", vmin=0, vmax=1)

    short = short_property_labels()
    ax.set_xticks(np.arange(len(PROPERTIES)))
    ax.set_xticklabels([short[p] for p in PROPERTIES],
                       rotation=25, ha="right", fontsize=18.0)
    ax.set_yticks([])
    ax.set_ylabel(f"Ionic liquids (n = {len(mat):,})")
    ax.set_title("Label presence matrix")

    legend_handles = [
        Patch(facecolor=HEATMAP_PRESENT, label="Labelled"),
        Patch(facecolor=HEATMAP_MISSING, label="Missing", edgecolor="grey",
              linewidth=0.4),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=18.0,
              frameon=True, facecolor="white",
              edgecolor="lightgrey", framealpha=0.85)


def plot_il_coverage(ax: plt.Axes, hist: pd.DataFrame) -> None:
    hist = hist.sort_values("N_Labels")
    n_props = len(PROPERTIES)
    bars = ax.bar(hist["N_Labels"], hist["N_IL"],
                  color=sns.color_palette("crest", n_colors=n_props),
                  edgecolor="black", linewidth=0.4)

    ax.set_xticks(range(1, n_props + 1))
    ax.set_xlabel("Number of property labels\nper ionic liquid")
    ax.set_ylabel("Count of ionic liquids")
    ax.set_title("IL label coverage")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.22)


def plot_temperature_distribution(ax: plt.Axes,
                                  df: pd.DataFrame,
                                  by_property: bool = True) -> None:
    """Histogram of measurement temperature, optionally split by property."""

    if not by_property:
        sns.histplot(df["Temperature_K"].dropna(),
                     ax=ax, bins=40, color="#4C72B0",
                     edgecolor="white", linewidth=0.3)
    else:
        long = []
        for p in PROPERTIES:
            mask = df[f"{p}_ActualValue"].notna()
            long.append(pd.DataFrame({
                "Temperature_K": df.loc[mask, "Temperature_K"],
                "Property": p,
            }))
        long_df = pd.concat(long, ignore_index=True).dropna(
            subset=["Temperature_K"])
        bins = np.linspace(150, 750, 40)
        bottom = np.zeros(len(bins) - 1)
        for p in PROPERTIES:
            vals = long_df.loc[long_df["Property"] == p, "Temperature_K"].to_numpy()
            counts, _ = np.histogram(vals, bins=bins)
            ax.bar(bins[:-1], counts, width=np.diff(bins),
                   bottom=bottom, align="edge",
                   color=PROPERTY_COLORS[p],
                   edgecolor="white", linewidth=0.2,
                   label=compact_property_legend_labels()[p])
            bottom += counts

    ax.set_xlim(150, 750)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Number of measurements")
    ax.set_title("Temperature distribution")
    if by_property:
        ax.legend(fontsize=18.0, loc="upper right",
                  ncol=1, handlelength=1.0, handletextpad=0.35,
                  borderpad=0.2, labelspacing=0.25)


def plot_pressure_distribution(ax: plt.Axes, df: pd.DataFrame) -> None:
    s = df["Pressure_kPa"].dropna()
    ax.hist(s, bins=np.logspace(np.log10(max(s.min(), 1.0)),
                                np.log10(s.max()), 40),
            color="#5B8FB9", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Pressure [kPa, log scale]")
    ax.set_ylabel("Number of measurements")
    ax.set_title(f"Pressure distribution (N = {len(s):,})")


def plot_family_bar(ax: plt.Axes, counts: pd.DataFrame,
                    column: str, title: str, top_n: Optional[int] = None,
                    palette_name: str = "Spectral",
                    label_map: Optional[Dict[str, str]] = None) -> None:
    counts = counts.copy()
    if top_n is not None and len(counts) > top_n:
        keep = counts.iloc[:top_n].copy()
        other_n = counts.iloc[top_n:]["N_IL"].sum()
        other_pct = counts.iloc[top_n:]["Pct_IL"].sum()
        keep = pd.concat([
            keep,
            pd.DataFrame([{column: f"Other (n={len(counts) - top_n})",
                           "N_IL": other_n,
                           "Pct_IL": other_pct}]),
        ], ignore_index=True)
        counts = keep

    counts = counts.sort_values("N_IL", ascending=True)
    palette = sns.color_palette(palette_name, n_colors=len(counts))
    y_labels = counts[column].map(label_map).fillna(counts[column]) if label_map else counts[column]
    ax.barh(y_labels, counts["N_IL"],
            color=palette, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Number of ionic liquids")
    ax.set_xlim(0, max(counts["N_IL"]) * 1.08)
    ax.set_title(title)


def plot_value_distributions(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Box-plot of every property's value distribution.

    Because the six properties span very different magnitudes and units,
    each property's values are rescaled to a common 0–1 range using a
    robust min/max (1st & 99th percentile of the raw or log-transformed
    series) so that distribution shapes are visually comparable.  Tick
    labels carry the unit; ``[log]`` indicates a log-rescaling.
    """

    long_records: List[Dict[str, object]] = []
    tick_labels: List[str] = []
    positions: List[int] = []

    for i, prop in enumerate(PROPERTIES):
        col = f"{prop}_ActualValue"
        s = df[col].dropna()
        if s.empty:
            continue
        if prop in LOG_SCALE_PROPERTIES:
            s = s[s > 0]
            ref = np.log10(s)
            unit_str = f"[{PROPERTY_UNITS[prop]}, log]"
        else:
            ref = s
            unit_str = f"[{PROPERTY_UNITS[prop]}]"

        q_lo, q_hi = ref.quantile([0.01, 0.99])
        if q_hi <= q_lo:
            q_lo, q_hi = ref.min(), ref.max()
        norm = ((ref - q_lo) / (q_hi - q_lo)).clip(-0.1, 1.1)
        for v in norm:
            long_records.append({"x": i, "value": v})

        positions.append(i)
        tick_labels.append(compact_property_value_labels()[prop])

    long_df = pd.DataFrame(long_records)
    sns.boxplot(data=long_df, x="x", y="value", hue="x", ax=ax,
                palette=[PROPERTY_COLORS[p] for p in PROPERTIES],
                width=0.55, linewidth=0.6, fliersize=1.0,
                legend=False)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="center",
                       rotation_mode="anchor", fontsize=18.0)
    ax.tick_params(axis="x", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Robust-rescaled value\n(0 = 1st pct, 1 = 99th pct)")
    ax.set_title("Property value distributions")
    ax.axhline(0.0, color="grey", lw=0.4, ls=":")
    ax.axhline(1.0, color="grey", lw=0.4, ls=":")
    ax.set_ylim(-0.2, 1.2)


def plot_value_distributions_per_property(df: pd.DataFrame,
                                          fig_dir: Path,
                                          dpi: int) -> None:
    """One small histogram per property in a 2 × 3 grid."""

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    axes_flat = axes.flatten()
    for ax, prop in zip(axes_flat, PROPERTIES):
        col = f"{prop}_ActualValue"
        s = df[col].dropna()
        log = prop in LOG_SCALE_PROPERTIES
        if log:
            s = s[s > 0]
            bins = np.logspace(np.log10(s.min()), np.log10(s.max()), 40)
        else:
            bins = 40
        ax.hist(s, bins=bins, color=PROPERTY_COLORS[prop],
                edgecolor="white", linewidth=0.3)
        if log:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(_format_log10))
        ax.set_title(f"{PROPERTY_DISPLAY[prop]}\n[{PROPERTY_UNITS[prop]}]",
                     fontsize=18.0)
        ax.set_ylabel("Count")
        ax.set_xlabel("Value")
        style_axes(ax)

    fig.tight_layout()
    save_figure(fig, fig_dir, "figF2_value_histograms_per_property", dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Standalone single-panel figures
# --------------------------------------------------------------------------- #
def make_individual_figures(tables: Tables, fig_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    plot_property_counts(ax, tables.property_sample_counts)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "figA_property_counts", dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.0, 4.0))
    plot_label_heatmap(ax, tables.il_label_coverage)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "figB_label_missing_heatmap", dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    plot_il_coverage(ax, tables.il_coverage_histogram)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "figC_il_label_coverage", dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    plot_temperature_distribution(ax, tables.dataset_with_family)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "figD_temperature_distribution", dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    plot_family_bar(ax, tables.cation_family_counts,
                    "Cation_Family", "Cation family distribution",
                    label_map=cation_family_short_labels())
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "figE_cation_family", dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    plot_value_distributions(ax, tables.dataset_with_family)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "figF_property_value_distributions", dpi)
    plt.close(fig)

    # Bonus: anion family (top-15) and pressure
    fig, ax = plt.subplots(figsize=(3.6, 4.4))
    plot_family_bar(ax, tables.anion_family_counts, "Anion_Family",
                    "Anion family distribution (top 15)",
                    top_n=15, palette_name="cubehelix")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "figG_anion_family", dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    plot_pressure_distribution(ax, tables.dataset_with_family)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "figH_pressure_distribution", dpi)
    plt.close(fig)

    plot_value_distributions_per_property(
        tables.dataset_with_family, fig_dir, dpi)


# --------------------------------------------------------------------------- #
# Composite (A–F) figure
# --------------------------------------------------------------------------- #
def make_composite_figure(tables: Tables, fig_dir: Path, dpi: int,
                          name: str = "Figure_dataset_overview") -> None:
    """Composite 2 × 3 layout for landscape double-column journal pages.

    Panel arrangement::

        A | B | C   (sample count) | (label heatmap)  | (IL coverage)
        D | E | F   (temperature)  | (cation family)  | (value distributions)
    """

    fig = plt.figure(figsize=(16.0, 10.0))
    gs = gridspec.GridSpec(
        nrows=2, ncols=3,
        figure=fig,
        wspace=0.35, hspace=0.42,
        left=0.07, right=0.985, top=0.93, bottom=0.16,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    plot_property_counts(ax_a, tables.property_sample_counts)
    plot_label_heatmap(ax_b, tables.il_label_coverage)
    plot_il_coverage(ax_c, tables.il_coverage_histogram)
    plot_temperature_distribution(ax_d, tables.dataset_with_family)
    plot_family_bar(ax_e, tables.cation_family_counts,
                    "Cation_Family", "Cation family distribution",
                    label_map=cation_family_short_labels())
    plot_value_distributions(ax_f, tables.dataset_with_family)
    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]:
        style_axes(ax)

    save_figure(fig, fig_dir, name, dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def make_all_figures(tables: Tables, fig_dir: Path, dpi: int) -> None:
    make_individual_figures(tables, fig_dir, dpi)
    make_composite_figure(tables, fig_dir, dpi)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the IL dataset.")
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/"
                     "ionic_liquid_6_properties_values_errors_ilthermo_strict.xlsx"),
        help="Excel input (used only if tables need to be regenerated).")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("exp1_dataset_analysis/outputs"),
        help="Root output directory containing tables/ and figures/.")
    parser.add_argument(
        "--tables-dir", type=Path, default=None,
        help="Directory of CSV tables (defaults to <output-dir>/tables).")
    parser.add_argument(
        "--figures-dir", type=Path, default=None,
        help="Directory to write figures (defaults to <output-dir>/figures).")
    parser.add_argument("--sheet", default="Merged")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font", default=DEFAULT_FONT_FAMILY)
    parser.add_argument("--rerun-analysis", action="store_true",
                        help="Force re-running the analysis pipeline.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    tables_dir = args.tables_dir or (args.output_dir / "tables")
    figures_dir = args.figures_dir or (args.output_dir / "figures")

    ensure_tables(tables_dir, args.input, args.output_dir,
                  sheet_name=args.sheet, rerun=args.rerun_analysis)

    configure_style(font_family=args.font)
    tables = load_tables(tables_dir)
    make_all_figures(tables, figures_dir, args.dpi)


if __name__ == "__main__":
    main()
