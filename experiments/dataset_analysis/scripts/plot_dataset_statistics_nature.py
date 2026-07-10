"""Nature-style dataset overview figure for the curated IL benchmark.

This script reads the analysis tables produced by ``analyze_dataset.py`` and
generates a compact multi-panel figure for the manuscript.  If the tables are
missing, it reruns the analysis from the current v2 interpolated workbook.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_dataset import (  # noqa: E402
    LOG_SCALE_PROPERTIES,
    PROPERTIES,
    PROPERTY_UNITS,
    Paths,
    run_analysis,
)

logger = logging.getLogger("plot_dataset_statistics_nature")

DEFAULT_INPUT = Path(
    "data/processed/"
    "ionic_liquid_6_properties_values_errors_ilthermo_strict_v2_interpolated.xlsx"
)
DEFAULT_OUTPUT_DIR = Path("experiments/dataset_analysis/outputs_v2_interpolated_nature")
DEFAULT_LATEX_COPY = Path("LaTex-MIPGraph/Fig/dataset_statistics.png")
DEFAULT_FONT_FAMILY = "Arial"

PROPERTY_LABELS: Dict[str, str] = {
    "Density": "Density",
    "ElectricalConductivity": "Elec. cond.",
    "HeatCapacity": "Heat cap.",
    "SurfaceTension": "Surf. tens.",
    "ThermalConductivity": "Therm. cond.",
    "Viscosity": "Viscosity",
}
PROPERTY_SHORT: Dict[str, str] = {
    "Density": "Density",
    "ElectricalConductivity": "EC",
    "HeatCapacity": "HC",
    "SurfaceTension": "ST",
    "ThermalConductivity": "TC",
    "Viscosity": "Visc.",
}
PROPERTY_COLORS: Dict[str, str] = {
    "Density": "#4C78A8",
    "ElectricalConductivity": "#59A14F",
    "HeatCapacity": "#F2BE5C",
    "SurfaceTension": "#B07AA1",
    "ThermalConductivity": "#76B7B2",
    "Viscosity": "#E15759",
}
SOURCE_ORDER: Sequence[str] = (
    "observed",
    "exact_condition_copy",
    "temperature_interpolation",
)
SOURCE_LABELS: Dict[str, str] = {
    "observed": "Observed",
    "exact_condition_copy": "Exact-condition copy",
    "temperature_interpolation": "Temperature interpolation",
}
SOURCE_COLORS: Dict[str, str] = {
    "observed": "#2F3A45",
    "exact_condition_copy": "#7AA6A1",
    "temperature_interpolation": "#D7A541",
}


@dataclass
class Tables:
    property_sample_counts: pd.DataFrame
    label_source_counts: pd.DataFrame
    il_label_coverage: pd.DataFrame
    il_coverage_histogram: pd.DataFrame
    cation_family_counts: pd.DataFrame
    anion_family_counts: pd.DataFrame
    summary_statistics: pd.DataFrame
    dataset_with_family: pd.DataFrame


def configure_style(font_family: str = DEFAULT_FONT_FAMILY,
                    base_size: float = 7.4) -> None:
    """Configure a restrained, editable Nature-style matplotlib theme."""

    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    if font_family not in available:
        logger.warning("Font %r not available; falling back to DejaVu Sans.",
                       font_family)
        font_family = "DejaVu Sans"

    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update({
        "font.family": font_family,
        "font.sans-serif": [font_family, "DejaVu Sans"],
        "font.size": base_size,
        "axes.titlesize": base_size + 0.6,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 0.3,
        "ytick.labelsize": base_size - 0.3,
        "legend.fontsize": base_size - 0.5,
        "legend.frameon": False,
        "axes.linewidth": 0.45,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "lines.linewidth": 0.85,
        "patch.linewidth": 0.35,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.default": "regular",
    })


def required_tables() -> Sequence[str]:
    return (
        "property_sample_counts.csv",
        "label_source_counts.csv",
        "il_label_coverage.csv",
        "il_coverage_histogram.csv",
        "cation_family_counts.csv",
        "anion_family_counts.csv",
        "summary_statistics.csv",
    )


def dataset_table_exists(tables_dir: Path) -> bool:
    return ((tables_dir / "dataset_with_family.parquet").exists()
            or (tables_dir / "dataset_with_family.csv").exists())


def ensure_tables(tables_dir: Path,
                  input_xlsx: Path,
                  output_dir: Path,
                  sheet_name: str,
                  rerun: bool) -> None:
    missing = [name for name in required_tables()
               if not (tables_dir / name).exists()]
    if rerun or missing or not dataset_table_exists(tables_dir):
        if missing:
            logger.info("Missing analysis tables: %s", ", ".join(missing))
        logger.info("Running dataset analysis from %s", input_xlsx)
        run_analysis(Paths(input_xlsx=input_xlsx,
                           output_dir=output_dir,
                           sheet_name=sheet_name))


def load_dataset_with_family(tables_dir: Path) -> pd.DataFrame:
    parquet = tables_dir / "dataset_with_family.parquet"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not read parquet (%s); using CSV.", exc)
    return pd.read_csv(tables_dir / "dataset_with_family.csv")


def load_tables(tables_dir: Path) -> Tables:
    return Tables(
        property_sample_counts=pd.read_csv(tables_dir / "property_sample_counts.csv"),
        label_source_counts=pd.read_csv(tables_dir / "label_source_counts.csv"),
        il_label_coverage=pd.read_csv(tables_dir / "il_label_coverage.csv"),
        il_coverage_histogram=pd.read_csv(tables_dir / "il_coverage_histogram.csv"),
        cation_family_counts=pd.read_csv(tables_dir / "cation_family_counts.csv"),
        anion_family_counts=pd.read_csv(tables_dir / "anion_family_counts.csv"),
        summary_statistics=pd.read_csv(tables_dir / "summary_statistics.csv"),
        dataset_with_family=load_dataset_with_family(tables_dir),
    )


def style_axes(ax: plt.Axes, grid_axis: Optional[str] = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#4A4A4A")
        ax.spines[side].set_linewidth(0.45)
    ax.tick_params(direction="out", colors="#303030", pad=1.5)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#E6E8EB", linewidth=0.35)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str,
                x: float = -0.10, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.0, fontweight="bold", color="black")


def format_count(value: float) -> str:
    value = float(value)
    if value >= 100000:
        return f"{value / 1000:.0f}k"
    if value >= 10000:
        return f"{value / 1000:.1f}k"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{int(value)}"


def _k_formatter(x: float, _pos: int) -> str:
    if abs(x) >= 1000:
        return f"{x / 1000:.0f}k"
    return f"{int(x)}"


def plot_summary_band(ax: plt.Axes, tables: Tables) -> None:
    ax.set_axis_off()
    panel_label(ax, "a", x=-0.015, y=0.93)

    df = tables.dataset_with_family
    counts = tables.property_sample_counts
    source = tables.label_source_counts
    n_records = len(df)
    n_ils = int(df["IL_SMILES"].nunique())
    n_labels = int(counts["N_Measurements"].sum())
    source_totals = (source.groupby("Source_Category")["N_Labels"]
                     .sum()
                     .reindex(SOURCE_ORDER, fill_value=0))

    ax.text(0.02, 0.83, "Sparse multi-property ionic-liquid benchmark",
            transform=ax.transAxes, fontsize=8.4, fontweight="bold",
            ha="left", va="center")

    metrics = (
        (n_records, "condition\nrecords"),
        (n_ils, "unique ionic\nliquids"),
        (n_labels, "available\nproperty labels"),
    )
    x0 = 0.03
    for i, (value, label) in enumerate(metrics):
        x = x0 + i * 0.17
        ax.text(x, 0.48, f"{value:,}", transform=ax.transAxes,
                fontsize=11.2, fontweight="bold", ha="left", va="center",
                color="#111111")
        ax.text(x, 0.17, label, transform=ax.transAxes, fontsize=6.5,
                ha="left", va="center", color="#505A63", linespacing=1.05)

    bar_x, bar_y, bar_w, bar_h = 0.58, 0.42, 0.36, 0.13
    total = max(float(source_totals.sum()), 1.0)
    left = bar_x
    for category in SOURCE_ORDER:
        width = bar_w * float(source_totals.loc[category]) / total
        ax.add_patch(Rectangle((left, bar_y), width, bar_h,
                               transform=ax.transAxes,
                               facecolor=SOURCE_COLORS[category],
                               edgecolor="white", linewidth=0.35))
        left += width
    ax.text(bar_x, 0.70, "source-aware label expansion",
            transform=ax.transAxes, fontsize=7.0, fontweight="bold",
            ha="left", va="center")
    legend_layout = (
        ("observed", bar_x, 0.19),
        ("exact_condition_copy", bar_x + 0.145, 0.19),
        ("temperature_interpolation", bar_x, 0.04),
    )
    for category, legend_x, legend_y in legend_layout:
        ax.add_patch(Rectangle((legend_x, legend_y - 0.03), 0.015, 0.055,
                               transform=ax.transAxes,
                               facecolor=SOURCE_COLORS[category],
                               edgecolor="none"))
        ax.text(legend_x + 0.02, legend_y,
                f"{SOURCE_LABELS[category]} ({format_count(source_totals.loc[category])})",
                transform=ax.transAxes, fontsize=6.0, ha="left", va="center",
                color="#404850")


def plot_label_source_counts(ax: plt.Axes, source_counts: pd.DataFrame) -> None:
    pivot = (source_counts
             .pivot(index="Property", columns="Source_Category", values="N_Labels")
             .reindex(PROPERTIES)
             .fillna(0.0))
    y = np.arange(len(PROPERTIES))
    left = np.zeros(len(PROPERTIES))
    for category in SOURCE_ORDER:
        values = pivot.get(category, pd.Series(0.0, index=pivot.index)).to_numpy()
        ax.barh(y, values, left=left, height=0.62,
                color=SOURCE_COLORS[category], edgecolor="white", linewidth=0.35,
                label=SOURCE_LABELS[category])
        left += values

    for yi, total in zip(y, left):
        ax.text(total + max(left) * 0.015, yi, f"{int(total):,}",
                ha="left", va="center", fontsize=6.3, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([PROPERTY_LABELS[p] for p in PROPERTIES])
    ax.invert_yaxis()
    ax.set_xlabel("Available labels")
    ax.set_title("Label availability after curation")
    ax.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
    ax.set_xlim(0, max(left) * 1.22)
    style_axes(ax, grid_axis="x")
    panel_label(ax, "b")


def plot_label_matrix(ax: plt.Axes, coverage: pd.DataFrame) -> None:
    cov = coverage.copy()
    cov = cov.sort_values(["N_Labels", *PROPERTIES],
                          ascending=[False, *([False] * len(PROPERTIES))])
    data = cov[list(PROPERTIES)].astype(int).to_numpy()
    cmap = mpl.colors.ListedColormap(["#F0F2F3", "#234F77"])
    ax.imshow(data, aspect="auto", interpolation="nearest",
              cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(PROPERTIES)))
    ax.set_xticklabels([PROPERTY_SHORT[p] for p in PROPERTIES], rotation=0)
    ax.set_yticks([0, len(cov) - 1])
    ax.set_yticklabels(["high", "low"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Property")
    ax.set_ylabel("Label coverage")
    ax.set_title("Sparse label matrix")
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "c")


def plot_coverage_histogram(ax: plt.Axes, hist: pd.DataFrame) -> None:
    hist = hist.sort_values("N_Labels")
    colors = sns.light_palette("#2F6F8F", n_colors=len(hist) + 2)[2:]
    bars = ax.bar(hist["N_Labels"], hist["N_IL"], width=0.72,
                  color=colors, edgecolor="white", linewidth=0.4)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + max(hist["N_IL"]) * 0.025,
                f"{int(height)}", ha="center", va="bottom", fontsize=6.2)
    ax.set_xlabel("Labels per IL")
    ax.set_ylabel("Number of ILs")
    ax.set_title("Per-IL coverage")
    ax.set_xticks(range(1, len(PROPERTIES) + 1))
    ax.set_ylim(0, max(hist["N_IL"]) * 1.18)
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    style_axes(ax, grid_axis="y")
    panel_label(ax, "d")


def plot_temperature_distribution(ax: plt.Axes, df: pd.DataFrame) -> None:
    all_t = df["Temperature_K"].dropna()
    lo = max(150.0, float(np.floor(all_t.quantile(0.005) / 10.0) * 10.0))
    hi = min(750.0, float(np.ceil(all_t.quantile(0.995) / 10.0) * 10.0))
    bins = np.linspace(lo, hi, 38)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for prop in PROPERTIES:
        vals = df.loc[df[f"{prop}_ActualValue"].notna(), "Temperature_K"].dropna()
        counts, _ = np.histogram(vals, bins=bins)
        ax.plot(centers, counts, color=PROPERTY_COLORS[prop],
                label=PROPERTY_SHORT[prop])
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Labels per bin")
    ax.set_title("Condition coverage")
    ax.legend(loc="upper right", ncol=2, columnspacing=0.6,
              handlelength=1.0, handletextpad=0.25, borderaxespad=0.2)
    style_axes(ax, grid_axis="y")
    panel_label(ax, "e")


def plot_pressure_distribution(ax: plt.Axes, df: pd.DataFrame) -> None:
    pressure = df["Pressure_kPa"].dropna()
    positive = pressure[pressure > 0]
    if positive.empty:
        ax.text(0.5, 0.5, "No pressure records", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_axis_off()
        return

    ambient = int(np.isclose(positive.to_numpy(), 101.325, atol=1e-6).sum())
    non_ambient = int(len(pressure) - ambient)
    missing = int(df["Pressure_kPa"].isna().sum())
    labels = ["Ambient\n101.325 kPa", "Non-ambient\nreported", "Missing"]
    values = np.array([ambient, non_ambient, missing], dtype=float)
    colors = ["#6F7D8C", "#A9B7C2", "#D7A541"]
    y = np.arange(len(values))
    bars = ax.barh(y, values, color=colors, edgecolor="white",
                   linewidth=0.35, height=0.62)
    for bar, value in zip(bars, values):
        ax.text(value + values.max() * 0.025,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value):,}",
                ha="left", va="center", fontsize=6.2, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Records")
    ax.set_title("Pressure coverage")
    ax.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
    ax.set_xlim(0, values.max() * 1.28)
    ax.text(0.98, 0.08,
            f"range {positive.min():g}-{positive.max():,.0f} kPa",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.9, color="#505A63")
    style_axes(ax, grid_axis="x")
    panel_label(ax, "h")


def collapse_top_families(counts: pd.DataFrame,
                          column: str,
                          top_n: int,
                          prefix: str) -> pd.DataFrame:
    ordered = counts.sort_values("N_IL", ascending=False).reset_index(drop=True)
    keep = ordered.iloc[:top_n].copy()
    if len(ordered) > top_n:
        other = pd.DataFrame([{
            column: "Other",
            "N_IL": int(ordered.iloc[top_n:]["N_IL"].sum()),
            "Pct_IL": float(ordered.iloc[top_n:]["Pct_IL"].sum()),
        }])
        keep = pd.concat([keep, other], ignore_index=True)
    keep["Ion"] = prefix
    keep["Family"] = keep[column].astype(str)
    return (keep[["Ion", "Family", "N_IL", "Pct_IL"]]
            .groupby(["Ion", "Family"], as_index=False)
            .agg({"N_IL": "sum", "Pct_IL": "sum"})
            .sort_values("N_IL", ascending=False)
            .reset_index(drop=True))


def short_family_name(name: str) -> str:
    mapping = {
        "Quaternary ammonium": "Quat. ammonium",
        "Protic ammonium": "Protic ammonium",
        "Pyrrolidinium": "Pyrrolidinium",
        "Imidazolium": "Imidazolium",
        "Phosphonium": "Phosphonium",
        "Carboxylate": "Carboxylate",
        "Alkyl sulfate": "Alkyl sulfate",
        "Alkyl sulfonate": "Alkyl sulfonate",
        "Dicyanamide": "Dicyanamide",
    }
    return mapping.get(name, name)


def plot_ion_family_distribution(ax: plt.Axes,
                                 cation_counts: pd.DataFrame,
                                 anion_counts: pd.DataFrame) -> None:
    cat = collapse_top_families(cation_counts, "Cation_Family", 4, "Cation")
    ani = collapse_top_families(anion_counts, "Anion_Family", 4, "Anion")
    rows = pd.concat([cat, ani], ignore_index=True)
    rows["Label"] = rows["Ion"].str[0] + ": " + rows["Family"].map(short_family_name)
    rows = rows.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(rows))
    colors = rows["Ion"].map({"Cation": "#447C7A", "Anion": "#C49033"}).tolist()
    ax.barh(y, rows["N_IL"], color=colors, edgecolor="white", linewidth=0.35,
            height=0.64)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["Label"])
    ax.tick_params(axis="y", labelsize=5.45)
    ax.set_xlabel("Number of ILs")
    ax.set_title("Ion-family diversity")
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.legend(handles=[
        Patch(facecolor="#447C7A", label="Cation family"),
        Patch(facecolor="#C49033", label="Anion family"),
    ], loc="lower right", handlelength=1.0, handletextpad=0.35,
        labelspacing=0.25)
    style_axes(ax, grid_axis="x")
    panel_label(ax, "f")


def plot_value_distributions(ax: plt.Axes, df: pd.DataFrame) -> None:
    data = []
    labels = []
    colors = []
    for prop in PROPERTIES:
        values = df[f"{prop}_ActualValue"].dropna()
        if prop in LOG_SCALE_PROPERTIES:
            values = values[values > 0]
            ref = np.log10(values)
        else:
            ref = values
        if ref.empty:
            continue
        q_lo, q_hi = ref.quantile([0.01, 0.99])
        if q_hi <= q_lo:
            q_lo, q_hi = ref.min(), ref.max()
        scaled = ((ref - q_lo) / (q_hi - q_lo)).clip(-0.15, 1.15)
        data.append(scaled.to_numpy())
        unit = PROPERTY_UNITS[prop]
        suffix = ", log" if prop in LOG_SCALE_PROPERTIES else ""
        labels.append(f"{PROPERTY_SHORT[prop]}\n({unit}{suffix})")
        colors.append(PROPERTY_COLORS[prop])

    bp = ax.boxplot(data, patch_artist=True, widths=0.58, showfliers=False,
                    medianprops={"color": "#1E1E1E", "linewidth": 0.7},
                    whiskerprops={"color": "#4A4A4A", "linewidth": 0.55},
                    capprops={"color": "#4A4A4A", "linewidth": 0.55},
                    boxprops={"edgecolor": "#3A3A3A", "linewidth": 0.55})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Robust-scaled value")
    ax.set_title("Property value ranges")
    ax.axhline(0, color="#8F969C", linewidth=0.4, linestyle=":")
    ax.axhline(1, color="#8F969C", linewidth=0.4, linestyle=":")
    ax.set_ylim(-0.18, 1.18)
    style_axes(ax, grid_axis="y")
    panel_label(ax, "g")


def save_figure(fig: plt.Figure, fig_dir: Path, name: str, dpi: int) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "pdf": {},
        "svg": {},
        "png": {"dpi": dpi},
        "tiff": {"dpi": dpi},
    }
    for ext, kwargs in exports.items():
        out = fig_dir / f"{name}.{ext}"
        fig.savefig(out, **kwargs)
        logger.info("Saved %s", out)


def copy_exports(fig_dir: Path, name: str, target_png: Optional[Path]) -> None:
    if target_png is None:
        return
    target_png.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg", "tiff"):
        src = fig_dir / f"{name}.{ext}"
        if src.exists():
            dst = target_png.with_suffix(f".{ext}")
            shutil.copy2(src, dst)
            logger.info("Copied %s -> %s", src, dst)


def make_composite_figure(tables: Tables,
                          fig_dir: Path,
                          name: str,
                          dpi: int) -> None:
    fig = plt.figure(figsize=(6.35, 6.10))
    gs = gridspec.GridSpec(
        nrows=4, ncols=6, figure=fig,
        height_ratios=[0.58, 1.42, 1.28, 1.48],
        hspace=0.72, wspace=0.88,
        left=0.085, right=0.985, top=0.975, bottom=0.075,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0:3])
    ax_c = fig.add_subplot(gs[1, 3:6])
    ax_d = fig.add_subplot(gs[2, 0:2])
    ax_e = fig.add_subplot(gs[2, 2:4])
    ax_f = fig.add_subplot(gs[2, 4:6])
    ax_g = fig.add_subplot(gs[3, 0:4])
    ax_h = fig.add_subplot(gs[3, 4:6])

    plot_summary_band(ax_a, tables)
    plot_label_source_counts(ax_b, tables.label_source_counts)
    plot_label_matrix(ax_c, tables.il_label_coverage)
    plot_coverage_histogram(ax_d, tables.il_coverage_histogram)
    plot_temperature_distribution(ax_e, tables.dataset_with_family)
    plot_ion_family_distribution(ax_f,
                                 tables.cation_family_counts,
                                 tables.anion_family_counts)
    plot_value_distributions(ax_g, tables.dataset_with_family)
    plot_pressure_distribution(ax_h, tables.dataset_with_family)

    save_figure(fig, fig_dir, name, dpi)
    plt.close(fig)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Nature-style IL dataset statistics figure.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--sheet", default="Merged")
    parser.add_argument("--name", default="Figure_dataset_overview_nature")
    parser.add_argument("--copy-to", type=Path, default=DEFAULT_LATEX_COPY)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--font", default=DEFAULT_FONT_FAMILY)
    parser.add_argument("--rerun-analysis", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    tables_dir = args.tables_dir or (args.output_dir / "tables")
    figures_dir = args.figures_dir or (args.output_dir / "figures")
    ensure_tables(tables_dir=tables_dir,
                  input_xlsx=args.input,
                  output_dir=args.output_dir,
                  sheet_name=args.sheet,
                  rerun=args.rerun_analysis)
    configure_style(args.font)
    tables = load_tables(tables_dir)
    make_composite_figure(tables, figures_dir, args.name, args.dpi)
    copy_exports(figures_dir, args.name, args.copy_to)


if __name__ == "__main__":
    main()
