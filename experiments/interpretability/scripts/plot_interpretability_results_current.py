"""Create a current-model interpretability figure from final test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROPERTY_ORDER = [
    "Density",
    "Viscosity",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
]
PROPERTY_SHORT = {
    "Density": "Density",
    "Viscosity": "Visc.",
    "ElectricalConductivity": "EC",
    "HeatCapacity": "HC",
    "SurfaceTension": "ST",
    "ThermalConductivity": "TC",
}
PROPERTY_COLORS = {
    "Density": "#4C78A8",
    "Viscosity": "#E15759",
    "ElectricalConductivity": "#59A14F",
    "HeatCapacity": "#F2BE5C",
    "SurfaceTension": "#B07AA1",
    "ThermalConductivity": "#76B7B2",
}
CATION_COLORS = {
    "Imidazolium": "#447C7A",
    "Pyridinium": "#6A8EAE",
    "Phosphonium": "#C49033",
    "Quaternary ammonium": "#A66C9A",
    "Protic ammonium": "#7A7A7A",
    "Cholinium": "#7AA36F",
    "Other": "#C8CDD2",
}
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "mipgraph_error_landscape",
    ["#F7FAFA", "#D9E9E6", "#A9D1CB", "#66A8AA", "#2E6F8E"],
)
CORR_CMAP = LinearSegmentedColormap.from_list(
    "mipgraph_corr",
    ["#4C78A8", "#F7F7F2", "#C65B5B"],
)


def configure_style() -> None:
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    font_family = "Arial" if "Arial" in available else "DejaVu Sans"
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update({
        "font.family": font_family,
        "font.sans-serif": [font_family, "DejaVu Sans"],
        "font.size": 6.7,
        "axes.titlesize": 7.2,
        "axes.labelsize": 6.7,
        "xtick.labelsize": 5.9,
        "ytick.labelsize": 5.9,
        "legend.fontsize": 5.8,
        "legend.frameon": False,
        "axes.linewidth": 0.45,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
    })


def style_axes(ax: plt.Axes, grid_axis: Optional[str] = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#3F3F3F")
        ax.spines[side].set_linewidth(0.45)
    ax.tick_params(direction="out", pad=1.2)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#E6E8EB", linewidth=0.35)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=8.4, fontweight="bold")


def safe_log(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.log(np.clip(arr, 1e-12, None))


def merge_families(df: pd.DataFrame, family_path: Path) -> pd.DataFrame:
    fam = pd.read_csv(family_path)
    keep = ["IL_SMILES", "Cation_Family", "Anion_Family"]
    fam = fam[keep].drop_duplicates("IL_SMILES")
    out = df.merge(fam, on="IL_SMILES", how="left")
    out["Cation_Family"] = out["Cation_Family"].fillna("Other")
    out["Anion_Family"] = out["Anion_Family"].fillna("Other")
    return out


def add_long_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["positive_prediction"] = (out["y_true"] > 0) & (out["y_pred"] > 0)
    out["signed_log_error"] = np.nan
    mask = out["positive_prediction"]
    out.loc[mask, "signed_log_error"] = (
        np.log(out.loc[mask, "y_pred"]) - np.log(out.loc[mask, "y_true"])
    )
    out["abs_log_error"] = out["signed_log_error"].abs()
    return out


def add_wide_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for prop in PROPERTY_ORDER:
        true_col = f"{prop}_true"
        pred_col = f"{prop}_pred"
        err_col = f"{prop}_abs_log_error"
        if true_col not in out.columns:
            out[err_col] = np.nan
            continue
        mask = out[true_col].notna() & out[pred_col].notna()
        mask &= (out[true_col] > 0) & (out[pred_col] > 0)
        out[err_col] = np.nan
        out.loc[mask, err_col] = (
            np.log(out.loc[mask, pred_col]) - np.log(out.loc[mask, true_col])
        ).abs()
    return out


def property_metrics(metrics_log_path: Path,
                     metrics_json_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_log_path)
    with metrics_json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for prop in PROPERTY_ORDER:
        row = metrics[metrics["property"] == prop].iloc[0].to_dict()
        row["label_count"] = int(raw[prop]["label_count"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_response_manifold(ax: plt.Axes, wide: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [f"{p}_pred" for p in PROPERTY_ORDER]
    features = np.column_stack([safe_log(wide[col]) for col in pred_cols])
    features = StandardScaler().fit_transform(features)
    coords = PCA(n_components=2, random_state=42).fit_transform(features)
    out = wide[["sample_id", "IL_Name", "IL_SMILES", "Temperature_K",
                "Pressure_kPa", "Cation_Family", "Anion_Family"]].copy()
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1]
    top = wide["Cation_Family"].value_counts().head(6).index.tolist()
    out["display_family"] = out["Cation_Family"].where(
        out["Cation_Family"].isin(top), "Other"
    )
    for fam, group in out.groupby("display_family"):
        ax.scatter(group["PC1"], group["PC2"], s=5.5,
                   color=CATION_COLORS.get(fam, "#C8CDD2"),
                   alpha=0.62, edgecolors="none", label=fam)
    ax.set_xlabel("Predicted fingerprint PC1")
    ax.set_ylabel("Predicted fingerprint PC2")
    ax.set_title("Multi-property response manifold")
    ax.legend(loc="best", markerscale=1.6, handletextpad=0.25,
              labelspacing=0.2, ncol=1)
    style_axes(ax)
    panel_label(ax, "a")
    return out


def plot_property_difficulty(ax: plt.Axes, metrics: pd.DataFrame) -> pd.DataFrame:
    for _, row in metrics.iterrows():
        prop = row["property"]
        size = 22 + 0.055 * float(row["label_count"])
        ax.scatter(row["log_NMAE"], row["log_R2"], s=size,
                   color=PROPERTY_COLORS[prop], edgecolor="white",
                   linewidth=0.45, alpha=0.9)
        ax.text(row["log_NMAE"] + 0.003, row["log_R2"],
                PROPERTY_SHORT[prop], ha="left", va="center", fontsize=6.2)
    ax.set_xlabel("log-space NMAE")
    ax.set_ylabel(r"log-space $R^2$")
    ax.set_title("Property difficulty")
    ax.set_xlim(0.025, 0.13)
    ax.set_ylim(0.94, 1.0)
    style_axes(ax, "both")
    panel_label(ax, "b")
    return metrics.copy()


def plot_error_distribution(ax: plt.Axes, long: pd.DataFrame) -> pd.DataFrame:
    data = []
    labels = []
    for prop in PROPERTY_ORDER:
        vals = long.loc[long["property"] == prop, "abs_log_error"].dropna()
        data.append(vals.to_numpy())
        labels.append(PROPERTY_SHORT[prop])
    bp = ax.boxplot(data, patch_artist=True, widths=0.58, showfliers=False,
                    medianprops={"color": "#202020", "linewidth": 0.65},
                    boxprops={"edgecolor": "#3A3A3A", "linewidth": 0.5},
                    whiskerprops={"color": "#3A3A3A", "linewidth": 0.45},
                    capprops={"color": "#3A3A3A", "linewidth": 0.45})
    for patch, prop in zip(bp["boxes"], PROPERTY_ORDER):
        patch.set_facecolor(PROPERTY_COLORS[prop])
        patch.set_alpha(0.72)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$|\Delta \log y|$")
    ax.set_title("Error distribution")
    style_axes(ax, "y")
    panel_label(ax, "c")
    return long[["property", "abs_log_error", "positive_prediction"]].copy()


def heatmap_table(ax: plt.Axes,
                  table: pd.DataFrame,
                  title: str,
                  label: str,
                  cbar_label: str = r"median $|\Delta \log y|$",
                  vmax: Optional[float] = None) -> None:
    if vmax is None:
        finite = table.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        vmax = float(np.nanpercentile(finite, 92)) if len(finite) else 0.2
        vmax = max(vmax, 0.05)
    sns.heatmap(table, ax=ax, cmap=ERROR_CMAP, vmin=0, vmax=vmax,
                linewidths=0.35, linecolor="white", cbar=True,
                cbar_kws={"label": cbar_label, "shrink": 0.78, "pad": 0.02})
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    panel_label(ax, label)


def plot_temperature_heatmap(ax: plt.Axes, long: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 280, 300, 320, 360, np.inf]
    labels = ["<280", "280-300", "300-320", "320-360", ">360"]
    out = long.dropna(subset=["abs_log_error"]).copy()
    out["temperature_bin"] = pd.cut(out["Temperature_K"], bins=bins,
                                    labels=labels, right=False)
    table = (out.pivot_table(index="property", columns="temperature_bin",
                             values="abs_log_error", aggfunc="median",
                             observed=False)
             .reindex(PROPERTY_ORDER))
    table.index = [PROPERTY_SHORT[p] for p in table.index]
    heatmap_table(ax, table, "Temperature-dependent error", "d")
    return out[["property", "Temperature_K", "temperature_bin",
                "abs_log_error"]].copy()


def plot_pressure_heatmap(ax: plt.Axes, long: pd.DataFrame) -> pd.DataFrame:
    out = long.dropna(subset=["abs_log_error"]).copy()
    out["pressure_category"] = "non-ambient"
    out.loc[out["Pressure_kPa"].isna(), "pressure_category"] = "missing"
    ambient = np.isclose(out["Pressure_kPa"].fillna(-9999), 101.325, atol=1e-6)
    out.loc[ambient, "pressure_category"] = "ambient"
    categories = ["ambient", "non-ambient", "missing"]
    table = (out.pivot_table(index="property", columns="pressure_category",
                             values="abs_log_error", aggfunc="median",
                             observed=False)
             .reindex(PROPERTY_ORDER)[categories])
    table.index = [PROPERTY_SHORT[p] for p in table.index]
    heatmap_table(ax, table, "Pressure-dependent error", "e")
    return out[["property", "Pressure_kPa", "pressure_category",
                "abs_log_error"]].copy()


def plot_error_correlation(ax: plt.Axes, wide: pd.DataFrame) -> pd.DataFrame:
    cols = [f"{p}_abs_log_error" for p in PROPERTY_ORDER]
    corr = wide[cols].corr(method="spearman", min_periods=20)
    corr.index = [PROPERTY_SHORT[p] for p in PROPERTY_ORDER]
    corr.columns = [PROPERTY_SHORT[p] for p in PROPERTY_ORDER]
    sns.heatmap(corr, ax=ax, cmap=CORR_CMAP, vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"fontsize": 5.2},
                linewidths=0.35, linecolor="white",
                cbar_kws={"label": r"Spearman $\rho$", "shrink": 0.78,
                          "pad": 0.02})
    ax.set_title("Shared error modes")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    panel_label(ax, "f")
    return corr.reset_index(names="property")


def family_heatmap(ax: plt.Axes,
                   long: pd.DataFrame,
                   family_col: str,
                   top_n: int,
                   title: str,
                   label: str) -> pd.DataFrame:
    out = long.dropna(subset=["abs_log_error"]).copy()
    top = out[family_col].value_counts().head(top_n).index.tolist()
    out["display_family"] = out[family_col].where(
        out[family_col].isin(top), "Other"
    )
    table = (out.pivot_table(index="display_family", columns="property",
                             values="abs_log_error", aggfunc="median",
                             observed=False)
             .reindex(columns=PROPERTY_ORDER))
    order = out["display_family"].value_counts().index.tolist()
    table = table.reindex(order)
    table.columns = [PROPERTY_SHORT[p] for p in table.columns]
    heatmap_table(ax, table, title, label)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", labelsize=5.4)
    return out[[family_col, "display_family", "property",
                "abs_log_error"]].copy()


def write_sources(output_dir: Path, sources: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for panel, df in sources.items():
        df.to_csv(output_dir / f"interpretability_results_source_data_{panel}.csv",
                  index=False)


def create_figure(pred_long_path: Path,
                  pred_wide_path: Path,
                  metrics_log_path: Path,
                  metrics_json_path: Path,
                  family_path: Path,
                  output_dir: Path,
                  source_data_dir: Path,
                  name: str,
                  dpi: int) -> None:
    long = pd.read_csv(pred_long_path)
    wide = pd.read_csv(pred_wide_path)
    long = merge_families(long, family_path)
    wide = merge_families(wide, family_path)
    long = add_long_errors(long)
    wide = add_wide_errors(wide)
    metrics = property_metrics(metrics_log_path, metrics_json_path)

    configure_style()
    fig = plt.figure(figsize=(7.2, 8.15))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.05, 1.0, 1.18],
                          hspace=0.72, wspace=0.72,
                          left=0.075, right=0.985,
                          top=0.965, bottom=0.07)
    axes = {
        "A": fig.add_subplot(gs[0, 0:2]),
        "B": fig.add_subplot(gs[0, 2:4]),
        "C": fig.add_subplot(gs[0, 4:6]),
        "D": fig.add_subplot(gs[1, 0:2]),
        "E": fig.add_subplot(gs[1, 2:4]),
        "F": fig.add_subplot(gs[1, 4:6]),
        "G": fig.add_subplot(gs[2, 0:3]),
        "H": fig.add_subplot(gs[2, 3:6]),
    }

    sources = {
        "A": plot_response_manifold(axes["A"], wide),
        "B": plot_property_difficulty(axes["B"], metrics),
        "C": plot_error_distribution(axes["C"], long),
        "D": plot_temperature_heatmap(axes["D"], long),
        "E": plot_pressure_heatmap(axes["E"], long),
        "F": plot_error_correlation(axes["F"], wide),
        "G": family_heatmap(axes["G"], long, "Cation_Family", 6,
                            "Cation-family error landscape", "g"),
        "H": family_heatmap(axes["H"], long, "Anion_Family", 7,
                            "Anion-family error landscape", "h"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": dpi},
        "tiff": {"dpi": dpi},
    }.items():
        fig.savefig(output_dir / f"{name}.{ext}", **kwargs)
    plt.close(fig)
    write_sources(source_data_dir, sources)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot current MIPGraph interpretability figure.")
    root = Path("il_property_prediction/outputs/"
                "fg_transformer_random_point_seed42_noamp")
    run = "unimol2_fg_transformer_random_point_seed42_noamp_resume56"
    parser.add_argument("--pred-long", type=Path,
                        default=root / "predictions" / run / "test_predictions.csv")
    parser.add_argument("--pred-wide", type=Path,
                        default=root / "predictions" / run / "test_predictions_wide.csv")
    parser.add_argument("--metrics-log", type=Path,
                        default=root / "metrics" / run / "test_metrics_log.csv")
    parser.add_argument("--metrics-json", type=Path,
                        default=root / "metrics" / run / "test_metrics.json")
    parser.add_argument("--family-table", type=Path,
                        default=Path("experiments/dataset_analysis/"
                                     "outputs_v2_interpolated_nature/tables/"
                                     "per_il_family_assignment.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("LaTex-MIPGraph/Fig"))
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=Path(
            "experiments/manuscript_figure_source_data/"
            "interpretability_feature_importance_4x3"
        ),
    )
    parser.add_argument("--name", default="interpretability_results")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    create_figure(args.pred_long, args.pred_wide, args.metrics_log,
                  args.metrics_json, args.family_table,
                  args.output_dir, args.source_data_dir,
                  args.name, args.dpi)
    print(f"Wrote {args.output_dir / (args.name + '.png')}")


if __name__ == "__main__":
    main()
