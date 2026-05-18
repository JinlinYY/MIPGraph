from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from metrics import compute_metrics_by_groups


PROPERTY_ORDER = [
    "Density",
    "Viscosity",
    "ElectricalConductivity",
    "SurfaceTension",
    "HeatCapacity",
    "ThermalConductivity",
]
PROPERTY_LABELS = {
    "Density": "Dens.",
    "Viscosity": "Visc.",
    "ElectricalConductivity": "EC",
    "SurfaceTension": "ST",
    "HeatCapacity": "Cp",
    "ThermalConductivity": "TC",
}
PROPERTY_PALETTE = sns.color_palette("Set2", n_colors=len(PROPERTY_ORDER))
PROPERTY_COLORS = dict(zip(PROPERTY_ORDER, PROPERTY_PALETTE))
MAIN_FONT_SIZE = 21
MAIN_ANNOTATION_SIZE = 13
REFERENCE_LINE_COLOR = "#5F6368"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    default_input = repo_root / "il_property_prediction" / "outputs" / "predictions" / "finetune_viscosity_from_weak_seed42"
    default_output = repo_root / "exp2_performance" / "outputs"
    parser = argparse.ArgumentParser(description="Generate publication-quality performance plots.")
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--prediction-file", default="test_predictions.csv")
    parser.add_argument("--split", default="test", help="Split to visualize.")
    return parser.parse_args()


def setup_style(font_family: str = "Arial", base_size: float = 18.0) -> None:
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    if font_family not in available:
        font_family = "DejaVu Sans"

    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": [font_family, "DejaVu Sans"],
            "font.size": base_size,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": base_size,
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "legend.fontsize": base_size - 2,
            "legend.frameon": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
        }
    )


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def save_both(fig: plt.Figure, out_stem: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(pad=0.25)
    for suffix in (".png", ".pdf"):
        out_path = out_stem.with_suffix(suffix)
        try:
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        except PermissionError:
            print(f"Skipped locked figure: {out_path}")
    plt.close(fig)


def sync_legacy_png_aliases(fig_dir: Path) -> None:
    aliases = {
        "main_figure.png": ["main_figure_2.png"],
        "parity_plots.png": [
            "main_figure_parityplot分成6张图_总图2x2.png",
            "main_figure_parityplot分成6张图_总图3x3.png",
        ],
        "property_wise_performance.png": ["main_figure_六个指标合一张图.png"],
        "residual_distributions.png": ["main_figure_残差图6项性质分开.png"],
    }
    for src_name, dst_names in aliases.items():
        src = fig_dir / src_name
        if not src.exists():
            continue
        for dst_name in dst_names:
            dst = fig_dir / dst_name
            if not dst.exists():
                continue
            try:
                shutil.copyfile(src, dst)
            except PermissionError:
                print(f"Skipped locked legacy figure alias: {dst}")


def load_prediction_df(input_dir: Path, output_dir: Path, prediction_file: str) -> pd.DataFrame:
    cached = output_dir / "tables" / "predictions_long.csv"
    if cached.exists():
        df = pd.read_csv(cached)
    else:
        pred_path = input_dir / prediction_file
        df = pd.read_csv(pred_path)
    required = {"property", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        missing = sorted(required.difference(df.columns))
        raise ValueError(f"Missing required columns in prediction file: {missing}")
    if "split" not in df.columns:
        df["split"] = "test"
    df = df.dropna(subset=["property", "y_true", "y_pred"]).copy()
    df["residual"] = df["y_pred"] - df["y_true"]
    return df


def get_metrics_for_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split.lower() == "all":
        use_df = df
    else:
        use_df = df[df["split"] == split]
    metrics = compute_metrics_by_groups(use_df, ["property"])
    if metrics.empty:
        raise ValueError(f"No rows found for split={split!r}")
    return metrics.sort_values("property").reset_index(drop=True)


def ordered_properties(properties: list[str] | pd.Series | np.ndarray) -> list[str]:
    available = set(properties)
    ordered = [prop for prop in PROPERTY_ORDER if prop in available]
    ordered.extend(sorted(prop for prop in available if prop not in PROPERTY_ORDER))
    return ordered


def property_label(prop: str) -> str:
    return PROPERTY_LABELS.get(prop, prop)


def property_color(prop: str):
    if prop in PROPERTY_COLORS:
        return PROPERTY_COLORS[prop]
    idx = abs(hash(prop)) % len(PROPERTY_PALETTE)
    return PROPERTY_PALETTE[idx]


def plot_parity_panels(df: pd.DataFrame, fig_dir: Path) -> None:
    properties = ordered_properties(df["property"].unique())
    n_props = len(properties)
    n_cols = 3
    n_rows = int(np.ceil(n_props / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.2, 5.8), constrained_layout=False)
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for idx, prop in enumerate(properties):
        ax = axes[idx // n_cols, idx % n_cols]
        sub = df[df["property"] == prop]
        ax.scatter(sub["y_true"], sub["y_pred"], s=12, alpha=0.6, color=property_color(prop), edgecolor="none")
        vmin = min(sub["y_true"].min(), sub["y_pred"].min())
        vmax = max(sub["y_true"].max(), sub["y_pred"].max())
        ax.plot([vmin, vmax], [vmin, vmax], "--", lw=1.0, color=REFERENCE_LINE_COLOR)
        ax.set_title(property_label(prop))
        ax.set_xlabel("True log")
        ax.set_ylabel("Pred. log")
        style_axes(ax)
    for idx in range(n_props, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    save_both(fig, fig_dir / "parity_plots")


def plot_residual_panels(df: pd.DataFrame, fig_dir: Path) -> None:
    properties = ordered_properties(df["property"].unique())
    n_props = len(properties)
    n_cols = 3
    n_rows = int(np.ceil(n_props / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.2, 5.8), constrained_layout=False)
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for idx, prop in enumerate(properties):
        ax = axes[idx // n_cols, idx % n_cols]
        sub = df[df["property"] == prop]
        sns.histplot(sub["residual"], bins=28, kde=True, ax=ax, color=property_color(prop))
        ax.axvline(0.0, ls="--", lw=1.0, color=REFERENCE_LINE_COLOR)
        ax.set_title(property_label(prop))
        ax.set_xlabel("Log residual (Pred - True)")
        ax.set_ylabel("Count")
        style_axes(ax)
    for idx in range(n_props, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    save_both(fig, fig_dir / "residual_distributions")


def plot_property_bar(metrics: pd.DataFrame, fig_dir: Path) -> None:
    metrics = metrics.copy()
    metrics["property_label"] = metrics["property"].map(property_label)
    order = [property_label(prop) for prop in ordered_properties(metrics["property"].unique())]
    palette = [property_color(prop) for prop in ordered_properties(metrics["property"].unique())]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    sns.barplot(data=metrics, x="property_label", y="MAE", ax=axes[0], order=order, palette=palette, hue="property_label", legend=False)
    sns.barplot(data=metrics, x="property_label", y="RMSE", ax=axes[1], order=order, palette=palette, hue="property_label", legend=False)
    sns.barplot(data=metrics, x="property_label", y="R2", ax=axes[2], order=order, palette=palette, hue="property_label", legend=False)
    for ax, name in zip(axes, ["Log MAE", "Log RMSE", "Log R2"]):
        ax.set_title(name)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        style_axes(ax)
    save_both(fig, fig_dir / "property_wise_performance")


def plot_radar(metrics: pd.DataFrame, fig_dir: Path) -> None:
    metrics = metrics.copy()
    metrics["property"] = pd.Categorical(metrics["property"], categories=PROPERTY_ORDER, ordered=True)
    metrics = metrics.sort_values("property")
    labels = [property_label(str(prop)) for prop in metrics["property"].tolist()]
    mae = metrics["MAE"].to_numpy()
    rmse = metrics["RMSE"].to_numpy()
    r2 = np.clip(metrics["R2"].to_numpy(), 0.0, 1.0)

    # Convert errors to "higher is better" scores in [0, 1].
    nmae = mae / (mae.max() + 1e-12)
    nrmse = rmse / (rmse.max() + 1e-12)
    mae_score = 1.0 - nmae
    rmse_score = 1.0 - nrmse

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    def closed(values: np.ndarray) -> np.ndarray:
        return np.concatenate([values, values[:1]])

    fig = plt.figure(figsize=(3.5, 3.2))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, closed(r2), lw=1.6, color=PROPERTY_PALETTE[0], label="R2")
    ax.fill(angles, closed(r2), alpha=0.15, color=PROPERTY_PALETTE[0])
    ax.plot(angles, closed(mae_score), lw=1.4, color=PROPERTY_PALETTE[1], label="1-NMAE")
    ax.plot(angles, closed(rmse_score), lw=1.4, color=PROPERTY_PALETTE[2], label="1-NRMSE")
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Radar of Property Metrics", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=False)
    style_axes(ax)
    save_both(fig, fig_dir / "radar_chart")


def build_main_summary(df: pd.DataFrame, metrics: pd.DataFrame, properties: list[str]) -> pd.DataFrame:
    metric_cols = ["property", "MAE", "RMSE", "R2", "N"]
    summary = metrics[metric_cols].copy()
    spread_rows: list[dict[str, float | str]] = []
    for prop in properties:
        sub = df[df["property"] == prop]
        if sub.empty:
            spread_rows.append(
                {
                    "property": prop,
                    "true_range": np.nan,
                    "residual_iqr": np.nan,
                    "residual_iqr_norm": np.nan,
                }
            )
            continue
        true_range = float(sub["y_true"].max() - sub["y_true"].min())
        residual_iqr = float(sub["residual"].quantile(0.75) - sub["residual"].quantile(0.25))
        spread_rows.append(
            {
                "property": prop,
                "true_range": true_range,
                "residual_iqr": residual_iqr,
                "residual_iqr_norm": residual_iqr / true_range if true_range > 0 else np.nan,
            }
        )

    summary = summary.merge(pd.DataFrame(spread_rows), on="property", how="right")
    summary["NMAE"] = summary["MAE"] / summary["true_range"].replace(0.0, np.nan)
    summary["R2_plot"] = summary["R2"].clip(lower=0.0, upper=1.0).fillna(0.0)

    max_nmae = summary["NMAE"].max(skipna=True)
    max_spread = summary["residual_iqr_norm"].max(skipna=True)
    summary["NMAE_plot"] = summary["NMAE"] / max_nmae if pd.notna(max_nmae) and max_nmae > 0 else 0.0
    summary["IQR_plot"] = (
        summary["residual_iqr_norm"] / max_spread
        if pd.notna(max_spread) and max_spread > 0
        else 0.0
    )
    summary["property_label"] = summary["property"].map(property_label)
    summary["property"] = pd.Categorical(summary["property"], categories=properties, ordered=True)
    return summary.sort_values("property").reset_index(drop=True)


def plot_summary_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    y_base = np.arange(len(summary))
    metric_specs = [
        ("R2_plot", "R$^2$", 0.22, "o"),
        ("NMAE_plot", "NMAE (scaled)", 0.00, "s"),
        ("IQR_plot", "Resid. IQR (scaled)", -0.22, "D"),
    ]

    for idx, row in summary.iterrows():
        color = property_color(str(row["property"]))
        ax.hlines(y_base[idx], 0.0, 1.0, color="#EAECEE", lw=0.8, zorder=0)
        for col, _, offset, marker in metric_specs:
            value = float(row[col]) if pd.notna(row[col]) else 0.0
            ax.plot([0.0, value], [y_base[idx] + offset, y_base[idx] + offset], color=color, lw=1.2, alpha=0.75)
            ax.scatter(value, y_base[idx] + offset, s=78, marker=marker, color=color, edgecolor="black", linewidth=0.45, zorder=3)

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.65, len(summary) - 0.35)
    ax.invert_yaxis()
    ax.set_yticks(y_base)
    ax.set_yticklabels(summary["property_label"])
    ax.set_xlabel("R$^2$ and scaled error metrics")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis="both", labelsize=MAIN_FONT_SIZE, pad=2)
    ax.grid(axis="x", color="#EAECEE", linewidth=0.7)
    ax.grid(axis="y", visible=False)

    handles = [
        Line2D([0], [0], marker=marker, color="black", linestyle="None", markersize=7.5, label=label)
        for _, label, _, marker in metric_specs
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        fontsize=14,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.2,
    )
    style_axes(ax)


def plot_main_layout(df: pd.DataFrame, metrics: pd.DataFrame, fig_dir: Path) -> None:
    fig = plt.figure(figsize=(14.0, 11.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.9], wspace=0.56, hspace=0.70)

    test_df = df
    if "split" in df.columns:
        test_only = df[df["split"] == "test"].copy()
        if not test_only.empty:
            test_df = test_only

    parity_props = [prop for prop in PROPERTY_ORDER if prop in test_df["property"].unique()]
    for idx, prop in enumerate(parity_props):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        sub = test_df[test_df["property"] == prop]

        if not sub.empty:
            ax.scatter(sub["y_true"], sub["y_pred"], s=18, alpha=0.65, color=property_color(prop), edgecolor="none")
            vmin = min(sub["y_true"].min(), sub["y_pred"].min())
            vmax = max(sub["y_true"].max(), sub["y_pred"].max())
            ax.plot([vmin, vmax], [vmin, vmax], "--", lw=1.2, color=REFERENCE_LINE_COLOR)
            y_true = sub["y_true"].to_numpy()
            y_pred = sub["y_pred"].to_numpy()
            ss_res = np.sum((y_pred - y_true) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            mae = float(np.mean(np.abs(y_pred - y_true)))
            n = len(sub)
            stat_text = f"{property_label(prop)}\nR2={r2:.3f}\nMAE={mae:.3f}\nN={n}"
        else:
            stat_text = f"{property_label(prop)}\nR2=NA\nMAE=NA\nN=0"

        ax.text(
            0.03,
            0.97,
            stat_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=MAIN_ANNOTATION_SIZE,
            linespacing=1.05,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.78, edgecolor="none"),
        )
        ax.set_xlabel("True log", fontsize=MAIN_FONT_SIZE, labelpad=3)
        ax.set_ylabel("Pred. log", fontsize=MAIN_FONT_SIZE, labelpad=3)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="both", labelsize=MAIN_FONT_SIZE, pad=2)
        style_axes(ax)

    summary_ax = fig.add_subplot(gs[2, :])
    summary = build_main_summary(test_df, metrics, parity_props)
    plot_summary_panel(summary_ax, summary)

    save_both(fig, fig_dir / "main_figure")


def main() -> None:
    args = parse_args()
    setup_style()

    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_df = load_prediction_df(args.input_dir, args.output_dir, args.prediction_file)
    df = all_df if args.split.lower() == "all" else all_df[all_df["split"] == args.split].copy()
    if df.empty:
        raise ValueError(f"No data found for split={args.split!r}")

    metrics = get_metrics_for_split(df, split="all")
    plot_parity_panels(df, fig_dir)
    plot_residual_panels(df, fig_dir)
    plot_property_bar(metrics, fig_dir)
    plot_radar(metrics, fig_dir)
    plot_main_layout(df, metrics, fig_dir)
    sync_legacy_png_aliases(fig_dir)
    print(f"Saved figures to: {fig_dir}")


if __name__ == "__main__":
    main()
