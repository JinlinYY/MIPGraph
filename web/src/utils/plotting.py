from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parity_plots(predictions_csv: str | Path, figure_dir: str | Path) -> None:
    df = pd.read_csv(predictions_csv)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for prop, g in df.groupby("property"):
        g = g[["y_true", "y_pred"]].replace([np.inf, -np.inf], np.nan).dropna()
        if g.empty:
            continue
        values = np.concatenate([g["y_true"].to_numpy(), g["y_pred"].to_numpy()])
        full_min, full_max = float(np.min(values)), float(np.max(values))
        q_min, q_max = np.quantile(values, [0.01, 0.99])
        if not np.isfinite(q_min) or not np.isfinite(q_max) or q_min == q_max:
            q_min, q_max = full_min, full_max
        pad = 0.05 * max(q_max - q_min, 1e-12)
        lo, hi = float(q_min - pad), float(q_max + pad)
        plt.figure(figsize=(5.2, 5.2))
        plt.scatter(g["y_true"], g["y_pred"], s=12, alpha=0.55, edgecolors="none")
        plt.plot([lo, hi], [lo, hi], "k--", lw=1)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{prop} parity")
        plt.grid(alpha=0.25, lw=0.5)
        plt.tight_layout()
        plt.savefig(figure_dir / f"parity_{prop}.png", dpi=200)
        plt.close()


def log_parity_plots(predictions_csv: str | Path, figure_dir: str | Path) -> None:
    df = pd.read_csv(predictions_csv)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for prop, g in df.groupby("property"):
        g = g[["y_true", "y_pred"]].replace([np.inf, -np.inf], np.nan).dropna()
        g = g[(g["y_true"] > 0) & (g["y_pred"] > 0)]
        if g.empty:
            continue
        true_log = np.log10(g["y_true"].to_numpy())
        pred_log = np.log10(g["y_pred"].to_numpy())
        values = np.concatenate([true_log, pred_log])
        q_min, q_max = np.quantile(values, [0.01, 0.99])
        if not np.isfinite(q_min) or not np.isfinite(q_max) or q_min == q_max:
            q_min, q_max = float(np.min(values)), float(np.max(values))
        pad = 0.05 * max(q_max - q_min, 1e-12)
        lo, hi = float(q_min - pad), float(q_max + pad)
        plt.figure(figsize=(5.2, 5.2))
        plt.scatter(true_log, pred_log, s=12, alpha=0.55, edgecolors="none")
        plt.plot([lo, hi], [lo, hi], "k--", lw=1)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("log10(True)")
        plt.ylabel("log10(Predicted)")
        plt.title(f"{prop} log-parity")
        plt.grid(alpha=0.25, lw=0.5)
        plt.tight_layout()
        plt.savefig(figure_dir / f"parity_log_{prop}.png", dpi=200)
        plt.close()


def error_distribution_plots(predictions_csv: str | Path, figure_dir: str | Path) -> None:
    df = pd.read_csv(predictions_csv)
    figure_dir = Path(figure_dir)
    for prop, g in df.groupby("property"):
        plt.figure(figsize=(5, 4))
        plt.hist(g["absolute_error"], bins=40)
        plt.xlabel("Absolute error")
        plt.ylabel("Count")
        plt.title(prop)
        plt.tight_layout()
        plt.savefig(figure_dir / f"error_distribution_{prop}.png", dpi=200)
        plt.close()


def residual_plots(predictions_csv: str | Path, figure_dir: str | Path) -> None:
    df = pd.read_csv(predictions_csv)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for prop, g in df.groupby("property"):
        g = g[["y_true", "y_pred"]].replace([np.inf, -np.inf], np.nan).dropna()
        if g.empty:
            continue
        residual = g["y_pred"] - g["y_true"]
        plt.figure(figsize=(5.5, 4.2))
        plt.scatter(g["y_true"], residual, s=12, alpha=0.55, edgecolors="none")
        plt.axhline(0.0, color="k", ls="--", lw=1)
        plt.xlabel("True")
        plt.ylabel("Prediction residual")
        plt.title(f"{prop} residuals")
        plt.grid(alpha=0.25, lw=0.5)
        plt.tight_layout()
        plt.savefig(figure_dir / f"residual_{prop}.png", dpi=200)
        plt.close()


def error_quantile_summary(predictions_csv: str | Path, output_csv: str | Path) -> None:
    df = pd.read_csv(predictions_csv)
    rows = []
    for prop, g in df.groupby("property"):
        err = g["absolute_error"].replace([np.inf, -np.inf], np.nan).dropna()
        if err.empty:
            continue
        rows.append(
            {
                "property": prop,
                "count": len(err),
                "mae": err.mean(),
                "median_abs_error": err.median(),
                "p90_abs_error": err.quantile(0.90),
                "p95_abs_error": err.quantile(0.95),
                "p99_abs_error": err.quantile(0.99),
                "max_abs_error": err.max(),
            }
        )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def ablation_barplot(summary_csv: str | Path, output_png: str | Path, metric: str = "macro_MAE") -> None:
    df = pd.read_csv(summary_csv)
    if metric not in df.columns:
        return
    plt.figure(figsize=(10, 4))
    plt.bar(df["experiment"], df[metric])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def label_availability_heatmap(clean_csv: str | Path, figure_path: str | Path, properties: list[str], max_rows: int = 2000) -> None:
    df = pd.read_csv(clean_csv)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    mat = df[[f"{p}_mask" for p in properties]].to_numpy()
    plt.figure(figsize=(7, 5))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.xticks(range(len(properties)), properties, rotation=45, ha="right")
    plt.yticks([])
    plt.colorbar(label="label available")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def normalized_metric_barplot(metrics_json: str | Path, figure_path: str | Path, metric: str = "NMAE_mean") -> None:
    with Path(metrics_json).open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    props = [k for k, v in metrics.items() if isinstance(v, dict) and metric in v]
    if not props:
        return
    vals = [metrics[p][metric] for p in props]
    plt.figure(figsize=(8, 4))
    plt.bar(props, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.tight_layout()
    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200)
    plt.close()
