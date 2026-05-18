from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils.io import load_config, resolve_path


def _metric_for_property(metrics: dict, prop: str, metric: str):
    item = metrics.get(prop, {})
    if metric in item:
        return item[metric]
    if metric.startswith("log_"):
        return metrics.get("log_space", {}).get(prop, {}).get(metric)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate generalization metrics and draw a split-strategy comparison figure.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split-types", default="random_point,il_pair,cation_family,anion_family")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--run-prefix", default="generalization")
    parser.add_argument("--variants", default="base,finetune_weak")
    parser.add_argument("--metric", default="NMAE_mean")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-png", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    metric_dir = resolve_path(cfg["outputs"]["metric_dir"], base)
    figure_dir = resolve_path(cfg["outputs"]["figure_dir"], base)
    split_types = [item.strip() for item in args.split_types.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    properties = cfg["properties"]["names"]
    rows = []

    for split_type in split_types:
        for seed in seeds:
            base_run = f"{args.run_prefix}_{split_type}_seed{seed}"
            for variant in variants:
                run_name = base_run if variant == "base" else f"{base_run}_{variant}"
                metrics_path = metric_dir / run_name / "test_metrics.json"
                if not metrics_path.exists():
                    continue
                with metrics_path.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
                for prop in properties:
                    value = _metric_for_property(metrics, prop, args.metric)
                    rows.append(
                        {
                            "split_type": split_type,
                            "seed": seed,
                            "variant": variant,
                            "property": prop,
                            "metric": args.metric,
                            "value": value,
                            "run_name": run_name,
                        }
                    )

    out_csv = Path(args.output_csv) if args.output_csv else metric_dir / f"{args.run_prefix}_generalization_{args.metric}.csv"
    out_png = Path(args.output_png) if args.output_png else figure_dir / f"{args.run_prefix}_generalization_{args.metric}.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    if df.empty:
        print(f"No metrics found. Wrote empty table: {out_csv}")
        return

    summary = df.groupby(["split_type", "variant", "property"], as_index=False)["value"].mean()
    n_rows = len(variants)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, max(4, 3.6 * n_rows)), sharex=True)
    if n_rows == 1:
        axes = [axes]
    x = np.arange(len(properties))
    width = 0.8 / max(len(split_types), 1)
    for ax, variant in zip(axes, variants):
        sub = summary[summary["variant"] == variant]
        for i, split_type in enumerate(split_types):
            vals = []
            for prop in properties:
                hit = sub[(sub["split_type"] == split_type) & (sub["property"] == prop)]["value"]
                vals.append(float(hit.iloc[0]) if len(hit) else np.nan)
            ax.bar(x + (i - (len(split_types) - 1) / 2) * width, vals, width=width, label=split_type)
        ax.set_title(variant)
        ax.set_ylabel(args.metric)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=2, fontsize=9)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(properties, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print({"csv": str(out_csv), "figure": str(out_png), "rows": len(df)})


if __name__ == "__main__":
    main()
