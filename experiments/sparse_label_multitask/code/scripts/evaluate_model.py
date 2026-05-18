from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataset import ILPropertyDataset
from src.data.scaler import fit_scalers
from src.data.split import load_split
from src.models.iptnet import IPTNet
from src.training.evaluate import evaluate_model
from src.utils.io import load_config, resolve_path, save_json, save_log_metrics_csv, save_metrics_csv
from src.utils.plotting import (
    error_distribution_plots,
    error_quantile_summary,
    log_parity_plots,
    normalized_metric_barplot,
    parity_plots,
    residual_plots,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--wide-as-main",
        action="store_true",
        help="Write the wide one-row-per-ILTP table to test_predictions.csv and save long format as test_predictions_long.csv.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    ckpt = torch.load(resolve_path(args.checkpoint, base), map_location="cpu", weights_only=False)
    model = IPTNet(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    arrays_path = resolve_path(cfg["data"]["arrays_path"], base)
    clean_csv = resolve_path(cfg["data"]["clean_csv"], base)
    graph_cache = resolve_path(cfg["data"]["graph_cache_path"], base)
    split_path = resolve_path(cfg["data"]["processed_dir"], base) / "splits" / f"{cfg['data'].get('split_type','il_level')}_seed{cfg['data']['seed']}.json"
    split = load_split(split_path)
    arrays = dict(np.load(arrays_path, allow_pickle=True))
    cond_scaler = ckpt["condition_scaler"]
    target_scaler = ckpt["target_scaler"]
    condition = cond_scaler.transform(arrays["temperature"], arrays["pressure"])
    y_scaled = target_scaler.transform(arrays["y"], arrays["mask"])
    weights = target_scaler.error_weights(arrays["y"], arrays["y_error"], arrays["mask"], arrays["error_mask"])
    ds = ILPropertyDataset(clean_csv, arrays_path, graph_cache, split[args.split], condition, y_scaled, weights)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    metrics, pred_df, pred_wide_df = evaluate_model(model, loader, device, target_scaler, pd.read_csv(clean_csv), args.split, cfg["properties"]["names"])
    out = ckpt["config"]["outputs"]
    metrics_path = resolve_path(out["metric_dir"], base) / f"{args.split}_metrics.json"
    save_json(metrics, metrics_path)
    save_metrics_csv(metrics, metrics_path.with_suffix(".csv"), cfg["properties"]["names"])
    save_log_metrics_csv(metrics, metrics_path.with_name(f"{args.split}_metrics_log.csv"), cfg["properties"]["names"])
    pred_dir = resolve_path(out["prediction_dir"], base)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{args.split}_predictions.csv"
    wide_path = pred_dir / f"{args.split}_predictions_wide.csv"
    long_path = pred_dir / f"{args.split}_predictions_long.csv"
    if args.wide_as_main:
        pred_wide_df.to_csv(pred_path, index=False)
        pred_df.to_csv(long_path, index=False)
        plot_source = long_path
    else:
        pred_df.to_csv(pred_path, index=False)
        pred_wide_df.to_csv(wide_path, index=False)
        plot_source = pred_path
    parity_plots(plot_source, resolve_path(out["figure_dir"], base))
    log_parity_plots(plot_source, resolve_path(out["figure_dir"], base))
    error_distribution_plots(plot_source, resolve_path(out["figure_dir"], base))
    residual_plots(plot_source, resolve_path(out["figure_dir"], base))
    error_quantile_summary(plot_source, pred_dir / f"{args.split}_error_quantiles.csv")
    normalized_metric_barplot(metrics_path, resolve_path(out["figure_dir"], base) / f"{args.split}_normalized_NMAE_mean.png")
    print(metrics)


if __name__ == "__main__":
    main()
