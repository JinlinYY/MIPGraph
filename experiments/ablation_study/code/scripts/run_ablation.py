from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils.io import load_config, resolve_path
from src.utils.plotting import ablation_barplot


ABLATIONS = {
    "A0_full_3d_iptnet": {
        "description": "Full model with ErrorValue-aware loss and all structural components.",
        "changes": {},
    },
    "A1_no_cross_ion_edges": {
        "description": "Remove virtual cation-anion interaction edges from the graph cache.",
        "changes": {"model.use_cross_ion_edges": False},
        "requires_graph_cache": True,
    },
    "A2_no_mechanistic_latents": {
        "description": "Replace packing/cohesion/transport/thermal latent decomposition with one shared latent.",
        "changes": {"model.use_mechanistic_latent_heads": False},
    },
    "A3_plain_mlp_decoder": {
        "description": "Replace the thermodynamic structured decoder with a plain MLP readout.",
        "changes": {"model.use_structured_decoder": False},
    },
    "A4_no_error_weighting": {
        "description": "Disable ErrorValue-based measurement-error-aware loss weights.",
        "changes": {"loss.use_error_weight": False},
    },
    "A5_no_condition_film": {
        "description": "Replace FiLM condition modulation with concatenation-based condition fusion.",
        "changes": {"model.use_condition_film": False},
    },
    "A6_no_property_latent_gating": {
        "description": "Remove the property-specific latent path by disabling learned gates and zeroing property latent inputs.",
        "changes": {"model.use_property_latent_gating": False, "model.zero_property_latents": True},
    },
    "A7_no_global_descriptors": {
        "description": "Remove RDKit global ion-pair descriptor fusion.",
        "changes": {"model.use_global_descriptors": False},
    },
    "A8_no_interaction_encoder": {
        "description": "Remove explicit cation-anion interaction features: cross-ion edges, bilinear encoder, product term, and difference term.",
        "changes": {"model.use_cross_ion_edges": False, "model.use_interaction_encoder": False, "model.use_pairwise_ion_features": False},
        "requires_graph_cache": True,
    },
    "A9_no_e3_geometry": {
        "description": "Remove geometric edge information by disabling E(3)-invariant RBF encoding and zeroing distance edge fields.",
        "changes": {"model.use_e3_invariant_geometry": False, "model.strip_edge_geometry": True},
    },
    "A10_row_level_split_leakage_control": {
        "description": "Row-level split control; leakage-prone and not a component ablation.",
        "changes": {"data.split_type": "row_level"},
    },
}


SUMMARY_KEYS = [
    "macro_MAE",
    "weighted_MAE",
    "macro_RMSE",
    "weighted_RMSE",
    "macro_R2",
    "weighted_R2",
    "macro_log_MAE",
    "weighted_log_MAE",
    "macro_log_RMSE",
    "weighted_log_RMSE",
    "macro_log_R2",
    "weighted_log_R2",
    "macro_NMAE_mean",
    "weighted_NMAE_mean",
    "macro_NRMSE_mean",
    "weighted_NRMSE_mean",
]


def set_nested(cfg: dict, dotted: str, value):
    cur = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def flatten_metrics(metrics: dict, property_names: list[str]) -> dict:
    row = {key: metrics.get(key) for key in SUMMARY_KEYS}
    for prop in property_names:
        prop_metrics = metrics.get(prop, {})
        log_metrics = metrics.get("log_space", {}).get(prop, {})
        for key in ["MAE", "RMSE", "R2", "MAPE", "NMAE_mean", "NRMSE_mean", "label_count"]:
            row[f"{prop}_{key}"] = prop_metrics.get(key)
        for key in ["log_MAE", "log_RMSE", "log_R2"]:
            row[f"{prop}_{key}"] = log_metrics.get(key)
    return row


def add_delta_columns(df: pd.DataFrame, baseline_name: str = "A0_full_3d_iptnet") -> pd.DataFrame:
    if df.empty or baseline_name not in set(df["experiment"]):
        return df
    baseline = df.loc[df["experiment"] == baseline_name].iloc[0]
    for metric in SUMMARY_KEYS:
        if metric not in df.columns:
            continue
        base_value = baseline.get(metric)
        if pd.isna(base_value):
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        df[f"delta_{metric}_vs_full"] = values - float(base_value)
        if "MAE" in metric or "RMSE" in metric or "NMAE" in metric or "NRMSE" in metric:
            df[f"relative_{metric}_increase_vs_full"] = (values - float(base_value)) / abs(float(base_value))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-training", action="store_true", help="Actually train every ablation. Default only builds configs and summarizes existing metrics.")
    parser.add_argument("--experiments", default=None, help="Comma-separated ablation names. Default: all.")
    parser.add_argument("--rebuild-graph-cache", action="store_true", help="Rebuild graph caches for graph-structure ablations even if the cache already exists.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    ab_dir = resolve_path(cfg["outputs"]["ablation_dir"], base)
    ab_dir.mkdir(parents=True, exist_ok=True)
    selected = set(item.strip() for item in args.experiments.split(",") if item.strip()) if args.experiments else None
    rows = []
    for name, spec in ABLATIONS.items():
        if selected is not None and name not in selected:
            continue
        changes = spec.get("changes", {})
        exp_cfg = copy.deepcopy(cfg)
        exp_cfg.pop("_config_path", None)
        exp_cfg.pop("_base_dir", None)
        for key in ["raw_path", "processed_dir", "clean_csv", "arrays_path", "graph_cache_path", "failed_smiles_path"]:
            if key in exp_cfg["data"]:
                exp_cfg["data"][key] = str(resolve_path(exp_cfg["data"][key], base))
        exp_output = resolve_path(Path(cfg["outputs"]["ablation_dir"]) / name, base)
        exp_cfg["outputs"]["output_dir"] = str(exp_output)
        exp_cfg["outputs"]["checkpoint_dir"] = str(exp_output / "checkpoints")
        exp_cfg["outputs"]["log_dir"] = str(exp_output / "logs")
        exp_cfg["outputs"]["metric_dir"] = str(exp_output / "metrics")
        exp_cfg["outputs"]["prediction_dir"] = str(exp_output / "predictions")
        exp_cfg["outputs"]["figure_dir"] = str(exp_output / "figures")
        if spec.get("requires_graph_cache"):
            exp_cfg["data"]["graph_cache_path"] = str(exp_output / "graph_cache.pt")
            exp_cfg["data"]["failed_smiles_path"] = str(exp_output / "failed_smiles.csv")
        for dotted, value in changes.items():
            set_nested(exp_cfg, dotted, value)
        cfg_path = ab_dir / f"{name}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(exp_cfg, f, sort_keys=False)
        if args.run_training:
            graph_cache_path = Path(exp_cfg["data"]["graph_cache_path"])
            if spec.get("requires_graph_cache") and (args.rebuild_graph_cache or not graph_cache_path.exists()):
                subprocess.run([sys.executable, str(PROJECT_DIR / "scripts" / "build_graph_cache.py"), "--config", str(cfg_path)], check=True, cwd=PROJECT_DIR)
            subprocess.run([sys.executable, str(PROJECT_DIR / "scripts" / "train_iptnet.py"), "--config", str(cfg_path), "--seed", str(cfg["data"]["seed"])], check=True, cwd=PROJECT_DIR)
        metric_path = resolve_path(exp_cfg["outputs"]["metric_dir"], base) / "test_metrics.json"
        row = {
            "experiment": name,
            "description": spec.get("description", ""),
            "changes": json.dumps(changes, ensure_ascii=False, sort_keys=True),
            "config_path": str(cfg_path),
            "metric_path": str(metric_path),
        }
        if metric_path.exists():
            with metric_path.open("r", encoding="utf-8") as f:
                m = json.load(f)
            row.update(flatten_metrics(m, exp_cfg["properties"]["names"]))
            row["status"] = "done"
        else:
            row["status"] = "config_only"
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = ab_dir / "ablation_summary.csv"
    if selected is not None and summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[~existing["experiment"].isin(summary["experiment"])]
        summary = pd.concat([existing, summary], ignore_index=True, sort=False)
        order = {name: i for i, name in enumerate(ABLATIONS)}
        summary["_order"] = summary["experiment"].map(order).fillna(len(order))
        summary = summary.sort_values("_order").drop(columns=["_order"])
    summary = add_delta_columns(summary)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if "macro_MAE" in summary.columns and summary["macro_MAE"].notna().any():
        ablation_barplot(summary_path, ab_dir / "ablation_barplot.png")
    print(summary)


if __name__ == "__main__":
    main()
