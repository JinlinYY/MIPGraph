from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.split import create_il_level_split, create_row_level_split, load_split
from src.training.trainer import Trainer
from src.utils.io import load_config, resolve_path
from src.utils.seed import set_seed


def _resolve_output_dirs(cfg: dict, base: str | Path, run_name: str) -> None:
    cfg["outputs"]["run_name"] = run_name
    for key, value in list(cfg["outputs"].items()):
        if key == "run_name":
            continue
        resolved = resolve_path(value, base)
        if key.endswith("_dir"):
            resolved = resolved / run_name
        cfg["outputs"][key] = str(resolved)


def _ensure_split(cfg: dict, clean_csv: Path, processed_dir: Path, seed: int) -> Path:
    split_name = cfg["data"].get("split_type", "il_level")
    split_path = processed_dir / "splits" / f"{split_name}_seed{seed}.json"
    if split_path.exists():
        return split_path
    if split_name == "row_level":
        create_row_level_split(clean_csv, processed_dir, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], seed)
    else:
        create_il_level_split(clean_csv, processed_dir, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], seed)
    return split_path


def _masked_arrays(
    arrays: dict,
    train_indices: list[int],
    property_idx: int,
    fraction: float,
    seed: int,
    thermal_only: bool,
) -> dict:
    out = {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in arrays.items()}
    mask = out["mask"].copy()
    error_mask = out["error_mask"].copy()
    train_idx = np.asarray(train_indices, dtype=np.int64)
    valid = train_idx[mask[train_idx, property_idx] > 0]
    keep_n = int(np.ceil(len(valid) * fraction))
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(valid, size=keep_n, replace=False).tolist()) if keep_n > 0 else set()
    drop = [idx for idx in valid.tolist() if idx not in keep]
    if drop:
        mask[np.asarray(drop, dtype=np.int64), property_idx] = 0.0
        error_mask[np.asarray(drop, dtype=np.int64), property_idx] = 0.0
    if thermal_only:
        other = [idx for idx in range(mask.shape[1]) if idx != property_idx]
        mask[np.ix_(train_idx, other)] = 0.0
        error_mask[np.ix_(train_idx, other)] = 0.0
    out["mask"] = mask
    out["error_mask"] = error_mask
    out["_kept_thermal_train_labels"] = np.asarray([len(keep)], dtype=np.int64)
    out["_available_thermal_train_labels"] = np.asarray([len(valid)], dtype=np.int64)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ThermalConductivity low-label robustness experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fractions", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint for fine-tuning instead of training from scratch.")
    parser.add_argument("--thermal-only", action="store_true", help="Mask other training labels so only ThermalConductivity supervises training.")
    parser.add_argument("--run-prefix", default="low_data_tc")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    seed = args.seed if args.seed is not None else cfg["data"]["seed"]
    set_seed(seed)
    arrays_path = resolve_path(cfg["data"]["arrays_path"], base)
    clean_csv = resolve_path(cfg["data"]["clean_csv"], base)
    processed_dir = resolve_path(cfg["data"]["processed_dir"], base)
    graph_cache = resolve_path(cfg["data"]["graph_cache_path"], base)
    split_path = _ensure_split(cfg, clean_csv, processed_dir, seed)
    split = load_split(split_path)
    property_names = cfg["properties"]["names"]
    property_idx = property_names.index("ThermalConductivity")
    arrays_base = dict(np.load(arrays_path, allow_pickle=True))

    output_root = resolve_path("outputs/low_data_thermal_conductivity", base)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    fractions = [float(item.strip()) for item in args.fractions.split(",") if item.strip()]
    for fraction in fractions:
        cfg_run = copy.deepcopy(cfg)
        cfg_run["data"]["seed"] = seed
        cfg_run["training"]["seed"] = seed
        if args.epochs is not None:
            cfg_run["training"]["epochs"] = args.epochs
        pct = int(round(fraction * 100))
        run_name = f"{args.run_prefix}_{pct}"
        arrays_run = _masked_arrays(arrays_base, split["train"], property_idx, fraction, seed + pct, args.thermal_only)
        arrays_out = output_root / f"arrays_thermal_{pct}.npz"
        np.savez_compressed(arrays_out, **arrays_run)
        cfg_run["data"]["arrays_path"] = str(arrays_out)
        _resolve_output_dirs(cfg_run, base, run_name)
        scalers = None
        ckpt = None
        if args.checkpoint:
            ckpt = torch.load(resolve_path(args.checkpoint, base), map_location="cpu", weights_only=False)
            condition_scaler = ckpt.get("condition_scaler")
            target_scaler = ckpt.get("target_scaler")
            if condition_scaler is not None and target_scaler is not None:
                y_scaled = target_scaler.transform(arrays_run["y"], arrays_run["mask"])
                condition = condition_scaler.transform(arrays_run["temperature"], arrays_run["pressure"])
                error_weights = target_scaler.error_weights(
                    arrays_run["y"],
                    arrays_run["y_error"],
                    arrays_run["mask"],
                    arrays_run["error_mask"],
                    cfg_run["loss"].get("error_weight_clip_min", 0.1),
                    cfg_run["loss"].get("error_weight_clip_max", 10.0),
                )
                scalers = (condition_scaler, target_scaler, y_scaled, condition, error_weights)
        trainer = Trainer(cfg_run, arrays_run, clean_csv, graph_cache, split, scalers=scalers)
        if ckpt is not None:
            missing, unexpected = trainer.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if missing or unexpected:
                print({"fraction": fraction, "missing_keys": missing, "unexpected_keys": unexpected})
        metrics = trainer.train()
        tc = metrics.get("ThermalConductivity", {})
        unc = metrics.get("uncertainty", {}).get("ThermalConductivity", {})
        rows.append(
            {
                "fraction": fraction,
                "kept_train_labels": int(arrays_run["_kept_thermal_train_labels"][0]),
                "available_train_labels": int(arrays_run["_available_thermal_train_labels"][0]),
                "MAE": tc.get("MAE"),
                "RMSE": tc.get("RMSE"),
                "R2": tc.get("R2"),
                "NMAE": tc.get("NMAE"),
                "label_count": tc.get("label_count"),
                "mean_pred_std": unc.get("mean_pred_std"),
                "coverage_90": unc.get("coverage_90"),
                "run_name": run_name,
            }
        )
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_root / "results.csv", index=False)
    with (output_root / "results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(results_df)


if __name__ == "__main__":
    main()
