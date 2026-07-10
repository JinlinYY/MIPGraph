from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.repair import apply_max_train_value_repair
from src.data.split import create_il_level_split, create_row_level_split, load_split
from src.training.trainer import Trainer
from src.utils.io import load_config, resolve_path, save_json
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--clean-csv", default=None)
    parser.add_argument("--arrays-path", default=None)
    parser.add_argument("--graph-cache", default=None)
    parser.add_argument("--split-path", default=None, help="Optional precomputed split JSON, relative to the project directory.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--unimol2-feature-cache", default=None)
    parser.add_argument("--unimol2-weight-dir", default=None)
    parser.add_argument("--unfreeze-unimol2-layers", type=int, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--resume-checkpoint", default=None, help="Optional checkpoint whose model weights are loaded before training.")
    parser.add_argument("--enable-amp", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--versioned-best-checkpoints", action="store_true")
    parser.add_argument("--monitor-space", choices=["raw", "log"], default="log")
    parser.add_argument("--disable-property-coupling", action="store_true")
    parser.add_argument("--disable-property-adapters", action="store_true")
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--balance-properties", action="store_true")
    parser.add_argument("--interpolation-label-weight", type=float, default=None)
    parser.add_argument(
        "--target-scaler-mask",
        choices=["mask", "evaluation_mask"],
        default=None,
        help="Fit target scaler on the full training mask or on original/evaluation labels only.",
    )
    parser.add_argument(
        "--repair-viscosity-action",
        choices=["none", "drop", "downweight"],
        default="none",
        help="Repair train-only Viscosity labels above --repair-viscosity-max-train.",
    )
    parser.add_argument("--repair-viscosity-max-train", type=float, default=1000.0)
    parser.add_argument("--repair-viscosity-downweight", type=float, default=0.05)
    parser.add_argument("--il-balance-properties", default="")
    parser.add_argument("--il-balance-power", type=float, default=1.0)
    parser.add_argument("--augment-properties", default="")
    parser.add_argument("--augment-points-per-interval", type=int, default=1)
    parser.add_argument("--augment-max-temperature-gap", type=float, default=40.0)
    parser.add_argument("--augment-sample-weight", type=float, default=0.5)
    parser.add_argument("--augment-max-samples-per-property", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    seed = args.seed if args.seed is not None else cfg["training"].get("seed", cfg["data"]["seed"])
    cfg["training"]["seed"] = seed
    cfg["data"]["seed"] = seed
    for key, value in (
        ("epochs", args.epochs),
        ("lr", args.lr),
        ("patience", args.patience),
        ("batch_size", args.batch_size),
        ("validate_every", args.validate_every),
        ("num_workers", args.num_workers),
    ):
        if value is not None:
            cfg["training"][key] = value
    if args.num_workers == 0:
        cfg["training"]["persistent_workers"] = False
    cfg["training"]["monitor_metric"] = "macro_log_val_score" if args.monitor_space == "log" else "macro_R2"
    cfg["training"]["monitor_mode"] = "max"
    cfg["training"]["evaluate_test"] = not args.skip_test_evaluation
    cfg["training"]["save_last_checkpoint"] = False
    cfg["training"]["versioned_best_checkpoints"] = bool(args.versioned_best_checkpoints)
    cfg["training"]["atomic_checkpoint_save"] = False
    if args.model_name:
        cfg["model"]["name"] = args.model_name
    if args.unimol2_feature_cache:
        cfg["model"]["unimol2_feature_cache_path"] = str(Path(args.unimol2_feature_cache).resolve())
    if args.unimol2_weight_dir:
        cfg["model"]["unimol2_weight_dir"] = str(Path(args.unimol2_weight_dir).resolve())
    if args.unfreeze_unimol2_layers is not None:
        cfg["model"]["freeze_unimol2_backbone"] = True
        cfg["model"]["unimol2_unfreeze_last_n_layers"] = args.unfreeze_unimol2_layers
    if args.backbone_lr is not None:
        cfg["training"]["backbone_lr"] = args.backbone_lr
    if args.resume_checkpoint:
        cfg["training"]["resume_checkpoint"] = str(Path(args.resume_checkpoint).resolve())
        cfg["training"]["evaluate_initial"] = True
    if args.enable_amp:
        cfg["training"]["use_amp"] = True
    if args.disable_amp:
        cfg["training"]["use_amp"] = False
    if args.disable_property_coupling:
        cfg["model"]["use_property_coupling"] = False
    if args.disable_property_adapters:
        cfg["model"]["use_property_adapters"] = False
    if args.balance_properties:
        cfg["loss"]["balance_properties"] = True
    if args.target_scaler_mask is not None:
        cfg["loss"]["target_scaler_mask"] = args.target_scaler_mask
    il_balance_properties = [item.strip() for item in args.il_balance_properties.split(",") if item.strip()]
    if il_balance_properties:
        cfg["loss"]["il_balance_properties"] = il_balance_properties
        cfg["loss"]["il_balance_power"] = args.il_balance_power
    augment_properties = [item.strip() for item in args.augment_properties.split(",") if item.strip()]
    bad_augmented = [name for name in augment_properties if name not in cfg["properties"]["names"]]
    if bad_augmented:
        raise ValueError(f"Unknown augmentation properties: {bad_augmented}")
    cfg["augmentation"] = {
        "enabled": bool(augment_properties),
        "method": "same_il_pressure_temperature_log_interpolation",
        "properties": augment_properties,
        "points_per_interval": args.augment_points_per_interval,
        "max_temperature_gap": args.augment_max_temperature_gap,
        "pressure_round_decimals": 1,
        "sample_weight": args.augment_sample_weight,
        "max_samples_per_property": args.augment_max_samples_per_property,
        "seed": seed,
    }
    if args.run_name:
        cfg["outputs"]["run_name"] = args.run_name
    if args.output_root:
        root = Path(args.output_root)
        cfg["outputs"]["output_dir"] = str(root)
        cfg["outputs"]["checkpoint_dir"] = str(root / "checkpoints")
        cfg["outputs"]["log_dir"] = str(root / "logs")
        cfg["outputs"]["metric_dir"] = str(root / "metrics")
        cfg["outputs"]["prediction_dir"] = str(root / "predictions")
        cfg["outputs"]["figure_dir"] = str(root / "figures")
    set_seed(seed)
    arrays_path = resolve_path(args.arrays_path or cfg["data"]["arrays_path"], base)
    clean_csv = resolve_path(args.clean_csv or cfg["data"]["clean_csv"], base)
    processed_dir = resolve_path(cfg["data"]["processed_dir"], base)
    graph_cache = resolve_path(args.graph_cache or cfg["data"]["graph_cache_path"], base)
    if not arrays_path.exists() or not clean_csv.exists():
        raise FileNotFoundError("Preprocessed files are missing. Run scripts/preprocess_data.py first.")
    if not graph_cache.exists():
        raise FileNotFoundError("Graph cache is missing. Run scripts/build_graph_cache.py first.")
    if args.split_path:
        split_path = resolve_path(args.split_path, base)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
    else:
        split_name = cfg["data"].get("split_type", "il_level")
        split_path = processed_dir / "splits" / f"{split_name}_seed{seed}.json"
        if not split_path.exists():
            if split_name == "row_level":
                create_row_level_split(clean_csv, processed_dir, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], seed)
            else:
                create_il_level_split(clean_csv, processed_dir, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], seed)
    cfg["data"]["arrays_path"] = str(arrays_path)
    cfg["data"]["clean_csv"] = str(clean_csv)
    cfg["data"]["graph_cache_path"] = str(graph_cache)
    cfg["data"]["split_path"] = str(split_path)
    run_name = cfg["outputs"].get("run_name", "")
    for k, v in cfg["outputs"].items():
        if k == "run_name":
            continue
        resolved = resolve_path(v, base)
        if run_name and run_name != "default" and k.endswith("_dir"):
            resolved = resolved / run_name
        cfg["outputs"][k] = str(resolved)
    split = load_split(split_path)
    arrays = dict(np.load(arrays_path, allow_pickle=True))
    if args.interpolation_label_weight is not None and "label_weight" in arrays:
        label_weight = np.asarray(arrays["label_weight"], dtype=np.float32).copy()
        label_weight[np.isclose(label_weight, 0.25)] = float(args.interpolation_label_weight)
        arrays["label_weight"] = label_weight
    repair_report = apply_max_train_value_repair(
        arrays,
        split["train"],
        "Viscosity",
        args.repair_viscosity_max_train,
        args.repair_viscosity_action,
        args.repair_viscosity_downweight,
    )
    trainer = Trainer(cfg, arrays, clean_csv, graph_cache, split)
    metrics = trainer.train()
    manifest = {
        "run_name": run_name,
        "best_checkpoint": str(trainer.best_checkpoint_path),
        "split_path": str(split_path),
        "test_selected": False,
        "target_scaler_mask": cfg["loss"].get("target_scaler_mask", "mask"),
        "data_repair": repair_report,
    }
    save_json(manifest, Path(cfg["outputs"]["metric_dir"]) / "run_manifest.json")
    print(metrics)
    print(manifest)


if __name__ == "__main__":
    main()
