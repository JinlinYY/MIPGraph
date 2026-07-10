from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.repair import apply_max_train_value_repair
from src.data.scaler import fit_scalers
from src.data.split import create_il_level_split, create_row_level_split, load_split
from src.training.trainer import Trainer
from src.utils.io import load_config, resolve_path, save_json
from src.utils.seed import set_seed


PROPERTY_BRANCH_PREFIXES = {
    "Density": (
        "property_adapters.adapters.Density",
        "decoder.heads.Density",
        "physics_moe.router_heads.Density",
    ),
    "Viscosity": (
        "property_adapters.adapters.Viscosity",
        "decoder.heads.Viscosity",
        "physics_moe.router_heads.Viscosity",
    ),
    "ElectricalConductivity": (
        "property_adapters.adapters.ElectricalConductivity",
        "decoder.heads.ElectricalConductivity",
        "physics_moe.router_heads.ElectricalConductivity",
    ),
    "HeatCapacity": (
        "property_adapters.adapters.HeatCapacity",
        "decoder.heads.HeatCapacity",
        "physics_moe.router_heads.HeatCapacity",
    ),
    "SurfaceTension": (
        "property_adapters.adapters.SurfaceTension",
        "decoder.heads.SurfaceTension",
        "physics_moe.router_heads.SurfaceTension",
    ),
    "ThermalConductivity": (
        "property_adapters.adapters.ThermalConductivity",
        "decoder.heads.ThermalConductivity",
        "physics_moe.router_heads.ThermalConductivity",
    ),
}

MIPGRAPH_SHARED_PREFIXES = (
    "ion_encoder.cation_projection",
    "ion_encoder.anion_projection",
    "atom_interaction",
    "interaction",
    "interaction_fusion",
    "project",
    "global_descriptor_encoder",
    "functional_group_encoder",
    "transformer_fusion",
    "condition",
    "physics_moe",
)
MIPGRAPH_UPPER_PREFIXES = (
    *MIPGRAPH_SHARED_PREFIXES,
    "property_adapters",
    "decoder",
)
LEGACY_UNUSED_PREFIXES = (
    "interaction.",
    "interaction_fusion.",
)


def _resolve_output_dirs(cfg: dict, base: str | Path, run_name: str) -> None:
    cfg["outputs"]["run_name"] = run_name
    for key, value in list(cfg["outputs"].items()):
        if key == "run_name":
            continue
        resolved = resolve_path(value, base)
        if key.endswith("_dir"):
            resolved = resolved / run_name
        cfg["outputs"][key] = str(resolved)


def _set_output_root(cfg: dict, output_root: str | None) -> None:
    if not output_root:
        return
    root = Path(output_root)
    cfg["outputs"]["output_dir"] = str(root)
    cfg["outputs"]["checkpoint_dir"] = str(root / "checkpoints")
    cfg["outputs"]["log_dir"] = str(root / "logs")
    cfg["outputs"]["metric_dir"] = str(root / "metrics")
    cfg["outputs"]["prediction_dir"] = str(root / "predictions")
    cfg["outputs"]["figure_dir"] = str(root / "figures")


def _set_property_weights(cfg: dict, focus_properties: list[str], focus_weight: float, background_weight: float) -> None:
    names = cfg["properties"]["names"]
    bad = [p for p in focus_properties if p not in names]
    if bad:
        raise ValueError(f"Unknown properties: {bad}. Valid names: {names}")
    cfg["loss"]["property_loss_weights"] = {name: (focus_weight if name in focus_properties else background_weight) for name in names}


def _freeze_for_finetune(model: torch.nn.Module, mode: str, target_property: str | None = None) -> None:
    if mode == "all":
        return
    if mode == "graph_frozen":
        mode = "encoder_frozen"
    for param in model.parameters():
        param.requires_grad = False
    if mode in {"property_branch", "property_adapter_branch"}:
        if not target_property:
            raise ValueError(f"--freeze-mode {mode} requires --target-property")
        train_prefixes = PROPERTY_BRANCH_PREFIXES[target_property]
        for name, param in model.named_parameters():
            if name.startswith(train_prefixes):
                param.requires_grad = True
        return
    if mode == "target_branch_plus_shared":
        if not target_property:
            raise ValueError("--freeze-mode target_branch_plus_shared requires --target-property")
        train_prefixes = (*MIPGRAPH_SHARED_PREFIXES, *PROPERTY_BRANCH_PREFIXES[target_property])
        for name, param in model.named_parameters():
            if name.startswith(train_prefixes):
                param.requires_grad = True
        return
    train_prefixes = {
        "decoder": ("decoder",),
        "adapters_decoder": ("property_adapters", "decoder"),
        "decoder_condition": ("property_adapters", "decoder", "condition"),
        "head_latent_condition": ("property_adapters", "decoder", "condition", "physics_moe"),
        "encoder_frozen": MIPGRAPH_UPPER_PREFIXES,
    }[mode]
    for name, param in model.named_parameters():
        if name.startswith(train_prefixes):
            param.requires_grad = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a MIPGraph checkpoint on selected weak properties.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clean-csv", default=None)
    parser.add_argument("--arrays-path", default=None)
    parser.add_argument("--graph-cache", default=None)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--properties", default="ElectricalConductivity,SurfaceTension,Viscosity")
    parser.add_argument(
        "--target-property",
        choices=[
            "Density",
            "Viscosity",
            "ElectricalConductivity",
            "HeatCapacity",
            "SurfaceTension",
            "ThermalConductivity",
        ],
        default=None,
        help="Train a property specialist and select checkpoints in the requested monitor space.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed used only for the train/val/test split. Defaults to --seed for backward compatibility.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root. Useful for specialists, e.g. outputs/property_specialists.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--enable-property-adapters",
        action="store_true",
        help="Add zero-initialized per-property adapters when loading a checkpoint that did not have them.",
    )
    parser.add_argument("--property-adapter-dim", type=int, default=64)
    parser.add_argument("--property-adapter-dropout", type=float, default=None)
    parser.add_argument("--interpolation-label-weight", type=float, default=None)
    parser.add_argument(
        "--target-scaler-mask",
        choices=["mask", "evaluation_mask"],
        default=None,
        help="Fit target scaler on the full training mask or original/evaluation labels only when no checkpoint scaler is reused.",
    )
    parser.add_argument(
        "--repair-viscosity-action",
        choices=["none", "drop", "downweight"],
        default="none",
        help="Repair train-only Viscosity labels above --repair-viscosity-max-train.",
    )
    parser.add_argument("--repair-viscosity-max-train", type=float, default=1000.0)
    parser.add_argument("--repair-viscosity-downweight", type=float, default=0.05)
    parser.add_argument("--il-balanced-loss", action="store_true")
    parser.add_argument("--il-balance-power", type=float, default=1.0)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--enable-amp", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--include-val-in-train",
        action="store_true",
        help="Merge the validation indices into the training split for final fixed-schedule refinement.",
    )
    parser.add_argument(
        "--use-checkpoint-model-config",
        action="store_true",
        help="Reuse model and chemistry settings saved in the checkpoint while keeping the new training/loss/output settings.",
    )
    parser.add_argument("--focus-weight", type=float, default=4.0)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument(
        "--protect-properties",
        default="",
        help="Comma-separated properties protected by teacher distillation during fine-tuning.",
    )
    parser.add_argument(
        "--distill-checkpoint",
        default=None,
        help="Teacher checkpoint used to keep protected property outputs from drifting.",
    )
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--monitor-space", choices=["raw", "log"], default="log")
    parser.add_argument(
        "--monitor-objective",
        choices=["score", "mae", "nmae", "r2"],
        default="score",
        help="Validation objective for selecting the best fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--disable-property-coupling",
        action="store_true",
        help="Compatibility flag for older commands; current MIPGraph heads are already independent.",
    )
    parser.add_argument(
        "--augment-properties",
        default="",
        help="Comma-separated low-data properties augmented by same-IL temperature interpolation on the train split only.",
    )
    parser.add_argument("--augment-points-per-interval", type=int, default=1)
    parser.add_argument("--augment-max-temperature-gap", type=float, default=40.0)
    parser.add_argument("--augment-sample-weight", type=float, default=0.5)
    parser.add_argument("--augment-max-samples-per-property", type=int, default=0)
    parser.add_argument("--tail-property", default=None, help="Optionally upweight high-value labels for this property.")
    parser.add_argument("--tail-threshold-scaled", type=float, default=1.0)
    parser.add_argument("--tail-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--freeze-mode",
        choices=[
            "all",
            "encoder_frozen",
            "graph_frozen",
            "head_latent_condition",
            "decoder_condition",
            "decoder",
            "adapters_decoder",
            "property_branch",
            "property_adapter_branch",
            "target_branch_plus_shared",
        ],
        default="encoder_frozen",
        help="encoder_frozen keeps the Uni-Mol2 backbone fixed and fine-tunes MIPGraph upper modules.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    seed = args.seed if args.seed is not None else cfg["training"].get("seed", cfg["data"]["seed"])
    split_seed = args.split_seed if args.split_seed is not None else seed
    cfg["data"]["seed"] = split_seed
    cfg["training"]["seed"] = seed
    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["lr"] = args.lr
    cfg["training"]["patience"] = args.patience
    cfg["training"]["save_last_checkpoint"] = False
    cfg["training"]["versioned_best_checkpoints"] = False
    cfg["training"]["atomic_checkpoint_save"] = False
    cfg["training"]["evaluate_test"] = not args.skip_test_evaluation
    if args.enable_amp:
        cfg["training"]["use_amp"] = True
    if args.disable_amp:
        cfg["training"]["use_amp"] = False
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.validate_every is not None:
        cfg["training"]["validate_every"] = args.validate_every
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
        if args.num_workers == 0:
            cfg["training"]["persistent_workers"] = False
    monitor_mode = "max"
    if args.target_property:
        if args.monitor_objective == "score":
            cfg["training"]["monitor_metric"] = "target_log_val_score" if args.monitor_space == "log" else "target_val_score"
            monitor_mode = "max"
        elif args.monitor_objective == "mae":
            cfg["training"]["monitor_metric"] = "target_log_MAE" if args.monitor_space == "log" else "target_MAE"
            monitor_mode = "min"
        elif args.monitor_objective == "nmae":
            cfg["training"]["monitor_metric"] = "target_log_NMAE" if args.monitor_space == "log" else "target_NMAE"
            monitor_mode = "min"
        elif args.monitor_objective == "r2":
            cfg["training"]["monitor_metric"] = "target_log_R2" if args.monitor_space == "log" else "target_R2"
            monitor_mode = "max"
        cfg["training"]["monitor_target_property"] = args.target_property
    else:
        cfg["training"]["monitor_metric"] = "focus_macro_log_R2" if args.monitor_space == "log" else "focus_macro_R2"
    cfg["training"]["monitor_mode"] = monitor_mode
    set_seed(seed)

    focus_properties = [args.target_property] if args.target_property else [p.strip() for p in args.properties.split(",") if p.strip()]
    cfg["training"]["monitor_properties"] = focus_properties
    run_name = args.run_name or (
        f"property_specialist_{args.target_property}_seed{seed}" if args.target_property else f"finetune_{'_'.join(focus_properties)}_seed{seed}"
    )
    _set_property_weights(cfg, focus_properties, args.focus_weight, args.background_weight)
    protect_properties = [p.strip() for p in args.protect_properties.split(",") if p.strip()]
    bad_protected = [p for p in protect_properties if p not in cfg["properties"]["names"]]
    if bad_protected:
        raise ValueError(f"Unknown protected properties: {bad_protected}. Valid names: {cfg['properties']['names']}")
    cfg["training"]["protect_properties"] = protect_properties
    cfg["training"]["distill_checkpoint"] = args.distill_checkpoint
    cfg["training"]["distill_weight"] = args.distill_weight
    augment_properties = [p.strip() for p in args.augment_properties.split(",") if p.strip()]
    bad_augmented = [p for p in augment_properties if p not in cfg["properties"]["names"]]
    if bad_augmented:
        raise ValueError(f"Unknown augmentation properties: {bad_augmented}. Valid names: {cfg['properties']['names']}")
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
    if args.tail_property:
        cfg["loss"]["high_value_weighting"] = {
            args.tail_property: {
                "threshold_scaled": args.tail_threshold_scaled,
                "multiplier": args.tail_multiplier,
            }
        }
    if args.target_scaler_mask is not None:
        cfg["loss"]["target_scaler_mask"] = args.target_scaler_mask
    _set_output_root(cfg, args.output_root)
    _resolve_output_dirs(cfg, base, run_name)

    arrays_path = resolve_path(args.arrays_path or cfg["data"]["arrays_path"], base)
    clean_csv = resolve_path(args.clean_csv or cfg["data"]["clean_csv"], base)
    processed_dir = resolve_path(cfg["data"]["processed_dir"], base)
    graph_cache = resolve_path(args.graph_cache or cfg["data"]["graph_cache_path"], base)
    split_name = cfg["data"].get("split_type", "il_level")
    split_path = (
        resolve_path(args.split_path, base)
        if args.split_path
        else processed_dir / "splits" / f"{split_name}_seed{split_seed}.json"
    )
    if not split_path.exists():
        if split_name == "row_level":
            create_row_level_split(
                clean_csv,
                processed_dir,
                cfg["data"]["train_ratio"],
                cfg["data"]["val_ratio"],
                cfg["data"]["test_ratio"],
                split_seed,
            )
        else:
            create_il_level_split(
                clean_csv,
                processed_dir,
                cfg["data"]["train_ratio"],
                cfg["data"]["val_ratio"],
                cfg["data"]["test_ratio"],
                split_seed,
            )

    ckpt = torch.load(resolve_path(args.checkpoint, base), map_location="cpu", weights_only=False)
    if args.use_checkpoint_model_config and ckpt.get("config"):
        ckpt_cfg = ckpt["config"]
        if "model" in ckpt_cfg:
            cfg["model"] = copy.deepcopy(ckpt_cfg["model"])
        if "chem" in ckpt_cfg:
            cfg["chem"] = copy.deepcopy(ckpt_cfg["chem"])
    if args.enable_property_adapters:
        cfg["model"]["use_property_adapters"] = True
        cfg["model"]["property_adapter_dim"] = args.property_adapter_dim
        cfg["model"]["property_adapter_zero_init"] = True
        if args.property_adapter_dropout is not None:
            cfg["model"]["property_adapter_dropout"] = args.property_adapter_dropout
    if args.disable_property_coupling:
        cfg["model"]["use_property_coupling"] = False
    if args.il_balanced_loss:
        if not args.target_property:
            raise ValueError("--il-balanced-loss requires --target-property")
        cfg["loss"]["il_balance_properties"] = [args.target_property]
        cfg["loss"]["il_balance_power"] = args.il_balance_power
    split = load_split(split_path)
    if args.include_val_in_train:
        split = copy.deepcopy(split)
        split["train"] = list(dict.fromkeys(list(split["train"]) + list(split.get("val", []))))

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
    condition_scaler = ckpt.get("condition_scaler")
    target_scaler = ckpt.get("target_scaler")
    if condition_scaler is None or target_scaler is None:
        condition_scaler, target_scaler, y_scaled, condition, error_weights = fit_scalers(
            arrays,
            split["train"],
            cfg["loss"].get("error_weight_clip_min", 0.1),
            cfg["loss"].get("error_weight_clip_max", 10.0),
            cfg["loss"].get("target_scaler_mask", "mask"),
        )
    else:
        y_scaled = target_scaler.transform(arrays["y"], arrays["mask"])
        condition = condition_scaler.transform(arrays["temperature"], arrays["pressure"])
        error_weights = target_scaler.error_weights(
            arrays["y"],
            arrays["y_error"],
            arrays["mask"],
            arrays["error_mask"],
            cfg["loss"].get("error_weight_clip_min", 0.1),
            cfg["loss"].get("error_weight_clip_max", 10.0),
        )
        if "label_weight" in arrays:
            error_weights *= np.asarray(arrays["label_weight"], dtype=np.float32)
    cfg["data"]["arrays_path"] = str(arrays_path)
    cfg["data"]["clean_csv"] = str(clean_csv)
    cfg["data"]["graph_cache_path"] = str(graph_cache)
    cfg["data"]["split_path"] = str(split_path)
    trainer = Trainer(cfg, arrays, clean_csv, graph_cache, split, scalers=(condition_scaler, target_scaler, y_scaled, condition, error_weights))
    missing, unexpected = trainer.model.load_state_dict(ckpt["model_state_dict"], strict=False)
    adapter_missing_allowed = bool(cfg["model"].get("use_property_adapters", False))

    def is_allowed_missing(key: str) -> bool:
        if adapter_missing_allowed and key.startswith("property_adapters."):
            return True
        return False

    invalid_missing = [key for key in missing if not is_allowed_missing(key)]
    invalid_unexpected = [
        key for key in unexpected if not key.startswith(LEGACY_UNUSED_PREFIXES)
    ]
    if invalid_missing or invalid_unexpected:
        raise RuntimeError(
            f"Checkpoint is incompatible with the requested model: "
            f"missing={invalid_missing}, unexpected={invalid_unexpected}"
        )
    if missing or unexpected:
        print({"missing_keys": missing, "unexpected_keys": unexpected})
    _freeze_for_finetune(trainer.model, args.freeze_mode, args.target_property)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(
        {
            "run_name": run_name,
            "seed": seed,
            "split_seed": split_seed,
            "target_property": args.target_property,
            "focus_properties": focus_properties,
            "property_loss_weights": cfg["loss"]["property_loss_weights"],
            "freeze_mode": args.freeze_mode,
            "include_val_in_train": args.include_val_in_train,
            "protect_properties": protect_properties,
            "distill_checkpoint": args.distill_checkpoint,
            "distill_weight": args.distill_weight,
            "use_property_coupling": cfg["model"].get("use_property_coupling", True),
            "use_property_adapters": cfg["model"].get("use_property_adapters", False),
            "augmentation": cfg["augmentation"],
            "trainable_parameters": trainable,
            "total_parameters": total,
        }
    )
    metrics = trainer.train()
    manifest = {
        "run_name": run_name,
        "target_property": args.target_property,
        "best_checkpoint": str(trainer.best_checkpoint_path),
        "split_path": str(split_path),
        "test_selected": False,
        "target_scaler_mask": cfg["loss"].get("target_scaler_mask", "mask"),
        "data_repair": repair_report,
        "use_property_adapters": cfg["model"].get("use_property_adapters", False),
    }
    save_json(manifest, Path(cfg["outputs"]["metric_dir"]) / "run_manifest.json")
    print(metrics)
    print(manifest)


if __name__ == "__main__":
    main()
