from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.scaler import fit_scalers
from src.data.split import create_split, load_split
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


def _set_property_weights(cfg: dict, focus_properties: list[str], focus_weight: float, background_weight: float) -> None:
    names = cfg["properties"]["names"]
    bad = [p for p in focus_properties if p not in names]
    if bad:
        raise ValueError(f"Unknown properties: {bad}. Valid names: {names}")
    cfg["loss"]["property_loss_weights"] = {name: (focus_weight if name in focus_properties else background_weight) for name in names}


def _freeze_for_finetune(model: torch.nn.Module, mode: str) -> None:
    if mode == "all":
        return
    for param in model.parameters():
        param.requires_grad = False
    train_prefixes = {
        "decoder": ("decoder",),
        "decoder_condition": ("decoder", "condition"),
        "head_latent_condition": ("decoder", "condition", "latents"),
        "graph_frozen": ("interaction", "project", "global_descriptor_encoder", "latents", "condition", "decoder"),
    }[mode]
    for name, param in model.named_parameters():
        if name.startswith(train_prefixes):
            param.requires_grad = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune an IPTNet checkpoint on selected weak properties.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--properties", default="ElectricalConductivity,SurfaceTension,Viscosity")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--focus-weight", type=float, default=4.0)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument("--monitor-space", choices=["raw", "log"], default="log")
    parser.add_argument("--tail-property", default=None, help="Optionally upweight high-value labels for this property.")
    parser.add_argument("--tail-threshold-scaled", type=float, default=1.0)
    parser.add_argument("--tail-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--freeze-mode",
        choices=["all", "graph_frozen", "head_latent_condition", "decoder_condition", "decoder"],
        default="graph_frozen",
        help="graph_frozen keeps the expensive graph encoder fixed and fine-tunes the higher-level property modules.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    seed = args.seed if args.seed is not None else cfg["data"]["seed"]
    cfg["data"]["seed"] = seed
    cfg["training"]["seed"] = seed
    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["lr"] = args.lr
    cfg["training"]["patience"] = args.patience
    cfg["training"]["monitor_metric"] = "focus_macro_log_R2" if args.monitor_space == "log" else "focus_macro_R2"
    cfg["training"]["monitor_mode"] = "max"
    set_seed(seed)

    focus_properties = [p.strip() for p in args.properties.split(",") if p.strip()]
    cfg["training"]["monitor_properties"] = focus_properties
    run_name = args.run_name or f"finetune_{'_'.join(focus_properties)}_seed{seed}"
    _set_property_weights(cfg, focus_properties, args.focus_weight, args.background_weight)
    if args.tail_property:
        cfg["loss"]["high_value_weighting"] = {
            args.tail_property: {
                "threshold_scaled": args.tail_threshold_scaled,
                "multiplier": args.tail_multiplier,
            }
        }
    _resolve_output_dirs(cfg, base, run_name)

    arrays_path = resolve_path(cfg["data"]["arrays_path"], base)
    clean_csv = resolve_path(cfg["data"]["clean_csv"], base)
    processed_dir = resolve_path(cfg["data"]["processed_dir"], base)
    graph_cache = resolve_path(cfg["data"]["graph_cache_path"], base)
    split_name = cfg["data"].get("split_type", "il_level")
    split_path = processed_dir / "splits" / f"{split_name}_seed{seed}.json"
    if not split_path.exists():
        create_split(split_name, clean_csv, processed_dir, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], seed)

    ckpt = torch.load(resolve_path(args.checkpoint, base), map_location="cpu", weights_only=False)
    arrays = dict(np.load(arrays_path, allow_pickle=True))
    condition_scaler = ckpt.get("condition_scaler")
    target_scaler = ckpt.get("target_scaler")
    if condition_scaler is None or target_scaler is None:
        condition_scaler, target_scaler, y_scaled, condition, error_weights = fit_scalers(
            arrays,
            load_split(split_path)["train"],
            cfg["loss"].get("error_weight_clip_min", 0.1),
            cfg["loss"].get("error_weight_clip_max", 10.0),
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
    cfg["data"]["arrays_path"] = str(arrays_path)
    trainer = Trainer(cfg, arrays, clean_csv, graph_cache, load_split(split_path), scalers=(condition_scaler, target_scaler, y_scaled, condition, error_weights))
    missing, unexpected = trainer.model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print({"missing_keys": missing, "unexpected_keys": unexpected})
    _freeze_for_finetune(trainer.model, args.freeze_mode)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(
        {
            "run_name": run_name,
            "focus_properties": focus_properties,
            "property_loss_weights": cfg["loss"]["property_loss_weights"],
            "freeze_mode": args.freeze_mode,
            "trainable_parameters": trainable,
            "total_parameters": total,
        }
    )
    print(trainer.train())


if __name__ == "__main__":
    main()
