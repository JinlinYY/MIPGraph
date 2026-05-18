from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils.io import load_config, resolve_path


def _absolute_data_paths(cfg: dict, base: str | Path) -> None:
    for key in ["raw_path", "processed_dir", "clean_csv", "arrays_path", "graph_cache_path", "failed_smiles_path"]:
        if key in cfg["data"]:
            cfg["data"][key] = str(resolve_path(cfg["data"][key], base))


def _absolute_output_roots(cfg: dict, base: str | Path) -> None:
    for key in ["output_dir", "checkpoint_dir", "log_dir", "metric_dir", "prediction_dir", "figure_dir", "ablation_dir"]:
        if key in cfg["outputs"]:
            cfg["outputs"][key] = str(resolve_path(cfg["outputs"][key], base))


def _write_config(cfg: dict, path: Path) -> Path:
    cfg = dict(cfg)
    cfg.pop("_config_path", None)
    cfg.pop("_base_dir", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or run split-strategy generalization experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split-types", default="random_point,il_pair,cation_family,anion_family")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--run-prefix", default="generalization")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Override base-training epochs for smoke tests or short runs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override base-training batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers; 0 is useful for Windows smoke tests.")
    parser.add_argument("--validate-every", type=int, default=None, help="Override validation frequency.")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience.")
    parser.add_argument("--with-finetune", action="store_true")
    parser.add_argument("--finetune-properties", default="ElectricalConductivity,SurfaceTension,Viscosity")
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-freeze-mode", default="graph_frozen")
    args = parser.parse_args()

    cfg0 = load_config(args.config)
    base = cfg0["_base_dir"]
    split_types = [item.strip() for item in args.split_types.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    config_dir = resolve_path(cfg0["outputs"]["output_dir"], base) / "generalization_configs"
    commands = []

    for split_type in split_types:
        for seed in seeds:
            cfg = load_config(args.config)
            _absolute_data_paths(cfg, base)
            _absolute_output_roots(cfg, base)
            cfg["data"]["split_type"] = split_type
            cfg["data"]["seed"] = seed
            cfg["training"]["seed"] = seed
            if args.epochs is not None:
                cfg["training"]["epochs"] = args.epochs
            if args.batch_size is not None:
                cfg["training"]["batch_size"] = args.batch_size
            if args.num_workers is not None:
                cfg["training"]["num_workers"] = args.num_workers
                if args.num_workers == 0:
                    cfg["training"]["persistent_workers"] = False
            if args.validate_every is not None:
                cfg["training"]["validate_every"] = args.validate_every
            if args.patience is not None:
                cfg["training"]["patience"] = args.patience
            run_name = f"{args.run_prefix}_{split_type}_seed{seed}"
            cfg["outputs"]["run_name"] = "default"
            cfg_path = _write_config(cfg, config_dir / f"{run_name}.yaml")
            metrics_path = resolve_path(cfg["outputs"]["metric_dir"], base) / run_name / "test_metrics.json"
            train_cmd = [
                sys.executable,
                str(PROJECT_DIR / "scripts" / "train_iptnet.py"),
                "--config",
                str(cfg_path),
                "--seed",
                str(seed),
                "--run-name",
                run_name,
            ]
            commands.append(train_cmd)
            if args.run_training and not (args.skip_existing and metrics_path.exists()):
                subprocess.run(train_cmd, cwd=PROJECT_DIR, check=True)

            if args.with_finetune:
                ckpt = resolve_path(cfg["outputs"]["checkpoint_dir"], base) / run_name / "best_model.pt"
                ft_run_name = f"{run_name}_finetune_weak"
                ft_metrics_path = resolve_path(cfg["outputs"]["metric_dir"], base) / ft_run_name / "test_metrics.json"
                ft_cmd = [
                    sys.executable,
                    str(PROJECT_DIR / "scripts" / "fine_tune_properties.py"),
                    "--config",
                    str(cfg_path),
                    "--checkpoint",
                    str(ckpt),
                    "--properties",
                    args.finetune_properties,
                    "--seed",
                    str(seed),
                    "--run-name",
                    ft_run_name,
                    "--epochs",
                    str(args.finetune_epochs),
                    "--lr",
                    str(args.finetune_lr),
                    "--freeze-mode",
                    args.finetune_freeze_mode,
                ]
                commands.append(ft_cmd)
                if args.run_training and not (args.skip_existing and ft_metrics_path.exists()):
                    subprocess.run(ft_cmd, cwd=PROJECT_DIR, check=True)

    print("Prepared commands:")
    for cmd in commands:
        print(" ".join(cmd))


if __name__ == "__main__":
    main()
