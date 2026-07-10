from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataset import PROPERTY_NAMES
from src.utils.io import load_config, load_json, resolve_path, save_json


def _parse_props(text: str) -> list[str]:
    props = [item.strip() for item in text.split(",") if item.strip()]
    bad = [prop for prop in props if prop not in PROPERTY_NAMES]
    if bad:
        raise ValueError(f"Unknown properties: {bad}. Valid properties: {PROPERTY_NAMES}")
    return props


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one property-adapter specialist per property and merge them into one checkpoint."
    )
    parser.add_argument("--config", default="configs/physics_moe.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--properties", default=",".join(PROPERTY_NAMES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--run-prefix", default="adapter_specialist")
    parser.add_argument("--output-root", default="outputs/physics_moe_property_adapter_specialists_seed42")
    parser.add_argument("--merged-output", default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validate-every", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--focus-weight", type=float, default=1.0)
    parser.add_argument("--background-weight", type=float, default=0.0)
    parser.add_argument("--freeze-mode", default="property_adapter_branch")
    parser.add_argument("--monitor-space", choices=["raw", "log"], default="log")
    parser.add_argument("--target-scaler-mask", choices=["mask", "evaluation_mask"], default="evaluation_mask")
    parser.add_argument("--repair-viscosity-action", choices=["none", "drop", "downweight"], default="drop")
    parser.add_argument("--repair-viscosity-max-train", type=float, default=1000.0)
    parser.add_argument("--repair-viscosity-downweight", type=float, default=0.05)
    parser.add_argument("--property-adapter-dim", type=int, default=64)
    parser.add_argument("--property-adapter-dropout", type=float, default=0.1)
    parser.add_argument("--il-balanced-loss", action="store_true")
    parser.add_argument("--clean-csv", default=None)
    parser.add_argument("--arrays-path", default=None)
    parser.add_argument("--graph-cache", default=None)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    properties = _parse_props(args.properties)
    output_root = resolve_path(args.output_root, base)
    fine_tune_script = PROJECT_DIR / "scripts" / "fine_tune_properties.py"
    merge_script = PROJECT_DIR / "scripts" / "merge_property_specialists.py"
    selection = {
        "base_checkpoint": str(resolve_path(args.checkpoint, base)),
        "properties": {},
    }

    for prop in properties:
        run_name = f"{args.run_prefix}_{prop}_seed{args.seed}"
        manifest_path = output_root / "metrics" / run_name / "run_manifest.json"
        if args.skip_existing and manifest_path.exists():
            manifest = load_json(manifest_path)
            selection["properties"][prop] = {
                "source": run_name,
                "checkpoint": manifest["best_checkpoint"],
            }
            print(f"skip existing: {run_name}")
            continue
        cmd = [
            sys.executable,
            str(fine_tune_script),
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
            "--target-property",
            prop,
            "--seed",
            str(args.seed),
            "--split-seed",
            str(args.split_seed),
            "--run-name",
            run_name,
            "--output-root",
            args.output_root,
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--validate-every",
            str(args.validate_every),
            "--num-workers",
            str(args.num_workers),
            "--focus-weight",
            str(args.focus_weight),
            "--background-weight",
            str(args.background_weight),
            "--freeze-mode",
            args.freeze_mode,
            "--monitor-space",
            args.monitor_space,
            "--target-scaler-mask",
            args.target_scaler_mask,
            "--repair-viscosity-action",
            args.repair_viscosity_action,
            "--repair-viscosity-max-train",
            str(args.repair_viscosity_max_train),
            "--repair-viscosity-downweight",
            str(args.repair_viscosity_downweight),
            "--enable-property-adapters",
            "--property-adapter-dim",
            str(args.property_adapter_dim),
            "--property-adapter-dropout",
            str(args.property_adapter_dropout),
            "--use-checkpoint-model-config",
            "--skip-test-evaluation",
        ]
        if args.il_balanced_loss:
            cmd.extend(["--il-balanced-loss", "--il-balance-power", "1.0"])
        for option, value in (
            ("--clean-csv", args.clean_csv),
            ("--arrays-path", args.arrays_path),
            ("--graph-cache", args.graph_cache),
            ("--split-path", args.split_path),
        ):
            if value:
                cmd.extend([option, value])
        print("running:", " ".join(cmd))
        subprocess.run(cmd, cwd=base, check=True)
        manifest = load_json(manifest_path)
        selection["properties"][prop] = {
            "source": run_name,
            "checkpoint": manifest["best_checkpoint"],
        }

    selection_path = output_root / "selected_property_adapter_checkpoints.json"
    save_json(selection, selection_path)
    merged_output = (
        resolve_path(args.merged_output, base)
        if args.merged_output
        else output_root / "merged" / f"{args.run_prefix}_merged_seed{args.seed}.pt"
    )
    merge_cmd = [
        sys.executable,
        str(merge_script),
        "--base-checkpoint",
        args.checkpoint,
        "--selection-json",
        str(selection_path),
        "--output-checkpoint",
        str(merged_output),
    ]
    print("running:", " ".join(merge_cmd))
    subprocess.run(merge_cmd, cwd=base, check=True)
    print({"selection": str(selection_path), "merged_checkpoint": str(merged_output)})


if __name__ == "__main__":
    main()
