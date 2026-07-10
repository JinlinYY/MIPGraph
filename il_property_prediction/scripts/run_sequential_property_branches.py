from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataset import PROPERTY_NAMES
from src.utils.io import load_config, resolve_path


def _parse_props(text: str) -> list[str]:
    props = [item.strip() for item in text.split(",") if item.strip()]
    bad = [prop for prop in props if prop not in PROPERTY_NAMES]
    if bad:
        raise ValueError(f"Unknown properties: {bad}. Valid properties: {PROPERTY_NAMES}")
    return props


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequentially fine-tune one property branch at a time from a base MIPGraph checkpoint."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/finetune_viscosity_from_weak_seed42/best_model.pt",
        help="Starting best MIPGraph checkpoint.",
    )
    parser.add_argument(
        "--property-order",
        default="Density,Viscosity,ElectricalConductivity,HeatCapacity,SurfaceTension,ThermalConductivity",
        help="Comma-separated order. Each step starts from the previous step's best checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed used only for the train/val/test split. Defaults to --seed.",
    )
    parser.add_argument("--run-prefix", default="sequential_branch")
    parser.add_argument("--output-root", default="outputs/property_branch_sequence")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--focus-weight", type=float, default=1.0)
    parser.add_argument(
        "--background-weight",
        type=float,
        default=0.0,
        help="Use 0.0 for target-only branch optimization.",
    )
    parser.add_argument(
        "--freeze-mode",
        choices=["property_branch", "target_branch_plus_shared", "graph_frozen", "decoder_condition", "head_latent_condition"],
        default="property_branch",
        help="Use target_branch_plus_shared for higher GPU utilization while freezing non-target decoder branches.",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=1.0,
        help="Teacher-distillation weight for properties completed in earlier steps.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--monitor-space", choices=["raw", "log"], default="log")
    parser.add_argument(
        "--disable-property-coupling",
        action="store_true",
        help="Use independent property equations with molecular, temperature, and pressure inputs only.",
    )
    parser.add_argument("--augment-properties", default="")
    parser.add_argument("--augment-points-per-interval", type=int, default=1)
    parser.add_argument("--augment-max-temperature-gap", type=float, default=40.0)
    parser.add_argument("--augment-sample-weight", type=float, default=0.5)
    parser.add_argument("--augment-max-samples-per-property", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    fine_tune_script = PROJECT_DIR / "scripts" / "fine_tune_properties.py"
    properties = _parse_props(args.property_order)
    augmentation_properties = set(_parse_props(args.augment_properties)) if args.augment_properties.strip() else set()
    current_checkpoint = args.checkpoint
    completed_properties: list[str] = []

    for step, prop in enumerate(properties, start=1):
        run_name = f"{args.run_prefix}_step{step:02d}_{prop}_seed{args.seed}"
        best_path = resolve_path(args.output_root, base) / "checkpoints" / run_name / "best_model.pt"
        if args.skip_existing and best_path.exists():
            print(f"skip existing: {run_name}")
            current_checkpoint = str(best_path)
            continue
        cmd = [
            sys.executable,
            str(fine_tune_script),
            "--config",
            args.config,
            "--checkpoint",
            current_checkpoint,
            "--target-property",
            prop,
            "--seed",
            str(args.seed),
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
            "--focus-weight",
            str(args.focus_weight),
            "--background-weight",
            str(args.background_weight),
            "--freeze-mode",
            args.freeze_mode,
            "--num-workers",
            str(args.num_workers),
            "--monitor-space",
            args.monitor_space,
            "--use-checkpoint-model-config",
        ]
        if args.disable_property_coupling:
            cmd.append("--disable-property-coupling")
        if prop in augmentation_properties:
            cmd.extend(
                [
                    "--augment-properties",
                    prop,
                    "--augment-points-per-interval",
                    str(args.augment_points_per_interval),
                    "--augment-max-temperature-gap",
                    str(args.augment_max_temperature_gap),
                    "--augment-sample-weight",
                    str(args.augment_sample_weight),
                    "--augment-max-samples-per-property",
                    str(args.augment_max_samples_per_property),
                ]
            )
        if args.split_seed is not None:
            cmd.extend(["--split-seed", str(args.split_seed)])
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
        if args.validate_every is not None:
            cmd.extend(["--validate-every", str(args.validate_every)])
        if completed_properties and args.distill_weight > 0.0:
            cmd.extend(
                [
                    "--protect-properties",
                    ",".join(completed_properties),
                    "--distill-checkpoint",
                    current_checkpoint,
                    "--distill-weight",
                    str(args.distill_weight),
                ]
            )
        print("running:", " ".join(cmd))
        subprocess.run(cmd, cwd=base, check=True)
        current_checkpoint = str(best_path)
        completed_properties.append(prop)

    print(f"final_checkpoint: {current_checkpoint}")


if __name__ == "__main__":
    main()
