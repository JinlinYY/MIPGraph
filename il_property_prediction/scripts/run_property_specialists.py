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


DEFAULT_SEEDS = "0,1,2,3,4,42,123,2024"


def _parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_csv_props(text: str) -> list[str]:
    props = [item.strip() for item in text.split(",") if item.strip()]
    bad = [prop for prop in props if prop not in PROPERTY_NAMES]
    if bad:
        raise ValueError(f"Unknown properties: {bad}. Valid properties: {PROPERTY_NAMES}")
    return props


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train property-specialized MIPGraph fine-tune runs from an existing best checkpoint."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--clean-csv", default=None)
    parser.add_argument("--arrays-path", default=None)
    parser.add_argument("--graph-cache", default=None)
    parser.add_argument("--split-path", default=None)
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/finetune_viscosity_from_weak_seed42/best_model.pt",
        help="Base MIPGraph checkpoint. v2 adapters are not enabled by this script.",
    )
    parser.add_argument("--properties", default=",".join(PROPERTY_NAMES))
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Fixed data split shared by all training seeds.",
    )
    parser.add_argument("--output-root", default="outputs/property_specialists")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validate-every", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--enable-amp", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--focus-weight", type=float, default=4.0)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument(
        "--freeze-mode",
        choices=["all", "encoder_frozen", "graph_frozen", "head_latent_condition", "decoder_condition", "decoder", "property_branch"],
        default="encoder_frozen",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--interpolation-label-weight", type=float, default=None)
    parser.add_argument("--il-balanced-loss", action="store_true")
    parser.add_argument("--il-balance-power", type=float, default=1.0)
    parser.add_argument("--monitor-space", choices=["raw", "log"], default="log")
    parser.add_argument(
        "--disable-property-coupling",
        action="store_true",
        help="Remove explicit property-to-property decoder terms.",
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
    properties = _parse_csv_props(args.properties)
    augmentation_properties = set(_parse_csv_props(args.augment_properties)) if args.augment_properties.strip() else set()
    seeds = _parse_csv_ints(args.seeds)
    fine_tune_script = PROJECT_DIR / "scripts" / "fine_tune_properties.py"
    active: list[tuple[str, subprocess.Popen]] = []

    def wait_one() -> None:
        run_name, process = active.pop(0)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, run_name)

    for prop in properties:
        for seed in seeds:
            run_name = f"property_specialist_{prop}_seed{seed}"
            metrics_path = resolve_path(args.output_root, base) / "metrics" / run_name / "test_metrics.json"
            manifest_path = resolve_path(args.output_root, base) / "metrics" / run_name / "run_manifest.json"
            if args.skip_existing and manifest_path.exists() and metrics_path.exists():
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
                str(seed),
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
            if args.interpolation_label_weight is not None:
                cmd.extend(["--interpolation-label-weight", str(args.interpolation_label_weight)])
            if args.il_balanced_loss:
                cmd.extend(["--il-balanced-loss", "--il-balance-power", str(args.il_balance_power)])
            for option, value in (
                ("--clean-csv", args.clean_csv),
                ("--arrays-path", args.arrays_path),
                ("--graph-cache", args.graph_cache),
                ("--split-path", args.split_path),
            ):
                if value:
                    cmd.extend([option, value])
            if args.disable_property_coupling:
                cmd.append("--disable-property-coupling")
            if args.skip_test_evaluation:
                cmd.append("--skip-test-evaluation")
            if args.enable_amp:
                cmd.append("--enable-amp")
            if args.disable_amp:
                cmd.append("--disable-amp")
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
            print("running:", " ".join(cmd))
            if args.max_parallel <= 1:
                subprocess.run(cmd, cwd=base, check=True)
            else:
                while len(active) >= args.max_parallel:
                    wait_one()
                active.append((run_name, subprocess.Popen(cmd, cwd=base)))

    while active:
        wait_one()


if __name__ == "__main__":
    main()
