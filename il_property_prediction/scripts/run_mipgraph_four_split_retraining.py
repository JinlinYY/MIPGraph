from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]


@dataclass(frozen=True)
class SplitCase:
    key: str
    label: str
    split_path: str
    base_checkpoint: str
    base_val_metrics: str
    target_properties: tuple[str, ...]


CASES = [
    SplitCase(
        key="random_point",
        label="Random-point",
        split_path="data/processed/splits/row_level_seed42.json",
        base_checkpoint=(
            "outputs/fg_transformer_random_point_seed42_noamp/checkpoints/"
            "unimol2_fg_transformer_random_point_seed42_noamp_resume56/"
            "best_model_pid73748_epoch088.pt"
        ),
        base_val_metrics=(
            "outputs/fg_transformer_random_point_seed42_noamp/metrics/"
            "unimol2_fg_transformer_random_point_seed42_noamp_resume56/val_metrics.json"
        ),
        target_properties=(
            "Density",
            "ElectricalConductivity",
            "HeatCapacity",
            "SurfaceTension",
            "ThermalConductivity",
        ),
    ),
    SplitCase(
        key="random_il_level",
        label="Random IL-level",
        split_path="data/processed/splits/il_level_seed42.json",
        base_checkpoint="outputs/split_strategy_comparison_seed42/checkpoints/il_level_random_seed42/best_model.pt",
        base_val_metrics="outputs/split_strategy_comparison_seed42/metrics/il_level_random_seed42/val_metrics.json",
        target_properties=("HeatCapacity", "SurfaceTension"),
    ),
    SplitCase(
        key="property_balanced_il_level",
        label="Property-balanced IL-level",
        split_path="data/processed/splits/il_level_property_balanced_seed42.json",
        base_checkpoint=(
            "outputs/split_strategy_comparison_seed42/checkpoints/"
            "il_level_property_balanced_seed42/best_model.pt"
        ),
        base_val_metrics=(
            "outputs/split_strategy_comparison_seed42/metrics/"
            "il_level_property_balanced_seed42/val_metrics.json"
        ),
        target_properties=(
            "Density",
            "ElectricalConductivity",
            "HeatCapacity",
            "SurfaceTension",
            "ThermalConductivity",
        ),
    ),
    SplitCase(
        key="ion_family",
        label="Ion-family",
        split_path="data/processed/splits/il_level_family_pair_seed42.json",
        base_checkpoint="outputs/split_strategy_comparison_seed42/checkpoints/il_level_family_pair_seed42/best_model.pt",
        base_val_metrics="outputs/split_strategy_comparison_seed42/metrics/il_level_family_pair_seed42/val_metrics.json",
        target_properties=("HeatCapacity", "SurfaceTension", "ThermalConductivity"),
    ),
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_props(text: str) -> tuple[str, ...]:
    props = tuple(item.strip() for item in text.split(",") if item.strip())
    bad = [prop for prop in props if prop not in PROPERTY_NAMES]
    if bad:
        raise ValueError(f"Unknown properties: {bad}. Valid properties: {PROPERTY_NAMES}")
    return props


def val_score(metrics_path: Path, prop: str) -> float:
    metrics = load_json(metrics_path)
    item = metrics["log_space"][prop]
    return float(item["log_R2"]) - 0.2 * float(item["log_NMAE"])


def val_log_mae(metrics_path: Path, prop: str) -> float:
    metrics = load_json(metrics_path)
    return float(metrics["log_space"][prop]["log_MAE"])


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("running:", subprocess.list2cmdline(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def resolve_project(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_DIR / p).resolve()


def completed_val_metrics(root: Path, run_name: str) -> Path:
    return root / "metrics" / run_name / "val_metrics.json"


def completed_manifest(root: Path, run_name: str) -> Path:
    return root / "metrics" / run_name / "run_manifest.json"


def train_specialists(case: SplitCase, args: argparse.Namespace, output_root: Path) -> dict[str, list[dict]]:
    fine_tune = PROJECT_DIR / "scripts" / "fine_tune_properties.py"
    results: dict[str, list[dict]] = {prop: [] for prop in PROPERTY_NAMES}
    base_checkpoint = resolve_project(case.base_checkpoint)
    base_val = resolve_project(case.base_val_metrics)
    for prop in PROPERTY_NAMES:
        results[prop].append(
            {
                "source": "base",
                "checkpoint": str(base_checkpoint),
                "val_metrics": str(base_val),
                "val_score": val_score(base_val, prop),
                "val_log_MAE": val_log_mae(base_val, prop),
            }
        )

    for prop in case.target_properties:
        for seed in args.seeds:
            run_name = f"{case.key}_specialist_{prop}_seed{seed}"
            val_path = completed_val_metrics(output_root, run_name)
            manifest_path = completed_manifest(output_root, run_name)
            if not (args.skip_existing and val_path.exists() and manifest_path.exists()):
                cmd = [
                    sys.executable,
                    str(fine_tune),
                    "--config",
                    args.config,
                    "--checkpoint",
                    str(base_checkpoint),
                    "--split-path",
                    str(resolve_project(case.split_path)),
                    "--target-property",
                    prop,
                    "--seed",
                    str(seed),
                    "--split-seed",
                    "42",
                    "--run-name",
                    run_name,
                    "--output-root",
                    str(output_root),
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
                    "log",
                    "--monitor-objective",
                    args.monitor_objective,
                    "--use-checkpoint-model-config",
                    "--skip-test-evaluation",
                ]
                if args.enable_amp:
                    cmd.append("--enable-amp")
                if args.il_balanced_loss:
                    cmd.extend(["--il-balanced-loss", "--il-balance-power", str(args.il_balance_power)])
                if prop in args.augment_properties:
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
                run_command(cmd, args.dry_run)
            if args.dry_run:
                continue
            manifest = load_json(manifest_path)
            results[prop].append(
                {
                    "source": run_name,
                    "checkpoint": manifest["best_checkpoint"],
                    "val_metrics": str(val_path),
                    "val_score": val_score(val_path, prop),
                    "val_log_MAE": val_log_mae(val_path, prop),
                }
            )
    return results


def candidate_rank_value(item: dict, objective: str) -> float:
    if objective == "val_score":
        return -float(item["val_score"])
    if objective == "val_log_mae":
        return float(item["val_log_MAE"])
    raise ValueError(f"Unknown selection objective: {objective}")


def build_selection(case: SplitCase, candidates: dict[str, list[dict]], output_root: Path, objective: str) -> Path:
    selected = {}
    for prop in PROPERTY_NAMES:
        items = sorted(candidates[prop], key=lambda item: candidate_rank_value(item, objective))
        selected[prop] = items[0]
    doc = {
        "case": case.key,
        "label": case.label,
        "selection_rule": (
            "validation only: maximize val_log_R2 - 0.2 * val_log_NMAE"
            if objective == "val_score"
            else "validation only: minimize val_log_MAE"
        ),
        "selection_objective": objective,
        "test_used_for_selection": False,
        "target_properties": list(case.target_properties),
        "properties": selected,
        "all_candidates": candidates,
    }
    path = output_root / "selected_checkpoints.json"
    save_json(doc, path)
    return path


def evaluate_selection(case: SplitCase, selection_path: Path, output_root: Path, args: argparse.Namespace) -> None:
    evaluator = PROJECT_DIR / "scripts" / "evaluate_selected_property_checkpoints.py"
    final_root = output_root / "final"
    cmd = [
        sys.executable,
        str(evaluator),
        "--selection",
        str(selection_path),
        "--clean-csv",
        str(PROJECT_DIR / "data" / "processed" / "il_multiprop_clean.csv"),
        "--arrays-path",
        str(PROJECT_DIR / "data" / "processed" / "il_multiprop_arrays.npz"),
        "--graph-cache",
        str(PROJECT_DIR / "data" / "processed" / "graph_cache_fg.pt"),
        "--split-path",
        str(resolve_project(case.split_path)),
        "--output-root",
        str(final_root),
        "--batch-size",
        str(args.eval_batch_size),
    ]
    run_command(cmd, args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain validation-selected MIPGraph property specialists on the four split strategies."
    )
    parser.add_argument("--config", default="configs/physics_moe_fg_transformer.yaml")
    parser.add_argument("--output-root", default="outputs/mipgraph_split_optimization_seed42/retraining")
    parser.add_argument("--cases", default=",".join(case.key for case in CASES))
    parser.add_argument("--extra-properties", default="")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--focus-weight", type=float, default=4.0)
    parser.add_argument("--background-weight", type=float, default=0.0)
    parser.add_argument("--freeze-mode", default="property_branch")
    parser.add_argument("--enable-amp", action="store_true", default=True)
    parser.add_argument("--no-enable-amp", action="store_false", dest="enable_amp")
    parser.add_argument("--il-balanced-loss", action="store_true")
    parser.add_argument("--il-balance-power", type=float, default=1.0)
    parser.add_argument("--augment-properties", default="")
    parser.add_argument("--augment-points-per-interval", type=int, default=1)
    parser.add_argument("--augment-max-temperature-gap", type=float, default=40.0)
    parser.add_argument("--augment-sample-weight", type=float, default=0.5)
    parser.add_argument("--augment-max-samples-per-property", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--selection-objective", choices=["val_log_mae", "val_score"], default="val_log_mae")
    parser.add_argument("--monitor-objective", choices=["score", "mae", "nmae", "r2"], default="mae")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected_case_keys = {item.strip() for item in args.cases.split(",") if item.strip()}
    case_by_key = {case.key: case for case in CASES}
    bad_cases = selected_case_keys - set(case_by_key)
    if bad_cases:
        raise ValueError(f"Unknown cases: {sorted(bad_cases)}")
    args.seeds = parse_csv_ints(args.seeds)
    extra_properties = set(parse_csv_props(args.extra_properties)) if args.extra_properties.strip() else set()
    args.augment_properties = set(parse_csv_props(args.augment_properties)) if args.augment_properties.strip() else set()

    root = resolve_project(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for case in CASES:
        if case.key not in selected_case_keys:
            continue
        if extra_properties:
            case = SplitCase(
                key=case.key,
                label=case.label,
                split_path=case.split_path,
                base_checkpoint=case.base_checkpoint,
                base_val_metrics=case.base_val_metrics,
                target_properties=tuple(dict.fromkeys((*case.target_properties, *extra_properties))),
            )
        case_root = root / case.key
        case_root.mkdir(parents=True, exist_ok=True)
        candidates = train_specialists(case, args, case_root)
        if args.dry_run:
            continue
        selection_path = build_selection(case, candidates, case_root, args.selection_objective)
        evaluate_selection(case, selection_path, case_root, args)
        summary[case.key] = {
            "selection": str(selection_path),
            "final_metrics": str(case_root / "final" / "selected_test_metrics.csv"),
        }
    if not args.dry_run:
        save_json(summary, root / "summary.json")
    print({"complete": not args.dry_run, "summary": summary}, flush=True)


if __name__ == "__main__":
    main()
