from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "il_property_prediction"
RUNNER = Path(__file__).resolve().parent / "run_baseline_comparison.py"


DEFAULT_CASES = {
    "random_point": PROJECT_DIR / "data" / "processed_ilthermo_interpolated" / "splits" / "row_level_seed42.json",
    "random_il_level": PROJECT_DIR / "data" / "processed_ilthermo_interpolated" / "splits" / "il_level_seed42.json",
    "property_balanced_il_level": PROJECT_DIR
    / "data"
    / "processed_ilthermo_interpolated"
    / "splits"
    / "il_level_property_balanced_seed42.json",
    "ion_family": PROJECT_DIR / "data" / "processed" / "splits" / "il_level_family_pair_seed42.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline comparison for multiple split strategies.")
    parser.add_argument("--models", default="all", help="Comma-separated model list or 'all'.")
    parser.add_argument(
        "--cases",
        default="random_point,random_il_level,property_balanced_il_level,ion_family",
        help=f"Comma-separated cases. Valid cases: {','.join(DEFAULT_CASES)}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DIR / "outputs" / "baseline_comparison_by_split_seed42",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tree-n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def selected_cases(text: str) -> list[str]:
    cases = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [case for case in cases if case not in DEFAULT_CASES]
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}. Valid cases: {sorted(DEFAULT_CASES)}")
    return cases


def main() -> None:
    args = parse_args()
    for case in selected_cases(args.cases):
        split_path = DEFAULT_CASES[case]
        output_root = args.output_root / case
        command = [
            sys.executable,
            str(RUNNER),
            "--models",
            args.models,
            "--split-path",
            str(split_path),
            "--output-root",
            str(output_root),
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--tree-n-estimators",
            str(args.tree_n_estimators),
            "--n-jobs",
            str(args.n_jobs),
            "--device",
            args.device,
        ]
        if args.skip_existing:
            command.append("--skip-existing")
        if args.dry_run:
            command.append("--dry-run")
        if args.allow_download:
            command.append("--allow-download")
        print({"case": case, "split_path": str(split_path), "output_root": str(output_root)})
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
