from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils.io import load_config, resolve_path


def _flatten_metrics(metrics: dict, prefix: str = "") -> dict:
    rows = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            rows.update(_flatten_metrics(value, name))
        else:
            rows[name] = value
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IL-level training for multiple random seeds and aggregate test metrics.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--run-prefix", default="seed")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    train_script = PROJECT_DIR / "scripts" / "train_mipgraphnet.py"
    summary_rows = []

    for seed in seeds:
        run_name = f"{args.run_prefix}_{seed}"
        metrics_path = resolve_path(cfg["outputs"]["metric_dir"], base) / run_name / "test_metrics.json"
        if not (args.skip_existing and metrics_path.exists()):
            cmd = [
                sys.executable,
                str(train_script),
                "--config",
                args.config,
                "--seed",
                str(seed),
                "--run-name",
                run_name,
            ]
            print("running:", " ".join(cmd))
            subprocess.run(cmd, cwd=base, check=True)
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            row = {"seed": seed, "run_name": run_name}
            row.update(_flatten_metrics(metrics))
            summary_rows.append(row)
        else:
            print(f"warning: metrics not found for seed {seed}: {metrics_path}")

    if not summary_rows:
        return
    out_path = resolve_path(cfg["outputs"]["metric_dir"], base) / f"{args.run_prefix}_multi_seed_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    fieldnames = ["seed", "run_name"] + [key for key in fieldnames if key not in {"seed", "run_name"}]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
