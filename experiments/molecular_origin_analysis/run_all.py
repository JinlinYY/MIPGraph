"""Unified Windows-compatible entrypoint for the molecular-origin analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.molecular_origin_analysis.src.pipeline import AnalysisPipeline


STAGES = [
    "all",
    "inspect",
    "extract",
    "association",
    "attribution",
    "attention",
    "counterfactual",
    "applicability",
    "rules",
    "figures",
    "manuscript",
    "validate",
]


def parse_args() -> argparse.Namespace:
    module_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the non-invasive MIPGraph molecular-origin workflow."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=module_root / "config" / "analysis_config.yaml",
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--set", dest="set_values", action="append", default=[])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = {}
    for item in args.set_values:
        if "=" not in item:
            raise ValueError(f"--set requires KEY=VALUE, received {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    if args.device is not None:
        overrides["model.device"] = args.device
    if args.batch_size is not None:
        overrides["model.batch_size"] = args.batch_size
    if args.max_samples is not None:
        overrides["data.max_samples"] = args.max_samples
    pipeline = AnalysisPipeline(args.config, overrides=overrides, verbose=args.verbose)
    if args.stage == "all":
        result = pipeline.run_all(force=args.force)
    else:
        stage_result = pipeline.run_stage(args.stage, force=args.force)
        manifest = pipeline.write_reproducibility_manifest()
        result = {
            "stage": args.stage,
            "result_type": type(stage_result).__name__,
            "manifest": str(manifest),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
