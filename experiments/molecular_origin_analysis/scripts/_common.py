"""Shared Windows-safe CLI entrypoint for individual analysis stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.molecular_origin_analysis.src.pipeline import AnalysisPipeline  # noqa: E402


def _overrides(arguments: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for item in arguments.set_values:
        if "=" not in item:
            raise ValueError(f"--set requires KEY=VALUE, received {item!r}")
        key, value = item.split("=", 1)
        output[key.strip()] = value.strip()
    if arguments.device is not None:
        output["model.device"] = arguments.device
    if arguments.batch_size is not None:
        output["model.batch_size"] = arguments.batch_size
    if arguments.max_samples is not None:
        output["data.max_samples"] = arguments.max_samples
    return output


def parser(description: str) -> argparse.ArgumentParser:
    module_root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description=description)
    result.add_argument(
        "--config",
        type=Path,
        default=module_root / "config" / "analysis_config.yaml",
    )
    result.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any dotted configuration field; repeat as needed.",
    )
    result.add_argument("--device", default=None)
    result.add_argument("--batch-size", type=int, default=None)
    result.add_argument("--max-samples", type=int, default=None)
    result.add_argument("--force", action="store_true")
    result.add_argument("--verbose", action="store_true")
    return result


def run_cli(stage: str, description: str) -> int:
    arguments = parser(description).parse_args()
    pipeline = AnalysisPipeline(
        arguments.config,
        overrides=_overrides(arguments),
        verbose=arguments.verbose,
    )
    result = pipeline.run_stage(stage, force=arguments.force)
    pipeline.write_reproducibility_manifest()
    print(json.dumps({"stage": stage, "result_type": type(result).__name__}, indent=2))
    return 0
