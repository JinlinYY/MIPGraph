"""Command-line entry point for the complete computational application case."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


CASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CASE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.computational_application_case.src.config import load_case_config  # noqa: E402
from experiments.computational_application_case.src.io_utils import prepare_output_tree  # noqa: E402
from experiments.computational_application_case.src.pipeline import CasePipeline, STEP_ORDER  # noqa: E402


LOGGER = logging.getLogger("computational_application_case")


def build_parser() -> argparse.ArgumentParser:
    """Build the documented command-line interface."""

    parser = argparse.ArgumentParser(
        description="Run MIPGraph-guided prospective screening of unseen ionic-liquid pairs."
    )
    parser.add_argument("--config", type=Path, default=CASE_DIR / "configs" / "default.yaml")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--checkpoints", nargs="+", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--only-step", choices=STEP_ORDER)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _configure_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _blocking_report(output_dir: Path, error: BaseException) -> Path:
    paths = prepare_output_tree(output_dir)
    target = paths["report"] / "blocking_report.md"
    lines = [
        "# Computational application case blocking report",
        "",
        f"- Exception type: `{type(error).__name__}`",
        f"- Message: `{error}`",
        "- The run stopped with a non-zero exit status; no missing result was replaced with synthetic data.",
        "- Inspect `logs/run.log` and the last completed JSON marker under `steps/`.",
        "- Minimum compatible remedy: correct the reported configuration, cache, checkpoint, schema, or adapter mismatch, then resume the same output directory.",
        "",
        "Deprecated web code used: No",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Load configuration, execute the selected steps, and return a process code."""

    args = build_parser().parse_args(argv)
    config_path = args.config
    if args.smoke_test and config_path.resolve() == (CASE_DIR / "configs" / "default.yaml").resolve():
        config_path = CASE_DIR / "configs" / "smoke_test.yaml"
    overrides = {
        "checkpoint": args.checkpoint,
        "checkpoints": args.checkpoints,
        "device": args.device,
        "batch_size": args.batch_size,
        "output_dir": args.output_dir,
    }
    output_dir = CASE_DIR / "outputs"
    try:
        config = load_case_config(config_path, overrides)
        output_dir = Path(config["_output_dir"])
        _configure_logging(output_dir / "logs" / "run.log", args.verbose)
        if args.smoke_test and not bool(config["figures"].get("simplified", False)):
            raise ValueError("--smoke-test requires a configuration with figures.simplified=true")
        _seed_everything(int(config["project"]["random_seed"]))
        LOGGER.info("Configuration: %s", config["_config_path"])
        LOGGER.info("Output directory: %s", output_dir)
        LOGGER.info("Device request: %s", config["model"]["device"])
        pipeline = CasePipeline(
            config,
            force=args.force,
            resume=args.resume,
            skip_figures=args.skip_figures,
            skip_report=args.skip_report,
        )
        results = pipeline.run(only_step=args.only_step)
        LOGGER.info("Completed steps: %s", ", ".join(results))
        stale_blocker = output_dir / "report" / "blocking_report.md"
        if stale_blocker.exists():
            stale_blocker.unlink()
        if args.only_step or args.skip_report:
            print(json.dumps({"status": "completed", "steps": list(results), "output_dir": str(output_dir)}, indent=2))
        return 0
    except Exception as error:
        if not LOGGER.handlers:
            _configure_logging(output_dir / "logs" / "run.log", args.verbose)
        report = _blocking_report(output_dir, error)
        LOGGER.error("Run failed: %s", error)
        if args.verbose:
            LOGGER.error("%s", traceback.format_exc())
        LOGGER.error("Blocking report: %s", report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
