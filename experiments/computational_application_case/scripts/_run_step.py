"""Shared standalone runner for named application-case steps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


CASE_DIR = Path(__file__).resolve().parents[1]
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from run_all import main as run_all_main  # noqa: E402


def run_step(step: str, argv: Sequence[str] | None = None) -> int:
    """Forward common standalone-script arguments to the main pipeline."""

    parser = argparse.ArgumentParser(description=f"Run only the {step} step.")
    parser.add_argument("--config", type=str, default=str(CASE_DIR / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    forwarded = ["--config", args.config, "--only-step", step]
    for flag, value in [
        ("--output-dir", args.output_dir),
        ("--checkpoint", args.checkpoint),
        ("--device", args.device),
        ("--batch-size", args.batch_size),
    ]:
        if value is not None:
            forwarded.extend([flag, str(value)])
    if args.force:
        forwarded.append("--force")
    if args.verbose:
        forwarded.append("--verbose")
    return run_all_main(forwarded)
