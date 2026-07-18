"""Run the case unit tests and persist their exit status for run summaries."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CASE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.computational_application_case.src.io_utils import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CASE_DIR / "outputs",
        help="Application output root receiving audit/unit_test_results.json.",
    )
    args = parser.parse_args()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        str(CASE_DIR / "tests"),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "status": "passed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    write_json(payload, args.output_dir.resolve() / "audit" / "unit_test_results.json")
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
