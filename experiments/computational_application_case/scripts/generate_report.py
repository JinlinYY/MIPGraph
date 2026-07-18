"""Regenerate case tables and report from persisted outputs."""
from __future__ import annotations
import sys
from pathlib import Path
CASE_DIR = Path(__file__).resolve().parents[1]
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))
from run_all import main as run_all_main
if __name__ == "__main__":
    code = run_all_main(["--only-step", "tables", *sys.argv[1:]])
    if code == 0:
        code = run_all_main(["--only-step", "report", "--resume", *sys.argv[1:]])
    raise SystemExit(code)
