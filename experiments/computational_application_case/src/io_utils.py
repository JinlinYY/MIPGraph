"""Schema-aware application-local output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


OUTPUT_SUBDIRECTORIES = (
    "data",
    "cache",
    "audit",
    "figures",
    "tables",
    "report",
    "logs",
    "steps",
)


def prepare_output_tree(output_dir: str | Path) -> dict[str, Path]:
    """Create and return the fixed application output subdirectories."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: root / name for name in OUTPUT_SUBDIRECTORIES}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    paths["root"] = root
    return paths


def validate_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    """Raise an actionable schema error when required columns are absent."""

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def write_json(payload: Any, path: str | Path) -> Path:
    """Write indented UTF-8 JSON, converting common NumPy scalar values."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")

    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=default)
    return target


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a UTF-8 CSV without an index."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def write_step_marker(
    step_dir: str | Path,
    step: str,
    payload: dict[str, Any],
) -> Path:
    """Persist successful step metadata used by resume mode."""

    return write_json({"step": step, "status": "completed", **payload}, Path(step_dir) / f"{step}.json")

