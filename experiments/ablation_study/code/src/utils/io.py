from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    base = Path(base_dir) if base_dir is not None else project_root()
    return (base / p).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        candidate = project_root() / path
        if candidate.exists():
            path = candidate
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    cfg["_base_dir"] = str(path.resolve().parent.parent)
    return cfg


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_metrics_csv(metrics: dict[str, Any], path: str | Path, property_names: list[str]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return ""

    rows = []
    for name in property_names:
        item = metrics.get(name, {})
        rows.append(
            {
                "property": name,
                "MAE": fmt(item.get("MAE")),
                "RMSE": fmt(item.get("RMSE")),
                "R2": fmt(item.get("R2")),
            }
        )
    rows.append(
        {
            "property": "Average",
            "MAE": fmt(metrics.get("macro_MAE")),
            "RMSE": fmt(metrics.get("macro_RMSE")),
            "R2": fmt(metrics.get("macro_R2")),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["property", "MAE", "RMSE", "R2"])
        writer.writeheader()
        writer.writerows(rows)


def save_log_metrics_csv(metrics: dict[str, Any], path: str | Path, property_names: list[str]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return ""

    log_metrics = metrics.get("log_space", {})
    rows = []
    for name in property_names:
        item = log_metrics.get(name, {})
        rows.append(
            {
                "property": name,
                "log_MAE": fmt(item.get("log_MAE")),
                "log_RMSE": fmt(item.get("log_RMSE")),
                "log_R2": fmt(item.get("log_R2")),
            }
        )
    rows.append(
        {
            "property": "Average",
            "log_MAE": fmt(log_metrics.get("macro_log_MAE")),
            "log_RMSE": fmt(log_metrics.get("macro_log_RMSE")),
            "log_R2": fmt(log_metrics.get("macro_log_R2")),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["property", "log_MAE", "log_RMSE", "log_R2"])
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
