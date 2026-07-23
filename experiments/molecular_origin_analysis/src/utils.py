"""Configuration, path, cache, logging, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


MODULE_ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating the inputs."""

    merged = json.loads(json.dumps(base, default=str))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_dotted(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = mapping
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot override {dotted_key!r}: {part!r} is not a mapping")
        target = child
    target[parts[-1]] = value


def load_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML and apply typed dotted-key overrides."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Analysis configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    for key, value in (overrides or {}).items():
        parsed = yaml.safe_load(value) if isinstance(value, str) else value
        _set_dotted(config, key, parsed)
    config["_config_path"] = str(config_path)
    config["_module_root"] = str(MODULE_ROOT)
    return config


def resolve_path(value: str | Path, base: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def ensure_within(path: str | Path, parent: str | Path) -> Path:
    resolved = Path(path).resolve()
    boundary = Path(parent).resolve()
    if not resolved.is_relative_to(boundary):
        raise ValueError(f"Path escapes the permitted module directory: {resolved}")
    return resolved


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(root: str | Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_status(root: str | Path) -> list[str]:
    try:
        output = subprocess.run(
            ["git", "status", "--short"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return [line for line in output.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        return []


def configure_logging(log_path: str | Path, verbose: bool = False) -> logging.Logger:
    path = ensure_within(log_path, MODULE_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("molecular_origin_analysis")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def write_json(path: str | Path, payload: Any) -> Path:
    target = ensure_within(path, MODULE_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a table, preserving the requested parquet name when supported."""

    target = ensure_within(path, MODULE_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(target, index=False)
            return target
        except (ImportError, ModuleNotFoundError):
            target = target.with_suffix(".csv")
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def read_table(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        return pd.read_parquet(target)
    return pd.read_csv(target)


def software_versions() -> dict[str, str | None]:
    modules = ["torch", "rdkit", "numpy", "pandas", "sklearn", "scipy", "matplotlib", "shap"]
    versions: dict[str, str | None] = {"python": os.sys.version.split()[0]}
    for name in modules:
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except (ImportError, OSError):
            versions[name] = None
    try:
        import torch

        versions["cuda"] = str(torch.version.cuda) if torch.cuda.is_available() else None
    except ImportError:
        versions["cuda"] = None
    return versions
