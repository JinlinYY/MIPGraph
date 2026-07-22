"""Build checkpoint-specific inputs for cross-protocol decision stability.

The formal primary candidate library is frozen before either sensitivity
checkpoint is evaluated.  Each sensitivity model is then run independently
with the training split that produced that checkpoint.  No prediction is
averaged across protocols.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CASE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.computational_application_case.src.config import (  # noqa: E402
    load_case_config,
    temperature_grid,
)
from experiments.computational_application_case.src.paths import (  # noqa: E402
    resolve_project_path,
)


PRIMARY_ROOT = CASE_DIR / "outputs_primary_audited"
PROTOCOL_CONFIGS = (
    CASE_DIR / "configs" / "protocol_stability_balanced.yaml",
    CASE_DIR / "configs" / "protocol_stability_ion_family.yaml",
)
IDENTITY_COLUMNS = (
    "candidate_id",
    "canonical_il_key",
    "cation_identity_key",
    "anion_identity_key",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_sha256(frame: pd.DataFrame) -> str:
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Candidate library lacks identity columns: {missing}")
    canonical = (
        frame[list(IDENTITY_COLUMNS)]
        .astype(str)
        .sort_values("candidate_id")
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    return hashlib.sha256(canonical).hexdigest()


def validate_protocol_manifest(
    manifest: dict[str, object],
    *,
    config_path: Path,
    training_split_path: Path,
    candidate_identity_digest: str,
    checkpoint_digest: str,
    candidate_count: int,
    temperature_grid_K: np.ndarray,
) -> None:
    """Reject stale sensitivity outputs before they can enter a decision matrix."""

    expected = {
        "config_path": str(config_path.resolve()),
        "training_split_path": str(training_split_path.resolve()),
        "candidate_identity_sha256": candidate_identity_digest,
        "checkpoint_sha256": checkpoint_digest,
        "candidate_count": int(candidate_count),
        "prediction_rows": int(candidate_count * len(temperature_grid_K)),
        "prediction_aggregation": "none; single checkpoint",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Protocol manifest field {key!r} is stale: "
                f"expected={value!r}, actual={manifest.get(key)!r}"
            )
    actual_grid = np.asarray(manifest.get("temperature_grid_K", []), dtype=float)
    if not np.array_equal(actual_grid, temperature_grid_K):
        raise RuntimeError("Protocol manifest has a stale temperature grid")


def run_step(script_name: str, config_path: Path, force: bool) -> None:
    command = [
        sys.executable,
        str(CASE_DIR / "scripts" / script_name),
        "--config",
        str(config_path),
    ]
    if force:
        command.append("--force")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_one(config_path: Path, frozen_library: pd.DataFrame, force: bool) -> dict[str, object]:
    config = load_case_config(config_path)
    output_root = resolve_project_path(PROJECT_ROOT, config["outputs"]["output_dir"])
    data_dir = output_root / "data"
    audit_dir = output_root / "audit"
    data_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    library_path = data_dir / "candidate_library.csv"
    shutil.copy2(PRIMARY_ROOT / "data" / "candidate_library.csv", library_path)

    run_step("audit_repository.py", config_path, force)
    run_step("run_property_inference.py", config_path, force)
    run_step("assess_applicability_domain.py", config_path, force)

    copied_library = pd.read_csv(library_path, low_memory=False)
    frozen_digest = identity_sha256(frozen_library)
    copied_digest = identity_sha256(copied_library)
    if copied_digest != frozen_digest:
        raise RuntimeError("Protocol candidate identities differ from the frozen primary library")

    predictions = pd.read_csv(data_dir / "property_predictions_long.csv", low_memory=False)
    predicted_identity = predictions[list(IDENTITY_COLUMNS)].drop_duplicates()
    if identity_sha256(predicted_identity) != frozen_digest:
        raise RuntimeError("Protocol predictions do not preserve the frozen ID--InChI mapping")
    expected_temperatures = np.unique(
        np.concatenate(
            (
                temperature_grid(config["conditions"]),
                temperature_grid(config["conditions"], extended=True),
            )
        )
    )
    actual_temperatures = np.sort(predictions["temperature_K"].unique())
    if not np.array_equal(actual_temperatures, expected_temperatures):
        raise RuntimeError(
            f"Incomplete protocol temperature grid: expected={expected_temperatures.tolist()}, "
            f"actual={actual_temperatures.tolist()}"
        )
    expected_rows = len(frozen_library) * len(expected_temperatures)
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"Incomplete protocol prediction table: expected {expected_rows}, got {len(predictions)}"
        )

    inference = json.loads((audit_dir / "inference_pipeline.json").read_text(encoding="utf-8"))
    expected_checkpoint = resolve_project_path(PROJECT_ROOT, config["model"]["checkpoint_path"])
    actual_checkpoint = Path(inference["checkpoint_path"]).resolve()
    if actual_checkpoint != expected_checkpoint.resolve():
        raise RuntimeError(
            f"Protocol checkpoint mismatch: expected {expected_checkpoint}, got {actual_checkpoint}"
        )
    manifest = {
        "config_path": str(config_path.resolve()),
        "output_root": str(output_root.resolve()),
        "checkpoint_path": str(expected_checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(expected_checkpoint),
        "training_split_path": str(
            resolve_project_path(PROJECT_ROOT, config["data"]["split_path"]).resolve()
        ),
        "candidate_identity_sha256": frozen_digest,
        "candidate_count": int(len(frozen_library)),
        "temperature_grid_K": actual_temperatures.tolist(),
        "prediction_rows": int(len(predictions)),
        "prediction_aggregation": "none; single checkpoint",
    }
    (audit_dir / "protocol_stability_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build independently auditable cross-protocol sensitivity outputs."
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    primary_library = pd.read_csv(PRIMARY_ROOT / "data" / "candidate_library.csv", low_memory=False)
    manifests = [build_one(path, primary_library, args.force) for path in PROTOCOL_CONFIGS]
    print(json.dumps(manifests, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
