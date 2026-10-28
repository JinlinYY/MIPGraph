"""Read-only discovery of the existing MIPGraph project assets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import MODULE_ROOT, ensure_within, resolve_path, software_versions


PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]


@dataclass(frozen=True)
class InspectionReport:
    project_root: Path
    module_root: Path
    output_root: Path
    model_class: str
    model_config: Path
    selected_checkpoint: Path
    selected_checkpoint_type: str
    requested_checkpoint_type: str
    checkpoint_candidates: tuple[Path, ...]
    data_paths: dict[str, Path]
    split_paths: dict[str, Path]
    selected_split: Path
    property_names: list[str]
    descriptor_dimensions: dict[str, int]
    descriptor_names_complete: bool
    cross_ion_attention_access: str
    router_access: str
    expert_output_access: str
    screening_paths: dict[str, Path]
    environment: dict[str, str | None]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "module_root": str(self.module_root),
            "output_root": str(self.output_root),
            "model_class": self.model_class,
            "model_config": str(self.model_config),
            "selected_checkpoint": str(self.selected_checkpoint),
            "selected_checkpoint_type": self.selected_checkpoint_type,
            "requested_checkpoint_type": self.requested_checkpoint_type,
            "checkpoint_candidates": [str(path) for path in self.checkpoint_candidates],
            "data_paths": {key: str(path) for key, path in self.data_paths.items()},
            "split_paths": {key: str(path) for key, path in self.split_paths.items()},
            "selected_split": str(self.selected_split),
            "property_names": self.property_names,
            "property_indices": {name: index for index, name in enumerate(self.property_names)},
            "descriptor_dimensions": self.descriptor_dimensions,
            "descriptor_names_complete": self.descriptor_names_complete,
            "cross_ion_attention_access": self.cross_ion_attention_access,
            "router_access": self.router_access,
            "expert_output_access": self.expert_output_access,
            "screening_paths": {key: str(path) for key, path in self.screening_paths.items()},
            "environment": self.environment,
            "warnings": list(self.warnings),
            "adapter_strategy": (
                "Import the original package under il_property_prediction.src, bind the historical "
                "checkpoint alias only during deserialization, and consume forward auxiliary tensors "
                "without modifying the model."
            ),
            "target_transform": "log(y + 1e-8), property-wise standardization",
            "inverse_transform": "exp(y_scaled * std + mean) - 1e-8",
        }


class ProjectAdapter:
    """Locate verified assets while enforcing the new module's output boundary."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.module_root = MODULE_ROOT.resolve()
        self.project_root = resolve_path(config["project"]["root"], self.module_root)
        if not self.project_root.is_dir():
            raise FileNotFoundError(f"Project root not found: {self.project_root}")
        self.output_root = ensure_within(
            resolve_path(config["project"]["output_root"], self.project_root),
            self.module_root,
        )

    @staticmethod
    def _checkpoint_type(path: Path) -> str:
        text = path.as_posix().lower()
        if "property_balanced" in text:
            return "property_balanced_il"
        if "family" in text:
            return "ion_family"
        if "il_level_random" in text or "random_il" in text:
            return "random_il"
        if "random_point" in text or "row_level" in text:
            return "random_point"
        return "unknown"

    def _find_checkpoints(self) -> tuple[Path, ...]:
        candidates: list[Path] = []
        roots = [
            self.project_root / "trained_weights",
            self.project_root / "il_property_prediction" / "outputs",
        ]
        for root in roots:
            if not root.is_dir():
                continue
            for pattern in ("*.pt", "*.pth", "*.ckpt"):
                candidates.extend(path.resolve() for path in root.rglob(pattern) if path.is_file())
        return tuple(sorted(set(candidates), key=lambda item: item.as_posix().lower()))

    def _select_checkpoint(
        self,
        candidates: tuple[Path, ...],
    ) -> tuple[Path, str, list[str]]:
        requested = str(self.config["model"].get("primary_checkpoint_type", "random_il"))
        explicit = self.config["model"].get("checkpoint", "auto")
        warnings: list[str] = []
        if explicit != "auto":
            selected = resolve_path(explicit, self.project_root)
            if not selected.is_file():
                raise FileNotFoundError(f"Configured checkpoint not found: {selected}")
            return selected, self._checkpoint_type(selected), warnings
        if not candidates:
            raise FileNotFoundError(
                "No checkpoint was found below trained_weights or il_property_prediction/outputs"
            )
        typed = [(path, self._checkpoint_type(path)) for path in candidates]
        matches = [path for path, checkpoint_type in typed if checkpoint_type == requested]
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime), requested, warnings
        priority = ["random_il", "property_balanced_il", "ion_family", "random_point", "unknown"]
        selected = next(
            path
            for candidate_type in priority
            for path, checkpoint_type in typed
            if checkpoint_type == candidate_type
        )
        selected_type = self._checkpoint_type(selected)
        warnings.append(
            f"Requested checkpoint type {requested!r} was unavailable; selected the only "
            f"compatible fallback {selected_type!r}: {selected}"
        )
        return selected, selected_type, warnings

    def inspect(self) -> InspectionReport:
        package = self.project_root / "il_property_prediction"
        model_file = package / "src" / "models" / "mipgraph.py"
        model_config = package / "configs" / "physics_moe_fg_transformer.yaml"
        required = [model_file, model_config]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Required MIPGraph source assets are missing: {missing}")

        data_dir = package / "data" / "processed"
        data_paths = {
            "clean_csv": data_dir / "il_multiprop_clean.csv",
            "arrays": data_dir / "il_multiprop_arrays.npz",
            "graph_cache": data_dir / "graph_cache_fg.pt",
            "unimol2_feature_cache": data_dir / "unimol2_ion_features.pt",
            "unique_ils": data_dir / "unique_ils.csv",
        }
        absent_data = [path for path in data_paths.values() if not path.is_file()]
        if absent_data:
            raise FileNotFoundError(f"Required processed data assets are missing: {absent_data}")

        split_dir = data_dir / "splits"
        split_paths = {
            "row_level": split_dir / "row_level_seed42.json",
            "random_il": split_dir / "il_level_seed42.json",
            "property_balanced_il": split_dir / "il_level_property_balanced_seed42.json",
            "ion_family": split_dir / "il_level_family_pair_seed42.json",
        }
        absent_splits = [path for path in split_paths.values() if not path.is_file()]
        if absent_splits:
            raise FileNotFoundError(f"Required split assets are missing: {absent_splits}")

        checkpoints = self._find_checkpoints()
        selected, selected_type, warnings = self._select_checkpoint(checkpoints)
        selected_split = (
            split_paths["row_level"]
            if selected_type == "random_point"
            else split_paths.get(selected_type, split_paths["random_il"])
        )
        screening_root = (
            self.project_root
            / "experiments"
            / "computational_application_case"
            / "chapter_results"
            / "csv"
        )
        screening_paths = {
            "candidate_trajectory_608": screening_root / "candidate_screening_trajectory_608.csv",
            "hard_and_pareto": screening_root / "pareto_rank1_all_candidates.csv",
            "top8": screening_root / "final_prioritized_candidates.csv",
            "cross_protocol": screening_root / "cross_protocol_decision_matrix.csv",
        }
        missing_screening = [key for key, path in screening_paths.items() if not path.is_file()]
        if missing_screening:
            warnings.append(f"Screening assets unavailable: {missing_screening}")

        return InspectionReport(
            project_root=self.project_root,
            module_root=self.module_root,
            output_root=self.output_root,
            model_class="src.models.mipgraph.MIPGraph",
            model_config=model_config,
            selected_checkpoint=selected,
            selected_checkpoint_type=selected_type,
            requested_checkpoint_type=str(
                self.config["model"].get("primary_checkpoint_type", "random_il")
            ),
            checkpoint_candidates=checkpoints,
            data_paths=data_paths,
            split_paths=split_paths,
            selected_split=selected_split,
            property_names=list(PROPERTY_NAMES),
            descriptor_dimensions={"global": 56, "functional_group": 80},
            descriptor_names_complete=True,
            cross_ion_attention_access="forward_auxiliary",
            router_access="forward_auxiliary",
            expert_output_access="forward_auxiliary",
            screening_paths=screening_paths,
            environment=software_versions(),
            warnings=tuple(warnings),
        )

    @staticmethod
    def load_report(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
