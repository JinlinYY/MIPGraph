"""Adapter around the current ``il_property_prediction`` MIPGraph runtime.

The adapter intentionally contains no model architecture.  It loads the
checkpoint-embedded configuration, calls the current model factory, calls the
current graph builder for unseen recombinations, and uses the checkpoint's
stored condition and target scalers.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .paths import resolve_project_path
from .schema import PROPERTY_NAMES, PROPERTY_UNITS


@dataclass
class InferenceResult:
    """Physical-unit predictions, reusable features, and explicit failures."""

    predictions: pd.DataFrame
    predictions_wide: pd.DataFrame
    features: pd.DataFrame
    failures: pd.DataFrame
    metadata: dict[str, Any]


class InferenceDataset(torch.utils.data.Dataset):
    """Minimal inference-only dataset preserving current graph field shapes."""

    def __init__(self, samples: Sequence[Data]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Data:
        return self.samples[index]


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"MIPGraph configuration is not a mapping: {path}")
    return payload


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class MIPGraphModelAdapter:
    """Load, validate, and run the audited six-property MIPGraph checkpoint."""

    def __init__(self, case_config: dict[str, Any], checkpoint_path: str | Path | None = None) -> None:
        self.case_config = copy.deepcopy(case_config)
        self.root = Path(case_config["_project_root"])
        self.package_root = self.root / "il_property_prediction"
        package_path = str(self.package_root.resolve())
        if package_path not in sys.path:
            sys.path.insert(0, package_path)
        # Checkpoint scaler objects were serialized under the current top-level
        # package name used by the training scripts.
        importlib.import_module("src.data.scaler")
        configured_checkpoint = checkpoint_path or case_config["model"].get("checkpoint_path")
        if configured_checkpoint is None:
            raise ValueError("An explicit six-property checkpoint is required")
        self.checkpoint_path = resolve_project_path(self.root, configured_checkpoint)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        checkpoint = _torch_load(self.checkpoint_path)
        required_keys = {
            "model_state_dict",
            "config",
            "condition_scaler",
            "target_scaler",
            "property_names",
        }
        missing = sorted(required_keys - set(checkpoint))
        if missing:
            raise ValueError(f"Checkpoint lacks required keys: {missing}")
        self.checkpoint = checkpoint
        self.property_names = list(checkpoint["property_names"])
        if self.property_names != list(PROPERTY_NAMES):
            raise ValueError(
                f"Unexpected property order: {self.property_names}; expected {list(PROPERTY_NAMES)}"
            )
        external_config = _load_yaml(
            resolve_project_path(self.root, case_config["model"]["config_path"])
        )
        self.model_config = copy.deepcopy(checkpoint["config"])
        if not isinstance(self.model_config, dict):
            raise ValueError("Checkpoint config must be a mapping")
        self.model_config["_base_dir"] = str(self.package_root.resolve())
        self.model_config["_config_path"] = str(
            resolve_project_path(self.root, case_config["model"]["config_path"])
        )
        self.model_config.setdefault("chem", copy.deepcopy(external_config.get("chem", {})))
        self.model_config["model"]["unimol2_feature_cache_path"] = str(
            resolve_project_path(
                self.root, case_config["model"]["unimol2_feature_cache_path"]
            )
        )
        configured_weight_dir = self.model_config["model"].get(
            "unimol2_weight_dir", external_config.get("model", {}).get("unimol2_weight_dir")
        )
        if configured_weight_dir is None:
            raise ValueError("Uni-Mol2 weight directory is absent from checkpoint config")
        weight_dir = Path(configured_weight_dir)
        if not weight_dir.is_absolute():
            weight_dir = self.package_root / weight_dir
        if not weight_dir.exists():
            fallback = self.package_root / "data" / "pretrained" / "unimol2"
            if not fallback.exists():
                raise FileNotFoundError(f"Uni-Mol2 weight directory not found: {weight_dir}")
            weight_dir = fallback
        self.model_config["model"]["unimol2_weight_dir"] = str(weight_dir.resolve())
        build_model = importlib.import_module("src.models.factory").build_model
        self.model = build_model(self.model_config)
        strict = bool(case_config["model"].get("strict_checkpoint_loading", True))
        incompatible = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        legacy_transformer_surplus_prefixes = ("interaction.", "interaction_fusion.")
        self.ignored_legacy_checkpoint_keys: list[str] = []
        if (
            str(self.model_config["model"].get("fusion_mode", "transformer")).lower()
            == "transformer"
            or bool(self.model_config["model"].get("use_transformer_fusion", False))
        ):
            self.ignored_legacy_checkpoint_keys = [
                key
                for key in unexpected_keys
                if key.startswith(legacy_transformer_surplus_prefixes)
            ]
        disallowed_unexpected = sorted(
            set(unexpected_keys) - set(self.ignored_legacy_checkpoint_keys)
        )
        if strict and (missing_keys or disallowed_unexpected):
            raise RuntimeError(
                "Checkpoint is incompatible with the current MIPGraph model: "
                f"missing={missing_keys}, unexpected={disallowed_unexpected}"
            )
        requested_device = str(case_config["model"].get("device", "auto"))
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {requested_device}")
        self.device = torch.device(requested_device)
        self.model.to(self.device)
        self.model.eval()
        self.condition_scaler = checkpoint["condition_scaler"]
        self.target_scaler = checkpoint["target_scaler"]
        self._feature_key_by_canonical = self._canonical_feature_key_map()
        self._build_graph = importlib.import_module(
            "src.chem.graph_featurizer"
        ).build_ion_pair_graph
        graph_identity = {
            "chem": self.model_config.get("chem", {}),
            "use_cross_ion_edges": self.model_config["model"].get(
                "use_cross_ion_edges", True
            ),
        }
        self.graph_config_fingerprint = hashlib.sha256(
            json.dumps(graph_identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        runtime_model_keys = {"unimol2_feature_cache_path", "unimol2_weight_dir"}
        model_structure_identity = {
            key: value
            for key, value in self.model_config["model"].items()
            if key not in runtime_model_keys
        }
        self.model_structure_fingerprint = hashlib.sha256(
            json.dumps(
                model_structure_identity,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]

    def _canonical_feature_key_map(self) -> dict[str, str]:
        canonicalize = importlib.import_module("src.chem.smiles_utils").canonicalize_smiles
        mapping: dict[str, str] = {}
        ambiguous: set[str] = set()
        feature_cache = self.model.ion_encoder.feature_cache
        for key in feature_cache:
            canonical, error = canonicalize(str(key))
            if canonical is None or error is not None:
                continue
            if canonical in mapping and mapping[canonical] != str(key):
                ambiguous.add(canonical)
            else:
                mapping[canonical] = str(key)
        for canonical in ambiguous:
            mapping.pop(canonical, None)
        return mapping

    def _ion_cache_key(self, smiles: str) -> str:
        feature_cache = self.model.ion_encoder.feature_cache
        if smiles in feature_cache:
            return smiles
        canonicalize = importlib.import_module("src.chem.smiles_utils").canonicalize_smiles
        canonical, error = canonicalize(smiles)
        if canonical is None:
            raise KeyError(f"Could not canonicalize ion for Uni-Mol2 cache: {error}")
        if canonical not in self._feature_key_by_canonical:
            raise KeyError(f"Uni-Mol2 feature cache is missing ion: {smiles}")
        return self._feature_key_by_canonical[canonical]

    def _graph_for_candidate(self, row: pd.Series) -> Data:
        cation = self._ion_cache_key(str(row["cation_smiles"]))
        anion = self._ion_cache_key(str(row["anion_smiles"]))
        model_smiles = f"{cation}.{anion}"
        chem = self.model_config["chem"]
        result = self._build_graph(
            model_smiles,
            use_3d=bool(chem.get("use_3d", True)),
            cutoff=float(chem.get("cross_ion_cutoff", 5.0)),
            seed=int(chem.get("seed", 42)),
            max_attempts=int(chem.get("max_conformer_attempts", 20)),
            optimize_method=str(chem.get("optimize_method", "UFF")),
            use_cross_edges=bool(self.model_config["model"].get("use_cross_ion_edges", True)),
            cross_ion_mode=str(chem.get("cross_ion_mode", "deterministic_2d")),
        )
        if result.data is None:
            raise RuntimeError(f"Current graph builder failed: {result.error}")
        result.data.model_il_smiles = model_smiles
        return result.data

    def _base_sample(
        self,
        graph: Data,
        temperature: float,
        pressure: float,
        sample_id: int,
        candidate_id: str,
    ) -> Data:
        sample = graph.clone()
        condition = self.condition_scaler.transform(
            np.asarray([temperature], dtype=np.float32),
            np.asarray([pressure], dtype=np.float32),
        )[0]
        sample.condition = torch.tensor(condition, dtype=torch.float32).view(1, 2)
        sample.raw_condition = torch.tensor(
            [temperature, pressure], dtype=torch.float32
        ).view(1, 2)
        sample.y = torch.zeros((1, len(self.property_names)), dtype=torch.float32)
        sample.y_raw = torch.zeros_like(sample.y)
        sample.mask = torch.zeros_like(sample.y)
        sample.eval_mask = torch.zeros_like(sample.y)
        sample.y_error = torch.zeros_like(sample.y)
        sample.error_mask = torch.zeros_like(sample.y)
        sample.error_weight = torch.ones_like(sample.y)
        sample.sample_id = torch.tensor([sample_id], dtype=torch.long)
        sample.is_augmented = torch.tensor([0], dtype=torch.long)
        sample.candidate_id = candidate_id
        return sample

    @staticmethod
    def _failure(
        row: pd.Series,
        stage: str,
        exception: Exception,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        summary = "".join(
            traceback.format_exception_only(type(exception), exception)
        ).strip()
        return {
            "candidate_id": row.get("candidate_id"),
            "cation_smiles": row.get("cation_smiles"),
            "anion_smiles": row.get("anion_smiles"),
            "temperature_K": temperature,
            "failed_stage": stage,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback_summary": summary,
            "excluded_from_analysis": True,
        }

    def predict(
        self,
        candidates: pd.DataFrame,
        temperatures: Sequence[float],
        pressure_kpa: float,
        application_cache_path: str | Path,
        training_temperature_range: tuple[float, float],
        force: bool = False,
    ) -> InferenceResult:
        """Predict six physical properties on the complete condition grid."""

        required = {
            "candidate_id",
            "candidate_type",
            "cation_smiles",
            "anion_smiles",
            "il_smiles",
            "canonical_il_key",
        }
        missing = sorted(required - set(candidates.columns))
        if missing:
            raise ValueError(f"Candidate table lacks required columns: {missing}")
        cache_path = Path(application_cache_path)
        graph_cache: dict[str, Data] = {}
        if cache_path.exists() and not force:
            payload = _torch_load(cache_path)
            if isinstance(payload, dict):
                graph_cache = payload
        failures: list[dict[str, Any]] = []
        graph_by_id: dict[str, Data] = {}
        for _, row in candidates.iterrows():
            candidate_id = str(row["candidate_id"])
            graph_cache_key = (
                f"{row['canonical_il_key']}::{self.graph_config_fingerprint}"
            )
            try:
                graph = graph_cache.get(graph_cache_key)
                if graph is None:
                    graph = self._graph_for_candidate(row)
                    graph_cache[graph_cache_key] = graph.cpu()
                graph_by_id[candidate_id] = graph
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                failures.append(self._failure(row, "graph_or_unimol_cache", exc))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph_cache, cache_path)
        samples: list[Data] = []
        sample_metadata: dict[int, dict[str, Any]] = {}
        for _, row in candidates.iterrows():
            candidate_id = str(row["candidate_id"])
            if candidate_id not in graph_by_id:
                continue
            for temperature in temperatures:
                sample_id = len(samples)
                sample = self._base_sample(
                    graph_by_id[candidate_id],
                    float(temperature),
                    float(pressure_kpa),
                    sample_id,
                    candidate_id,
                )
                samples.append(sample)
                sample_metadata[sample_id] = {
                    **row.to_dict(),
                    "model_il_smiles": str(sample.model_il_smiles),
                    "temperature_K": float(temperature),
                    "pressure_kPa": float(pressure_kpa),
                }
        if not samples:
            raise RuntimeError("No candidate-condition samples reached model inference")
        loader = DataLoader(
            InferenceDataset(samples),
            batch_size=int(self.case_config["model"]["batch_size"]),
            shuffle=False,
            num_workers=int(self.case_config["model"].get("num_workers", 0)),
        )
        prediction_rows: list[dict[str, Any]] = []
        embeddings: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for batch in loader:
                sample_ids = batch.sample_id.detach().cpu().numpy().reshape(-1).astype(int)
                try:
                    device_batch = batch.to(self.device)
                    scaled, auxiliary = self.model(device_batch)
                    physical = self.target_scaler.inverse_transform(
                        scaled.detach().cpu().numpy()
                    )
                    structure = auxiliary.get("h_structure")
                    structure_array = (
                        structure.detach().cpu().numpy() if structure is not None else None
                    )
                    for local_index, sample_id in enumerate(sample_ids):
                        metadata = sample_metadata[int(sample_id)]
                        values = physical[local_index]
                        row = {
                            **metadata,
                            "checkpoint_name": self.checkpoint_path.name,
                            "inference_status": "success",
                            "temperature_domain_flag": (
                                "in_domain"
                                if training_temperature_range[0]
                                <= float(metadata["temperature_K"])
                                <= training_temperature_range[1]
                                else "extrapolation"
                            ),
                        }
                        row.update(
                            {
                                name: float(values[index])
                                for index, name in enumerate(self.property_names)
                            }
                        )
                        prediction_rows.append(row)
                        candidate_id = str(metadata["candidate_id"])
                        if structure_array is not None and candidate_id not in embeddings:
                            embeddings[candidate_id] = structure_array[local_index]
                except (KeyError, RuntimeError, TypeError, ValueError) as batch_exc:
                    # Isolate a failing batch without manufacturing predictions for
                    # unaffected samples.
                    for sample_id in sample_ids:
                        metadata = sample_metadata[int(sample_id)]
                        source_row = pd.Series(metadata)
                        try:
                            single = DataLoader(
                                InferenceDataset([samples[int(sample_id)]]),
                                batch_size=1,
                                shuffle=False,
                            )
                            single_batch = next(iter(single)).to(self.device)
                            scaled, auxiliary = self.model(single_batch)
                            physical = self.target_scaler.inverse_transform(
                                scaled.detach().cpu().numpy()
                            )[0]
                            result_row = {
                                **metadata,
                                "checkpoint_name": self.checkpoint_path.name,
                                "inference_status": "success",
                                "temperature_domain_flag": (
                                    "in_domain"
                                    if training_temperature_range[0]
                                    <= float(metadata["temperature_K"])
                                    <= training_temperature_range[1]
                                    else "extrapolation"
                                ),
                            }
                            result_row.update(
                                {
                                    name: float(physical[index])
                                    for index, name in enumerate(self.property_names)
                                }
                            )
                            prediction_rows.append(result_row)
                            structure = auxiliary.get("h_structure")
                            candidate_id = str(metadata["candidate_id"])
                            if structure is not None and candidate_id not in embeddings:
                                embeddings[candidate_id] = structure.detach().cpu().numpy()[0]
                        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                            failures.append(
                                self._failure(
                                    source_row,
                                    "forward_inference",
                                    exc,
                                    float(metadata["temperature_K"]),
                                )
                            )
                    del batch_exc
        predictions = pd.DataFrame(prediction_rows)
        if predictions.empty:
            raise RuntimeError("MIPGraph did not produce any physical-unit predictions")
        predictions = predictions.sort_values(
            ["candidate_id", "temperature_K"]
        ).reset_index(drop=True)
        wide = predictions.pivot(
            index=[
                "candidate_id",
                "candidate_type",
                "cation_smiles",
                "anion_smiles",
                "il_smiles",
            ],
            columns="temperature_K",
            values=self.property_names,
        )
        wide.columns = [f"{name}_{temperature:g}K" for name, temperature in wide.columns]
        wide = wide.reset_index()
        feature_rows: list[dict[str, Any]] = []
        for _, candidate in candidates.iterrows():
            candidate_id = str(candidate["candidate_id"])
            if candidate_id not in graph_by_id:
                continue
            graph = graph_by_id[candidate_id]
            row: dict[str, Any] = {"candidate_id": candidate_id}
            global_values = graph.global_desc.detach().cpu().numpy().reshape(-1)
            fg_values = graph.functional_group_desc.detach().cpu().numpy().reshape(-1)
            row.update({f"global_desc_{index}": float(value) for index, value in enumerate(global_values)})
            row.update({f"fg_desc_{index}": float(value) for index, value in enumerate(fg_values)})
            if candidate_id in embeddings:
                row.update(
                    {
                        f"embedding_{index}": float(value)
                        for index, value in enumerate(embeddings[candidate_id])
                    }
                )
            feature_rows.append(row)
        metadata = {
            "model_class": f"{type(self.model).__module__}.{type(self.model).__name__}",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_epoch": int(self.checkpoint.get("epoch", -1)),
            "checkpoint_state_dict_key": "model_state_dict",
            "strict_checkpoint_loading": bool(
                self.case_config["model"].get("strict_checkpoint_loading", True)
            ),
            "ignored_legacy_checkpoint_keys": self.ignored_legacy_checkpoint_keys,
            "legacy_key_compatibility_reason": (
                "The selected historical transformer-fusion checkpoint contains dormant "
                "bilinear interaction parameters. Every current transformer-model key remains required."
                if self.ignored_legacy_checkpoint_keys
                else "not_applicable"
            ),
            "property_order": self.property_names,
            "property_units": dict(PROPERTY_UNITS),
            "target_inverse_transform": "exp(y_scaled * target_std + target_mean) - 1e-8",
            "target_epsilon": float(self.target_scaler.eps),
            "condition_scaler_class": (
                f"{type(self.condition_scaler).__module__}."
                f"{type(self.condition_scaler).__name__}"
            ),
            "target_scaler_class": (
                f"{type(self.target_scaler).__module__}."
                f"{type(self.target_scaler).__name__}"
            ),
            "condition_scaler": dict(vars(self.condition_scaler)),
            "target_means": self.target_scaler.means.tolist(),
            "target_stds": self.target_scaler.stds.tolist(),
            "graph_feature_dimension": int(next(iter(graph_by_id.values())).x.shape[1]),
            "global_descriptor_dimension": int(next(iter(graph_by_id.values())).global_desc.shape[-1]),
            "functional_group_dimension": int(next(iter(graph_by_id.values())).functional_group_desc.shape[-1]),
            "device": str(self.device),
            "unimol2_feature_cache": self.model.ion_encoder.feature_cache_path,
            "application_graph_cache": str(cache_path),
            "graph_config_fingerprint": self.graph_config_fingerprint,
            "model_structure_fingerprint": self.model_structure_fingerprint,
            "embedding_available": bool(embeddings),
            "deprecated_archive_used": False,
        }
        return InferenceResult(
            predictions=predictions,
            predictions_wide=wide,
            features=pd.DataFrame(feature_rows),
            failures=pd.DataFrame(
                failures,
                columns=[
                    "candidate_id",
                    "cation_smiles",
                    "anion_smiles",
                    "temperature_K",
                    "failed_stage",
                    "exception_type",
                    "exception_message",
                    "traceback_summary",
                    "excluded_from_analysis",
                ],
            ),
            metadata=metadata,
        )
