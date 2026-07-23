"""Non-invasive loading and inference for the existing trained MIPGraph."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem

from .data_adapter import AnalysisData
from .project_adapter import InspectionReport
from .utils import file_sha256, stable_hash


MECHANISM_NAMES = ["packing", "cohesion", "transport", "thermal"]


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dataclass
class ModelOutputs:
    predictions: np.ndarray
    property_names: list[str]
    latent_arrays: dict[str, np.ndarray]
    attention_summary: pd.DataFrame
    metadata: dict[str, Any]


class ModelAdapter:
    """Load a checkpoint and expose verified auxiliary tensors without source edits."""

    def __init__(self, config: dict[str, Any], inspection: InspectionReport) -> None:
        self.config = config
        self.inspection = inspection
        self._bind_original_package_alias()
        checkpoint = _torch_load(inspection.selected_checkpoint)
        required = {
            "model_state_dict",
            "config",
            "condition_scaler",
            "target_scaler",
            "property_names",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError(f"Checkpoint is missing required fields: {missing}")
        self.checkpoint = checkpoint
        self.property_names = list(checkpoint["property_names"])
        if self.property_names != inspection.property_names:
            raise ValueError(
                f"Checkpoint property order {self.property_names} does not match "
                f"the audited order {inspection.property_names}"
            )
        self.condition_scaler = checkpoint["condition_scaler"]
        self.target_scaler = checkpoint["target_scaler"]
        self.model_config = copy.deepcopy(checkpoint["config"])
        self.model_config["_base_dir"] = str(
            (inspection.project_root / "il_property_prediction").resolve()
        )
        self.model_config["model"]["unimol2_feature_cache_path"] = str(
            inspection.data_paths["unimol2_feature_cache"].resolve()
        )
        weight_dir = Path(self.model_config["model"].get("unimol2_weight_dir", "data/pretrained/unimol2"))
        if not weight_dir.is_absolute():
            weight_dir = Path(self.model_config["_base_dir"]) / weight_dir
        if not weight_dir.is_dir():
            raise FileNotFoundError(f"Uni-Mol2 weight directory not found: {weight_dir}")
        self.model_config["model"]["unimol2_weight_dir"] = str(weight_dir.resolve())
        factory = importlib.import_module("il_property_prediction.src.models.factory")
        self.model = factory.build_model(self.model_config)
        incompatible = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not key.startswith(("interaction.", "interaction_fusion."))
        ]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                "Checkpoint/model mismatch: "
                f"missing={list(incompatible.missing_keys)}, unexpected={unexpected}"
            )
        requested = str(config["model"].get("device", "auto"))
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
        self.device = torch.device(requested)
        self.model.to(self.device)
        self.model.eval()
        self._atom_category_cache: dict[
            tuple[str, int],
            dict[str, torch.Tensor],
        ] = {}

    @staticmethod
    def _bind_original_package_alias() -> None:
        """Bind the historical ``src`` pickle name to the original project only."""

        original_src = importlib.import_module("il_property_prediction.src")
        sys.modules["src"] = original_src
        for suffix in ("data", "data.scaler"):
            module = importlib.import_module(f"il_property_prediction.src.{suffix}")
            sys.modules[f"src.{suffix}"] = module

    def make_loader(self, data: AnalysisData) -> DataLoader:
        """Build the original aligned evaluation loader for an audited split."""

        dataset_module = importlib.import_module("il_property_prediction.src.data.dataset")
        condition = self.condition_scaler.transform(
            data.arrays["temperature"],
            data.arrays["pressure"],
        )
        y_scaled = self.target_scaler.transform(data.arrays["y"], data.arrays["mask"])
        weights = np.ones_like(data.arrays["y"], dtype=np.float32)
        dataset = dataset_module.ILPropertyDataset(
            self.inspection.data_paths["clean_csv"],
            self.inspection.data_paths["arrays"],
            data.graph_cache_path,
            data.indices,
            condition,
            y_scaled,
            weights,
        )
        return DataLoader(
            dataset,
            batch_size=int(self.config["model"].get("batch_size", 64)),
            shuffle=False,
            num_workers=int(self.config["model"].get("num_workers", 0)),
        )

    def _rdkit_atom_categories(
        self,
        smiles: str,
        padded_atom_count: int,
    ) -> dict[str, torch.Tensor]:
        cache_key = (smiles, padded_atom_count)
        if cache_key in self._atom_category_cache:
            return self._atom_category_cache[cache_key]
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not parse attention ion: {smiles}")
        molecule = Chem.RemoveHs(molecule)
        if molecule.GetNumAtoms() > padded_atom_count:
            raise ValueError(
                f"Attention atom dimension {padded_atom_count} is smaller than "
                f"the RDKit atom count {molecule.GetNumAtoms()} for {smiles}"
            )

        def atom_vector(indices: set[int]) -> torch.Tensor:
            vector = torch.zeros(padded_atom_count, dtype=torch.bool)
            for index in indices:
                vector[index] = True
            return vector

        def matched_atoms(smarts: str) -> set[int]:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                raise ValueError(f"Invalid internal SMARTS pattern: {smarts}")
            return {
                int(index)
                for match in molecule.GetSubstructMatches(pattern)
                for index in match
            }

        alkyl_carbons = {
            atom.GetIdx()
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic()
        }
        categories = {
            "alkyl_carbon": atom_vector(alkyl_carbons),
            "sulfonyl": atom_vector(matched_atoms("[S](=O)(=O)")),
            "carboxylate": atom_vector(matched_atoms("[CX3](=O)[O-]")),
        }
        self._atom_category_cache[cache_key] = categories
        return categories

    def _attention_records(
        self,
        sample_ids: np.ndarray,
        auxiliary: dict[str, Any],
        atom_inputs: tuple[torch.Tensor, ...] | None,
        ion_smiles: list[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        if atom_inputs is None:
            return []
        c_mask = atom_inputs[1].detach()
        c_chem = atom_inputs[2].detach()
        a_mask = atom_inputs[4].detach()
        a_chem = atom_inputs[5].detach()
        attention = 0.5 * (
            auxiliary["cation_to_anion_attention"]
            + auxiliary["anion_to_cation_attention"].transpose(1, 2)
        )
        valid = c_mask.unsqueeze(2) & a_mask.unsqueeze(1)
        c_charge = c_chem[..., 0].unsqueeze(2)
        a_charge = a_chem[..., 0].unsqueeze(1)
        hbond = (
            c_chem[..., 1].unsqueeze(2) * a_chem[..., 2].unsqueeze(1)
            + c_chem[..., 2].unsqueeze(2) * a_chem[..., 1].unsqueeze(1)
        ) > 0
        aromatic = (
            c_chem[..., 3].unsqueeze(2) * a_chem[..., 3].unsqueeze(1)
        ) > 0
        charge_complementary = (c_charge * a_charge) < 0
        high_charge = (c_charge * a_charge).abs() > 0
        rows: list[dict[str, Any]] = []
        for batch_index, (sample_id, smiles_pair) in enumerate(
            zip(sample_ids, ion_smiles)
        ):
            cation_smiles, anion_smiles = smiles_pair
            c_rdkit = self._rdkit_atom_categories(
                cation_smiles,
                c_mask.shape[1],
            )
            a_rdkit = self._rdkit_atom_categories(
                anion_smiles,
                a_mask.shape[1],
            )
            c_atomic_number = c_chem[batch_index, :, 4]
            a_atomic_number = a_chem[batch_index, :, 4]
            c_fluorine = c_atomic_number == 9
            a_fluorine = a_atomic_number == 9
            c_other_halogen = (
                (c_atomic_number == 17)
                | (c_atomic_number == 35)
                | (c_atomic_number == 53)
            )
            a_other_halogen = (
                (a_atomic_number == 17)
                | (a_atomic_number == 35)
                | (a_atomic_number == 53)
            )
            anion_polar = (
                (a_chem[batch_index, :, 2] > 0)
                | (a_chem[batch_index, :, 0].abs() > 0)
            )
            sample_masks = {
                "charge-complementary pairs": charge_complementary[batch_index],
                "high-charge-magnitude pairs": (
                    c_charge[batch_index].abs()
                    + a_charge[batch_index].abs()
                )
                >= 1,
                "H-bond-compatible pairs": hbond[batch_index],
                "aromatic-contact pairs": aromatic[batch_index],
                "cation-aromatic-core--anion-polar-site pairs": (
                    (c_chem[batch_index, :, 3] > 0).unsqueeze(1)
                    & anion_polar.unsqueeze(0)
                ),
                "alkyl-carbon--anion-polar-site pairs": (
                    c_rdkit["alkyl_carbon"].to(c_mask.device).unsqueeze(1)
                    & anion_polar.unsqueeze(0)
                ),
                "fluorine-containing pairs": (
                    c_fluorine.unsqueeze(1) | a_fluorine.unsqueeze(0)
                ),
                "sulfonyl-site-containing pairs": (
                    c_rdkit["sulfonyl"].to(c_mask.device).unsqueeze(1)
                    | a_rdkit["sulfonyl"].to(a_mask.device).unsqueeze(0)
                ),
                "carboxylate-site-containing pairs": (
                    c_rdkit["carboxylate"].to(c_mask.device).unsqueeze(1)
                    | a_rdkit["carboxylate"].to(a_mask.device).unsqueeze(0)
                ),
                "non-fluorine-halogen-containing pairs": (
                    c_other_halogen.unsqueeze(1)
                    | a_other_halogen.unsqueeze(0)
                ),
            }
            explained = torch.zeros_like(valid[batch_index])
            for mask in sample_masks.values():
                explained |= mask
            sample_masks["other atom pairs"] = ~explained
            for category, mask in sample_masks.items():
                pair_mask = valid[batch_index] & mask
                pair_count = int(pair_mask.sum().detach().cpu())
                mass = float((attention[batch_index] * pair_mask).sum().detach().cpu())
                rows.append(
                    {
                        "sample_id": int(sample_id),
                        "interaction_category": category,
                        "pair_count": pair_count,
                        "attention_mass": mass,
                        "attention_per_pair": mass / max(pair_count, 1),
                        "interpretation_scope": "shared model attention; not an interaction energy",
                    }
                )
        return rows

    def predict(self, data: AnalysisData, gradients: bool = False) -> ModelOutputs:
        loader = self.make_loader(data)
        predictions: list[np.ndarray] = []
        collected: dict[str, list[np.ndarray]] = {
            "sample_id": [],
            "cation_embedding": [],
            "anion_embedding": [],
            "ion_pair_embedding": [],
            "condition_modulated_embedding": [],
            "router_weights": [],
            "expert_outputs": [],
            "global_descriptors": [],
            "functional_group_descriptors": [],
        }
        attention_rows: list[dict[str, Any]] = []
        hook_state: dict[str, Any] = {}
        ion_smiles_by_id = {
            int(index): (str(cation), str(anion))
            for index, cation, anion in zip(
                data.indices,
                data.frame["cation_smiles"],
                data.frame["anion_smiles"],
            )
        }

        def condition_hook(_module, _inputs, output):
            hook_state["condition"] = output[0]

        def atom_hook(_module, inputs, _output):
            hook_state["atom_inputs"] = tuple(item.detach() for item in inputs)

        handles = [self.model.condition.register_forward_hook(condition_hook)]
        if hasattr(self.model, "atom_interaction"):
            handles.append(self.model.atom_interaction.register_forward_hook(atom_hook))
        context = torch.enable_grad() if gradients else torch.no_grad()
        try:
            with context:
                for batch in loader:
                    sample_ids = (
                        batch.sample_id.detach().cpu().numpy().reshape(-1).astype(np.int64)
                    )
                    batch = batch.to(self.device)
                    scaled, auxiliary = self.model(batch)
                    physical = self.target_scaler.inverse_transform(
                        scaled.detach().cpu().numpy()
                    )
                    if not np.isfinite(physical).all():
                        raise FloatingPointError("Inverse-transformed predictions contain non-finite values")
                    predictions.append(physical)
                    collected["sample_id"].append(sample_ids)
                    tensor_map = {
                        "cation_embedding": auxiliary["h_cation"],
                        "anion_embedding": auxiliary["h_anion"],
                        "ion_pair_embedding": auxiliary["h_structure"],
                        "condition_modulated_embedding": hook_state["condition"],
                        "router_weights": auxiliary["gates"],
                        "global_descriptors": batch.global_desc.view(len(sample_ids), -1),
                        "functional_group_descriptors": batch.functional_group_desc.view(
                            len(sample_ids), -1
                        ),
                    }
                    for name, tensor in tensor_map.items():
                        collected[name].append(tensor.detach().cpu().numpy())
                    experts = torch.stack(
                        [auxiliary["latents"][name] for name in MECHANISM_NAMES],
                        dim=1,
                    )
                    collected["expert_outputs"].append(experts.detach().cpu().numpy())
                    attention_rows.extend(
                        self._attention_records(
                            sample_ids,
                            auxiliary,
                            hook_state.get("atom_inputs"),
                            [ion_smiles_by_id[int(value)] for value in sample_ids],
                        )
                    )
        finally:
            for handle in handles:
                handle.remove()

        if not predictions:
            raise RuntimeError("No samples reached MIPGraph inference")
        latent_arrays = {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in collected.items()
            if chunks
        }
        expected_ids = np.asarray(data.indices, dtype=np.int64)
        if not np.array_equal(latent_arrays["sample_id"], expected_ids):
            raise RuntimeError("Inference sample order is not aligned with the requested split indices")
        metadata = {
            "model_class": f"{type(self.model).__module__}.{type(self.model).__name__}",
            "checkpoint": str(self.inspection.selected_checkpoint),
            "checkpoint_sha256": file_sha256(self.inspection.selected_checkpoint),
            "checkpoint_epoch": int(self.checkpoint.get("epoch", -1)),
            "checkpoint_type": self.inspection.selected_checkpoint_type,
            "split": data.split_name,
            "split_path": str(data.split_path),
            "sample_count": len(data.indices),
            "property_order": self.property_names,
            "target_transform": "log + property-wise standardization",
            "target_inverse_transform": "exp(y_scaled * std + mean) - 1e-8",
            "device": str(self.device),
            "model_config_hash": stable_hash(self.model_config),
            "attention_scope": "shared cross-ion attention; property conditioning requires gradients",
            "mechanism_scope": (
                "packing/cohesion/transport/thermal are mechanism-associated latent channels, "
                "not independently measured physical quantities"
            ),
        }
        return ModelOutputs(
            predictions=np.concatenate(predictions, axis=0),
            property_names=self.property_names,
            latent_arrays=latent_arrays,
            attention_summary=pd.DataFrame(attention_rows),
            metadata=metadata,
        )

    def _feature_cache_key(self, smiles: str) -> str:
        cache = self.model.ion_encoder.feature_cache
        if smiles in cache:
            return smiles
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not parse ion SMILES: {smiles}")
        canonical = Chem.MolToSmiles(molecule, canonical=True)
        matches = []
        for key in cache:
            candidate = Chem.MolFromSmiles(str(key))
            if candidate is not None and Chem.MolToSmiles(candidate, canonical=True) == canonical:
                matches.append(str(key))
        if len(matches) != 1:
            raise KeyError(
                f"Uni-Mol2 cache has {len(matches)} canonical matches for ion {smiles!r}"
            )
        return matches[0]

    def predict_ion_pairs(
        self,
        candidates: pd.DataFrame,
        temperatures_k: list[float],
        pressure_kpa: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Predict validated ion pairs whose ions exist in the frozen feature cache."""

        required = {"candidate_id", "cation_smiles", "anion_smiles"}
        missing = sorted(required - set(candidates.columns))
        if missing:
            raise ValueError(f"Candidate table lacks required fields: {missing}")
        graph_builder = importlib.import_module(
            "il_property_prediction.src.chem.graph_featurizer"
        ).build_ion_pair_graph
        chem = self.model_config.get("chem", {})
        samples: list[Data] = []
        identities: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for row in candidates.to_dict("records"):
            try:
                cation = self._feature_cache_key(str(row["cation_smiles"]))
                anion = self._feature_cache_key(str(row["anion_smiles"]))
                model_smiles = f"{cation}.{anion}"
                result = graph_builder(
                    model_smiles,
                    use_3d=bool(chem.get("use_3d", True)),
                    cutoff=float(chem.get("cross_ion_cutoff", 5.0)),
                    seed=int(chem.get("seed", 42)),
                    max_attempts=int(chem.get("max_conformer_attempts", 20)),
                    optimize_method=str(chem.get("optimize_method", "UFF")),
                    use_cross_edges=bool(
                        self.model_config["model"].get("use_cross_ion_edges", True)
                    ),
                    cross_ion_mode=str(
                        chem.get("cross_ion_mode", "deterministic_2d")
                    ),
                )
                if result.data is None:
                    raise RuntimeError(str(result.error))
                global_descriptor = (
                    result.data.global_desc.detach().cpu().numpy().reshape(-1)
                )
                functional_descriptor = (
                    result.data.functional_group_desc.detach().cpu().numpy().reshape(-1)
                )
                descriptor_payload = {
                    **{
                        f"global_descriptor_{index}": float(value)
                        for index, value in enumerate(global_descriptor)
                    },
                    **{
                        f"functional_group_descriptor_{index}": float(value)
                        for index, value in enumerate(functional_descriptor)
                    },
                }
                for temperature in temperatures_k:
                    sample = result.data.clone()
                    condition = self.condition_scaler.transform(
                        np.asarray([temperature], dtype=np.float32),
                        np.asarray([pressure_kpa], dtype=np.float32),
                    )[0]
                    sample.condition = torch.tensor(condition, dtype=torch.float32).view(1, 2)
                    sample.raw_condition = torch.tensor(
                        [temperature, pressure_kpa], dtype=torch.float32
                    ).view(1, 2)
                    zeros = torch.zeros((1, len(self.property_names)), dtype=torch.float32)
                    sample.y = zeros.clone()
                    sample.y_raw = zeros.clone()
                    sample.mask = zeros.clone()
                    sample.eval_mask = zeros.clone()
                    sample.y_error = zeros.clone()
                    sample.error_mask = zeros.clone()
                    sample.error_weight = torch.ones_like(zeros)
                    sample.sample_id = torch.tensor([len(samples)], dtype=torch.long)
                    sample.is_augmented = torch.tensor([0], dtype=torch.long)
                    sample.smiles = model_smiles
                    sample.il_name = str(row["candidate_id"])
                    samples.append(sample)
                    identities.append(
                        {
                            **row,
                            "temperature_K": float(temperature),
                            "pressure_kPa": float(pressure_kpa),
                            "model_il_smiles": model_smiles,
                            **descriptor_payload,
                        }
                    )
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                failures.append(
                    {
                        **row,
                        "failed_stage": "graph_or_feature_cache",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )
        if not samples:
            return pd.DataFrame(), pd.DataFrame(failures)

        class _Dataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(samples)

            def __getitem__(self, index: int) -> Data:
                return samples[index]

        loader = DataLoader(
            _Dataset(),
            batch_size=int(self.config["model"].get("batch_size", 64)),
            shuffle=False,
            num_workers=0,
        )
        prediction_rows: list[dict[str, Any]] = []
        offset = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                scaled, _ = self.model(batch)
                physical = self.target_scaler.inverse_transform(
                    scaled.detach().cpu().numpy()
                )
                for local_index, values in enumerate(physical):
                    metadata = identities[offset + local_index]
                    prediction_rows.append(
                        {
                            **metadata,
                            **{
                                property_name: float(values[property_index])
                                for property_index, property_name in enumerate(
                                    self.property_names
                                )
                            },
                        }
                    )
                offset += len(physical)
        return pd.DataFrame(prediction_rows), pd.DataFrame(failures)
