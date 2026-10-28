"""Descriptor naming, aligned output assembly, and compact cache persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from il_property_prediction.src.chem.functional_groups import (
    ION_FUNCTIONAL_GROUP_NAMES,
    PAIR_FUNCTIONAL_GROUP_NAMES,
)

from .data_adapter import AnalysisData
from .model_adapter import ModelOutputs
from .utils import MODULE_ROOT, ensure_within, read_table, write_json, write_table


ION_DESCRIPTOR_NAMES = [
    "molecular_weight_scaled",
    "exact_molecular_weight_scaled",
    "heavy_atom_count_scaled",
    "tpsa_scaled",
    "hbond_donor_count_scaled",
    "hbond_acceptor_count_scaled",
    "rotatable_bond_count_scaled",
    "ring_count_scaled",
    "aromatic_ring_count_scaled",
    "aromatic_atom_fraction",
    "formal_charge_scaled",
    "logp_scaled",
    "radius_of_gyration_scaled",
    "principal_moment_0_scaled",
    "principal_moment_1_scaled",
    "principal_moment_2_scaled",
    "asphericity",
    "carbon_fraction",
    "nitrogen_fraction",
    "oxygen_fraction",
    "fluorine_fraction",
    "phosphorus_sulfur_fraction",
    "halogen_fraction",
    "charged_atom_fraction",
]
PAIR_DESCRIPTOR_NAMES = [
    "total_heavy_atom_count_scaled",
    "cation_to_anion_heavy_atom_ratio",
    "anion_to_cation_heavy_atom_ratio",
    "total_molecular_weight_scaled",
    "total_tpsa_scaled",
    "net_formal_charge",
    "charge_product_scaled",
    "radius_of_gyration_difference",
]
DESCRIPTOR_NAMES = (
    [f"cation_{name}" for name in ION_DESCRIPTOR_NAMES]
    + [f"anion_{name}" for name in ION_DESCRIPTOR_NAMES]
    + [f"pair_{name}" for name in PAIR_DESCRIPTOR_NAMES]
)
_RAW_FUNCTIONAL_GROUP_NAMES = (
    [f"cation_{name}" for name in ION_FUNCTIONAL_GROUP_NAMES]
    + [f"anion_{name}" for name in ION_FUNCTIONAL_GROUP_NAMES]
    + [f"pair_{name}" for name in PAIR_FUNCTIONAL_GROUP_NAMES]
)
FUNCTIONAL_GROUP_NAMES = [
    (
        f"{name}_functional_group"
        if name in set(DESCRIPTOR_NAMES)
        else name
    )
    for name in _RAW_FUNCTIONAL_GROUP_NAMES
]


@dataclass
class FeatureBundle:
    records: pd.DataFrame
    descriptors: pd.DataFrame
    latent_arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    auxiliary_tables: dict[str, pd.DataFrame] = field(default_factory=dict)


class FeatureExtractor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def from_model_outputs(
        self,
        data: AnalysisData,
        outputs: ModelOutputs,
    ) -> FeatureBundle:
        if outputs.predictions.shape != (len(data.indices), len(outputs.property_names)):
            raise ValueError("Prediction dimensions do not match the selected data rows")
        records = data.frame.copy()
        for index, property_name in enumerate(outputs.property_names):
            records[f"prediction_{property_name}"] = outputs.predictions[:, index]
            actual = records[f"{property_name}_ActualValue"]
            records[f"residual_{property_name}"] = actual - outputs.predictions[:, index]
        global_values = outputs.latent_arrays["global_descriptors"]
        fg_values = outputs.latent_arrays["functional_group_descriptors"]
        if global_values.shape[1] != len(DESCRIPTOR_NAMES):
            raise ValueError(
                f"Global descriptor dimension {global_values.shape[1]} does not match "
                f"the source-derived name count {len(DESCRIPTOR_NAMES)}"
            )
        if fg_values.shape[1] != len(FUNCTIONAL_GROUP_NAMES):
            raise ValueError(
                f"Functional-group dimension {fg_values.shape[1]} does not match "
                f"the source-derived name count {len(FUNCTIONAL_GROUP_NAMES)}"
            )
        descriptors = pd.DataFrame(
            np.concatenate([global_values, fg_values], axis=1),
            columns=DESCRIPTOR_NAMES + FUNCTIONAL_GROUP_NAMES,
        )
        descriptors.insert(0, "sample_id", records["sample_id"].to_numpy())
        if not np.array_equal(
            descriptors["sample_id"].to_numpy(),
            records["sample_id"].to_numpy(),
        ):
            raise RuntimeError("Descriptor and observation identities are misaligned")
        metadata = dict(outputs.metadata)
        metadata.update(
            {
                "global_descriptor_names": DESCRIPTOR_NAMES,
                "functional_group_names": FUNCTIONAL_GROUP_NAMES,
                "descriptor_name_source": (
                    "Exact index order audited from global_descriptors.py and functional_groups.py; "
                    "four cross-source duplicate names carry a _functional_group suffix"
                ),
            }
        )
        return FeatureBundle(
            records=records,
            descriptors=descriptors,
            latent_arrays=outputs.latent_arrays,
            metadata=metadata,
            auxiliary_tables={"cross_ion_attention": outputs.attention_summary},
        )

    @staticmethod
    def descriptors_from_graph_cache(data: AnalysisData) -> pd.DataFrame:
        """Extract source-ordered descriptors without running the neural network."""

        try:
            graph_cache = torch.load(
                data.graph_cache_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            graph_cache = torch.load(data.graph_cache_path, map_location="cpu")
        rows: list[np.ndarray] = []
        for smiles in data.frame["IL_SMILES"].astype(str):
            graph = graph_cache.get(smiles)
            if graph is None:
                raise KeyError(f"Graph cache lacks descriptor source for {smiles}")
            global_values = graph.global_desc.detach().cpu().numpy().reshape(-1)
            functional_values = (
                graph.functional_group_desc.detach().cpu().numpy().reshape(-1)
            )
            rows.append(np.concatenate([global_values, functional_values]))
        descriptors = pd.DataFrame(
            np.asarray(rows, dtype=np.float32),
            columns=DESCRIPTOR_NAMES + FUNCTIONAL_GROUP_NAMES,
        )
        descriptors.insert(0, "sample_id", data.frame["sample_id"].to_numpy())
        return descriptors

    @staticmethod
    def save_bundle(bundle: FeatureBundle, base_path: str | Path) -> dict[str, Path]:
        base = ensure_within(base_path, MODULE_ROOT)
        base.parent.mkdir(parents=True, exist_ok=True)
        records_path = write_table(
            bundle.records,
            base.with_name(f"{base.name}_records.parquet"),
        )
        descriptors_path = write_table(
            bundle.descriptors,
            base.with_name(f"{base.name}_descriptors.parquet"),
        )
        arrays_path = base.with_name(f"{base.name}_latents.npz")
        np.savez_compressed(arrays_path, **bundle.latent_arrays)
        auxiliary_paths = {
            name: write_table(
                frame,
                base.with_name(f"{base.name}_{name}.parquet"),
            )
            for name, frame in bundle.auxiliary_tables.items()
            if not frame.empty
        }
        metadata = dict(bundle.metadata)
        metadata["cache_files"] = {
            "records": str(records_path),
            "descriptors": str(descriptors_path),
            "latent_arrays": str(arrays_path),
            "auxiliary": {key: str(path) for key, path in auxiliary_paths.items()},
        }
        metadata_path = write_json(
            base.with_name(f"{base.name}_metadata.json"),
            metadata,
        )
        return {
            "records": records_path,
            "descriptors": descriptors_path,
            "latent_arrays": arrays_path,
            "metadata": metadata_path,
            **{f"auxiliary_{key}": path for key, path in auxiliary_paths.items()},
        }

    @staticmethod
    def load_bundle(base_path: str | Path) -> FeatureBundle:
        base = Path(base_path).resolve()
        metadata_path = base.with_name(f"{base.name}_metadata.json")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        files = metadata["cache_files"]
        with np.load(files["latent_arrays"], allow_pickle=False) as arrays:
            latent_arrays = {key: arrays[key] for key in arrays.files}
        auxiliary = {
            key: read_table(path)
            for key, path in files.get("auxiliary", {}).items()
        }
        return FeatureBundle(
            records=read_table(files["records"]),
            descriptors=read_table(files["descriptors"]),
            latent_arrays=latent_arrays,
            metadata=metadata,
            auxiliary_tables=auxiliary,
        )
