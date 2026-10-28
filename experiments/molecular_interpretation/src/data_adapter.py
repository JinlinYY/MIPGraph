"""Aligned loading of original observations, splits, identities, and graphs."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from .project_adapter import InspectionReport


@dataclass
class AnalysisData:
    frame: pd.DataFrame
    arrays: dict[str, np.ndarray]
    indices: list[int]
    graph_cache_path: Path
    split_path: Path
    split_name: str


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _cation_family(smiles: str) -> str:
    text = smiles.lower()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unparsed"
    patterns = [
        ("imidazolium", "[n+]1cc[n,c]c1"),
        ("pyridinium", "[n+]1ccccc1"),
        ("phosphonium", "[PX4+]"),
        ("quaternary_ammonium", "[NX4+]"),
        ("sulfonium", "[SX3+]"),
    ]
    for name, smarts in patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return name
    if "[nh+]" in text:
        return "protonated_ammonium"
    return "other_cation"


def _anion_family(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unparsed"
    atomic_numbers = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    if mol.GetNumAtoms() == 1 and atomic_numbers & {9, 17, 35, 53}:
        return "halide"
    patterns = [
        ("bis_sulfonimide", "[N-]([S](=O)(=O))[S](=O)(=O)"),
        ("hexafluorophosphate", "[P-](F)(F)(F)(F)(F)F"),
        ("tetrafluoroborate", "[B-](F)(F)(F)F"),
        ("carboxylate", "[CX3](=O)[O-]"),
        ("sulfonate", "[SX4](=O)(=O)[O-]"),
        ("phosphate", "[PX4](=O)([O-,O])([O-,O])([O-,O])"),
    ]
    for name, smarts in patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return name
    return "other_anion"


class DataAdapter:
    """Load a configured split while preserving original row alignment."""

    def __init__(self, config: dict[str, Any], inspection: InspectionReport) -> None:
        self.config = config
        self.inspection = inspection

    def load(
        self,
        split_name: str | None = None,
        max_samples: int | None = None,
    ) -> AnalysisData:
        split_name = split_name or str(self.config["data"].get("split", "test"))
        with self.inspection.selected_split.open("r", encoding="utf-8") as handle:
            split = json.load(handle)
        if split_name not in split:
            raise KeyError(
                f"Split {split_name!r} is absent from {self.inspection.selected_split}"
            )
        indices = [int(value) for value in split[split_name]]
        configured_limit = int(self.config["data"].get("max_samples", 0))
        limit = configured_limit if max_samples is None else int(max_samples)
        if limit > 0:
            indices = indices[:limit]

        frame_all = pd.read_csv(self.inspection.data_paths["clean_csv"])
        with np.load(self.inspection.data_paths["arrays"], allow_pickle=True) as payload:
            arrays = {key: payload[key] for key in payload.files}
        if len(frame_all) != arrays["y"].shape[0]:
            raise ValueError("Clean CSV and preprocessed arrays have different row counts")
        if max(indices, default=-1) >= len(frame_all):
            raise IndexError("Split index exceeds the preprocessed dataset")

        graph_cache = _torch_load(self.inspection.data_paths["graph_cache"])
        frame = frame_all.iloc[indices].copy().reset_index(drop=True)
        frame["_row_index"] = indices
        cations: list[str] = []
        anions: list[str] = []
        for smiles in frame["IL_SMILES"].astype(str):
            graph = graph_cache.get(smiles)
            if graph is None:
                raise KeyError(f"Graph cache lacks an indexed data row: {smiles}")
            cations.append(str(graph.cation_smiles))
            anions.append(str(graph.anion_smiles))
        frame["cation_smiles"] = cations
        frame["anion_smiles"] = anions
        frame["cation_family"] = [_cation_family(smiles) for smiles in cations]
        frame["anion_family"] = [_anion_family(smiles) for smiles in anions]
        frame["data_split"] = split_name
        frame["checkpoint_type"] = self.inspection.selected_checkpoint_type
        frame["split_path"] = str(self.inspection.selected_split)
        return AnalysisData(
            frame=frame,
            arrays=arrays,
            indices=indices,
            graph_cache_path=self.inspection.data_paths["graph_cache"],
            split_path=self.inspection.selected_split,
            split_name=split_name,
        )

    def load_screening_assets(self) -> dict[str, pd.DataFrame]:
        tables: dict[str, pd.DataFrame] = {}
        for name, path in self.inspection.screening_paths.items():
            if path.is_file():
                tables[name] = pd.read_csv(path)
        return tables
