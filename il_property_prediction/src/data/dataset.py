from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]


def safe_torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ILPropertyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        clean_csv: str | Path,
        arrays_path: str | Path,
        graph_cache_path: str | Path,
        indices: Sequence[int],
        condition: np.ndarray,
        y_scaled: np.ndarray,
        error_weights: np.ndarray,
        augmented_samples: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.df = pd.read_csv(clean_csv)
        with np.load(arrays_path, allow_pickle=True) as arrays:
            self.arrays = {key: arrays[key] for key in arrays.files}
        self.graph_cache = safe_torch_load(graph_cache_path)
        self.indices = [int(i) for i in indices]
        self.condition = condition.astype(np.float32)
        self.y_scaled = y_scaled.astype(np.float32)
        self.error_weights = error_weights.astype(np.float32)
        self.augmented_samples = list(augmented_samples or [])

    def __len__(self) -> int:
        return len(self.indices) + len(self.augmented_samples)

    def __getitem__(self, item: int) -> Data:
        if item >= len(self.indices):
            return self._augmented_item(self.augmented_samples[item - len(self.indices)], item - len(self.indices))
        idx = self.indices[item]
        row = self.df.iloc[idx]
        smiles = str(row["IL_SMILES"])
        if smiles not in self.graph_cache:
            raise KeyError(f"Graph cache missing IL_SMILES: {smiles}")
        graph = self.graph_cache[smiles].clone()
        graph.condition = torch.tensor(self.condition[idx], dtype=torch.float32).view(1, 2)
        graph.raw_condition = torch.tensor(
            [row["Temperature_K"], row["Pressure_kPa"] if pd.notna(row["Pressure_kPa"]) else np.nan],
            dtype=torch.float32,
        ).view(1, 2)
        graph.y = torch.tensor(self.y_scaled[idx], dtype=torch.float32).view(1, -1)
        graph.y_raw = torch.tensor(self.arrays["y"][idx], dtype=torch.float32).view(1, -1)
        graph.mask = torch.tensor(self.arrays["mask"][idx], dtype=torch.float32).view(1, -1)
        evaluation_mask = self.arrays.get("evaluation_mask", self.arrays["mask"])
        graph.eval_mask = torch.tensor(evaluation_mask[idx], dtype=torch.float32).view(1, -1)
        graph.y_error = torch.tensor(self.arrays["y_error"][idx], dtype=torch.float32).view(1, -1)
        graph.error_mask = torch.tensor(self.arrays["error_mask"][idx], dtype=torch.float32).view(1, -1)
        graph.error_weight = torch.tensor(self.error_weights[idx], dtype=torch.float32).view(1, -1)
        graph.sample_id = torch.tensor([idx], dtype=torch.long)
        graph.is_augmented = torch.tensor([0], dtype=torch.long)
        graph.smiles = smiles
        graph.il_name = str(row["IL_Name"])
        return graph

    def _augmented_item(self, sample: dict[str, Any], augmented_index: int) -> Data:
        source_idx = int(sample["source_idx"])
        row = self.df.iloc[source_idx]
        smiles = str(row["IL_SMILES"])
        if smiles not in self.graph_cache:
            raise KeyError(f"Graph cache missing IL_SMILES: {smiles}")
        graph = self.graph_cache[smiles].clone()
        graph.condition = torch.tensor(sample["condition"], dtype=torch.float32).view(1, 2)
        graph.raw_condition = torch.tensor(
            [sample["temperature"], sample["pressure"]], dtype=torch.float32
        ).view(1, 2)
        graph.y = torch.tensor(sample["y_scaled"], dtype=torch.float32).view(1, -1)
        graph.y_raw = torch.tensor(sample["y_raw"], dtype=torch.float32).view(1, -1)
        graph.mask = torch.tensor(sample["mask"], dtype=torch.float32).view(1, -1)
        graph.eval_mask = torch.zeros_like(graph.mask)
        graph.y_error = torch.zeros_like(graph.y)
        graph.error_mask = torch.zeros_like(graph.mask)
        graph.error_weight = torch.tensor(sample["error_weight"], dtype=torch.float32).view(1, -1)
        graph.sample_id = torch.tensor([-(augmented_index + 1)], dtype=torch.long)
        graph.is_augmented = torch.tensor([1], dtype=torch.long)
        graph.smiles = smiles
        graph.il_name = str(row["IL_Name"])
        return graph
