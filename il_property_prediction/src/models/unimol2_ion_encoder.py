from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def _safe_torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class UniMol2IonEncoder(nn.Module):
    """Shared Uni-Mol2 backbone with role-specific cation and anion projections."""

    def __init__(
        self,
        hidden_dim: int,
        feature_cache_path: str | Path,
        weight_dir: str | Path,
        model_size: str = "84m",
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        projection_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        feature_cache_path = Path(feature_cache_path).resolve()
        weight_dir = Path(weight_dir).resolve()
        if not feature_cache_path.exists():
            raise FileNotFoundError(
                f"Uni-Mol2 ion feature cache not found: {feature_cache_path}. "
                "Run scripts/build_unimol2_ion_cache.py first."
            )
        os.environ["UNIMOL_WEIGHT_DIR"] = str(weight_dir)
        from unimol_tools.models.unimolv2 import UniMolV2Model

        self.backbone = UniMolV2Model(output_dim=1, model_size=model_size, remove_hs=True)
        backbone_dim = int(self.backbone.args.encoder_embed_dim)
        self.backbone.classification_head = nn.Identity()
        self.cation_projection = self._projection(backbone_dim, hidden_dim, projection_dropout)
        self.anion_projection = self._projection(backbone_dim, hidden_dim, projection_dropout)
        self.feature_cache_path = str(feature_cache_path)
        cache_payload = _safe_torch_load(feature_cache_path)
        self.feature_cache = cache_payload.get("features", cache_payload)
        if not isinstance(self.feature_cache, dict) or not self.feature_cache:
            raise ValueError(f"Invalid Uni-Mol2 ion feature cache: {feature_cache_path}")
        self.chemistry_cache = cache_payload.get("chemistry")
        if not isinstance(self.chemistry_cache, dict) or not self.chemistry_cache:
            raise ValueError(
                f"Uni-Mol2 cache does not contain atom chemistry annotations: {feature_cache_path}. "
                "Rebuild it with scripts/build_unimol2_ion_cache.py --force."
            )
        self.freeze_backbone = bool(freeze_backbone)
        self.unfreeze_last_n_layers = int(unfreeze_last_n_layers)
        self._configure_trainable_layers()

    @staticmethod
    def _projection(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def _configure_trainable_layers(self) -> None:
        if not self.freeze_backbone:
            return
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if self.unfreeze_last_n_layers <= 0:
            return
        layers = self.backbone.encoder.layers
        if self.unfreeze_last_n_layers > len(layers):
            raise ValueError(
                f"unimol2_unfreeze_last_n_layers={self.unfreeze_last_n_layers} exceeds "
                f"the {len(layers)} layers in Uni-Mol2-{self.backbone.model_size}"
            )
        for layer in layers[-self.unfreeze_last_n_layers :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    @staticmethod
    def _deduplicate(smiles: list[str]) -> tuple[list[str], torch.Tensor]:
        unique: list[str] = []
        index_by_smiles: dict[str, int] = {}
        inverse = []
        for item in smiles:
            item = str(item)
            if item not in index_by_smiles:
                index_by_smiles[item] = len(unique)
                unique.append(item)
            inverse.append(index_by_smiles[item])
        return unique, torch.tensor(inverse, dtype=torch.long)

    def _collate(
        self, smiles: list[str], device: torch.device
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[str]]:
        unique_smiles, inverse = self._deduplicate(smiles)
        missing = [item for item in unique_smiles if item not in self.feature_cache]
        if missing:
            preview = ", ".join(missing[:3])
            raise KeyError(f"Uni-Mol2 feature cache is missing {len(missing)} ions: {preview}")
        samples = [(self.feature_cache[item], 0.0) for item in unique_smiles]
        features, _ = self.backbone.batch_collate_fn(samples)
        features = {name: value.to(device, non_blocking=True) for name, value in features.items()}
        return features, inverse.to(device, non_blocking=True), unique_smiles

    def _encode_unique(self, features: dict[str, torch.Tensor]) -> dict[str, Any]:
        backbone_trainable = any(parameter.requires_grad for parameter in self.backbone.parameters())
        if backbone_trainable:
            output = self.backbone(**features, return_repr=True, return_atomic_reprs=True)
        else:
            with torch.no_grad():
                output = self.backbone(**features, return_repr=True, return_atomic_reprs=True)
        return output

    def forward(self, batch):
        cation_smiles = [str(item) for item in batch.cation_smiles]
        anion_smiles = [str(item) for item in batch.anion_smiles]
        batch_size = len(cation_smiles)
        features, inverse, unique_smiles = self._collate(cation_smiles + anion_smiles, batch.x.device)
        output = self._encode_unique(features)
        unique_representations = output["cls_repr"]
        representations = unique_representations.index_select(0, inverse)
        h_cation = self.cation_projection(representations[:batch_size])
        h_anion = self.anion_projection(representations[batch_size:])
        h_pair_summary = 0.5 * (h_cation + h_anion)

        atomic_representations = pad_sequence(output["atomic_reprs"], batch_first=True)
        atom_lengths = torch.tensor(
            [item.size(0) for item in output["atomic_reprs"]],
            device=atomic_representations.device,
        )
        atom_positions = torch.arange(atomic_representations.size(1), device=atomic_representations.device)
        atom_mask = atom_positions.unsqueeze(0) < atom_lengths.unsqueeze(1)
        chemistry_items = [
            torch.as_tensor(self.chemistry_cache[item], dtype=atomic_representations.dtype)
            for item in unique_smiles
        ]
        chemistry = pad_sequence(chemistry_items, batch_first=True).to(
            atomic_representations.device, non_blocking=True
        )
        if chemistry.shape[:2] != atomic_representations.shape[:2]:
            raise ValueError("Padded Uni-Mol2 atomic representations and chemistry annotations are misaligned")
        atomic_representations = atomic_representations.index_select(0, inverse)
        chemistry = chemistry.index_select(0, inverse)
        atom_mask = atom_mask.index_select(0, inverse)
        atom_data = {
            "unique_representations": unique_representations,
            "cation_atoms": atomic_representations[:batch_size],
            "cation_mask": atom_mask[:batch_size],
            "cation_chemistry": chemistry[:batch_size],
            "anion_atoms": atomic_representations[batch_size:],
            "anion_mask": atom_mask[batch_size:],
            "anion_chemistry": chemistry[batch_size:],
        }
        return atom_data, h_pair_summary, h_cation, h_anion
