from __future__ import annotations

import torch
from torch import nn


LATENT_NAMES = ["packing", "cohesion", "transport", "thermal"]
PROPERTY_NAMES = ["Density", "ElectricalConductivity", "HeatCapacity", "SurfaceTension", "ThermalConductivity", "Viscosity"]


def _latent_mlp(hidden_dim: int, latent_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(latent_dim, latent_dim))


class MechanisticLatentHeads(nn.Module):
    def __init__(self, hidden_dim: int = 256, latent_dim: int = 256, dropout: float = 0.15, use_property_gating: bool = True) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({name: _latent_mlp(hidden_dim, latent_dim, dropout) for name in LATENT_NAMES})
        self.use_property_gating = use_property_gating
        priors = torch.tensor(
            [
                [0.55, 0.25, 0.10, 0.10],
                [0.20, 0.25, 0.45, 0.10],
                [0.20, 0.25, 0.10, 0.45],
                [0.20, 0.45, 0.10, 0.25],
                [0.35, 0.20, 0.10, 0.35],
                [0.33, 0.33, 0.29, 0.05],
            ],
            dtype=torch.float32,
        )
        self.gate_logits = nn.Parameter(torch.log(priors))

    def forward(self, h_structure: torch.Tensor):
        latent = {name: head(h_structure) for name, head in self.heads.items()}
        stacked = torch.stack([latent[name] for name in LATENT_NAMES], dim=1)
        if self.use_property_gating:
            gates = torch.softmax(self.gate_logits, dim=-1)
        else:
            gates = torch.full_like(self.gate_logits, 1.0 / len(LATENT_NAMES))
        z_props = torch.einsum("pl,bld->bpd", gates, stacked)
        latent["property_latents"] = z_props
        latent["gates"] = gates
        return latent
