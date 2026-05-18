from __future__ import annotations

import torch
from torch import nn


LATENT_NAMES = ["packing", "cohesion", "transport", "thermal"]
PROPERTY_NAMES = ["Density", "ElectricalConductivity", "HeatCapacity", "SurfaceTension", "ThermalConductivity", "Viscosity"]


def _latent_mlp(hidden_dim: int, latent_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(latent_dim, latent_dim))


class MechanisticLatentHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        dropout: float = 0.15,
        use_property_gating: bool = True,
        zero_property_latents: bool = False,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({name: _latent_mlp(hidden_dim, latent_dim, dropout) for name in LATENT_NAMES})
        self.use_property_gating = use_property_gating
        self.zero_property_latents = zero_property_latents
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
        if self.zero_property_latents:
            z_props = torch.zeros_like(z_props)
        latent["property_latents"] = z_props
        latent["gates"] = gates
        return latent


class SharedLatentHeads(nn.Module):
    """A non-mechanistic control that removes the four-factor decomposition."""

    def __init__(self, hidden_dim: int = 256, latent_dim: int = 256, dropout: float = 0.15) -> None:
        super().__init__()
        self.head = _latent_mlp(hidden_dim, latent_dim, dropout)

    def forward(self, h_structure: torch.Tensor):
        z = self.head(h_structure)
        latent = {name: z for name in LATENT_NAMES}
        latent["property_latents"] = z.unsqueeze(1).expand(-1, len(PROPERTY_NAMES), -1)
        latent["gates"] = torch.full(
            (len(PROPERTY_NAMES), len(LATENT_NAMES)),
            1.0 / len(LATENT_NAMES),
            dtype=z.dtype,
            device=z.device,
        )
        return latent
