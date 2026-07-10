from __future__ import annotations

import torch
from torch import nn


PROPERTY_NAMES = ["Density", "ElectricalConductivity", "HeatCapacity", "SurfaceTension", "ThermalConductivity", "Viscosity"]


class PropertyAdapterBank(nn.Module):
    """Small per-property residual adapters over a shared representation."""

    def __init__(
        self,
        hidden_dim: int,
        adapter_dim: int = 64,
        dropout: float = 0.1,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict(
            {
                prop: nn.Sequential(
                    nn.Linear(hidden_dim, adapter_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(adapter_dim, hidden_dim),
                )
                for prop in PROPERTY_NAMES
            }
        )
        if zero_init:
            for adapter in self.adapters.values():
                output = adapter[-1]
                if isinstance(output, nn.Linear):
                    nn.init.zeros_(output.weight)
                    nn.init.zeros_(output.bias)

    def forward(self, h_shared: torch.Tensor) -> torch.Tensor:
        return torch.stack([h_shared + self.adapters[prop](h_shared) for prop in PROPERTY_NAMES], dim=1)
