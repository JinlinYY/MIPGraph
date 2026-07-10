from __future__ import annotations

import torch
from torch import nn


PROPERTY_NAMES = ["Density", "ElectricalConductivity", "HeatCapacity", "SurfaceTension", "ThermalConductivity", "Viscosity"]


def _property_head(in_dim: int, hidden_dim: int, depth: int, dropout: float) -> nn.Sequential:
    if depth < 1:
        raise ValueError("property_head_depth must be at least 1")
    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(depth):
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
        )
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)


class IndependentPropertyDecoder(nn.Module):
    """Six independent property heads conditioned only on structure, physics latents, T and P."""

    def __init__(
        self,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        head_hidden_dim: int = 512,
        head_depth: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        condition_basis_dim = 6
        in_dim = hidden_dim * 2 + latent_dim + condition_basis_dim
        self.heads = nn.ModuleDict(
            {name: _property_head(in_dim, head_hidden_dim, head_depth, dropout) for name in PROPERTY_NAMES}
        )

    @staticmethod
    def _condition_basis(condition: torch.Tensor, raw_condition: torch.Tensor) -> torch.Tensor:
        t_norm = condition[:, 0]
        p_norm = condition[:, 1]
        t_kelvin = torch.nan_to_num(raw_condition[:, 0], nan=298.15).clamp_min(1.0)
        log_t_ratio = torch.log(t_kelvin / 298.15)
        inverse_t_ratio = 298.15 / t_kelvin - 1.0
        return torch.stack(
            [t_norm, p_norm, t_norm.square(), t_norm * p_norm, log_t_ratio, inverse_t_ratio],
            dim=-1,
        )

    def forward(
        self,
        h_conditioned: torch.Tensor,
        h_condition: torch.Tensor,
        property_latents: torch.Tensor,
        condition: torch.Tensor,
        raw_condition: torch.Tensor,
        h_property_conditioned: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        basis = self._condition_basis(condition, raw_condition)
        predictions = []
        for property_index, property_name in enumerate(PROPERTY_NAMES):
            h_head = (
                h_property_conditioned[:, property_index]
                if h_property_conditioned is not None
                else h_conditioned
            )
            head_input = torch.cat(
                [h_head, h_condition, property_latents[:, property_index], basis],
                dim=-1,
            )
            predictions.append(self.heads[property_name](head_input).squeeze(-1))
        output = torch.stack(predictions, dim=-1)
        return output, {
            "physical": output,
            "residual": torch.zeros_like(output),
            "condition_basis": basis,
        }
