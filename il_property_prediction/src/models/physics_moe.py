from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


MECHANISM_NAMES = ["packing", "cohesion", "transport", "thermal"]
PROPERTY_NAMES = ["Density", "ElectricalConductivity", "HeatCapacity", "SurfaceTension", "ThermalConductivity", "Viscosity"]


def mechanism_priors() -> torch.Tensor:
    return torch.tensor(
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


def _expert(hidden_dim: int, latent_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, latent_dim),
        nn.LayerNorm(latent_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(latent_dim, latent_dim),
        nn.LayerNorm(latent_dim),
        nn.SiLU(),
    )


class PhysicsLatentMoE(nn.Module):
    """Condition-aware sparse routing over mechanism-oriented latent experts."""

    def __init__(
        self,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        dropout: float = 0.15,
        top_k: int = 2,
        router_temperature: float = 1.0,
        prior_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > len(MECHANISM_NAMES):
            raise ValueError(f"physics_moe_top_k must be between 1 and {len(MECHANISM_NAMES)}")
        if router_temperature <= 0:
            raise ValueError("physics_moe_router_temperature must be positive")
        self.top_k = top_k
        self.router_temperature = float(router_temperature)
        self.prior_strength = float(prior_strength)
        self.experts = nn.ModuleDict({name: _expert(hidden_dim, latent_dim, dropout) for name in MECHANISM_NAMES})
        router_hidden = max(hidden_dim, latent_dim)
        self.router_trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, router_hidden),
            nn.LayerNorm(router_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.router_heads = nn.ModuleDict(
            {name: nn.Linear(router_hidden, len(MECHANISM_NAMES)) for name in PROPERTY_NAMES}
        )
        priors = mechanism_priors()
        self.register_buffer("priors", priors)

    def _sparse_gates(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dense_gates = torch.softmax(logits / self.router_temperature, dim=-1)
        if self.top_k == len(MECHANISM_NAMES):
            return dense_gates, torch.ones_like(dense_gates, dtype=torch.bool)
        _, indices = torch.topk(dense_gates, self.top_k, dim=-1)
        selected = torch.zeros_like(dense_gates, dtype=torch.bool).scatter_(-1, indices, True)
        sparse = dense_gates * selected
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return sparse, selected

    def forward(
        self,
        h_structure: torch.Tensor,
        h_conditioned: torch.Tensor,
        h_property_conditioned: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent = {name: expert(h_structure) for name, expert in self.experts.items()}
        stacked = torch.stack([latent[name] for name in MECHANISM_NAMES], dim=1)
        if h_property_conditioned is None:
            router_features = self.router_trunk(torch.cat([h_structure, h_conditioned], dim=-1))
            dynamic_logits = torch.stack(
                [self.router_heads[name](router_features) for name in PROPERTY_NAMES],
                dim=1,
            )
        else:
            h_structure_by_property = h_structure.unsqueeze(1).expand(-1, len(PROPERTY_NAMES), -1)
            router_features = self.router_trunk(torch.cat([h_structure_by_property, h_property_conditioned], dim=-1))
            dynamic_logits = torch.stack(
                [self.router_heads[name](router_features[:, property_idx]) for property_idx, name in enumerate(PROPERTY_NAMES)],
                dim=1,
            )
        prior_logits = torch.log(self.priors.clamp_min(1e-8)).unsqueeze(0)
        logits = dynamic_logits + self.prior_strength * prior_logits
        gates, selected = self._sparse_gates(logits)
        property_latents = torch.einsum("bpe,bed->bpd", gates, stacked)

        importance = gates.mean(dim=(0, 1))
        load = selected.float().mean(dim=(0, 1)) / float(self.top_k)
        load_balance_loss = len(MECHANISM_NAMES) * torch.sum(importance * load)
        mean_property_gates = gates.mean(dim=0).clamp_min(1e-8)
        prior_loss = F.kl_div(mean_property_gates.log(), self.priors, reduction="batchmean")

        latent.update(
            {
                "property_latents": property_latents,
                "gates": gates,
                "router_logits": logits,
                "router_selected": selected,
                "expert_importance": importance,
                "expert_load": load,
                "moe_load_balance_loss": load_balance_loss,
                "moe_prior_loss": prior_loss,
            }
        )
        return latent
