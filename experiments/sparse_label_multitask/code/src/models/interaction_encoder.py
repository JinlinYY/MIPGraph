from __future__ import annotations

import torch
from torch import nn


class InteractionEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.15) -> None:
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, h_cation: torch.Tensor, h_anion: torch.Tensor) -> torch.Tensor:
        return self.out(self.bilinear(h_cation, h_anion))


class StructureProjector(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.15, use_pairwise_ion_features: bool = True) -> None:
        super().__init__()
        self.use_pairwise_ion_features = use_pairwise_ion_features
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_cation: torch.Tensor, h_anion: torch.Tensor, h_interaction: torch.Tensor) -> torch.Tensor:
        if self.use_pairwise_ion_features:
            h_product = h_cation * h_anion
            h_difference = torch.abs(h_cation - h_anion)
        else:
            h_product = torch.zeros_like(h_cation)
            h_difference = torch.zeros_like(h_cation)
        h_pair = torch.cat([h_cation, h_anion, h_product, h_difference, h_interaction], dim=-1)
        return self.net(h_pair)
