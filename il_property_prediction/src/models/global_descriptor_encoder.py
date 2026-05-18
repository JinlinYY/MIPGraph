from __future__ import annotations

import torch
from torch import nn


class GlobalDescriptorEncoder(nn.Module):
    def __init__(self, descriptor_dim: int, hidden_dim: int = 256, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_structure: torch.Tensor, global_desc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_desc = self.encoder(global_desc)
        h_fused = self.fusion(torch.cat([h_structure, h_desc], dim=-1))
        return h_fused, h_desc
