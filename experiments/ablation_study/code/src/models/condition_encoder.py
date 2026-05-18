from __future__ import annotations

import torch
from torch import nn


class ConditionEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.15, use_film: bool = True) -> None:
        super().__init__()
        self.use_film = use_film
        self.encoder = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        if use_film:
            self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        else:
            self.concat = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, h_structure: torch.Tensor, condition: torch.Tensor):
        h_condition = self.encoder(condition)
        if self.use_film:
            gamma, beta = self.film(h_condition).chunk(2, dim=-1)
            h_cond = (1.0 + torch.tanh(gamma)) * h_structure + beta
        else:
            h_cond = self.concat(torch.cat([h_structure, h_condition], dim=-1))
        return h_cond, h_condition
