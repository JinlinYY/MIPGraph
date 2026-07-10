from __future__ import annotations

import torch
from torch import nn


class IonPairTransformerFusion(nn.Module):
    """Token-level fusion for cation, anion, interaction, and descriptor features."""

    def __init__(
        self,
        hidden_dim: int = 256,
        global_descriptor_dim: int = 56,
        functional_group_dim: int = 80,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_global_descriptor_token: bool = True,
        use_functional_group_token: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("Transformer fusion hidden_dim must be divisible by num_heads")
        self.use_global_descriptor_token = use_global_descriptor_token
        self.use_functional_group_token = use_functional_group_token
        self.pair_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_embedding = nn.Parameter(torch.zeros(1, 8, hidden_dim))
        self.product_projection = nn.Linear(hidden_dim, hidden_dim)
        self.difference_projection = nn.Linear(hidden_dim, hidden_dim)
        self.atom_interaction_projection = nn.Linear(hidden_dim, hidden_dim)
        self.global_projection = nn.Linear(global_descriptor_dim, hidden_dim)
        self.functional_group_projection = nn.Linear(functional_group_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h_cation: torch.Tensor,
        h_anion: torch.Tensor,
        h_atom_interaction: torch.Tensor | None = None,
        global_descriptor: torch.Tensor | None = None,
        functional_group_descriptor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = h_cation.size(0)
        tokens = [
            self.pair_token.expand(batch_size, -1, -1),
            h_cation.unsqueeze(1),
            h_anion.unsqueeze(1),
            self.product_projection(h_cation * h_anion).unsqueeze(1),
            self.difference_projection(torch.abs(h_cation - h_anion)).unsqueeze(1),
        ]
        token_names = ["pair", "cation", "anion", "product", "difference"]
        if h_atom_interaction is not None:
            tokens.append(self.atom_interaction_projection(h_atom_interaction).unsqueeze(1))
            token_names.append("atom_interaction")
        if self.use_global_descriptor_token and global_descriptor is not None:
            tokens.append(self.global_projection(global_descriptor).unsqueeze(1))
            token_names.append("global_descriptor")
        if self.use_functional_group_token and functional_group_descriptor is not None:
            tokens.append(self.functional_group_projection(functional_group_descriptor).unsqueeze(1))
            token_names.append("functional_group")

        token_tensor = torch.cat(tokens, dim=1)
        token_tensor = token_tensor + self.type_embedding[:, : token_tensor.size(1), :]
        encoded = self.encoder(token_tensor)
        encoded = self.norm(encoded)
        pair_out = encoded[:, 0]
        pooled = encoded[:, 1:].mean(dim=1)
        fused = self.out(torch.cat([pair_out, pooled], dim=-1))
        return fused, {
            "fusion_tokens": encoded.detach(),
            "fusion_token_count": torch.tensor([len(token_names)], device=encoded.device),
        }
