from __future__ import annotations

import torch
from torch import nn


class ChemistryBiasedCrossIonAttention(nn.Module):
    """Bidirectional atom-level attention between ions with chemistry-derived bias."""

    def __init__(
        self,
        atom_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("cross-ion attention hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cation_atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.anion_atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.cation_q = nn.Linear(hidden_dim, hidden_dim)
        self.cation_k = nn.Linear(hidden_dim, hidden_dim)
        self.cation_v = nn.Linear(hidden_dim, hidden_dim)
        self.anion_q = nn.Linear(hidden_dim, hidden_dim)
        self.anion_k = nn.Linear(hidden_dim, hidden_dim)
        self.anion_v = nn.Linear(hidden_dim, hidden_dim)
        self.cation_out = nn.Linear(hidden_dim, hidden_dim)
        self.anion_out = nn.Linear(hidden_dim, hidden_dim)
        self.cation_norm = nn.LayerNorm(hidden_dim)
        self.anion_norm = nn.LayerNorm(hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.chemistry_bias = nn.Linear(4, num_heads, bias=False)
        initial_bias = torch.tensor([0.5, 0.5, 0.15, 0.1], dtype=torch.float32)
        with torch.no_grad():
            self.chemistry_bias.weight.copy_(initial_bias.repeat(num_heads, 1))
        self.cation_pool = nn.Linear(hidden_dim, 1)
        self.anion_pool = nn.Linear(hidden_dim, 1)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _pair_bias(self, cation_chemistry: torch.Tensor, anion_chemistry: torch.Tensor) -> torch.Tensor:
        c_charge = cation_chemistry[..., 0].unsqueeze(2)
        a_charge = anion_chemistry[..., 0].unsqueeze(1)
        charge_product = c_charge * a_charge
        charge_complementarity = -charge_product
        hbond_compatibility = (
            cation_chemistry[..., 1].unsqueeze(2) * anion_chemistry[..., 2].unsqueeze(1)
            + cation_chemistry[..., 2].unsqueeze(2) * anion_chemistry[..., 1].unsqueeze(1)
        )
        aromatic_contact = (
            cation_chemistry[..., 3].unsqueeze(2) * anion_chemistry[..., 3].unsqueeze(1)
        )
        charge_strength = charge_product.abs()
        pair_features = torch.stack(
            [charge_complementarity, hbond_compatibility, aromatic_contact, charge_strength],
            dim=-1,
        )
        return self.chemistry_bias(pair_features).permute(0, 3, 1, 2)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, atoms, _ = tensor.shape
        return tensor.view(batch, atoms, self.num_heads, self.head_dim).transpose(1, 2)

    def _attend(
        self,
        query_atoms: torch.Tensor,
        key_atoms: torch.Tensor,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
        query_layer: nn.Linear,
        key_layer: nn.Linear,
        value_layer: nn.Linear,
        output_layer: nn.Linear,
        output_norm: nn.LayerNorm,
        pair_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._split_heads(query_layer(query_atoms))
        key = self._split_heads(key_layer(key_atoms))
        value = self._split_heads(value_layer(key_atoms))
        logits = torch.matmul(query, key.transpose(-1, -2)) * self.scale + pair_bias
        logits = logits.masked_fill(~key_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = self.attention_dropout(weights)
        context = torch.matmul(weights, value).transpose(1, 2).contiguous()
        context = context.view(query_atoms.size(0), query_atoms.size(1), -1)
        context = output_norm(query_atoms + self.output_dropout(output_layer(context)))
        context = context * query_mask.unsqueeze(-1).to(context.dtype)
        return context, weights

    @staticmethod
    def _pool(atoms: torch.Tensor, mask: torch.Tensor, scorer: nn.Linear) -> torch.Tensor:
        scores = scorer(atoms).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(weights.unsqueeze(-1) * atoms, dim=1)

    def forward(
        self,
        cation_atoms: torch.Tensor,
        cation_mask: torch.Tensor,
        cation_chemistry: torch.Tensor,
        anion_atoms: torch.Tensor,
        anion_mask: torch.Tensor,
        anion_chemistry: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cation_base = self.cation_atom_projection(cation_atoms)
        anion_base = self.anion_atom_projection(anion_atoms)
        pair_bias = self._pair_bias(cation_chemistry, anion_chemistry).to(cation_base.dtype)
        cation_cross, cation_attention = self._attend(
            cation_base,
            anion_base,
            cation_mask,
            anion_mask,
            self.cation_q,
            self.anion_k,
            self.anion_v,
            self.cation_out,
            self.cation_norm,
            pair_bias,
        )
        anion_cross, anion_attention = self._attend(
            anion_base,
            cation_base,
            anion_mask,
            cation_mask,
            self.anion_q,
            self.cation_k,
            self.cation_v,
            self.anion_out,
            self.anion_norm,
            pair_bias.transpose(-1, -2),
        )
        cation_pool = self._pool(cation_cross, cation_mask, self.cation_pool)
        anion_pool = self._pool(anion_cross, anion_mask, self.anion_pool)
        interaction = self.fusion(
            torch.cat(
                [
                    cation_pool,
                    anion_pool,
                    cation_pool * anion_pool,
                    torch.abs(cation_pool - anion_pool),
                ],
                dim=-1,
            )
        )
        cation_to_anion_mask = cation_mask.unsqueeze(-1) & anion_mask.unsqueeze(1)
        anion_to_cation_mask = anion_mask.unsqueeze(-1) & cation_mask.unsqueeze(1)
        return interaction, {
            "cation_to_anion_attention": (
                cation_attention.mean(dim=1) * cation_to_anion_mask.to(cation_attention.dtype)
            ).detach(),
            "anion_to_cation_attention": (
                anion_attention.mean(dim=1) * anion_to_cation_mask.to(anion_attention.dtype)
            ).detach(),
            "cross_ion_pair_bias": (
                pair_bias.mean(dim=1) * cation_to_anion_mask.to(pair_bias.dtype)
            ).detach(),
        }
