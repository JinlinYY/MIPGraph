from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import AttentionalAggregation, GINEConv, global_add_pool, global_mean_pool


def mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))


class GraphEncoder3D(nn.Module):
    def __init__(
        self,
        atom_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.15,
        pooling: str = "attention",
        use_e3_invariant_geometry: bool = True,
        rbf_num_centers: int = 32,
        rbf_cutoff: float = 6.0,
    ) -> None:
        super().__init__()
        self.use_e3_invariant_geometry = use_e3_invariant_geometry
        self.rbf_cutoff = float(rbf_cutoff)
        self.rbf_num_centers = int(rbf_num_centers)
        self.node_in = nn.Linear(atom_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)
        if use_e3_invariant_geometry:
            centers = torch.linspace(0.0, self.rbf_cutoff, self.rbf_num_centers)
            self.register_buffer("rbf_centers", centers)
            self.rbf_gamma = nn.Parameter(torch.tensor(10.0))
            self.geometry_encoder = nn.Sequential(
                nn.Linear(self.rbf_num_centers + 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_mlp = mlp(hidden_dim, hidden_dim, hidden_dim, dropout)
            self.layers.append(GINEConv(conv_mlp, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        if pooling == "attention":
            self.att_pool = AttentionalAggregation(nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)))

    def _geometry_features(self, data) -> torch.Tensor:
        row, col = data.edge_index
        distance = torch.norm(data.pos[row] - data.pos[col], dim=-1).clamp_min(0.0)
        edge_type = data.edge_attr[:, 6] if data.edge_attr.size(1) > 6 else torch.zeros_like(distance)
        covalent_mask = (edge_type < 0.5).float()
        has_3d = getattr(data, "has_3d", None)
        if has_3d is not None and hasattr(data, "batch"):
            graph_has_3d = data.has_3d.view(-1).float()
            edge_has_3d = graph_has_3d[data.batch[row]]
        else:
            edge_has_3d = torch.ones_like(distance)
        active = covalent_mask * edge_has_3d
        diff = distance.unsqueeze(-1) - self.rbf_centers.to(distance.device).unsqueeze(0)
        rbf = torch.exp(-torch.abs(self.rbf_gamma) * diff.pow(2)) * active.unsqueeze(-1)
        geom_scalar = torch.stack(
            [
                (distance / max(self.rbf_cutoff, 1e-6)).clamp(0.0, 1.0) * active,
                active,
            ],
            dim=-1,
        )
        return torch.cat([rbf, geom_scalar], dim=-1)

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "add":
            return global_add_pool(h, batch)
        if self.pooling == "attention":
            return self.att_pool(h, batch)
        return global_mean_pool(h, batch)

    def forward(self, data):
        h = self.node_in(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        if self.use_e3_invariant_geometry:
            edge_attr = edge_attr + self.geometry_encoder(self._geometry_features(data))
        for conv, norm in zip(self.layers, self.norms):
            h_new = conv(h, data.edge_index, edge_attr)
            h = norm(h + self.dropout(h_new))
            h = torch.nn.functional.silu(h)
        batch = data.batch
        h_graph = self._pool(h, batch)
        frag = data.fragment_id
        h_cation = self._pool(h * (frag == 0).float().unsqueeze(-1), batch)
        h_anion = self._pool(h * (frag == 1).float().unsqueeze(-1), batch)
        return h, h_graph, h_cation, h_anion
