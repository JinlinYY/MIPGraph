from __future__ import annotations

from torch import nn

from .condition_encoder import ConditionEncoder
from .global_descriptor_encoder import GlobalDescriptorEncoder
from .graph_encoder_3d import GraphEncoder3D
from .interaction_encoder import InteractionEncoder, StructureProjector
from .latent_heads import MechanisticLatentHeads
from .thermodynamic_decoder import ThermodynamicStructuredDecoder


class MIPGraphNet(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        m = config["model"]
        hidden = int(m["hidden_dim"])
        latent = int(m.get("latent_dim", hidden))
        dropout = float(m.get("dropout", 0.15))
        edge_type_index = int(m.get("edge_type_index", 6 if m.get("name") == "3D-IPTNet" else 9))
        self.graph = GraphEncoder3D(
            m.get("atom_feature_dim", 45),
            m.get("edge_feature_dim", 12),
            hidden,
            int(m.get("num_layers", 4)),
            dropout,
            m.get("pooling", "attention"),
            bool(m.get("use_e3_invariant_geometry", True)),
            int(m.get("rbf_num_centers", 32)),
            float(m.get("rbf_cutoff", 6.0)),
            edge_type_index,
        )
        self.interaction = InteractionEncoder(hidden, dropout)
        self.project = StructureProjector(hidden, dropout)
        self.use_global_descriptors = bool(m.get("use_global_descriptors", True))
        if self.use_global_descriptors:
            self.global_descriptor_encoder = GlobalDescriptorEncoder(int(m.get("global_descriptor_dim", 56)), hidden, dropout)
        self.latents = MechanisticLatentHeads(hidden, latent, dropout, bool(m.get("use_property_latent_gating", True)))
        self.condition = ConditionEncoder(hidden, dropout, bool(m.get("use_condition_film", True)))
        self.decoder = ThermodynamicStructuredDecoder(
            hidden,
            latent,
            float(m.get("residual_scale", 0.1)),
            dropout,
            bool(m.get("use_structured_decoder", True)),
            bool(m.get("use_neural_residual", True)),
        )

    def forward(self, batch):
        atom_h, h_graph, h_cation, h_anion = self.graph(batch)
        h_inter = self.interaction(h_cation, h_anion)
        h_structure = self.project(h_cation, h_anion, h_inter)
        h_desc = None
        if self.use_global_descriptors and hasattr(batch, "global_desc"):
            global_desc = batch.global_desc.view(h_structure.size(0), -1)
            h_structure, h_desc = self.global_descriptor_encoder(h_structure, global_desc)
        latent = self.latents(h_structure)
        h_cond, h_condition = self.condition(h_structure, batch.condition.view(-1, 2))
        y, aux = self.decoder(h_cond, latent, batch.condition.view(-1, 2), batch.raw_condition.view(-1, 2))
        aux.update({"h_graph": h_graph, "h_structure": h_structure, "h_condition": h_condition, "h_global_desc": h_desc, "latents": latent})
        return y, aux
