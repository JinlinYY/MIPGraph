from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .condition_encoder import ConditionEncoder
from .cross_ion_attention import ChemistryBiasedCrossIonAttention
from .global_descriptor_encoder import GlobalDescriptorEncoder
from .independent_property_decoder import IndependentPropertyDecoder
from .interaction_encoder import InteractionEncoder, StructureProjector
from .ion_pair_transformer_fusion import IonPairTransformerFusion
from .physics_moe import PhysicsLatentMoE
from .property_adapters import PropertyAdapterBank
from .unimol2_ion_encoder import UniMol2IonEncoder


class MIPGraph(nn.Module):
    """Uni-Mol2-based MIPGraph with ion fusion, physical latent routing, and independent heads."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        hidden_dim = int(model_cfg["hidden_dim"])
        latent_dim = int(model_cfg.get("latent_dim", hidden_dim))
        dropout = float(model_cfg.get("dropout", 0.15))
        base_dir = Path(config.get("_base_dir", "."))

        def model_path(key: str, default: str) -> Path:
            value = Path(model_cfg.get(key, default))
            return value if value.is_absolute() else (base_dir / value).resolve()

        self.ion_encoder = UniMol2IonEncoder(
            hidden_dim,
            model_path("unimol2_feature_cache_path", "data/processed/unimol2_ion_features.pt"),
            model_path("unimol2_weight_dir", "data/pretrained/unimol2"),
            str(model_cfg.get("unimol2_model_size", "84m")),
            bool(model_cfg.get("freeze_unimol2_backbone", True)),
            int(model_cfg.get("unimol2_unfreeze_last_n_layers", 0)),
            float(model_cfg.get("unimol2_projection_dropout", dropout)),
        )
        fusion_mode = str(
            model_cfg.get(
                "fusion_mode",
                "transformer" if bool(model_cfg.get("use_transformer_fusion", False)) else "bilinear",
            )
        ).lower()
        fusion_aliases = {
            "feature_concat": "concat",
            "concat": "concat",
            "bilinear": "bilinear",
            "transformer": "transformer",
        }
        if fusion_mode not in fusion_aliases:
            raise ValueError(
                f"Unknown fusion_mode={fusion_mode!r}. Use one of: concat, bilinear, transformer."
            )
        self.fusion_mode = fusion_aliases[fusion_mode]
        if self.fusion_mode == "bilinear":
            self.interaction = InteractionEncoder(hidden_dim, dropout)
        self.use_atom_cross_attention = bool(model_cfg.get("use_atom_cross_attention", False))
        if self.use_atom_cross_attention:
            self.atom_interaction = ChemistryBiasedCrossIonAttention(
                int(self.ion_encoder.backbone.args.encoder_embed_dim),
                hidden_dim,
                int(model_cfg.get("cross_attention_heads", 8)),
                float(model_cfg.get("cross_attention_dropout", 0.1)),
            )
        if self.use_atom_cross_attention and self.fusion_mode == "bilinear":
            self.interaction_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.project = StructureProjector(hidden_dim, dropout)
        self.use_global_descriptors = bool(model_cfg.get("use_global_descriptors", True))
        if self.use_global_descriptors:
            self.global_descriptor_encoder = GlobalDescriptorEncoder(
                int(model_cfg.get("global_descriptor_dim", 56)), hidden_dim, dropout
            )
        self.use_functional_group_descriptors = bool(model_cfg.get("use_functional_group_descriptors", False))
        self.functional_group_dim = int(model_cfg.get("functional_group_dim", 80))
        if self.use_functional_group_descriptors:
            self.functional_group_encoder = GlobalDescriptorEncoder(
                self.functional_group_dim, hidden_dim, dropout
            )
        self.use_transformer_fusion = self.fusion_mode == "transformer"
        if self.use_transformer_fusion:
            self.transformer_fusion = IonPairTransformerFusion(
                hidden_dim=hidden_dim,
                global_descriptor_dim=int(model_cfg.get("global_descriptor_dim", 56)),
                functional_group_dim=self.functional_group_dim,
                num_layers=int(model_cfg.get("transformer_fusion_layers", 2)),
                num_heads=int(model_cfg.get("transformer_fusion_heads", 8)),
                dropout=float(model_cfg.get("transformer_fusion_dropout", 0.1)),
                use_global_descriptor_token=self.use_global_descriptors,
                use_functional_group_token=self.use_functional_group_descriptors,
            )
        self.condition = ConditionEncoder(hidden_dim, dropout, bool(model_cfg.get("use_condition_film", True)))
        self.use_property_adapters = bool(model_cfg.get("use_property_adapters", False))
        if self.use_property_adapters:
            self.property_adapters = PropertyAdapterBank(
                hidden_dim,
                int(model_cfg.get("property_adapter_dim", 64)),
                float(model_cfg.get("property_adapter_dropout", dropout)),
                bool(model_cfg.get("property_adapter_zero_init", True)),
            )
        self.physics_moe = PhysicsLatentMoE(
            hidden_dim,
            latent_dim,
            dropout,
            int(model_cfg.get("physics_moe_top_k", 2)),
            float(model_cfg.get("physics_moe_router_temperature", 1.0)),
            float(model_cfg.get("physics_moe_prior_strength", 1.0)),
        )
        self.decoder = IndependentPropertyDecoder(
            hidden_dim,
            latent_dim,
            int(model_cfg.get("property_head_hidden_dim", hidden_dim * 2)),
            int(model_cfg.get("property_head_depth", 3)),
            float(model_cfg.get("property_head_dropout", dropout)),
        )

    def forward(self, batch):
        atom_data, h_graph, h_cation, h_anion = self.ion_encoder(batch)
        h_cls_interaction = None
        atom_interaction_aux = {}
        h_atom_interaction = None
        if self.use_atom_cross_attention:
            h_atom_interaction, atom_interaction_aux = self.atom_interaction(
                atom_data["cation_atoms"],
                atom_data["cation_mask"],
                atom_data["cation_chemistry"],
                atom_data["anion_atoms"],
                atom_data["anion_mask"],
                atom_data["anion_chemistry"],
            )
        global_descriptor = None
        if self.use_global_descriptors and hasattr(batch, "global_desc"):
            global_descriptor = batch.global_desc.view(h_cation.size(0), -1)
        functional_group_descriptor = None
        if self.use_functional_group_descriptors and hasattr(batch, "functional_group_desc"):
            functional_group_descriptor = batch.functional_group_desc.view(h_cation.size(0), -1)

        transformer_aux = {}
        if self.use_transformer_fusion:
            h_structure, transformer_aux = self.transformer_fusion(
                h_cation,
                h_anion,
                h_atom_interaction,
                global_descriptor,
                functional_group_descriptor,
            )
            h_interaction = h_atom_interaction
            h_descriptor = None
            h_functional_group = None
        else:
            if self.fusion_mode == "bilinear":
                h_cls_interaction = self.interaction(h_cation, h_anion)
                h_interaction = (
                    self.interaction_fusion(torch.cat([h_cls_interaction, h_atom_interaction], dim=-1))
                    if self.use_atom_cross_attention
                    else h_cls_interaction
                )
            elif self.fusion_mode == "concat":
                h_interaction = h_atom_interaction if h_atom_interaction is not None else torch.zeros_like(h_cation)
            else:
                raise RuntimeError(f"Unhandled fusion_mode={self.fusion_mode!r}")
            h_structure = self.project(h_cation, h_anion, h_interaction)
            h_descriptor = None
            if global_descriptor is not None:
                h_structure, h_descriptor = self.global_descriptor_encoder(h_structure, global_descriptor)
            h_functional_group = None
            if functional_group_descriptor is not None:
                h_structure, h_functional_group = self.functional_group_encoder(
                    h_structure, functional_group_descriptor
                )
        condition = batch.condition.view(-1, 2)
        raw_condition = batch.raw_condition.view(-1, 2)
        h_conditioned, h_condition = self.condition(h_structure, condition)
        h_property_conditioned = self.property_adapters(h_conditioned) if self.use_property_adapters else None
        latent = self.physics_moe(h_structure, h_conditioned, h_property_conditioned)
        prediction, aux = self.decoder(
            h_conditioned,
            h_condition,
            latent["property_latents"],
            condition,
            raw_condition,
            h_property_conditioned,
        )
        aux.update(
            {
                "atom_h": atom_data["unique_representations"],
                "h_graph": h_graph,
                "h_cation": h_cation,
                "h_anion": h_anion,
                "h_interaction": h_interaction,
                "h_cls_interaction": h_cls_interaction,
                "h_atom_interaction": h_atom_interaction,
                "h_structure": h_structure,
                "h_condition": h_condition,
                "h_property_conditioned": h_property_conditioned,
                "h_global_desc": h_descriptor,
                "h_functional_group": h_functional_group,
                "latents": latent,
                "gates": latent["gates"],
                "router_logits": latent["router_logits"],
                "router_selected": latent["router_selected"],
                "expert_importance": latent["expert_importance"],
                "expert_load": latent["expert_load"],
                "moe_load_balance_loss": latent["moe_load_balance_loss"],
                "moe_prior_loss": latent["moe_prior_loss"],
                "raw_condition": raw_condition,
                "condition": condition,
                "sample_id": getattr(batch, "sample_id", None),
                "smiles": getattr(batch, "smiles", None),
            }
        )
        aux.update(atom_interaction_aux)
        aux.update(transformer_aux)
        return prediction, aux
