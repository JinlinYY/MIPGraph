from __future__ import annotations

from torch import nn

from .mipgraph import MIPGraph


def build_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "MIPGraph")).lower()
    supported_names = {
        "mipgraph",
        "mipgraph_physics_moe",
        "mipgraphphysicsmoe",
        "physics_moe",
        "physicsmoe",
    }
    if name not in supported_names:
        raise ValueError(f"Unknown model name {model_cfg.get('name')!r}. Use 'MIPGraph'.")
    return MIPGraph(config)
