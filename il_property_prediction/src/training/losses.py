from __future__ import annotations

import torch
from torch import nn


class MaskedWeightedMSELoss(nn.Module):
    def __init__(
        self,
        use_error_weight: bool = True,
        balance_properties: bool = True,
        property_weights: list[float] | None = None,
        high_value_weighting: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self.use_error_weight = use_error_weight
        self.balance_properties = balance_properties
        if property_weights is None:
            property_weights = [1.0] * 6
        self.register_buffer("property_weights", torch.tensor(property_weights, dtype=torch.float32).view(1, -1))
        self.high_value_weighting = high_value_weighting or []

    def forward(self, pred, target, mask, weight=None, aux_outputs=None):
        mask = mask.float()
        weight = torch.ones_like(mask) if (weight is None or not self.use_error_weight) else weight.float()
        weight = weight * self.property_weights.to(pred.device)
        for item in self.high_value_weighting:
            idx = int(item["index"])
            threshold = float(item.get("threshold_scaled", 1.0))
            multiplier = float(item.get("multiplier", 1.0))
            valid = (mask[:, idx] > 0) & (target[:, idx] >= threshold)
            weight[:, idx] = torch.where(valid, weight[:, idx] * multiplier, weight[:, idx])
        element_loss = mask * weight * (pred - target) ** 2
        if self.balance_properties:
            counts = mask.sum(dim=0)
            present = counts > 0
            if not bool(present.any()):
                return element_loss.sum() * 0.0
            per_property = element_loss.sum(dim=0) / counts.clamp_min(1.0)
            return per_property[present].mean()
        return element_loss.sum() / mask.sum().clamp_min(1.0)


class MaskedWeightedMAELoss(nn.Module):
    def __init__(
        self,
        use_error_weight: bool = True,
        balance_properties: bool = True,
        property_weights: list[float] | None = None,
        high_value_weighting: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self.use_error_weight = use_error_weight
        self.balance_properties = balance_properties
        if property_weights is None:
            property_weights = [1.0] * 6
        self.register_buffer("property_weights", torch.tensor(property_weights, dtype=torch.float32).view(1, -1))
        self.high_value_weighting = high_value_weighting or []

    def forward(self, pred, target, mask, weight=None, aux_outputs=None):
        mask = mask.float()
        weight = torch.ones_like(mask) if (weight is None or not self.use_error_weight) else weight.float()
        weight = weight * self.property_weights.to(pred.device)
        for item in self.high_value_weighting:
            idx = int(item["index"])
            threshold = float(item.get("threshold_scaled", 1.0))
            multiplier = float(item.get("multiplier", 1.0))
            valid = (mask[:, idx] > 0) & (target[:, idx] >= threshold)
            weight[:, idx] = torch.where(valid, weight[:, idx] * multiplier, weight[:, idx])
        element_loss = mask * weight * torch.abs(pred - target)
        if self.balance_properties:
            counts = mask.sum(dim=0)
            present = counts > 0
            if not bool(present.any()):
                return element_loss.sum() * 0.0
            per_property = element_loss.sum(dim=0) / counts.clamp_min(1.0)
            return per_property[present].mean()
        return element_loss.sum() / mask.sum().clamp_min(1.0)


def latent_orthogonality_penalty(latents: dict[str, torch.Tensor]) -> torch.Tensor:
    names = ["packing", "cohesion", "transport", "thermal"]
    penalty = None
    for i, ni in enumerate(names):
        zi = torch.nn.functional.normalize(latents[ni], dim=-1)
        for nj in names[i + 1 :]:
            zj = torch.nn.functional.normalize(latents[nj], dim=-1)
            val = (zi * zj).sum(dim=-1).pow(2).mean()
            penalty = val if penalty is None else penalty + val
    return penalty if penalty is not None else torch.tensor(0.0)


def gate_entropy_penalty(gates: torch.Tensor) -> torch.Tensor:
    ent = -(gates * (gates + 1e-8).log()).sum(dim=-1).mean()
    return ent


def build_loss(config: dict) -> nn.Module:
    balance_properties = config["loss"].get("balance_properties", True)
    property_names = config["properties"]["names"]
    weights_cfg = config["loss"].get("property_loss_weights", {})
    property_weights = [float(weights_cfg.get(name, 1.0)) for name in property_names]
    high_value_cfg = config["loss"].get("high_value_weighting", {})
    high_value_weighting = []
    for name, item in high_value_cfg.items():
        if name not in property_names:
            continue
        high_value_weighting.append(
            {
                "index": property_names.index(name),
                "threshold_scaled": float(item.get("threshold_scaled", 1.0)),
                "multiplier": float(item.get("multiplier", 1.0)),
            }
        )
    if config["loss"].get("type", "masked_weighted_mse") == "masked_weighted_mae":
        return MaskedWeightedMAELoss(config["loss"].get("use_error_weight", True), balance_properties, property_weights, high_value_weighting)
    return MaskedWeightedMSELoss(config["loss"].get("use_error_weight", True), balance_properties, property_weights, high_value_weighting)
