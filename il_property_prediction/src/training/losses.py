from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F


PROPERTY_NAMES = ["Density", "ElectricalConductivity", "HeatCapacity", "SurfaceTension", "ThermalConductivity", "Viscosity"]
PROPERTY_INDEX = {name: idx for idx, name in enumerate(PROPERTY_NAMES)}


def _zero_like_loss(pred: torch.Tensor) -> torch.Tensor:
    return pred.sum() * 0.0


def _as_property_weights(property_weights: list[float] | None) -> torch.Tensor:
    if property_weights is None:
        property_weights = [1.0] * 6
    return torch.tensor(property_weights, dtype=torch.float32).view(1, -1)


def _prepare_weight(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None,
    use_error_weight: bool,
    property_weights: torch.Tensor,
    high_value_weighting: list[dict],
) -> torch.Tensor:
    out = torch.ones_like(mask) if (weight is None or not use_error_weight) else weight.float()
    out = out * property_weights.to(pred.device)
    for item in high_value_weighting:
        idx = int(item["index"])
        threshold = float(item.get("threshold_scaled", 1.0))
        multiplier = float(item.get("multiplier", 1.0))
        valid = (mask[:, idx] > 0) & (target[:, idx] >= threshold)
        out[:, idx] = torch.where(valid, out[:, idx] * multiplier, out[:, idx])
    return out


def _reduce_masked(element_loss: torch.Tensor, mask: torch.Tensor, balance_properties: bool) -> torch.Tensor:
    if balance_properties:
        counts = mask.sum(dim=0)
        present = counts > 0
        if not bool(present.any()):
            return element_loss.sum() * 0.0
        per_property = element_loss.sum(dim=0) / counts.clamp_min(1.0)
        return per_property[present].mean()
    return element_loss.sum() / mask.sum().clamp_min(1.0)


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
        self.register_buffer("property_weights", _as_property_weights(property_weights))
        self.high_value_weighting = high_value_weighting or []

    def forward(self, pred, target, mask, weight=None, aux_outputs=None):
        mask = mask.float()
        weight = _prepare_weight(pred, target, mask, weight, self.use_error_weight, self.property_weights, self.high_value_weighting)
        return _reduce_masked(mask * weight * (pred - target) ** 2, mask, self.balance_properties)


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
        self.register_buffer("property_weights", _as_property_weights(property_weights))
        self.high_value_weighting = high_value_weighting or []

    def forward(self, pred, target, mask, weight=None, aux_outputs=None):
        mask = mask.float()
        weight = _prepare_weight(pred, target, mask, weight, self.use_error_weight, self.property_weights, self.high_value_weighting)
        return _reduce_masked(mask * weight * torch.abs(pred - target), mask, self.balance_properties)


class MaskedHeteroscedasticGaussianNLLLoss(nn.Module):
    def __init__(
        self,
        use_error_weight: bool = True,
        balance_properties: bool = True,
        property_weights: list[float] | None = None,
        high_value_weighting: list[dict] | None = None,
        min_logvar: float = -8.0,
        max_logvar: float = 5.0,
    ) -> None:
        super().__init__()
        self.use_error_weight = use_error_weight
        self.balance_properties = balance_properties
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.register_buffer("property_weights", _as_property_weights(property_weights))
        self.high_value_weighting = high_value_weighting or []

    def forward(self, pred, target, mask, weight=None, aux_outputs=None):
        mask = mask.float()
        if aux_outputs is None or aux_outputs.get("logvar") is None:
            logvar = torch.zeros_like(pred)
        else:
            logvar = aux_outputs["logvar"].clamp(self.min_logvar, self.max_logvar)
        weight = _prepare_weight(pred, target, mask, weight, self.use_error_weight, self.property_weights, self.high_value_weighting)
        element = 0.5 * torch.exp(-logvar) * (pred - target) ** 2 + 0.5 * logvar
        return _reduce_masked(mask * weight * element, mask, self.balance_properties)


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
    return -(gates * (gates + 1e-8).log()).sum(dim=-1).mean()


def _temperature_gradient(pred: torch.Tensor, raw_condition: torch.Tensor | None, prop_idx: int) -> torch.Tensor | None:
    if raw_condition is None or not raw_condition.requires_grad:
        return None
    grad = torch.autograd.grad(
        pred[:, prop_idx].sum(),
        raw_condition,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        return None
    return grad[:, 0]


def _monotonic_decreasing_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    aux_outputs: dict | None,
    prop_idx: int,
    require_label: bool = True,
) -> torch.Tensor:
    aux_outputs = aux_outputs or {}
    grad_t = _temperature_gradient(pred, aux_outputs.get("raw_condition"), prop_idx)
    if grad_t is None:
        return _zero_like_loss(pred)
    valid = mask[:, prop_idx] > 0 if require_label else torch.ones_like(grad_t, dtype=torch.bool)
    if not bool(valid.any()):
        return _zero_like_loss(pred)
    return F.relu(grad_t[valid]).mean()


def _walden_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, aux_outputs: dict | None, config: dict) -> torch.Tensor:
    aux_outputs = aux_outputs or {}
    sigma_idx = PROPERTY_INDEX["ElectricalConductivity"]
    eta_idx = PROPERTY_INDEX["Viscosity"]
    both = (mask[:, sigma_idx] > 0) & (mask[:, eta_idx] > 0)
    alpha = aux_outputs.get("walden_alpha")
    if alpha is None:
        alpha = torch.tensor(float(config.get("walden_alpha_init", 1.0)), device=pred.device, dtype=pred.dtype)
    alpha = alpha.to(device=pred.device, dtype=pred.dtype)
    if bool(both.any()):
        pred_q = pred[:, sigma_idx] + alpha * pred[:, eta_idx]
        target_q = target[:, sigma_idx] + alpha * target[:, eta_idx]
        return F.smooth_l1_loss(pred_q[both], target_q[both])
    if not bool(config.get("walden_use_prediction_smoothness", False)):
        return _zero_like_loss(pred)
    smiles = aux_outputs.get("smiles")
    if not isinstance(smiles, (list, tuple)):
        return _zero_like_loss(pred)
    q = pred[:, sigma_idx] + alpha * pred[:, eta_idx]
    losses = []
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles):
        groups[str(smi)].append(idx)
    for group in groups.values():
        if len(group) < 2:
            continue
        values = q[torch.tensor(group, device=pred.device)]
        losses.append(F.smooth_l1_loss(values, values.mean().expand_as(values)))
    if not losses:
        return _zero_like_loss(pred)
    return torch.stack(losses).mean()


def _curve_consistency_loss(pred: torch.Tensor, mask: torch.Tensor, aux_outputs: dict | None, config: dict) -> torch.Tensor:
    aux_outputs = aux_outputs or {}
    smiles = aux_outputs.get("smiles")
    raw_condition = aux_outputs.get("raw_condition")
    if not isinstance(smiles, (list, tuple)) or raw_condition is None:
        return _zero_like_loss(pred)
    min_points = int(config.get("min_curve_points", 3))
    props = config.get("curve_properties", ["Viscosity", "SurfaceTension", "Density"])
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles):
        groups[str(smi)].append(idx)
    losses = []
    temp = raw_condition[:, 0]
    for prop in props:
        if prop not in PROPERTY_INDEX:
            continue
        pidx = PROPERTY_INDEX[prop]
        for group in groups.values():
            idx_tensor = torch.tensor(group, device=pred.device, dtype=torch.long)
            valid = mask[idx_tensor, pidx] > 0
            idx_tensor = idx_tensor[valid]
            if idx_tensor.numel() < min_points:
                continue
            order = torch.argsort(temp[idx_tensor])
            ordered = idx_tensor[order]
            dy = pred[ordered[1:], pidx] - pred[ordered[:-1], pidx]
            dt = (temp[ordered[1:]] - temp[ordered[:-1]]).clamp_min(1e-6)
            losses.append(F.relu(dy / dt).mean())
    if not losses:
        return _zero_like_loss(pred)
    return torch.stack(losses).mean()


class CompositeMIPGraphLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, config: dict, property_names: list[str]) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.loss_cfg = config.get("loss", {})
        self.property_names = property_names

    def forward(self, pred, target, mask, weight=None, aux_outputs=None):
        loss = self.base_loss(pred, target, mask, weight, aux_outputs)
        aux_outputs = aux_outputs or {}
        if self.loss_cfg.get("use_residual_penalty", False) and aux_outputs.get("residual") is not None:
            loss = loss + float(self.loss_cfg.get("residual_penalty_weight", 0.0)) * aux_outputs["residual"].pow(2).mean()
        if self.loss_cfg.get("use_latent_orthogonality", False) and aux_outputs.get("latents") is not None:
            loss = loss + float(self.loss_cfg.get("latent_orthogonality_weight", 0.0)) * latent_orthogonality_penalty(aux_outputs["latents"]).to(pred.device)
        if self.loss_cfg.get("use_gate_entropy_penalty", False) and aux_outputs.get("gates") is not None:
            loss = loss + float(self.loss_cfg.get("gate_entropy_weight", 0.0)) * gate_entropy_penalty(aux_outputs["gates"]).to(pred.device)
        if self.loss_cfg.get("use_moe_load_balance_loss", False) and aux_outputs.get("moe_load_balance_loss") is not None:
            loss = loss + float(self.loss_cfg.get("moe_load_balance_weight", 0.0)) * aux_outputs["moe_load_balance_loss"]
        if self.loss_cfg.get("use_moe_prior_loss", False) and aux_outputs.get("moe_prior_loss") is not None:
            loss = loss + float(self.loss_cfg.get("moe_prior_weight", 0.0)) * aux_outputs["moe_prior_loss"]

        if self.loss_cfg.get("use_monotonic_loss", False) or float(self.loss_cfg.get("viscosity_monotonic_weight", 0.0)) > 0:
            require_label = bool(self.loss_cfg.get("monotonic_require_label", True))
            mono = _monotonic_decreasing_loss(pred, mask.float(), aux_outputs, PROPERTY_INDEX["Viscosity"], require_label)
            loss = loss + float(self.loss_cfg.get("viscosity_monotonic_weight", 0.0)) * mono
        if self.loss_cfg.get("use_surface_tension_monotonic_reg", False) or float(self.loss_cfg.get("surface_tension_monotonic_weight", 0.0)) > 0:
            mono = _monotonic_decreasing_loss(pred, mask.float(), aux_outputs, PROPERTY_INDEX["SurfaceTension"], True)
            loss = loss + float(self.loss_cfg.get("surface_tension_monotonic_weight", 0.0)) * mono
        if self.loss_cfg.get("use_density_monotonic_reg", False) or float(self.loss_cfg.get("density_monotonic_weight", 0.0)) > 0:
            mono = _monotonic_decreasing_loss(pred, mask.float(), aux_outputs, PROPERTY_INDEX["Density"], True)
            loss = loss + float(self.loss_cfg.get("density_monotonic_weight", 0.0)) * mono
        if self.loss_cfg.get("use_walden_loss", False) or float(self.loss_cfg.get("walden_weight", 0.0)) > 0:
            loss = loss + float(self.loss_cfg.get("walden_weight", 0.0)) * _walden_loss(pred, target, mask.float(), aux_outputs, self.loss_cfg)
        if float(self.loss_cfg.get("curve_consistency_weight", 0.0)) > 0:
            loss = loss + float(self.loss_cfg.get("curve_consistency_weight", 0.0)) * _curve_consistency_loss(pred, mask.float(), aux_outputs, self.loss_cfg)
        return loss


def needs_temperature_grad(config: dict) -> bool:
    loss_cfg = config.get("loss", {})
    return any(
        [
            bool(loss_cfg.get("use_monotonic_loss", False)),
            float(loss_cfg.get("viscosity_monotonic_weight", 0.0)) > 0,
            bool(loss_cfg.get("use_surface_tension_monotonic_reg", False)),
            float(loss_cfg.get("surface_tension_monotonic_weight", 0.0)) > 0,
            bool(loss_cfg.get("use_density_monotonic_reg", False)),
            float(loss_cfg.get("density_monotonic_weight", 0.0)) > 0,
        ]
    )


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
    loss_type = config["loss"].get("type", "masked_weighted_mse")
    if loss_type == "masked_weighted_mae":
        base = MaskedWeightedMAELoss(config["loss"].get("use_error_weight", True), balance_properties, property_weights, high_value_weighting)
    elif loss_type == "heteroscedastic":
        base = MaskedHeteroscedasticGaussianNLLLoss(
            config["loss"].get("use_error_weight", True),
            balance_properties,
            property_weights,
            high_value_weighting,
            float(config["loss"].get("min_logvar", -8.0)),
            float(config["loss"].get("max_logvar", 5.0)),
        )
    else:
        base = MaskedWeightedMSELoss(config["loss"].get("use_error_weight", True), balance_properties, property_weights, high_value_weighting)
    return CompositeMIPGraphLoss(base, config, property_names)
