from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


PROPERTY_NAMES = ["Density", "ElectricalConductivity", "HeatCapacity", "SurfaceTension", "ThermalConductivity", "Viscosity"]


def head(in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))


class ThermodynamicStructuredDecoder(nn.Module):
    """Physics-inspired decoder that predicts standardized log-properties."""

    def __init__(self, hidden_dim: int = 256, latent_dim: int = 256, residual_scale: float = 0.1, dropout: float = 0.15, use_structured: bool = True, use_residual: bool = True) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.use_structured = use_structured
        self.use_residual = use_residual
        in_base = hidden_dim + latent_dim * 2
        self.rho_params = head(in_base, 4, hidden_dim, dropout)
        self.eta_params = head(hidden_dim + latent_dim * 3, 3, hidden_dim, dropout)
        self.sigma_params = head(hidden_dim + latent_dim * 3, 4, hidden_dim, dropout)
        self.cp_params = head(hidden_dim + latent_dim * 3, 3, hidden_dim, dropout)
        self.gamma_params = head(hidden_dim + latent_dim * 3, 4, hidden_dim, dropout)
        self.lambda_params = head(hidden_dim + latent_dim * 2, 4, hidden_dim, dropout)
        self.scores = nn.ModuleDict({name: head(latent_dim, 1, hidden_dim // 2, dropout) for name in ["packing", "cohesion", "transport", "thermal"]})
        self.residuals = nn.ModuleDict({p: head(hidden_dim + latent_dim, 1, hidden_dim, dropout) for p in PROPERTY_NAMES})
        self.plain = head(hidden_dim + latent_dim * 4 + 2, 6, hidden_dim, dropout)

    def _res(self, prop: str, h_cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.use_residual:
            return torch.zeros(h_cond.size(0), device=h_cond.device)
        return self.residuals[prop](torch.cat([h_cond, z], dim=-1)).squeeze(-1)

    def forward(self, h_cond: torch.Tensor, latent: dict[str, torch.Tensor], condition: torch.Tensor, raw_condition: torch.Tensor):
        packing, cohesion, transport, thermal = latent["packing"], latent["cohesion"], latent["transport"], latent["thermal"]
        prop_z = latent["property_latents"]
        t_raw = torch.nan_to_num(raw_condition[:, 0], nan=298.15).clamp_min(1.0)
        p_raw = torch.nan_to_num(raw_condition[:, 1], nan=101.325)
        t_norm = condition[:, 0]
        p_norm = condition[:, 1]
        log_t = torch.log(t_raw)
        inv_t = 1.0 / t_raw
        inv_t2 = inv_t * inv_t

        if not self.use_structured:
            y = self.plain(torch.cat([h_cond, packing, cohesion, transport, thermal, condition], dim=-1))
            return y, {"physical": y.detach() * 0.0, "residual": y.detach() * 0.0, "gates": latent.get("gates")}

        rho_a, rho_b, rho_c, rho_d = self.rho_params(torch.cat([h_cond, packing, cohesion], dim=-1)).unbind(-1)
        log_rho_phys = rho_a + rho_b * inv_t + rho_c * p_norm + rho_d * p_norm * inv_t
        log_rho = log_rho_phys + self.residual_scale * self._res("Density", h_cond, prop_z[:, 0])

        eta_a, eta_b, eta_c = self.eta_params(torch.cat([h_cond, transport, cohesion, packing], dim=-1)).unbind(-1)
        log_eta_phys = eta_a + F.softplus(eta_b) * inv_t + eta_c * inv_t2
        log_eta = log_eta_phys + self.residual_scale * self._res("Viscosity", h_cond, prop_z[:, 5])

        tr_score = self.scores["transport"](transport).squeeze(-1)
        pk_score = self.scores["packing"](packing).squeeze(-1)
        co_score = self.scores["cohesion"](cohesion).squeeze(-1)
        th_score = self.scores["thermal"](thermal).squeeze(-1)

        sig_a, sig_b, sig_alpha, sig_c = self.sigma_params(torch.cat([h_cond, transport, cohesion, packing], dim=-1)).unbind(-1)
        log_sigma_phys = sig_a + sig_b * log_t - F.softplus(sig_alpha) * log_eta + sig_c * tr_score + 0.1 * pk_score
        log_sigma = log_sigma_phys + self.residual_scale * self._res("ElectricalConductivity", h_cond, prop_z[:, 1])

        cp_a, cp_b, cp_c = self.cp_params(torch.cat([h_cond, thermal, cohesion, packing], dim=-1)).unbind(-1)
        log_cp_phys = cp_a + cp_b * t_norm + cp_c * t_norm * t_norm
        log_cp = log_cp_phys + self.residual_scale * self._res("HeatCapacity", h_cond, prop_z[:, 2])

        ga_a, ga_b, ga_c, ga_d = self.gamma_params(torch.cat([h_cond, cohesion, packing, thermal], dim=-1)).unbind(-1)
        log_gamma_phys = ga_a - F.softplus(ga_b) * t_norm + ga_c * log_rho + ga_d * co_score + 0.1 * th_score
        log_gamma = log_gamma_phys + self.residual_scale * self._res("SurfaceTension", h_cond, prop_z[:, 3])

        la_a, la_b, la_c, la_d = self.lambda_params(torch.cat([h_cond, packing, thermal], dim=-1)).unbind(-1)
        log_lambda_phys = la_a + la_b * log_rho + la_c * log_cp + la_d * t_norm + 0.1 * pk_score + 0.1 * th_score
        log_lambda = log_lambda_phys + self.residual_scale * self._res("ThermalConductivity", h_cond, prop_z[:, 4])

        y = torch.stack([log_rho, log_sigma, log_cp, log_gamma, log_lambda, log_eta], dim=-1)
        phys = torch.stack([log_rho_phys, log_sigma_phys, log_cp_phys, log_gamma_phys, log_lambda_phys, log_eta_phys], dim=-1)
        return y, {"physical": phys, "residual": y - phys, "gates": latent.get("gates")}
