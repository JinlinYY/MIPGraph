from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from .metrics import PROPERTY_NAMES, compute_log_space_metrics, compute_metrics, compute_uncertainty_metrics


@torch.no_grad()
def predict(model, loader, device, return_uncertainty: bool = False):
    model.eval()
    preds, targets, masks, sample_ids, logvars = [], [], [], [], []
    for batch in tqdm(loader, desc="Predict", leave=False):
        batch = batch.to(device)
        pred, aux = model(batch)
        preds.append(pred.cpu())
        targets.append(batch.y.cpu())
        masks.append(getattr(batch, "eval_mask", batch.mask).cpu())
        sample_ids.append(batch.sample_id.cpu())
        if return_uncertainty:
            if aux is not None and aux.get("logvar") is not None:
                logvars.append(aux["logvar"].detach().cpu())
            else:
                logvars.append(torch.full_like(pred.detach().cpu(), float("nan")))
    out = (
        torch.cat(preds, dim=0).numpy(),
        torch.cat(targets, dim=0).numpy(),
        torch.cat(masks, dim=0).numpy(),
        torch.cat(sample_ids, dim=0).numpy().reshape(-1),
    )
    if return_uncertainty:
        return (*out, torch.cat(logvars, dim=0).numpy())
    return out


def evaluate_model(model, loader, device, target_scaler, clean_df, split_name: str = "test", property_names: list[str] | None = None):
    property_names = property_names or PROPERTY_NAMES
    pred_scaled, true_scaled, mask, sample_ids, logvar_scaled = predict(model, loader, device, return_uncertainty=True)
    y_pred = target_scaler.inverse_transform(pred_scaled)
    y_true = target_scaler.inverse_transform(true_scaled)
    metrics = compute_metrics(y_true, y_pred, mask, property_names)
    y_pred_log = pred_scaled * target_scaler.stds[None, :] + target_scaler.means[None, :]
    y_true_log = true_scaled * target_scaler.stds[None, :] + target_scaler.means[None, :]
    log_metrics = compute_log_space_metrics(
        y_true_log,
        y_pred_log,
        mask,
        property_names,
        normalization_stds=target_scaler.stds,
    )
    metrics["log_space"] = log_metrics
    for key in [
        "macro_log_MAE",
        "macro_log_RMSE",
        "macro_log_NMAE",
        "weighted_log_MAE",
        "weighted_log_RMSE",
        "weighted_log_NMAE",
        "macro_log_R2",
        "weighted_log_R2",
    ]:
        metrics[key] = log_metrics.get(key)
    has_uncertainty = np.isfinite(logvar_scaled).any()
    pred_std = pred_std_log = pi90_lower = pi90_upper = None
    if has_uncertainty:
        logvar_scaled = np.clip(logvar_scaled, -20.0, 20.0)
        std_scaled = np.exp(0.5 * logvar_scaled)
        pred_std_log = std_scaled * target_scaler.stds[None, :]
        z90 = 1.6448536269514722
        pi90_lower = np.exp(y_pred_log - z90 * pred_std_log) - target_scaler.eps
        pi90_upper = np.exp(y_pred_log + z90 * pred_std_log) - target_scaler.eps
        pred_std = y_pred * pred_std_log
        metrics["uncertainty"] = compute_uncertainty_metrics(y_true, y_pred, mask, pred_std, pred_std_log, pi90_lower, pi90_upper, property_names)
    rows = []
    wide_rows = []
    for i, sid in enumerate(sample_ids):
        row = clean_df.iloc[int(sid)]
        wide = {
            "sample_id": int(sid),
            "IL_Name": row["IL_Name"],
            "IL_SMILES": row["IL_SMILES"],
            "Temperature_K": row["Temperature_K"],
            "Pressure_kPa": row["Pressure_kPa"],
            "split": split_name,
        }
        for j, prop in enumerate(property_names):
            wide[f"{prop}_pred"] = y_pred[i, j]
            if has_uncertainty:
                wide[f"{prop}_pred_std"] = pred_std[i, j]
                wide[f"{prop}_pred_std_log"] = pred_std_log[i, j]
                wide[f"{prop}_pi90_lower"] = pi90_lower[i, j]
                wide[f"{prop}_pi90_upper"] = pi90_upper[i, j]
            if mask[i, j] > 0:
                wide[f"{prop}_true"] = y_true[i, j]
                wide[f"{prop}_absolute_error"] = abs(y_pred[i, j] - y_true[i, j])
            else:
                wide[f"{prop}_true"] = None
                wide[f"{prop}_absolute_error"] = None
            if mask[i, j] <= 0:
                continue
            rows.append(
                {
                    "IL_Name": row["IL_Name"],
                    "IL_SMILES": row["IL_SMILES"],
                    "Temperature_K": row["Temperature_K"],
                    "Pressure_kPa": row["Pressure_kPa"],
                    "property": prop,
                    "y_true": y_true[i, j],
                    "y_pred": y_pred[i, j],
                    "absolute_error": abs(y_pred[i, j] - y_true[i, j]),
                    "split": split_name,
                }
            )
            if has_uncertainty:
                rows[-1].update(
                    {
                        "pred_std": pred_std[i, j],
                        "pred_std_log": pred_std_log[i, j],
                        "pi90_lower": pi90_lower[i, j],
                        "pi90_upper": pi90_upper[i, j],
                        "inside_pi90": bool(pi90_lower[i, j] <= y_true[i, j] <= pi90_upper[i, j]),
                    }
                )
        wide_rows.append(wide)
    return metrics, pd.DataFrame(rows), pd.DataFrame(wide_rows)
