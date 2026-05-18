from __future__ import annotations

import pandas as pd
import torch
from tqdm import tqdm

from .metrics import PROPERTY_NAMES, compute_log_space_metrics, compute_metrics


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets, masks, sample_ids = [], [], [], []
    for batch in tqdm(loader, desc="Predict", leave=False):
        batch = batch.to(device)
        pred, _ = model(batch)
        preds.append(pred.cpu())
        targets.append(batch.y.cpu())
        masks.append(batch.mask.cpu())
        sample_ids.append(batch.sample_id.cpu())
    return (
        torch.cat(preds, dim=0).numpy(),
        torch.cat(targets, dim=0).numpy(),
        torch.cat(masks, dim=0).numpy(),
        torch.cat(sample_ids, dim=0).numpy().reshape(-1),
    )


def evaluate_model(model, loader, device, target_scaler, clean_df, split_name: str = "test", property_names: list[str] | None = None):
    property_names = property_names or PROPERTY_NAMES
    pred_scaled, true_scaled, mask, sample_ids = predict(model, loader, device)
    y_pred = target_scaler.inverse_transform(pred_scaled)
    y_true = target_scaler.inverse_transform(true_scaled)
    metrics = compute_metrics(y_true, y_pred, mask, property_names)
    y_pred_log = pred_scaled * target_scaler.stds[None, :] + target_scaler.means[None, :]
    y_true_log = true_scaled * target_scaler.stds[None, :] + target_scaler.means[None, :]
    log_metrics = compute_log_space_metrics(y_true_log, y_pred_log, mask, property_names)
    metrics["log_space"] = log_metrics
    for key in ["macro_log_MAE", "macro_log_RMSE", "weighted_log_MAE", "weighted_log_RMSE", "macro_log_R2", "weighted_log_R2"]:
        metrics[key] = log_metrics.get(key)
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
        wide_rows.append(wide)
    return metrics, pd.DataFrame(rows), pd.DataFrame(wide_rows)
