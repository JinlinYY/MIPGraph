from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from ..chem.graph_featurizer import build_ion_pair_graph
from ..models.factory import build_model
from ..utils.io import resolve_path


class DesignPredictor:
    def __init__(self, config: dict, checkpoint: str | Path, device: str | None = None) -> None:
        base = config.get("_base_dir")
        ckpt = torch.load(resolve_path(checkpoint, base), map_location="cpu", weights_only=False)
        self.config = ckpt.get("config", config)
        self.property_names = ckpt.get("property_names", self.config["properties"]["names"])
        self.condition_scaler = ckpt["condition_scaler"]
        self.target_scaler = ckpt["target_scaler"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(self.config)
        missing, unexpected = self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            print({"missing_keys": missing, "unexpected_keys": unexpected})
        self.model.to(self.device)
        self.model.eval()

    def _build_data(self, smiles: str, temperature: float, pressure: float, sample_id: int):
        chem = self.config.get("chem", {})
        model_cfg = self.config.get("model", {})
        result = build_ion_pair_graph(
            smiles,
            use_3d=chem.get("use_3d", True),
            cutoff=chem.get("cross_ion_cutoff", 5.0),
            seed=chem.get("seed", 42),
            max_attempts=chem.get("max_conformer_attempts", 20),
            optimize_method=chem.get("optimize_method", "UFF"),
            use_cross_edges=model_cfg.get("use_cross_ion_edges", True),
            cross_ion_mode=chem.get("cross_ion_mode", "deterministic_2d"),
        )
        if result.data is None:
            return None, result.error
        condition = self.condition_scaler.transform(np.asarray([temperature], dtype=np.float32), np.asarray([pressure], dtype=np.float32))
        data = result.data
        data.condition = torch.tensor(condition, dtype=torch.float32).view(1, 2)
        data.raw_condition = torch.tensor([temperature, pressure], dtype=torch.float32).view(1, 2)
        data.y = torch.zeros(1, len(self.property_names), dtype=torch.float32)
        data.y_raw = torch.zeros(1, len(self.property_names), dtype=torch.float32)
        data.mask = torch.zeros(1, len(self.property_names), dtype=torch.float32)
        data.y_error = torch.zeros(1, len(self.property_names), dtype=torch.float32)
        data.error_mask = torch.zeros(1, len(self.property_names), dtype=torch.float32)
        data.error_weight = torch.ones(1, len(self.property_names), dtype=torch.float32)
        data.sample_id = torch.tensor([sample_id], dtype=torch.long)
        data.smiles = smiles
        data.il_name = smiles
        return data, None

    @torch.no_grad()
    def predict(self, candidates: list[dict], temperature: float, pressure: float, batch_size: int = 64) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[dict] = []
        failures: list[dict] = []
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start : start + batch_size]
            data_list = []
            records = []
            for offset, candidate in enumerate(chunk):
                sample_id = start + offset
                data, error = self._build_data(candidate["IL_SMILES"], temperature, pressure, sample_id)
                if data is None:
                    failures.append({**candidate, "reason": error})
                    continue
                data_list.append(data)
                records.append(candidate)
            if not data_list:
                continue
            batch = Batch.from_data_list(data_list).to(self.device)
            pred_scaled, aux = self.model(batch)
            pred_scaled_np = pred_scaled.detach().cpu().numpy()
            pred = self.target_scaler.inverse_transform(pred_scaled_np)
            log_pred = pred_scaled_np * self.target_scaler.stds[None, :] + self.target_scaler.means[None, :]
            logvar = aux.get("logvar") if aux is not None else None
            std_log = std = lower = upper = None
            if logvar is not None:
                std_scaled = np.exp(0.5 * np.clip(logvar.detach().cpu().numpy(), -20.0, 20.0))
                std_log = std_scaled * self.target_scaler.stds[None, :]
                std = pred * std_log
                z90 = 1.6448536269514722
                lower = np.exp(log_pred - z90 * std_log) - self.target_scaler.eps
                upper = np.exp(log_pred + z90 * std_log) - self.target_scaler.eps
            for row_idx, candidate in enumerate(records):
                row = {
                    "candidate_id": candidate["candidate_id"],
                    "cation_name": candidate["cation_name"],
                    "anion_name": candidate["anion_name"],
                    "IL_SMILES": candidate["IL_SMILES"],
                    "family": candidate["family"],
                    "Temperature_K": temperature,
                    "Pressure_kPa": pressure,
                }
                for prop_idx, prop in enumerate(self.property_names):
                    row[f"{prop}_pred"] = float(pred[row_idx, prop_idx])
                    if std_log is not None:
                        row[f"{prop}_pred_std"] = float(std[row_idx, prop_idx])
                        row[f"{prop}_pred_std_log"] = float(std_log[row_idx, prop_idx])
                        row[f"{prop}_pi90_lower"] = float(lower[row_idx, prop_idx])
                        row[f"{prop}_pi90_upper"] = float(upper[row_idx, prop_idx])
                rows.append(row)
        return pd.DataFrame(rows), pd.DataFrame(failures)
