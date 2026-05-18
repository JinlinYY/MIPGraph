"""供 UI 服务调用：加载 IPTNet 检查点并对离子液体 SMILES 批量前向（与训练代码一致）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.chem.graph_featurizer import build_ion_pair_graph  # noqa: E402
from src.models.iptnet import IPTNet  # noqa: E402
from src.utils.io import load_config, resolve_path  # noqa: E402

PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]

PROPERTY_UNITS: dict[str, str] = {
    "Density": "kg·m⁻³",
    "ElectricalConductivity": "S·m⁻¹",
    "HeatCapacity": "J·mol⁻¹·K⁻¹",
    "SurfaceTension": "mN·m⁻¹",
    "ThermalConductivity": "W·m⁻¹·K⁻¹",
    "Viscosity": "Pa·s",
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_bundle(config_path: Path, checkpoint_path: Path) -> tuple[dict, IPTNet, Any, Any, torch.device]:
    cfg = load_config(config_path)
    base = Path(cfg["_base_dir"])
    ckpt = torch.load(resolve_path(checkpoint_path, base), map_location="cpu", weights_only=False)
    model = IPTNet(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    device = _device()
    model.to(device)
    model.eval()
    return ckpt["config"], model, ckpt["condition_scaler"], ckpt["target_scaler"], device


def build_graph_for_smiles(smiles: str, chem_cfg: dict, model_cfg: dict):
    result = build_ion_pair_graph(
        smiles.strip(),
        use_3d=bool(chem_cfg.get("use_3d", True)),
        cutoff=float(chem_cfg.get("cross_ion_cutoff", 5.0)),
        seed=int(chem_cfg.get("seed", 42)),
        max_attempts=int(chem_cfg.get("max_conformer_attempts", 20)),
        optimize_method=str(chem_cfg.get("optimize_method", "UFF")),
        use_cross_edges=bool(model_cfg.get("use_cross_ion_edges", True)),
        cross_ion_mode=str(chem_cfg.get("cross_ion_mode", "deterministic_2d")),
    )
    if result.data is None:
        raise ValueError(result.error or "graph build failed")
    return result.data


def predict_batch_loaded(
    bundle: tuple[dict, IPTNet, Any, Any, torch.device],
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str | None]]:
    cfg, model, cond_scaler, target_scaler, device = bundle
    chem_cfg = cfg.get("chem", {})
    model_cfg = cfg.get("model", {})
    graphs: list = []
    errors: list[str | None] = []
    for row in rows:
        try:
            g = build_graph_for_smiles(str(row["IL_SMILES"]), chem_cfg, model_cfg)
            graphs.append(g)
            errors.append(None)
        except Exception as exc:  # noqa: BLE001
            graphs.append(None)
            errors.append(str(exc))

    valid_indices = [i for i, g in enumerate(graphs) if g is not None]
    if not valid_indices:
        return np.full((len(rows), 6), np.nan), errors

    batch_list = [graphs[i].clone() for i in valid_indices]
    temps = np.array([float(rows[i]["Temperature_K"]) for i in valid_indices], dtype=np.float64)
    pressures = np.array(
        [float(rows[i]["Pressure_kPa"]) if rows[i].get("Pressure_kPa") is not None else np.nan for i in valid_indices],
        dtype=np.float64,
    )
    cond = cond_scaler.transform(temps, pressures)
    raw = np.stack([temps, pressures], axis=-1).astype(np.float32)

    batch = Batch.from_data_list(batch_list).to(device)
    batch.condition = torch.tensor(cond, dtype=torch.float32, device=device)
    batch.raw_condition = torch.tensor(raw, dtype=torch.float32, device=device)

    with torch.inference_mode():
        pred_scaled, _ = model(batch)
    pred_scaled_np = pred_scaled.detach().float().cpu().numpy()
    pred_phys = target_scaler.inverse_transform(pred_scaled_np)

    out = np.full((len(rows), 6), np.nan, dtype=np.float64)
    for j, i in enumerate(valid_indices):
        out[i] = pred_phys[j]
    return out, errors
