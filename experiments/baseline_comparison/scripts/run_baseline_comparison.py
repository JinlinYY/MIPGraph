from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, global_mean_pool

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "il_property_prediction"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.scaler import fit_scalers  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]

TREE_MODELS = {"rf", "xgboost", "lgbm"}
GRAPH_MODELS = {"mpnn_concat", "gcn", "gat", "graphsage", "gin"}
ALL_MODELS = ["rf", "xgboost", "lgbm", "chemberta", "mpnn_concat", "gcn", "gat", "graphsage", "gin"]


@dataclass
class Paths:
    data_dir: Path
    clean_csv: Path
    arrays_path: Path
    split_path: Path
    graph_cache: Path
    output_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun baseline comparisons on the updated sparse IL dataset.")
    parser.add_argument("--models", default="all", help="Comma-separated models or 'all'.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "processed_ilthermo_interpolated")
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--graph-cache", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DIR / "outputs" / "baseline_comparison_random_point_seed42",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-scaler-mask", choices=["mask", "evaluation_mask"], default="mask")
    parser.add_argument("--tree-n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=1, help="CPU workers for tree baselines. Default avoids Windows joblib permission issues.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-grouped-graphs",
        action="store_true",
        help="Disable grouped graph training that reuses each IL graph embedding across condition rows.",
    )
    parser.add_argument(
        "--no-grouped-chemberta",
        action="store_true",
        help="Disable grouped ChemBERTa training that reuses each IL SMILES embedding across condition rows.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow HuggingFace downloads for ChemBERTa.")
    parser.add_argument("--chemberta-model", default="seyonec/ChemBERTa-zinc-base-v1")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    data_dir = args.data_dir.resolve()
    graph_cache = args.graph_cache
    if graph_cache is None:
        fg_cache = PROJECT_DIR / "data" / "processed" / "graph_cache_fg.pt"
        graph_cache = fg_cache if fg_cache.exists() else data_dir / "graph_cache.pt"
    split_path = args.split_path or data_dir / "splits" / "row_level_seed42.json"
    return Paths(
        data_dir=data_dir,
        clean_csv=data_dir / "il_multiprop_clean.csv",
        arrays_path=data_dir / "il_multiprop_arrays.npz",
        split_path=split_path.resolve(),
        graph_cache=graph_cache.resolve(),
        output_root=args.output_root.resolve(),
    )


def parse_model_list(value: str) -> list[str]:
    if value.lower() == "all":
        return ALL_MODELS.copy()
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise ValueError(f"Unknown baseline models: {unknown}. Valid models: {ALL_MODELS}")
    return models


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def fit_context(paths: Paths, split: dict, target_scaler_mask: str):
    arrays = dict(np.load(paths.arrays_path, allow_pickle=True))
    condition_scaler, target_scaler, y_scaled, condition, error_weights = fit_scalers(
        arrays,
        split["train"],
        clip_min=0.1,
        clip_max=10.0,
        target_mask_key=target_scaler_mask,
    )
    return arrays, condition_scaler, target_scaler, y_scaled, condition, error_weights


def condition_basis(temperature: np.ndarray, pressure: np.ndarray, condition: np.ndarray) -> np.ndarray:
    t = np.asarray(temperature, dtype=np.float32)
    p = np.asarray(pressure, dtype=np.float32)
    p = np.where(np.isfinite(p), p, 101.325)
    t_safe = np.maximum(t, 1e-6)
    return np.column_stack(
        [
            condition[:, 0],
            condition[:, 1],
            condition[:, 0] ** 2,
            condition[:, 0] * condition[:, 1],
            np.log(t_safe / 298.15),
            298.15 / t_safe - 1.0,
            p / 101.325 - 1.0,
        ]
    ).astype(np.float32)


def build_fixed_features(df: pd.DataFrame, arrays: dict, condition: np.ndarray, graph_cache: dict) -> np.ndarray:
    any_graph = next(iter(graph_cache.values()))
    global_dim = int(getattr(any_graph, "global_desc").numel()) if hasattr(any_graph, "global_desc") else 0
    fg_dim = int(getattr(any_graph, "functional_group_desc").numel()) if hasattr(any_graph, "functional_group_desc") else 0
    rows = []
    for smiles in df["IL_SMILES"].astype(str):
        graph = graph_cache.get(smiles)
        if graph is None:
            global_desc = np.zeros(global_dim, dtype=np.float32)
            fg_desc = np.zeros(fg_dim, dtype=np.float32)
        else:
            global_desc = (
                getattr(graph, "global_desc", torch.zeros(1, global_dim)).detach().cpu().numpy().reshape(-1)
                if global_dim
                else np.zeros(0, dtype=np.float32)
            )
            fg_desc = (
                getattr(graph, "functional_group_desc", torch.zeros(1, fg_dim)).detach().cpu().numpy().reshape(-1)
                if fg_dim
                else np.zeros(0, dtype=np.float32)
            )
        rows.append(np.concatenate([global_desc, fg_desc], axis=0))
    descriptor = np.asarray(rows, dtype=np.float32)
    cond = condition_basis(arrays["temperature"], arrays["pressure"], condition)
    return np.concatenate([descriptor, cond], axis=1)


def make_tree_model(model_name: str, seed: int, n_estimators: int, n_jobs: int):
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if model_name == "xgboost":
        if xgb is None:
            raise RuntimeError("xgboost is not installed in this Python environment.")
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if model_name == "lgbm":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed in this Python environment.")
        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    raise ValueError(model_name)


def compute_log_metrics(y_true: np.ndarray, pred_scaled: np.ndarray, mask: np.ndarray, target_scaler) -> pd.DataFrame:
    rows = []
    for j, prop in enumerate(PROPERTY_NAMES):
        valid = (mask[:, j] > 0) & np.isfinite(y_true[:, j]) & (y_true[:, j] > 0) & np.isfinite(pred_scaled[:, j])
        n = int(valid.sum())
        if n == 0:
            rows.append({"property": prop, "log_MAE": np.nan, "log_RMSE": np.nan, "log_R2": np.nan, "log_NMAE": np.nan})
            continue
        true_log = np.log(y_true[valid, j] + target_scaler.eps)
        pred_log = pred_scaled[valid, j] * target_scaler.stds[j] + target_scaler.means[j]
        err = pred_log - true_log
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        r2 = float(r2_score(true_log, pred_log)) if n > 1 else np.nan
        rows.append(
            {
                "property": prop,
                "log_MAE": mae,
                "log_RMSE": rmse,
                "log_R2": r2,
                "log_NMAE": float(mae / max(float(target_scaler.stds[j]), 1e-8)),
            }
        )
    frame = pd.DataFrame(rows)
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "property": "Average",
                        "log_MAE": float(frame["log_MAE"].mean()),
                        "log_RMSE": float(frame["log_RMSE"].mean()),
                        "log_R2": float(frame["log_R2"].mean()),
                        "log_NMAE": float(frame["log_NMAE"].mean()),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return frame


def prediction_long_df(df: pd.DataFrame, indices: Iterable[int], y: np.ndarray, pred_scaled: np.ndarray, mask: np.ndarray, target_scaler) -> pd.DataFrame:
    rows = []
    for local_i, idx in enumerate(indices):
        row = df.iloc[int(idx)]
        for j, prop in enumerate(PROPERTY_NAMES):
            if mask[local_i, j] <= 0:
                continue
            true_raw = float(y[local_i, j])
            pred_log = float(pred_scaled[local_i, j] * target_scaler.stds[j] + target_scaler.means[j])
            pred_raw = float(math.exp(pred_log) - target_scaler.eps)
            true_log = float(math.log(true_raw + target_scaler.eps))
            rows.append(
                {
                    "sample_id": int(idx),
                    "IL_Name": row["IL_Name"],
                    "IL_SMILES": row["IL_SMILES"],
                    "Temperature_K": row["Temperature_K"],
                    "Pressure_kPa": row["Pressure_kPa"],
                    "property": prop,
                    "y_true": true_raw,
                    "y_pred": pred_raw,
                    "y_true_log": true_log,
                    "y_pred_log": pred_log,
                    "absolute_error_log": abs(pred_log - true_log),
                    "split": "test",
                }
            )
    return pd.DataFrame(rows)


def write_result_tables(model_name: str, paths: Paths, metrics: pd.DataFrame, predictions: pd.DataFrame, manifest: dict) -> None:
    metric_dir = paths.output_root / "metrics" / model_name
    pred_dir = paths.output_root / "predictions" / model_name
    metric_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metric_dir / "test_metrics_log.csv", index=False)
    predictions.to_csv(pred_dir / "test_predictions.csv", index=False)
    save_json(manifest, metric_dir / "run_manifest.json")


def run_tree_baseline(model_name: str, args: argparse.Namespace, paths: Paths, df: pd.DataFrame, split: dict, arrays: dict, target_scaler, y_scaled, error_weights, features: np.ndarray) -> None:
    out_metrics = paths.output_root / "metrics" / model_name / "test_metrics_log.csv"
    if args.skip_existing and out_metrics.exists():
        print({"model": model_name, "status": "skip_existing", "metrics": str(out_metrics)})
        return
    pred_scaled = np.full_like(y_scaled, np.nan, dtype=np.float32)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)
    checkpoint_dir = paths.output_root / "checkpoints" / model_name
    if args.save_models:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for j, prop in enumerate(PROPERTY_NAMES):
        valid = (
            (arrays["mask"][train_idx, j] > 0)
            & np.isfinite(y_scaled[train_idx, j])
            & np.isfinite(features[train_idx]).all(axis=1)
        )
        prop_train_idx = train_idx[valid]
        if prop_train_idx.size == 0:
            continue
        model = make_tree_model(model_name, args.seed + j, args.tree_n_estimators, args.n_jobs)
        sample_weight = np.asarray(error_weights[prop_train_idx, j], dtype=np.float32)
        try:
            model.fit(features[prop_train_idx], y_scaled[prop_train_idx, j], sample_weight=sample_weight)
        except TypeError:
            model.fit(features[prop_train_idx], y_scaled[prop_train_idx, j])
        pred_scaled[:, j] = model.predict(features).astype(np.float32)
        if args.save_models:
            joblib.dump(model, checkpoint_dir / f"{prop}.joblib")
    test_mask = arrays.get("evaluation_mask", arrays["mask"])[test_idx]
    metrics = compute_log_metrics(arrays["y"][test_idx], pred_scaled[test_idx], test_mask, target_scaler)
    predictions = prediction_long_df(df, test_idx, arrays["y"][test_idx], pred_scaled[test_idx], test_mask, target_scaler)
    write_result_tables(
        model_name,
        paths,
        metrics,
        predictions,
        {
            "model": model_name,
            "kind": "tree",
            "data_dir": str(paths.data_dir),
            "split_path": str(paths.split_path),
            "target_scaler_mask": args.target_scaler_mask,
            "train_mask": "mask",
            "eval_mask": "evaluation_mask",
            "uses_label_weight": True,
            "features": "global_desc + optional functional_group_desc + condition_basis",
        },
    )
    print({"model": model_name, "status": "done", "metrics": str(out_metrics)})


class GraphBaselineDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        arrays: dict,
        graph_cache: dict,
        indices: list[int],
        y_scaled: np.ndarray,
        condition_features: np.ndarray,
        sample_weights: np.ndarray,
    ) -> None:
        self.df = df
        self.arrays = arrays
        self.graph_cache = graph_cache
        self.indices = [int(i) for i in indices]
        self.y_scaled = y_scaled.astype(np.float32)
        self.condition_features = condition_features.astype(np.float32)
        self.sample_weights = sample_weights.astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        smiles = str(self.df.iloc[idx]["IL_SMILES"])
        graph = self.graph_cache[smiles].clone()
        graph.y = torch.tensor(self.y_scaled[idx], dtype=torch.float32).view(1, -1)
        graph.mask = torch.tensor(self.arrays["mask"][idx], dtype=torch.float32).view(1, -1)
        graph.eval_mask = torch.tensor(self.arrays.get("evaluation_mask", self.arrays["mask"])[idx], dtype=torch.float32).view(1, -1)
        weights = self.sample_weights[idx].astype(np.float32)
        graph.error_weight = torch.tensor(weights, dtype=torch.float32).view(1, -1)
        graph.condition_features = torch.tensor(self.condition_features[idx], dtype=torch.float32).view(1, -1)
        graph.sample_id = torch.tensor([idx], dtype=torch.long)
        return graph


class GraphStructureDataset(Dataset):
    def __init__(self, smiles_list: list[str], graph_cache: dict, uid_by_smiles: dict[str, int]) -> None:
        self.smiles_list = list(smiles_list)
        self.graph_cache = graph_cache
        self.uid_by_smiles = uid_by_smiles

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, item: int):
        smiles = self.smiles_list[item]
        graph = self.graph_cache[smiles].clone()
        graph.graph_uid = torch.tensor([self.uid_by_smiles[smiles]], dtype=torch.long)
        return graph


def row_groups_by_smiles(df: pd.DataFrame, indices: list[int]) -> tuple[list[str], dict[str, np.ndarray]]:
    groups: dict[str, list[int]] = {}
    for idx in indices:
        smiles = str(df.iloc[int(idx)]["IL_SMILES"])
        groups.setdefault(smiles, []).append(int(idx))
    smiles_list = sorted(groups)
    return smiles_list, {smiles: np.asarray(groups[smiles], dtype=np.int64) for smiles in smiles_list}


class GraphRegressor(nn.Module):
    def __init__(self, model_name: str, in_dim: int, cond_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.model_name = model_name
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        for layer in range(layers):
            din = in_dim if layer == 0 else hidden_dim
            if model_name == "gcn":
                conv = GCNConv(din, hidden_dim)
            elif model_name == "gat":
                heads = 4
                conv = GATConv(din, hidden_dim // heads, heads=heads, concat=True, dropout=dropout)
            elif model_name in {"graphsage", "mpnn_concat"}:
                conv = SAGEConv(din, hidden_dim)
            elif model_name == "gin":
                conv = GINConv(
                    nn.Sequential(
                        nn.Linear(din, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
            else:
                raise ValueError(model_name)
            self.convs.append(conv)
        pooled_dim = hidden_dim * 2 if model_name == "mpnn_concat" else hidden_dim
        self.head = nn.Sequential(
            nn.Linear(pooled_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(PROPERTY_NAMES)),
        )

    def _edge_index(self, data):
        edge_index = data.edge_index
        if self.model_name != "mpnn_concat" or not hasattr(data, "fragment_id"):
            return edge_index
        frag = data.fragment_id
        keep = frag[edge_index[0]] == frag[edge_index[1]]
        return edge_index[:, keep]

    def _pool(self, x: torch.Tensor, data) -> torch.Tensor:
        if self.model_name != "mpnn_concat" or not hasattr(data, "fragment_id"):
            return global_mean_pool(x, data.batch)
        size = int(data.num_graphs)
        frag = data.fragment_id
        batch = data.batch
        parts = []
        for frag_id in (0, 1):
            mask = frag == frag_id
            if bool(mask.any()):
                parts.append(global_mean_pool(x[mask], batch[mask], size=size))
            else:
                parts.append(x.new_zeros((size, x.shape[-1])))
        return torch.cat(parts, dim=-1)

    def encode_graph(self, data) -> torch.Tensor:
        x = data.x.float()
        edge_index = self._edge_index(data)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        return self._pool(x, data)

    def predict_from_embedding(self, graph_embedding: torch.Tensor, condition_features: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([graph_embedding, condition_features.float()], dim=-1))

    def forward(self, data) -> torch.Tensor:
        pooled = self.encode_graph(data)
        return self.predict_from_embedding(pooled, data.condition_features)


def masked_weighted_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    active = mask > 0
    if not bool(active.any()):
        return pred.sum() * 0.0
    sq = (pred - target).pow(2) * mask * weight
    denom = mask.sum(dim=0).clamp_min(1.0)
    per_prop = sq.sum(dim=0) / denom
    prop_active = mask.sum(dim=0) > 0
    return per_prop[prop_active].mean()


@torch.no_grad()
def predict_graph_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, sample_ids = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).detach().cpu().numpy())
        sample_ids.append(batch.sample_id.detach().cpu().numpy().reshape(-1))
    return np.concatenate(sample_ids), np.concatenate(preds, axis=0)


def grouped_graph_loader(
    df: pd.DataFrame,
    graph_cache: dict,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> tuple[DataLoader, dict[int, np.ndarray]]:
    smiles_list, rows_by_smiles = row_groups_by_smiles(df, indices)
    missing = [smiles for smiles in smiles_list if smiles not in graph_cache]
    if missing:
        raise KeyError(f"Graph cache is missing {len(missing)} IL_SMILES; first missing: {missing[0]}")
    uid_by_smiles = {smiles: uid for uid, smiles in enumerate(smiles_list)}
    rows_by_uid = {uid_by_smiles[smiles]: rows for smiles, rows in rows_by_smiles.items()}
    dataset = GraphStructureDataset(smiles_list, graph_cache, uid_by_smiles)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, rows_by_uid


def expand_group_rows(uids: np.ndarray, rows_by_uid: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    row_parts = []
    position_parts = []
    for pos, uid_value in enumerate(uids):
        rows = rows_by_uid[int(uid_value)]
        row_parts.append(rows)
        position_parts.append(np.full(rows.shape[0], pos, dtype=np.int64))
    return np.concatenate(row_parts), np.concatenate(position_parts)


def train_grouped_graph_epoch(
    model: GraphRegressor,
    loader: DataLoader,
    rows_by_uid: dict[int, np.ndarray],
    device: torch.device,
    opt: torch.optim.Optimizer,
    y_scaled: torch.Tensor,
    mask: torch.Tensor,
    error_weights: torch.Tensor,
    condition_features: torch.Tensor,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        uids = batch.graph_uid.detach().cpu().numpy().reshape(-1)
        row_idx, graph_pos = expand_group_rows(uids, rows_by_uid)
        rows_t = torch.tensor(row_idx, dtype=torch.long, device=device)
        graph_pos_t = torch.tensor(graph_pos, dtype=torch.long, device=device)
        opt.zero_grad(set_to_none=True)
        graph_embedding = model.encode_graph(batch)
        pred = model.predict_from_embedding(graph_embedding[graph_pos_t], condition_features[rows_t])
        loss = masked_weighted_mse(pred, y_scaled[rows_t], mask[rows_t], error_weights[rows_t])
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan


@torch.no_grad()
def predict_grouped_graph_model(
    model: GraphRegressor,
    loader: DataLoader,
    rows_by_uid: dict[int, np.ndarray],
    device: torch.device,
    condition_features: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    sample_ids = []
    preds = []
    for batch in loader:
        batch = batch.to(device)
        uids = batch.graph_uid.detach().cpu().numpy().reshape(-1)
        row_idx, graph_pos = expand_group_rows(uids, rows_by_uid)
        rows_t = torch.tensor(row_idx, dtype=torch.long, device=device)
        graph_pos_t = torch.tensor(graph_pos, dtype=torch.long, device=device)
        graph_embedding = model.encode_graph(batch)
        pred = model.predict_from_embedding(graph_embedding[graph_pos_t], condition_features[rows_t])
        sample_ids.append(row_idx)
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(sample_ids), np.concatenate(preds, axis=0)


def evaluate_graph_score(model: nn.Module, loader: DataLoader, device: torch.device, arrays: dict, target_scaler) -> float:
    sample_ids, preds = predict_graph_model(model, loader, device)
    mask = arrays.get("evaluation_mask", arrays["mask"])[sample_ids]
    metrics = compute_log_metrics(arrays["y"][sample_ids], preds, mask, target_scaler)
    avg = metrics.loc[metrics["property"] == "Average"].iloc[0]
    return float(avg["log_R2"] - 0.2 * avg["log_NMAE"])


def evaluate_grouped_graph_score(
    model: GraphRegressor,
    loader: DataLoader,
    rows_by_uid: dict[int, np.ndarray],
    device: torch.device,
    arrays: dict,
    target_scaler,
    condition_features: torch.Tensor,
) -> float:
    sample_ids, preds = predict_grouped_graph_model(model, loader, rows_by_uid, device, condition_features)
    mask = arrays.get("evaluation_mask", arrays["mask"])[sample_ids]
    metrics = compute_log_metrics(arrays["y"][sample_ids], preds, mask, target_scaler)
    avg = metrics.loc[metrics["property"] == "Average"].iloc[0]
    return float(avg["log_R2"] - 0.2 * avg["log_NMAE"])


def run_graph_baseline(model_name: str, args: argparse.Namespace, paths: Paths, df: pd.DataFrame, split: dict, arrays: dict, target_scaler, y_scaled, error_weights, condition_features: np.ndarray, graph_cache: dict) -> None:
    out_metrics = paths.output_root / "metrics" / model_name / "test_metrics_log.csv"
    if args.skip_existing and out_metrics.exists():
        print({"model": model_name, "status": "skip_existing", "metrics": str(out_metrics)})
        return
    device = device_from_arg(args.device)
    first_graph = next(iter(graph_cache.values()))
    model = GraphRegressor(
        model_name,
        int(first_graph.x.shape[-1]),
        int(condition_features.shape[-1]),
        args.hidden_dim,
        args.layers,
        args.dropout,
    ).to(device)
    grouped_graphs = not args.no_grouped_graphs
    if grouped_graphs:
        train_loader, train_rows_by_uid = grouped_graph_loader(
            df, graph_cache, split["train"], args.batch_size, True, args.num_workers
        )
        val_loader, val_rows_by_uid = grouped_graph_loader(
            df, graph_cache, split["val"], args.batch_size, False, args.num_workers
        )
        test_loader, test_rows_by_uid = grouped_graph_loader(
            df, graph_cache, split["test"], args.batch_size, False, args.num_workers
        )
        y_scaled_t = torch.tensor(y_scaled.astype(np.float32), dtype=torch.float32, device=device)
        mask_t = torch.tensor(arrays["mask"].astype(np.float32), dtype=torch.float32, device=device)
        error_weights_t = torch.tensor(error_weights.astype(np.float32), dtype=torch.float32, device=device)
        condition_features_t = torch.tensor(condition_features.astype(np.float32), dtype=torch.float32, device=device)
    else:
        train_ds = GraphBaselineDataset(df, arrays, graph_cache, split["train"], y_scaled, condition_features, error_weights)
        val_ds = GraphBaselineDataset(df, arrays, graph_cache, split["val"], y_scaled, condition_features, error_weights)
        test_ds = GraphBaselineDataset(df, arrays, graph_cache, split["test"], y_scaled, condition_features, error_weights)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = -float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        if grouped_graphs:
            train_loss = train_grouped_graph_epoch(
                model,
                train_loader,
                train_rows_by_uid,
                device,
                opt,
                y_scaled_t,
                mask_t,
                error_weights_t,
                condition_features_t,
            )
            score = evaluate_grouped_graph_score(
                model,
                val_loader,
                val_rows_by_uid,
                device,
                arrays,
                target_scaler,
                condition_features_t,
            )
        else:
            model.train()
            losses = []
            for batch in train_loader:
                batch = batch.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(batch)
                loss = masked_weighted_mse(pred, batch.y, batch.mask, batch.error_weight)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(losses)) if losses else np.nan
            score = evaluate_graph_score(model, val_loader, device, arrays, target_scaler)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_score": score})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_dir = paths.output_root / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "args": vars(args),
            "best_val_score": best_score,
        },
        checkpoint_dir / "best_model.pt",
    )
    log_dir = paths.output_root / "logs" / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(log_dir / "train_log.csv", index=False)
    if grouped_graphs:
        sample_ids, pred_scaled = predict_grouped_graph_model(
            model, test_loader, test_rows_by_uid, device, condition_features_t
        )
    else:
        sample_ids, pred_scaled = predict_graph_model(model, test_loader, device)
    test_mask = arrays.get("evaluation_mask", arrays["mask"])[sample_ids]
    metrics = compute_log_metrics(arrays["y"][sample_ids], pred_scaled, test_mask, target_scaler)
    predictions = prediction_long_df(df, sample_ids, arrays["y"][sample_ids], pred_scaled, test_mask, target_scaler)
    write_result_tables(
        model_name,
        paths,
        metrics,
        predictions,
        {
            "model": model_name,
            "kind": "graph_neural_network",
            "data_dir": str(paths.data_dir),
            "split_path": str(paths.split_path),
            "target_scaler_mask": args.target_scaler_mask,
            "train_mask": "mask",
            "eval_mask": "evaluation_mask",
            "best_val_score": best_score,
            "epochs_completed": len(history),
            "graph_training_mode": "grouped_by_il_smiles" if grouped_graphs else "row_level",
        },
    )
    print({"model": model_name, "status": "done", "metrics": str(out_metrics), "best_val_score": best_score})


class SmilesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, arrays: dict, indices: list[int], y_scaled: np.ndarray, condition_features: np.ndarray, sample_weights: np.ndarray) -> None:
        self.df = df
        self.arrays = arrays
        self.indices = [int(i) for i in indices]
        self.y_scaled = y_scaled.astype(np.float32)
        self.condition_features = condition_features.astype(np.float32)
        self.sample_weights = sample_weights.astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        idx = self.indices[item]
        weights = self.sample_weights[idx].astype(np.float32)
        return {
            "sample_id": idx,
            "smiles": str(self.df.iloc[idx]["IL_SMILES"]),
            "y": self.y_scaled[idx].astype(np.float32),
            "mask": self.arrays["mask"][idx].astype(np.float32),
            "weight": weights.astype(np.float32),
            "condition_features": self.condition_features[idx].astype(np.float32),
        }


class UniqueSmilesDataset(Dataset):
    def __init__(self, smiles_list: list[str], uid_by_smiles: dict[str, int]) -> None:
        self.smiles_list = list(smiles_list)
        self.uid_by_smiles = uid_by_smiles

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, item: int) -> dict:
        smiles = self.smiles_list[item]
        return {"uid": self.uid_by_smiles[smiles], "smiles": smiles}


def grouped_smiles_loader(
    df: pd.DataFrame,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    collate_fn,
) -> tuple[torch.utils.data.DataLoader, dict[int, np.ndarray]]:
    smiles_list, rows_by_smiles = row_groups_by_smiles(df, indices)
    uid_by_smiles = {smiles: uid for uid, smiles in enumerate(smiles_list)}
    rows_by_uid = {uid_by_smiles[smiles]: rows for smiles, rows in rows_by_smiles.items()}
    dataset = UniqueSmilesDataset(smiles_list, uid_by_smiles)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader, rows_by_uid


def run_chemberta_baseline(model_name: str, args: argparse.Namespace, paths: Paths, df: pd.DataFrame, split: dict, arrays: dict, target_scaler, y_scaled, error_weights, condition_features: np.ndarray) -> None:
    out_metrics = paths.output_root / "metrics" / model_name / "test_metrics_log.csv"
    if args.skip_existing and out_metrics.exists():
        print({"model": model_name, "status": "skip_existing", "metrics": str(out_metrics)})
        return
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        save_json({"model": model_name, "status": "skipped", "reason": f"transformers import failed: {exc}"}, paths.output_root / "metrics" / model_name / "run_manifest.json")
        print({"model": model_name, "status": "skipped", "reason": str(exc)})
        return
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.chemberta_model, local_files_only=not args.allow_download)
        backbone = AutoModel.from_pretrained(args.chemberta_model, local_files_only=not args.allow_download)
    except Exception as exc:
        reason = f"ChemBERTa weights unavailable: {exc}"
        save_json({"model": model_name, "status": "skipped", "reason": reason}, paths.output_root / "metrics" / model_name / "run_manifest.json")
        print({"model": model_name, "status": "skipped", "reason": reason})
        return

    class ChemBertaRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            hidden = int(backbone.config.hidden_size)
            self.head = nn.Sequential(
                nn.Linear(hidden + condition_features.shape[-1], args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, len(PROPERTY_NAMES)),
            )

        def encode_smiles(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state[:, 0, :]

        def predict_from_embedding(self, smiles_embedding, cond):
            return self.head(torch.cat([smiles_embedding, cond], dim=-1))

        def forward(self, input_ids, attention_mask, cond):
            cls = self.encode_smiles(input_ids, attention_mask)
            return self.predict_from_embedding(cls, cond)

    def collate(batch):
        toks = tokenizer([b["smiles"] for b in batch], padding=True, truncation=True, max_length=256, return_tensors="pt")
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "condition_features": torch.tensor(np.stack([b["condition_features"] for b in batch]), dtype=torch.float32),
            "y": torch.tensor(np.stack([b["y"] for b in batch]), dtype=torch.float32),
            "mask": torch.tensor(np.stack([b["mask"] for b in batch]), dtype=torch.float32),
            "weight": torch.tensor(np.stack([b["weight"] for b in batch]), dtype=torch.float32),
            "sample_id": np.asarray([b["sample_id"] for b in batch], dtype=np.int64),
        }

    def collate_unique(batch):
        toks = tokenizer([b["smiles"] for b in batch], padding=True, truncation=True, max_length=256, return_tensors="pt")
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "uid": np.asarray([b["uid"] for b in batch], dtype=np.int64),
        }

    device = device_from_arg(args.device)
    model = ChemBertaRegressor().to(device)
    chemberta_batch_size = max(8, args.batch_size // 4)
    grouped_chemberta = not args.no_grouped_chemberta
    if grouped_chemberta:
        train_loader, train_rows_by_uid = grouped_smiles_loader(
            df, split["train"], chemberta_batch_size, True, collate_unique
        )
        val_loader, val_rows_by_uid = grouped_smiles_loader(
            df, split["val"], chemberta_batch_size, False, collate_unique
        )
        test_loader, test_rows_by_uid = grouped_smiles_loader(
            df, split["test"], chemberta_batch_size, False, collate_unique
        )
        y_scaled_t = torch.tensor(y_scaled.astype(np.float32), dtype=torch.float32, device=device)
        mask_t = torch.tensor(arrays["mask"].astype(np.float32), dtype=torch.float32, device=device)
        error_weights_t = torch.tensor(error_weights.astype(np.float32), dtype=torch.float32, device=device)
        condition_features_t = torch.tensor(condition_features.astype(np.float32), dtype=torch.float32, device=device)
    else:
        train_loader = torch.utils.data.DataLoader(SmilesDataset(df, arrays, split["train"], y_scaled, condition_features, error_weights), batch_size=chemberta_batch_size, shuffle=True, collate_fn=collate)
        val_loader = torch.utils.data.DataLoader(SmilesDataset(df, arrays, split["val"], y_scaled, condition_features, error_weights), batch_size=chemberta_batch_size, shuffle=False, collate_fn=collate)
        test_loader = torch.utils.data.DataLoader(SmilesDataset(df, arrays, split["test"], y_scaled, condition_features, error_weights), batch_size=chemberta_batch_size, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=min(args.lr, 2e-5), weight_decay=args.weight_decay)
    best_state, best_score, bad_epochs, history = None, -float("inf"), 0, []

    def predict_smiles(loader):
        model.eval()
        ids, preds = [], []
        with torch.no_grad():
            for batch in loader:
                pred = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["condition_features"].to(device))
                preds.append(pred.cpu().numpy())
                ids.append(batch["sample_id"])
        return np.concatenate(ids), np.concatenate(preds, axis=0)

    def predict_grouped_smiles(loader, rows_by_uid):
        model.eval()
        ids, preds = [], []
        with torch.no_grad():
            for batch in loader:
                uids = batch["uid"]
                row_idx, smiles_pos = expand_group_rows(uids, rows_by_uid)
                rows_t = torch.tensor(row_idx, dtype=torch.long, device=device)
                smiles_pos_t = torch.tensor(smiles_pos, dtype=torch.long, device=device)
                embeddings = model.encode_smiles(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                pred = model.predict_from_embedding(embeddings[smiles_pos_t], condition_features_t[rows_t])
                preds.append(pred.cpu().numpy())
                ids.append(row_idx)
        return np.concatenate(ids), np.concatenate(preds, axis=0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        if grouped_chemberta:
            for batch in train_loader:
                uids = batch["uid"]
                row_idx, smiles_pos = expand_group_rows(uids, train_rows_by_uid)
                rows_t = torch.tensor(row_idx, dtype=torch.long, device=device)
                smiles_pos_t = torch.tensor(smiles_pos, dtype=torch.long, device=device)
                opt.zero_grad(set_to_none=True)
                embeddings = model.encode_smiles(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                pred = model.predict_from_embedding(embeddings[smiles_pos_t], condition_features_t[rows_t])
                loss = masked_weighted_mse(pred, y_scaled_t[rows_t], mask_t[rows_t], error_weights_t[rows_t])
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            val_ids, val_pred = predict_grouped_smiles(val_loader, val_rows_by_uid)
        else:
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                pred = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["condition_features"].to(device))
                loss = masked_weighted_mse(pred, batch["y"].to(device), batch["mask"].to(device), batch["weight"].to(device))
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            val_ids, val_pred = predict_smiles(val_loader)
        val_mask = arrays.get("evaluation_mask", arrays["mask"])[val_ids]
        val_metrics = compute_log_metrics(arrays["y"][val_ids], val_pred, val_mask, target_scaler)
        avg = val_metrics.loc[val_metrics["property"] == "Average"].iloc[0]
        score = float(avg["log_R2"] - 0.2 * avg["log_NMAE"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else np.nan, "val_score": score})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_dir = paths.output_root / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "state_dict": model.state_dict(), "best_val_score": best_score}, checkpoint_dir / "best_model.pt")
    log_dir = paths.output_root / "logs" / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(log_dir / "train_log.csv", index=False)
    if grouped_chemberta:
        test_ids, pred_scaled = predict_grouped_smiles(test_loader, test_rows_by_uid)
    else:
        test_ids, pred_scaled = predict_smiles(test_loader)
    test_mask = arrays.get("evaluation_mask", arrays["mask"])[test_ids]
    metrics = compute_log_metrics(arrays["y"][test_ids], pred_scaled, test_mask, target_scaler)
    predictions = prediction_long_df(df, test_ids, arrays["y"][test_ids], pred_scaled, test_mask, target_scaler)
    write_result_tables(
        model_name,
        paths,
        metrics,
        predictions,
        {
            "model": model_name,
            "kind": "pretrained_smiles_transformer",
            "pretrained_model": args.chemberta_model,
            "allow_download": bool(args.allow_download),
            "data_dir": str(paths.data_dir),
            "split_path": str(paths.split_path),
            "target_scaler_mask": args.target_scaler_mask,
            "best_val_score": best_score,
            "epochs_completed": len(history),
            "smiles_training_mode": "grouped_by_il_smiles" if grouped_chemberta else "row_level",
        },
    )
    print({"model": model_name, "status": "done", "metrics": str(out_metrics), "best_val_score": best_score})


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    models = parse_model_list(args.models)
    print({"models": models, "paths": asdict(paths), "seed": args.seed})
    if args.dry_run:
        return
    for required in [paths.clean_csv, paths.arrays_path, paths.split_path, paths.graph_cache]:
        if not required.exists():
            raise FileNotFoundError(required)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    df = pd.read_csv(paths.clean_csv)
    split = load_json(paths.split_path)
    arrays, _, target_scaler, y_scaled, condition, error_weights = fit_context(paths, split, args.target_scaler_mask)
    condition_features = condition_basis(arrays["temperature"], arrays["pressure"], condition)
    graph_cache = safe_torch_load(paths.graph_cache)
    if not isinstance(graph_cache, dict) or not graph_cache:
        raise ValueError(f"Invalid graph cache: {paths.graph_cache}")
    features = build_fixed_features(df, arrays, condition, graph_cache)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "models": models,
            "data_dir": str(paths.data_dir),
            "clean_csv": str(paths.clean_csv),
            "arrays_path": str(paths.arrays_path),
            "split_path": str(paths.split_path),
            "graph_cache": str(paths.graph_cache),
            "target_scaler_mask": args.target_scaler_mask,
            "train_labels": "mask",
            "eval_labels": "evaluation_mask",
            "seed": args.seed,
        },
        paths.output_root / "run_manifest.json",
    )
    for model_name in models:
        if model_name in TREE_MODELS:
            run_tree_baseline(model_name, args, paths, df, split, arrays, target_scaler, y_scaled, error_weights, features)
        elif model_name in GRAPH_MODELS:
            run_graph_baseline(model_name, args, paths, df, split, arrays, target_scaler, y_scaled, error_weights, condition_features, graph_cache)
        elif model_name == "chemberta":
            run_chemberta_baseline(model_name, args, paths, df, split, arrays, target_scaler, y_scaled, error_weights, condition_features)
        else:  # pragma: no cover
            raise ValueError(model_name)


if __name__ == "__main__":
    main()
