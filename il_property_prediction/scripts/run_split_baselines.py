from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataset import ILPropertyDataset, safe_torch_load  # noqa: E402
from src.data.scaler import fit_scalers  # noqa: E402
from src.training.metrics import compute_log_space_metrics, compute_metrics  # noqa: E402
from src.utils.io import save_json, save_log_metrics_csv, save_metrics_csv  # noqa: E402


PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]

DEFAULT_CASES = {
    "random_il_level": "data/processed_ilthermo_interpolated/splits/il_level_seed42.json",
    "property_balanced_il_level": "data/processed_ilthermo_interpolated/splits/il_level_property_balanced_seed42.json",
    "ion_family": "data/processed/splits/il_level_family_pair_seed42.json",
}

CASE_LABELS = {
    "random_point": "Random-point",
    "random_il_level": "Random IL-level",
    "property_balanced_il_level": "Property-balanced IL-level",
    "ion_family": "Ion-family",
}

MODEL_LABELS = {
    "rf": "RF",
    "xgboost": "XGBoost",
    "lgbm": "LGBM",
    "chemberta": "ChemBERTa",
    "mpnn_concat": "MPNN-Concat",
    "gcn": "GCN",
    "gat": "GAT",
    "graphsage": "GraphSAGE",
    "gin": "GIN",
    "mipgraph": "MIPGraph",
}

GRAPH_BACKBONES = {"gcn", "gat", "graphsage", "gin"}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_DIR / p).resolve()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as arrays:
        return {key: arrays[key] for key in arrays.files}


def load_split(path: Path) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    return {key: [int(idx) for idx in value] for key, value in split.items()}


def condition_basis(condition: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    t_norm = condition[:, 0]
    p_norm = condition[:, 1]
    t_kelvin = np.nan_to_num(temperature.astype(np.float32), nan=298.15)
    t_kelvin = np.clip(t_kelvin, 1.0, None)
    log_t_ratio = np.log(t_kelvin / 298.15)
    inverse_t_ratio = 298.15 / t_kelvin - 1.0
    return np.stack(
        [t_norm, p_norm, t_norm * t_norm, t_norm * p_norm, log_t_ratio, inverse_t_ratio],
        axis=-1,
    ).astype(np.float32)


@dataclass
class BaselineData:
    df: pd.DataFrame
    arrays: dict[str, np.ndarray]
    split: dict[str, list[int]]
    target_scaler: Any
    y_scaled: np.ndarray
    condition: np.ndarray
    weights: np.ndarray
    base_features: np.ndarray
    feature_description: str

    @property
    def y_true_log(self) -> np.ndarray:
        y = self.arrays["y"].astype(np.float64)
        valid = np.isfinite(y) & (y > 0)
        out = np.zeros_like(y, dtype=np.float32)
        out[valid] = np.log(y[valid] + float(self.target_scaler.eps)).astype(np.float32)
        return out


def build_descriptor_features(df: pd.DataFrame, graph_cache_path: Path, include_functional_groups: bool) -> np.ndarray:
    graph_cache = safe_torch_load(graph_cache_path)
    by_smiles: dict[str, np.ndarray] = {}
    for smiles in df["IL_SMILES"].astype(str).unique():
        if smiles not in graph_cache:
            raise KeyError(f"Graph cache missing IL_SMILES: {smiles}")
        graph = graph_cache[smiles]
        parts = [graph.global_desc.detach().cpu().numpy().reshape(-1).astype(np.float32)]
        if include_functional_groups:
            if hasattr(graph, "functional_group_desc"):
                parts.append(graph.functional_group_desc.detach().cpu().numpy().reshape(-1).astype(np.float32))
            else:
                parts.append(np.zeros(80, dtype=np.float32))
        by_smiles[smiles] = np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0)
    return np.vstack([by_smiles[str(smiles)] for smiles in df["IL_SMILES"]]).astype(np.float32)


def prepare_data(
    data_dir: Path,
    split_path: Path,
    graph_cache_path: Path,
    include_functional_groups: bool,
) -> BaselineData:
    clean_csv = data_dir / "il_multiprop_clean.csv"
    arrays_path = data_dir / "il_multiprop_arrays.npz"
    df = pd.read_csv(clean_csv)
    arrays = load_npz(arrays_path)
    split = load_split(split_path)
    _, target_scaler, y_scaled, condition, weights = fit_scalers(
        arrays,
        split["train"],
        target_mask_key="mask",
    )
    descriptors = build_descriptor_features(df, graph_cache_path, include_functional_groups)
    basis = condition_basis(condition, arrays["temperature"])
    features = np.concatenate([descriptors, basis], axis=1)
    desc_text = "global_desc"
    if include_functional_groups:
        desc_text += " + functional_group_desc"
    desc_text += " + condition_basis"
    return BaselineData(
        df=df,
        arrays=arrays,
        split=split,
        target_scaler=target_scaler,
        y_scaled=y_scaled,
        condition=condition,
        weights=weights,
        base_features=features.astype(np.float32),
        feature_description=desc_text,
    )


def make_tree_model(name: str, seed: int, n_jobs: int):
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            verbosity=0,
        )
    if name == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
    raise ValueError(f"Unsupported tree model: {name}")


def fit_property_models(
    model_name: str,
    features: np.ndarray,
    data: BaselineData,
    seed: int,
    n_jobs: int,
    model_factory,
) -> np.ndarray:
    pred_scaled = np.full_like(data.y_scaled, np.nan, dtype=np.float32)
    train_idx = np.asarray(data.split["train"], dtype=np.int64)
    for prop_idx, prop_name in enumerate(PROPERTY_NAMES):
        valid = (
            (data.arrays["mask"][train_idx, prop_idx] > 0)
            & np.isfinite(data.y_scaled[train_idx, prop_idx])
            & np.isfinite(features[train_idx]).all(axis=1)
        )
        prop_train_idx = train_idx[valid]
        if len(prop_train_idx) < 2:
            print(f"[{model_name}] skip {prop_name}: only {len(prop_train_idx)} training labels")
            continue
        model = model_factory(seed + prop_idx, n_jobs)
        sample_weight = np.asarray(data.weights[prop_train_idx, prop_idx], dtype=np.float32)
        model.fit(
            features[prop_train_idx],
            data.y_scaled[prop_train_idx, prop_idx],
            sample_weight=sample_weight,
        )
        pred_scaled[:, prop_idx] = model.predict(features).astype(np.float32)
    return pred_scaled


def predictions_to_metrics(
    data: BaselineData,
    pred_scaled: np.ndarray,
    split_name: str = "test",
) -> tuple[dict[str, Any], pd.DataFrame]:
    pred_log = pred_scaled * data.target_scaler.stds[None, :] + data.target_scaler.means[None, :]
    pred_raw = np.exp(pred_log) - float(data.target_scaler.eps)
    y_raw = data.arrays["y"]
    y_true_log = data.y_true_log
    eval_idx = np.asarray(data.split[split_name], dtype=np.int64)
    eval_mask_full = data.arrays.get("evaluation_mask", data.arrays["mask"])
    eval_mask = np.zeros_like(eval_mask_full, dtype=np.float32)
    eval_mask[eval_idx] = eval_mask_full[eval_idx]
    raw_metrics = compute_metrics(y_raw, pred_raw, eval_mask, PROPERTY_NAMES)
    log_metrics = compute_log_space_metrics(
        y_true_log,
        pred_log,
        eval_mask,
        PROPERTY_NAMES,
        normalization_stds=data.target_scaler.stds,
    )
    metrics = copy.deepcopy(raw_metrics)
    metrics["log_space"] = log_metrics

    rows: list[dict[str, Any]] = []
    for idx in eval_idx:
        for prop_idx, prop_name in enumerate(PROPERTY_NAMES):
            if eval_mask_full[idx, prop_idx] <= 0:
                continue
            if not np.isfinite(pred_log[idx, prop_idx]):
                continue
            rows.append(
                {
                    "sample_id": int(data.df.iloc[idx].get("sample_id", idx)),
                    "IL_Name": data.df.iloc[idx]["IL_Name"],
                    "IL_SMILES": data.df.iloc[idx]["IL_SMILES"],
                    "Temperature_K": data.df.iloc[idx]["Temperature_K"],
                    "Pressure_kPa": data.df.iloc[idx]["Pressure_kPa"],
                    "property": prop_name,
                    "y_true": float(y_raw[idx, prop_idx]),
                    "y_pred": float(pred_raw[idx, prop_idx]),
                    "y_true_log": float(y_true_log[idx, prop_idx]),
                    "y_pred_log": float(pred_log[idx, prop_idx]),
                    "absolute_error_log": float(abs(pred_log[idx, prop_idx] - y_true_log[idx, prop_idx])),
                    "split": split_name,
                }
            )
    return metrics, pd.DataFrame(rows)


def write_model_outputs(
    output_root: Path,
    case_name: str,
    model_name: str,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    metric_dir = output_root / "metrics" / case_name / model_name
    pred_dir = output_root / "predictions" / case_name / model_name
    metric_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, metric_dir / "test_metrics.json")
    save_log_metrics_csv(metrics, metric_dir / "test_metrics_log.csv", PROPERTY_NAMES)
    save_metrics_csv(metrics, metric_dir / "test_metrics_raw.csv", PROPERTY_NAMES)
    save_json(manifest, metric_dir / "run_manifest.json")
    predictions.to_csv(pred_dir / "test_predictions.csv", index=False)


def run_tree_baseline(
    model_name: str,
    data: BaselineData,
    output_root: Path,
    case_name: str,
    split_path: Path,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    pred_scaled = fit_property_models(
        model_name,
        data.base_features,
        data,
        seed,
        n_jobs,
        lambda model_seed, jobs: make_tree_model(model_name, model_seed, jobs),
    )
    metrics, predictions = predictions_to_metrics(data, pred_scaled)
    manifest = {
        "model": model_name,
        "kind": "tree",
        "split_case": case_name,
        "split_path": str(split_path),
        "target_scaler_mask": "mask",
        "train_mask": "mask",
        "eval_mask": "evaluation_mask",
        "uses_label_weight": True,
        "features": data.feature_description,
        "seed": seed,
    }
    write_model_outputs(output_root, case_name, model_name, metrics, predictions, manifest)
    return metrics


def chemberta_embeddings(
    df: pd.DataFrame,
    model_name: str,
    batch_size: int,
    device: torch.device,
    local_files_only: bool,
    cache_path: Path | None,
) -> np.ndarray:
    if cache_path and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        smiles = cached["smiles"].astype(str).tolist()
        values = cached["embeddings"].astype(np.float32)
        by_smiles = dict(zip(smiles, values))
        return np.vstack([by_smiles[str(s)] for s in df["IL_SMILES"]]).astype(np.float32)

    from transformers import AutoModel, AutoTokenizer

    unique_smiles = df["IL_SMILES"].astype(str).unique().tolist()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    model.to(device)
    model.eval()
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(unique_smiles), batch_size):
            batch_smiles = unique_smiles[start : start + batch_size]
            encoded = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            cls = output.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
            embs.append(cls)
            print(f"[chemberta] embedded {min(start + batch_size, len(unique_smiles))}/{len(unique_smiles)}")
    values = np.vstack(embs).astype(np.float32)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, smiles=np.asarray(unique_smiles), embeddings=values)
    by_smiles = dict(zip(unique_smiles, values))
    return np.vstack([by_smiles[str(s)] for s in df["IL_SMILES"]]).astype(np.float32)


def run_chemberta_baseline(
    data: BaselineData,
    output_root: Path,
    case_name: str,
    split_path: Path,
    seed: int,
    n_jobs: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_path = output_root / "cache" / f"chemberta_{args.chemberta_model.replace('/', '__')}.npz"
    embeddings = chemberta_embeddings(
        data.df,
        args.chemberta_model,
        args.chemberta_batch_size,
        device,
        local_files_only=not args.allow_chemberta_download,
        cache_path=cache_path,
    )
    features = np.concatenate(
        [embeddings, condition_basis(data.condition, data.arrays["temperature"])],
        axis=1,
    ).astype(np.float32)
    pred_scaled = fit_property_models(
        "chemberta",
        features,
        data,
        seed,
        n_jobs,
        lambda model_seed, jobs: RandomForestRegressor(
            n_estimators=100,
            random_state=model_seed,
            n_jobs=jobs,
        ),
    )
    metrics, predictions = predictions_to_metrics(data, pred_scaled)
    manifest = {
        "model": "chemberta",
        "kind": "pretrained_smiles_transformer_rf",
        "pretrained_model": args.chemberta_model,
        "split_case": case_name,
        "split_path": str(split_path),
        "target_scaler_mask": "mask",
        "train_mask": "mask",
        "eval_mask": "evaluation_mask",
        "downstream_predictor": "RandomForestRegressor(n_estimators=100)",
        "features": "ChemBERTa [CLS] embedding + condition_basis",
        "seed": seed,
    }
    write_model_outputs(output_root, case_name, "chemberta", metrics, predictions, manifest)
    return metrics


class MPNNConcat(torch.nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.node_encoder = torch.nn.Linear(node_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 + 6, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, len(PROPERTY_NAMES)),
        )

    @staticmethod
    def _condition_basis(condition: torch.Tensor, raw_condition: torch.Tensor) -> torch.Tensor:
        t_norm = condition[:, 0]
        p_norm = condition[:, 1]
        t_kelvin = torch.nan_to_num(raw_condition[:, 0], nan=298.15).clamp_min(1.0)
        log_t_ratio = torch.log(t_kelvin / 298.15)
        inverse_t_ratio = 298.15 / t_kelvin - 1.0
        return torch.stack(
            [t_norm, p_norm, t_norm.square(), t_norm * p_norm, log_t_ratio, inverse_t_ratio],
            dim=-1,
        )

    def encode_graph(self, batch) -> torch.Tensor:
        from torch_geometric.nn import global_mean_pool

        x = torch.relu(self.node_encoder(batch.x))
        src, dst = batch.edge_index
        intra = batch.fragment_id[src] == batch.fragment_id[dst]
        edge_index = batch.edge_index[:, intra]
        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index))
        graph_count = int(batch.num_graphs)
        frag = batch.fragment_id
        cation_mask = frag == 0
        anion_mask = frag == 1
        cation = global_mean_pool(x[cation_mask], batch.batch[cation_mask], size=graph_count)
        anion = global_mean_pool(x[anion_mask], batch.batch[anion_mask], size=graph_count)
        return torch.cat([cation, anion], dim=-1)

    def predict_from_embedding(self, graph_embedding: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([graph_embedding, basis], dim=-1))

    def forward(self, batch):
        graph_embedding = self.encode_graph(batch)
        basis = self._condition_basis(batch.condition.view(-1, 2), batch.raw_condition.view(-1, 2))
        return self.predict_from_embedding(graph_embedding, basis)


class GraphBackboneRegressor(torch.nn.Module):
    """Plain whole-graph GNN baseline with a swappable message-passing layer."""

    def __init__(self, backbone: str, node_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

        self.backbone = backbone
        self.node_encoder = torch.nn.Linear(node_dim, hidden_dim)
        if backbone == "gcn":
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif backbone == "gat":
            self.conv1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        elif backbone == "graphsage":
            self.conv1 = SAGEConv(hidden_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        elif backbone == "gin":
            self.conv1 = GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.conv2 = GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                )
            )
        else:
            raise ValueError(f"Unsupported graph backbone: {backbone}")
        self.dropout = torch.nn.Dropout(dropout)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + 6, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, len(PROPERTY_NAMES)),
        )

    @staticmethod
    def _condition_basis(condition: torch.Tensor, raw_condition: torch.Tensor) -> torch.Tensor:
        return MPNNConcat._condition_basis(condition, raw_condition)

    def encode_graph(self, batch) -> torch.Tensor:
        from torch_geometric.nn import global_mean_pool

        x = torch.relu(self.node_encoder(batch.x))
        x = torch.relu(self.conv1(x, batch.edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, batch.edge_index))
        return global_mean_pool(x, batch.batch, size=int(batch.num_graphs))

    def predict_from_embedding(self, graph_embedding: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([graph_embedding, basis], dim=-1))

    def forward(self, batch):
        graph_embedding = self.encode_graph(batch)
        basis = self._condition_basis(batch.condition.view(-1, 2), batch.raw_condition.view(-1, 2))
        return self.predict_from_embedding(graph_embedding, basis)


def masked_weighted_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    valid_weight = mask * weight
    return ((pred - target).pow(2) * valid_weight).sum() / valid_weight.sum().clamp_min(1e-8)


def make_loader(
    data: BaselineData,
    data_dir: Path,
    graph_cache_path: Path,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
):
    from torch_geometric.loader import DataLoader

    dataset = ILPropertyDataset(
        data_dir / "il_multiprop_clean.csv",
        data_dir / "il_multiprop_arrays.npz",
        graph_cache_path,
        indices,
        data.condition,
        data.y_scaled,
        data.weights,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_mpnn(
    model: torch.nn.Module,
    loader,
    data: BaselineData,
    device: torch.device,
    split_name: str,
) -> tuple[dict[str, Any], pd.DataFrame, float]:
    pred_scaled = np.full_like(data.y_scaled, np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).detach().cpu().numpy().astype(np.float32)
            idx = batch.sample_id.detach().cpu().numpy().astype(np.int64)
            pred_scaled[idx] = pred
    metrics, predictions = predictions_to_metrics(data, pred_scaled, split_name=split_name)
    log_metrics = metrics["log_space"]
    score = float(log_metrics["macro_log_R2"] - 0.2 * log_metrics["macro_log_NMAE"])
    return metrics, predictions, score


def run_mpnn_concat_baseline(
    data: BaselineData,
    data_dir: Path,
    graph_cache_path: Path,
    output_root: Path,
    case_name: str,
    split_path: Path,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    sample_graph = next(iter(safe_torch_load(graph_cache_path).values()))
    model = MPNNConcat(int(sample_graph.x.shape[1]), args.mpnn_hidden_dim, args.mpnn_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mpnn_lr)
    train_loader = make_loader(data, data_dir, graph_cache_path, data.split["train"], args.mpnn_batch_size, True)
    val_loader = make_loader(data, data_dir, graph_cache_path, data.split["val"], args.mpnn_batch_size, False)
    test_loader = make_loader(data, data_dir, graph_cache_path, data.split["test"], args.mpnn_batch_size, False)

    best_state = None
    best_score = -math.inf
    best_epoch = -1
    bad_epochs = 0
    log_rows: list[dict[str, Any]] = []
    for epoch in range(args.mpnn_epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = masked_weighted_mse(pred, batch.y.view(-1, len(PROPERTY_NAMES)), batch.mask.view(-1, len(PROPERTY_NAMES)), batch.error_weight.view(-1, len(PROPERTY_NAMES)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.mpnn_grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics, _, val_score = evaluate_mpnn(model, val_loader, data, device, "val")
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else np.nan,
                "val_score": val_score,
                "val_macro_log_R2": val_metrics["log_space"]["macro_log_R2"],
                "val_macro_log_NMAE": val_metrics["log_space"]["macro_log_NMAE"],
            }
        )
        print(f"[mpnn_concat] epoch={epoch:03d} loss={log_rows[-1]['train_loss']:.5f} val_score={val_score:.5f}")
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.mpnn_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, predictions, _ = evaluate_mpnn(model, test_loader, data, device, "test")
    metric_dir = output_root / "metrics" / case_name / "mpnn_concat"
    checkpoint_dir = output_root / "checkpoints" / case_name / "mpnn_concat"
    log_dir = output_root / "logs" / case_name / "mpnn_concat"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
    pd.DataFrame(log_rows).to_csv(log_dir / "train_log.csv", index=False)
    manifest = {
        "model": "mpnn_concat",
        "kind": "graph_neural_network",
        "split_case": case_name,
        "split_path": str(split_path),
        "target_scaler_mask": "mask",
        "train_mask": "mask",
        "eval_mask": "evaluation_mask",
        "best_val_score": best_score,
        "best_epoch": best_epoch,
        "epochs_completed": len(log_rows),
        "hidden_dim": args.mpnn_hidden_dim,
        "batch_size": args.mpnn_batch_size,
        "learning_rate": args.mpnn_lr,
        "features": "separate fragment GCN pooling + condition_basis",
        "seed": seed,
    }
    write_model_outputs(output_root, case_name, "mpnn_concat", metrics, predictions, manifest)
    save_json(manifest, metric_dir / "run_manifest.json")
    return metrics


def run_graph_backbone_baseline(
    model_name: str,
    data: BaselineData,
    data_dir: Path,
    graph_cache_path: Path,
    output_root: Path,
    case_name: str,
    split_path: Path,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    sample_graph = next(iter(safe_torch_load(graph_cache_path).values()))
    model = GraphBackboneRegressor(
        model_name,
        int(sample_graph.x.shape[1]),
        args.mpnn_hidden_dim,
        args.mpnn_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mpnn_lr)
    train_loader = make_loader(data, data_dir, graph_cache_path, data.split["train"], args.mpnn_batch_size, True)
    val_loader = make_loader(data, data_dir, graph_cache_path, data.split["val"], args.mpnn_batch_size, False)
    test_loader = make_loader(data, data_dir, graph_cache_path, data.split["test"], args.mpnn_batch_size, False)

    best_state = None
    best_score = -math.inf
    best_epoch = -1
    bad_epochs = 0
    log_rows: list[dict[str, Any]] = []
    for epoch in range(args.mpnn_epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = masked_weighted_mse(
                pred,
                batch.y.view(-1, len(PROPERTY_NAMES)),
                batch.mask.view(-1, len(PROPERTY_NAMES)),
                batch.error_weight.view(-1, len(PROPERTY_NAMES)),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.mpnn_grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics, _, val_score = evaluate_mpnn(model, val_loader, data, device, "val")
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else np.nan,
                "val_score": val_score,
                "val_macro_log_R2": val_metrics["log_space"]["macro_log_R2"],
                "val_macro_log_NMAE": val_metrics["log_space"]["macro_log_NMAE"],
            }
        )
        print(f"[{model_name}] epoch={epoch:03d} loss={log_rows[-1]['train_loss']:.5f} val_score={val_score:.5f}")
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.mpnn_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, predictions, _ = evaluate_mpnn(model, test_loader, data, device, "test")
    metric_dir = output_root / "metrics" / case_name / model_name
    checkpoint_dir = output_root / "checkpoints" / case_name / model_name
    log_dir = output_root / "logs" / case_name / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
    pd.DataFrame(log_rows).to_csv(log_dir / "train_log.csv", index=False)
    manifest = {
        "model": model_name,
        "kind": "graph_neural_network",
        "split_case": case_name,
        "split_path": str(split_path),
        "target_scaler_mask": "mask",
        "train_mask": "mask",
        "eval_mask": "evaluation_mask",
        "best_val_score": best_score,
        "best_epoch": best_epoch,
        "epochs_completed": len(log_rows),
        "hidden_dim": args.mpnn_hidden_dim,
        "batch_size": args.mpnn_batch_size,
        "learning_rate": args.mpnn_lr,
        "features": f"whole ion-pair graph {MODEL_LABELS[model_name]} pooling + condition_basis",
        "seed": seed,
    }
    write_model_outputs(output_root, case_name, model_name, metrics, predictions, manifest)
    save_json(manifest, metric_dir / "run_manifest.json")
    return metrics


def read_average_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df.loc[df["property"].astype(str).str.lower() == "average"]
    if row.empty:
        return None
    item = row.iloc[0]
    return {
        "macro_log_MAE": float(item["log_MAE"]),
        "macro_log_RMSE": float(item["log_RMSE"]),
        "macro_log_R2": float(item["log_R2"]),
        "macro_log_NMAE": float(item["log_NMAE"]),
    }


def collect_summary(output_root: Path, random_point_root: Path, split_strategy_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    random_summary = random_point_root / "baseline_metrics_summary.csv"
    if random_summary.exists():
        df = pd.read_csv(random_summary, header=[0, 1], index_col=0)
        if "Average" in df.index:
            for model_name in MODEL_LABELS:
                if model_name in df["log_R2"].columns and model_name in df["log_NMAE"].columns:
                    rows.append(
                        {
                            "case": "random_point",
                            "split": CASE_LABELS["random_point"],
                            "model": model_name,
                            "model_label": MODEL_LABELS[model_name],
                            "macro_log_R2": float(df.loc["Average", ("log_R2", model_name)]),
                            "macro_log_NMAE": float(df.loc["Average", ("log_NMAE", model_name)]),
                            "macro_log_RMSE": float(df.loc["Average", ("log_RMSE", model_name)]),
                        }
                    )
    for case_name in DEFAULT_CASES:
        for model_name in [
            "rf",
            "xgboost",
            "lgbm",
            "chemberta",
            "mpnn_concat",
            "gcn",
            "gat",
            "graphsage",
            "gin",
        ]:
            avg = read_average_metrics(output_root / "metrics" / case_name / model_name / "test_metrics_log.csv")
            if avg is None:
                continue
            rows.append(
                {
                    "case": case_name,
                    "split": CASE_LABELS[case_name],
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "macro_log_R2": avg["macro_log_R2"],
                    "macro_log_NMAE": avg["macro_log_NMAE"],
                    "macro_log_RMSE": avg["macro_log_RMSE"],
                }
            )
        mipgraph_avg = read_average_metrics(split_strategy_root / "metrics" / f"{case_name if case_name != 'ion_family' else 'il_level_family_pair_seed42'}" / "test_metrics_log.csv")
        if mipgraph_avg is None:
            mapped = {
                "random_il_level": "il_level_random_seed42",
                "property_balanced_il_level": "il_level_property_balanced_seed42",
                "ion_family": "il_level_family_pair_seed42",
            }[case_name]
            mipgraph_avg = read_average_metrics(split_strategy_root / "metrics" / mapped / "test_metrics_log.csv")
        if mipgraph_avg is not None:
            rows.append(
                {
                    "case": case_name,
                    "split": CASE_LABELS[case_name],
                    "model": "mipgraph",
                    "model_label": MODEL_LABELS["mipgraph"],
                    "macro_log_R2": mipgraph_avg["macro_log_R2"],
                    "macro_log_NMAE": mipgraph_avg["macro_log_NMAE"],
                    "macro_log_RMSE": mipgraph_avg["macro_log_RMSE"],
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    order_cases = ["random_point", "random_il_level", "property_balanced_il_level", "ion_family"]
    order_models = [
        "rf",
        "xgboost",
        "lgbm",
        "chemberta",
        "mpnn_concat",
        "gcn",
        "gat",
        "graphsage",
        "gin",
        "mipgraph",
    ]
    summary["case_order"] = summary["case"].map({name: i for i, name in enumerate(order_cases)})
    summary["model_order"] = summary["model"].map({name: i for i, name in enumerate(order_models)})
    return summary.sort_values(["case_order", "model_order"]).drop(columns=["case_order", "model_order"])


def write_summary_table(summary: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "split_baseline_summary.csv", index=False)
    order_cases = ["random_point", "random_il_level", "property_balanced_il_level", "ion_family"]
    order_models = [
        "rf",
        "xgboost",
        "lgbm",
        "chemberta",
        "mpnn_concat",
        "gcn",
        "gat",
        "graphsage",
        "gin",
        "mipgraph",
    ]
    rows: list[dict[str, str]] = []
    for case_name in order_cases:
        row = {"Split": CASE_LABELS[case_name]}
        sub = summary[summary["case"] == case_name].set_index("model")
        for model_name in order_models:
            if model_name in sub.index:
                item = sub.loc[model_name]
                row[MODEL_LABELS[model_name]] = f"{item['macro_log_R2']:.4f}/{item['macro_log_NMAE']:.4f}"
            else:
                row[MODEL_LABELS[model_name]] = "--"
        rows.append(row)
    table_df = pd.DataFrame(rows)
    table_df.to_csv(output_root / "split_baseline_summary_wide.csv", index=False)
    columns = ["Split"] + [MODEL_LABELS[m] for m in order_models]
    with (output_root / "split_baseline_summary_table.tex").open("w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\caption{Macro-averaged split-strategy comparison of baseline models and MIPGraph. Each entry reports log-space $R^2$/NMAE on the test set.}\n")
        f.write("  \\label{tab:split_baseline_macro}\n")
        f.write("  \\scriptsize\n")
        f.write("  \\resizebox{\\textwidth}{!}{%\n")
        f.write("  \\begin{tabular}{l" + "c" * (len(columns) - 1) + "}\n")
        f.write("    \\hline\n")
        f.write("    " + " & ".join(columns) + " \\\\\n")
        f.write("    \\hline\n")
        for _, row in table_df.iterrows():
            f.write("    " + " & ".join(str(row[col]) for col in columns) + " \\\\\n")
        f.write("    \\hline\n")
        f.write("  \\end{tabular}%\n")
        f.write("  }\n")
        f.write("\\end{table*}\n")


def parse_cases(text: str) -> dict[str, Path]:
    selected = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in selected if item not in DEFAULT_CASES]
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}. Choose from {sorted(DEFAULT_CASES)}")
    return {name: resolve(DEFAULT_CASES[name]) for name in selected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-free split baselines for IL property prediction.")
    parser.add_argument("--data-dir", default="data/processed_ilthermo_interpolated")
    parser.add_argument("--graph-cache", default="data/processed/graph_cache_fg.pt")
    parser.add_argument("--output-root", default="outputs/split_baseline_comparison_seed42")
    parser.add_argument("--random-point-root", default="outputs/baseline_comparison_random_point_seed42")
    parser.add_argument("--split-strategy-root", default="outputs/split_strategy_comparison_seed42")
    parser.add_argument("--cases", default="random_il_level,property_balanced_il_level,ion_family")
    parser.add_argument("--models", default="rf,xgboost,lgbm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-functional-groups", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--chemberta-model", default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--chemberta-batch-size", type=int, default=32)
    parser.add_argument("--allow-chemberta-download", action="store_true")
    parser.add_argument("--mpnn-epochs", type=int, default=100)
    parser.add_argument("--mpnn-patience", type=int, default=20)
    parser.add_argument("--mpnn-batch-size", type=int, default=128)
    parser.add_argument("--mpnn-hidden-dim", type=int, default=64)
    parser.add_argument("--mpnn-dropout", type=float, default=0.1)
    parser.add_argument("--mpnn-lr", type=float, default=0.001)
    parser.add_argument("--mpnn-grad-clip", type=float, default=5.0)
    args = parser.parse_args()

    data_dir = resolve(args.data_dir)
    graph_cache_path = resolve(args.graph_cache)
    output_root = resolve(args.output_root)
    random_point_root = resolve(args.random_point_root)
    split_strategy_root = resolve(args.split_strategy_root)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    supported = {"rf", "xgboost", "lgbm", "chemberta", "mpnn_concat", *GRAPH_BACKBONES}
    unknown_models = [model for model in models if model not in supported]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}. Choose from {sorted(supported)}")

    if not args.summary_only:
        for case_name, split_path in parse_cases(args.cases).items():
            print(f"[case] {case_name}: {split_path}")
            data = prepare_data(
                data_dir,
                split_path,
                graph_cache_path,
                include_functional_groups=not args.no_functional_groups,
            )
            for model_name in models:
                metric_path = output_root / "metrics" / case_name / model_name / "test_metrics_log.csv"
                if args.skip_existing and metric_path.exists():
                    print(f"[skip] {case_name}/{model_name}: {metric_path}")
                    continue
                print(f"[run] {case_name}/{model_name}")
                if model_name in {"rf", "xgboost", "lgbm"}:
                    run_tree_baseline(model_name, data, output_root, case_name, split_path, args.seed, args.n_jobs)
                elif model_name == "chemberta":
                    run_chemberta_baseline(data, output_root, case_name, split_path, args.seed, args.n_jobs, args)
                elif model_name == "mpnn_concat":
                    run_mpnn_concat_baseline(data, data_dir, graph_cache_path, output_root, case_name, split_path, args.seed, args)
                elif model_name in GRAPH_BACKBONES:
                    run_graph_backbone_baseline(
                        model_name,
                        data,
                        data_dir,
                        graph_cache_path,
                        output_root,
                        case_name,
                        split_path,
                        args.seed,
                        args,
                    )

    summary = collect_summary(output_root, random_point_root, split_strategy_root)
    write_summary_table(summary, output_root)
    print(f"saved: {output_root / 'split_baseline_summary.csv'}")
    print(f"saved: {output_root / 'split_baseline_summary_table.tex'}")


if __name__ == "__main__":
    main()
