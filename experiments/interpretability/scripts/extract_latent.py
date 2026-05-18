

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import (
    LATENT_NAMES,
    PROPERTY_NAMES,
    add_il_project_to_path,
    default_checkpoint,
    default_data_paths,
    default_split_path,
    load_iptnet_checkpoint,
    output_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--split-path", type=Path, default=default_split_path(seed=42))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--include-splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to extract embeddings for.",
    )
    parser.add_argument(
        "--include-test-only",
        action="store_true",
        help="If set, embeddings are only extracted for the test split (overrides --include-splits).",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_dataset(
    clean_csv: Path,
    arrays_path: Path,
    graph_cache_path: Path,
    indices: np.ndarray,
    condition: np.ndarray,
    y_scaled: np.ndarray,
    error_weights: np.ndarray,
):
    add_il_project_to_path()
    from src.data.dataset import ILPropertyDataset

    return ILPropertyDataset(
        clean_csv=clean_csv,
        arrays_path=arrays_path,
        graph_cache_path=graph_cache_path,
        indices=indices,
        condition=condition,
        y_scaled=y_scaled,
        error_weights=error_weights,
    )


def attach_family_labels(meta_df: pd.DataFrame) -> pd.DataFrame:
    add_il_project_to_path()
    try:
        from family_classifier import classify_dataframe
    except ImportError as err:
        raise ImportError(
            "Could not import family_classifier from exp1_dataset_analysis/scripts."
        ) from err
    return classify_dataframe(meta_df, smiles_column="IL_SMILES")


def collect_split_indices(split: dict[str, list[int]], wanted: list[str]) -> tuple[np.ndarray, np.ndarray]:
    items: list[tuple[int, str]] = []
    for split_name in wanted:
        if split_name not in split:
            raise KeyError(f"Split file is missing key {split_name!r}; got {list(split)}")
        for idx in split[split_name]:
            items.append((int(idx), split_name))
    items.sort(key=lambda t: t[0])
    indices = np.array([t[0] for t in items], dtype=np.int64)
    labels = np.array([t[1] for t in items])
    return indices, labels


@torch.no_grad()
def run_extraction(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    storage: dict[str, list[np.ndarray]] = {
        "packing": [],
        "cohesion": [],
        "transport": [],
        "thermal": [],
        "h_structure": [],
        "h_graph": [],
        "h_condition": [],
        "h_global_desc": [],
        "property_latents": [],
        "y_pred_scaled": [],
        "sample_id": [],
    }
    gates: np.ndarray | None = None
    for batch in loader:
        batch = batch.to(device)
        pred, aux = model(batch)
        latents = aux["latents"]
        for key in LATENT_NAMES:
            storage[key].append(latents[key].detach().cpu().numpy())
        storage["property_latents"].append(latents["property_latents"].detach().cpu().numpy())
        storage["h_structure"].append(aux["h_structure"].detach().cpu().numpy())
        storage["h_graph"].append(aux["h_graph"].detach().cpu().numpy())
        storage["h_condition"].append(aux["h_condition"].detach().cpu().numpy())
        h_desc = aux.get("h_global_desc")
        if h_desc is not None:
            storage["h_global_desc"].append(h_desc.detach().cpu().numpy())
        storage["y_pred_scaled"].append(pred.detach().cpu().numpy())
        storage["sample_id"].append(batch.sample_id.detach().cpu().numpy().reshape(-1))
        if gates is None and "gates" in latents:
            gates = latents["gates"].detach().cpu().numpy()
    out: dict[str, np.ndarray] = {}
    for key, chunks in storage.items():
        if not chunks:
            continue
        out[key] = np.concatenate(chunks, axis=0)
    if gates is not None:
        out["gates"] = gates
    return out


def main() -> None:
    args = parse_args()
    if args.include_test_only:
        wanted_splits = ["test"]
    else:
        wanted_splits = list(args.include_splits)

    add_il_project_to_path()
    from torch_geometric.loader import DataLoader

    from src.data.scaler import fit_scalers

    paths = default_data_paths()
    out = output_dirs()

    split_path: Path = args.split_path
    with split_path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    split = {k: [int(i) for i in v] for k, v in split.items()}

    arrays = dict(np.load(paths["arrays_path"], allow_pickle=True))

    print(f"[extract_latent] loading checkpoint: {args.checkpoint}")
    model, ckpt = load_iptnet_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    model = model.to(device)

    cond_scaler = ckpt["condition_scaler"]
    target_scaler = ckpt["target_scaler"]

    if cond_scaler is not None and target_scaler is not None:
        condition = cond_scaler.transform(arrays["temperature"], arrays["pressure"])
        y_scaled = target_scaler.transform(arrays["y"], arrays["mask"])
        error_weights = target_scaler.error_weights(
            arrays["y"], arrays["y_error"], arrays["mask"], arrays["error_mask"]
        )
    else:
        cond_scaler, target_scaler, y_scaled, condition, error_weights = fit_scalers(arrays, split["train"])

    indices, split_labels = collect_split_indices(split, wanted_splits)
    print(f"[extract_latent] extracting for {len(indices)} rows across splits {wanted_splits}")

    dataset = build_dataset(
        clean_csv=paths["clean_csv"],
        arrays_path=paths["arrays_path"],
        graph_cache_path=paths["graph_cache_path"],
        indices=indices,
        condition=condition,
        y_scaled=y_scaled,
        error_weights=error_weights,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    extracted = run_extraction(model, loader, device)
    sample_ids = extracted["sample_id"].astype(np.int64)
    if not np.array_equal(sample_ids, indices):
        order = np.argsort(sample_ids)
        for key, value in extracted.items():
            if key == "gates":
                continue
            extracted[key] = value[order]
        sample_ids = extracted["sample_id"]
        split_label_lookup = dict(zip(indices.tolist(), split_labels.tolist()))
        split_labels = np.array([split_label_lookup[int(sid)] for sid in sample_ids])

    clean_df = pd.read_csv(paths["clean_csv"])
    selected = clean_df.iloc[sample_ids].reset_index(drop=True)

    meta = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "split": split_labels,
            "IL_Name": selected["IL_Name"].values,
            "IL_SMILES": selected["IL_SMILES"].values,
            "Cation_FullName": selected["Cation_FullName"].values,
            "Anion_FullName": selected["Anion_FullName"].values,
            "Temperature_K": selected["Temperature_K"].values,
            "Pressure_kPa": selected["Pressure_kPa"].values,
        }
    )
    raw_y = arrays["y"][sample_ids]
    raw_mask = arrays["mask"][sample_ids]
    for j, name in enumerate(PROPERTY_NAMES):
        col = raw_y[:, j].astype(float)
        col_mask = raw_mask[:, j] > 0
        meta[name] = np.where(col_mask, col, np.nan)
        meta[f"{name}_mask"] = col_mask.astype(np.int8)

    print("[extract_latent] classifying cation / anion families ...")
    meta = attach_family_labels(meta)

    embeddings_path = out["embeddings"] / "latent_embeddings.npz"
    metadata_path = out["embeddings"] / "latent_metadata.csv"
    payload = {k: v for k, v in extracted.items() if k != "sample_id"}
    payload["sample_id"] = sample_ids
    np.savez_compressed(embeddings_path, **payload)
    meta.to_csv(metadata_path, index=False)

    info = {
        "n_samples": int(len(sample_ids)),
        "splits": {name: int(np.sum(split_labels == name)) for name in sorted(set(split_labels.tolist()))},
        "latent_dim": int(extracted["packing"].shape[1]),
        "structure_dim": int(extracted["h_structure"].shape[1]),
        "checkpoint": str(args.checkpoint),
        "split_path": str(args.split_path),
        "embeddings_npz": str(embeddings_path),
        "metadata_csv": str(metadata_path),
    }
    summary_path = out["embeddings"] / "latent_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("[extract_latent] saved:")
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
