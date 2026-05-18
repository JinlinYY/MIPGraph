from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rdkit import Chem
from torch_geometric.data import Batch
from tqdm import tqdm

from _common import (
    PROPERTY_NAMES,
    add_il_project_to_path,
    collapse_minor_families,
    default_checkpoint,
    default_data_paths,
    default_split_path,
    load_iptnet_checkpoint,
    output_dirs,
    save_both,
    setup_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--split-path", type=Path, default=default_split_path(seed=42))
    parser.add_argument("--split", default="test", help="Which split to attribute on.")
    parser.add_argument(
        "--target",
        default="Viscosity",
        choices=PROPERTY_NAMES,
        help="Which property prediction to attribute.",
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--top-edges", type=int, default=20)
    parser.add_argument("--top-families", type=int, default=8)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def metadata_lookup() -> pd.DataFrame:
    out = output_dirs()
    metadata_path = out["embeddings"] / "latent_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "latent_metadata.csv not found — run extract_latent.py first."
        )
    return pd.read_csv(metadata_path)


def integrated_gradients_edges(
    model: torch.nn.Module,
    batch: Batch,
    target_index: int,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrated gradients of the target prediction wrt ``edge_attr``.

    Returns (attribution, prediction).  ``attribution`` has the same shape as
    ``edge_attr``.
    """

    edge_attr = batch.edge_attr.detach()
    baseline = torch.zeros_like(edge_attr)
    diff = edge_attr - baseline

    accum = torch.zeros_like(edge_attr)
    for k in range(n_steps):
        alpha = (k + 0.5) / n_steps
        scaled = baseline + alpha * diff
        scaled = scaled.clone().requires_grad_(True)
        batch.edge_attr = scaled
        pred, _ = model(batch)
        target = pred[:, target_index].sum()
        grad = torch.autograd.grad(target, scaled, retain_graph=False, create_graph=False)[0]
        accum = accum + grad.detach()
    accum = accum / n_steps
    attribution = accum * diff

    batch.edge_attr = edge_attr
    with torch.no_grad():
        pred, _ = model(batch)
    return attribution.detach(), pred.detach()


def edge_atomic_symbols(graph) -> tuple[list[str], list[str], list[str], list[str]]:
    smiles = graph.smiles
    cation_smi, anion_smi = (smiles.split(".", 1) + [""])[:2]
    if hasattr(graph, "cation_smiles"):
        cation_smi = graph.cation_smiles or cation_smi
    if hasattr(graph, "anion_smiles"):
        anion_smi = graph.anion_smiles or anion_smi
    cation_mol = Chem.MolFromSmiles(cation_smi) if cation_smi else None
    anion_mol = Chem.MolFromSmiles(anion_smi) if anion_smi else None
    cation_atoms = [a.GetSymbol() for a in cation_mol.GetAtoms()] if cation_mol else []
    cation_charges = [a.GetFormalCharge() for a in cation_mol.GetAtoms()] if cation_mol else []
    anion_atoms = [a.GetSymbol() for a in anion_mol.GetAtoms()] if anion_mol else []
    anion_charges = [a.GetFormalCharge() for a in anion_mol.GetAtoms()] if anion_mol else []
    return cation_atoms, cation_charges, anion_atoms, anion_charges


def select_samples(metadata: pd.DataFrame, split: str, n_samples: int, seed: int) -> pd.DataFrame:
    if split.lower() != "all":
        df = metadata.loc[metadata["split"] == split].copy()
    else:
        df = metadata.copy()
    if df.empty:
        raise ValueError(f"No rows for split={split!r}")
    df = df.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    take = min(n_samples, len(df))
    indices = rng.choice(len(df), size=take, replace=False)
    return df.iloc[sorted(indices.tolist())].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    setup_style()
    out = output_dirs()

    add_il_project_to_path()
    from src.data.dataset import ILPropertyDataset
    from src.data.scaler import fit_scalers

    paths = default_data_paths()
    with args.split_path.open("r", encoding="utf-8") as f:
        raw_split = json.load(f)
    split = {k: [int(i) for i in v] for k, v in raw_split.items()}

    arrays = dict(np.load(paths["arrays_path"], allow_pickle=True))

    print(f"[edge_attribution] loading checkpoint: {args.checkpoint}")
    model, ckpt = load_iptnet_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    model = model.to(device)
    model.eval()
    target_index = PROPERTY_NAMES.index(args.target)

    cond_scaler = ckpt["condition_scaler"]
    target_scaler = ckpt["target_scaler"]
    if cond_scaler is None or target_scaler is None:
        cond_scaler, target_scaler, y_scaled, condition, error_weights = fit_scalers(arrays, split["train"])
    else:
        condition = cond_scaler.transform(arrays["temperature"], arrays["pressure"])
        y_scaled = target_scaler.transform(arrays["y"], arrays["mask"])
        error_weights = target_scaler.error_weights(
            arrays["y"], arrays["y_error"], arrays["mask"], arrays["error_mask"]
        )

    metadata = metadata_lookup()
    selected = select_samples(metadata, args.split, args.n_samples, args.seed)
    selected_ids = selected["sample_id"].to_numpy(dtype=np.int64)

    dataset = ILPropertyDataset(
        clean_csv=paths["clean_csv"],
        arrays_path=paths["arrays_path"],
        graph_cache_path=paths["graph_cache_path"],
        indices=selected_ids,
        condition=condition,
        y_scaled=y_scaled,
        error_weights=error_weights,
    )

    summary_rows: list[dict] = []
    family_records: list[dict] = []
    top_edge_records: list[dict] = []

    for i in tqdm(range(len(dataset)), desc="IG"):
        graph = dataset[i]
        sample_id = int(graph.sample_id.item())
        meta_row = selected.loc[selected["sample_id"] == sample_id].iloc[0]
        batch = Batch.from_data_list([graph]).to(device)

        attribution, pred = integrated_gradients_edges(model, batch, target_index, args.ig_steps)
        edge_index = batch.edge_index.cpu().numpy()
        edge_attr_np = batch.edge_attr.cpu().numpy()
        attribution_np = attribution.cpu().numpy()
        per_edge_attr = attribution_np.sum(axis=-1)
        per_edge_abs = np.abs(per_edge_attr)
        # Column 9 of edge_attr is the explicit edge_type flag (0 = covalent, 1 = cross-ion).
        # We additionally require the edge to span a cation (frag=0) → anion (frag=1) boundary
        # because that is the geometric definition the IPTNet is built around.
        fragment_id = batch.fragment_id.cpu().numpy().reshape(-1)
        spans_ion_pair = (fragment_id[edge_index[0]] != fragment_id[edge_index[1]])
        edge_type_col = edge_attr_np[:, 9] if edge_attr_np.shape[1] > 9 else np.zeros(edge_attr_np.shape[0])
        is_cross = (edge_type_col >= 0.5) | spans_ion_pair

        covalent_total = float(per_edge_abs[~is_cross].sum())
        cross_total = float(per_edge_abs[is_cross].sum())
        total = covalent_total + cross_total + 1e-12

        summary_rows.append(
            {
                "sample_id": sample_id,
                "IL_Name": meta_row["IL_Name"],
                "IL_SMILES": meta_row["IL_SMILES"],
                "Cation_Family": meta_row.get("Cation_Family"),
                "Anion_Family": meta_row.get("Anion_Family"),
                "Temperature_K": float(meta_row["Temperature_K"]),
                "Pressure_kPa": float(meta_row.get("Pressure_kPa", np.nan)),
                "target": args.target,
                "y_pred_scaled": float(pred[0, target_index].item()),
                "covalent_abs_sum": covalent_total,
                "cross_ion_abs_sum": cross_total,
                "covalent_share": covalent_total / total,
                "cross_ion_share": cross_total / total,
                "n_covalent_edges": int(np.sum(~is_cross)),
                "n_cross_edges": int(np.sum(is_cross)),
            }
        )

        cation_atoms, cation_charges, anion_atoms, _ = edge_atomic_symbols(graph)
        cation_indices = np.where(fragment_id == 0)[0]
        anion_indices = np.where(fragment_id == 1)[0]
        n_cation_graph = int(len(cation_indices))

        if cross_total > 0 and is_cross.any():
            cross_idx = np.where(is_cross)[0]
            for e in cross_idx:
                a, b = edge_index[:, e].tolist()
                # ensure a is the cation-side atom
                if fragment_id[a] != 0:
                    a, b = b, a
                if fragment_id[a] != 0 or fragment_id[b] != 1:
                    continue
                local_a = int(a)
                local_b = int(b - n_cation_graph)
                cat_atom = cation_atoms[local_a] if local_a < len(cation_atoms) else "?"
                an_atom = anion_atoms[local_b] if local_b < len(anion_atoms) else "?"
                top_edge_records.append(
                    {
                        "sample_id": sample_id,
                        "IL_Name": meta_row["IL_Name"],
                        "Cation_Family": meta_row.get("Cation_Family"),
                        "Anion_Family": meta_row.get("Anion_Family"),
                        "cation_atom_idx": local_a,
                        "anion_atom_idx": local_b,
                        "cation_atom": cat_atom,
                        "anion_atom": an_atom,
                        "edge_attribution": float(per_edge_attr[e]),
                        "abs_attribution": float(per_edge_abs[e]),
                        "edge_type_flag": float(edge_type_col[e]),
                        "distance": float(edge_attr_np[e, 10]) if edge_attr_np.shape[1] > 10 else 0.0,
                        "inv_distance": float(edge_attr_np[e, 11]) if edge_attr_np.shape[1] > 11 else 0.0,
                    }
                )
            family_records.append(
                {
                    "Cation_Family": meta_row.get("Cation_Family"),
                    "Anion_Family": meta_row.get("Anion_Family"),
                    "cross_ion_abs_sum": cross_total,
                    "cross_ion_share": cross_total / total,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out["tables"] / "edge_attribution_summary.csv", index=False)

    top_edges_df = pd.DataFrame(top_edge_records)
    if not top_edges_df.empty:
        top_edges_df.sort_values("abs_attribution", ascending=False, inplace=True)
        top_edges_df.head(args.top_edges).to_csv(
            out["tables"] / f"top_cross_ion_edges_{args.target}.csv", index=False
        )

    family_df = pd.DataFrame(family_records)
    if not family_df.empty:
        family_df["Cation_Family"] = collapse_minor_families(
            family_df["Cation_Family"].fillna("Unknown"), top_k=args.top_families
        )
        family_df["Anion_Family"] = collapse_minor_families(
            family_df["Anion_Family"].fillna("Unknown"), top_k=args.top_families
        )
        family_pivot = family_df.groupby(["Cation_Family", "Anion_Family"]).agg(
            mean_share=("cross_ion_share", "mean"),
            mean_abs_attr=("cross_ion_abs_sum", "mean"),
            count=("cross_ion_abs_sum", "size"),
        ).reset_index()
        family_pivot.to_csv(out["tables"] / f"cross_ion_family_attribution_{args.target}.csv", index=False)
    else:
        family_pivot = pd.DataFrame()

    plot_overview(summary_df, top_edges_df, family_df, family_pivot, args, out)

    summary_path = out["tables"] / f"edge_attribution_meta_{args.target}.json"
    meta = {
        "target": args.target,
        "split": args.split,
        "n_samples": int(len(summary_df)),
        "ig_steps": int(args.ig_steps),
        "checkpoint": str(args.checkpoint),
        "covalent_share_mean": float(summary_df["covalent_share"].mean()) if not summary_df.empty else None,
        "cross_share_mean": float(summary_df["cross_ion_share"].mean()) if not summary_df.empty else None,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[edge_attribution] summary:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


def plot_overview(
    summary_df: pd.DataFrame,
    top_edges_df: pd.DataFrame,
    family_df: pd.DataFrame,
    family_pivot: pd.DataFrame,
    args: argparse.Namespace,
    out: dict[str, Path],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))

    ax = axes[0, 0]
    if not summary_df.empty:
        share_long = summary_df.melt(
            id_vars=["sample_id"],
            value_vars=["covalent_share", "cross_ion_share"],
            var_name="edge_kind",
            value_name="share",
        )
        share_long["edge_kind"] = share_long["edge_kind"].map(
            {"covalent_share": "Covalent", "cross_ion_share": "Cross-ion"}
        )
        sns.violinplot(
            data=share_long,
            x="edge_kind",
            y="share",
            hue="edge_kind",
            ax=ax,
            inner="quartile",
            cut=0,
            palette={"Covalent": "#4c78a8", "Cross-ion": "#e45756"},
            linewidth=0.7,
            legend=False,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Attribution share")
        ax.set_title(f"Attribution share to {args.target}")
    else:
        ax.set_axis_off()

    ax = axes[0, 1]
    if not summary_df.empty:
        sns.scatterplot(
            data=summary_df,
            x="covalent_abs_sum",
            y="cross_ion_abs_sum",
            hue="Cation_Family",
            alpha=0.75,
            s=14,
            ax=ax,
            palette="tab10",
            edgecolor="white",
            linewidth=0.2,
        )
        lim = float(max(summary_df["covalent_abs_sum"].max(), summary_df["cross_ion_abs_sum"].max()) * 1.05)
        ax.plot([0, lim], [0, lim], "--", lw=0.8, color="grey")
        ax.set_xlabel(r"$\Sigma$ |Attr| (covalent)")
        ax.set_ylabel(r"$\Sigma$ |Attr| (cross-ion)")
        ax.set_title("Per-IL attribution magnitude")
        ax.legend(title="Cation", fontsize=6, frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    else:
        ax.set_axis_off()

    ax = axes[1, 0]
    if not top_edges_df.empty:
        head = top_edges_df.head(args.top_edges).copy()
        head["label"] = (
            head["IL_Name"].astype(str).str.slice(0, 22)
            + " | "
            + head["cation_atom"]
            + "→"
            + head["anion_atom"]
        )
        sns.barplot(
            data=head.iloc[::-1],
            x="abs_attribution",
            y="label",
            color="#e45756",
            ax=ax,
        )
        ax.set_xlabel("|Edge attribution|")
        ax.set_ylabel("")
        ax.set_title(f"Top cross-ion edges ({args.target})")
        ax.tick_params(axis="y", labelsize=6)
    else:
        ax.set_axis_off()

    ax = axes[1, 1]
    if not family_pivot.empty:
        heat = family_pivot.pivot(index="Cation_Family", columns="Anion_Family", values="mean_abs_attr")
        sns.heatmap(
            heat,
            annot=True,
            fmt=".2f",
            cmap="rocket_r",
            cbar_kws={"shrink": 0.8, "label": "Mean Σ |Attr| cross-ion"},
            linewidths=0.4,
            linecolor="white",
            annot_kws={"fontsize": 6},
            ax=ax,
        )
        ax.set_title("Cation × anion attribution")
        ax.set_xlabel("Anion family")
        ax.set_ylabel("Cation family")
        ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.tick_params(axis="y", labelsize=6)
    else:
        ax.set_axis_off()

    fig.suptitle(f"Edge attribution analysis – target={args.target}", fontsize=10, y=0.995)
    save_both(fig, out["figures"] / f"fig_edge_attribution_{args.target}")


if __name__ == "__main__":
    main()
