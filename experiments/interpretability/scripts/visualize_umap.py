

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from _common import (
    LATENT_DISPLAY,
    LATENT_NAMES,
    PROPERTY_NAMES,
    collapse_minor_families,
    family_palette,
    output_dirs,
    safe_log10,
    save_both,
    setup_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--split", default="all", help="train/val/test/all")
    parser.add_argument(
        "--feature",
        default="h_structure",
        choices=["h_structure", "h_graph"] + LATENT_NAMES + ["concat_factors"],
        help="Which embedding field to project.",
    )
    parser.add_argument("--per-il", action="store_true", default=True, help="Average over T/P per IL.")
    parser.add_argument("--no-per-il", action="store_false", dest="per_il")
    parser.add_argument("--n-neighbors", type=int, default=25)
    parser.add_argument("--min-dist", type=float, default=0.25)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-families", type=int, default=8)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    out = output_dirs()
    embeddings = args.embeddings or (out["embeddings"] / "latent_embeddings.npz")
    metadata = args.metadata or (out["embeddings"] / "latent_metadata.csv")
    return embeddings, metadata


def gather_features(emb: np.lib.npyio.NpzFile, feature: str) -> np.ndarray:
    if feature == "concat_factors":
        return np.concatenate([emb[name] for name in LATENT_NAMES], axis=1)
    return emb[feature]


def per_il_aggregate(features: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    grouped = metadata.groupby("IL_SMILES", sort=False)
    rows: list[dict] = []
    feats: list[np.ndarray] = []
    numeric_cols = ["Temperature_K"] + PROPERTY_NAMES
    for smiles, sub in grouped:
        idx = sub.index.to_numpy()
        feats.append(features[idx].mean(axis=0))
        first = sub.iloc[0]
        entry = {
            "IL_SMILES": smiles,
            "IL_Name": first["IL_Name"],
            "Cation_FullName": first.get("Cation_FullName"),
            "Anion_FullName": first.get("Anion_FullName"),
            "Cation_Family": first.get("Cation_Family"),
            "Anion_Family": first.get("Anion_Family"),
            "n_measurements": int(len(sub)),
            "split": first.get("split"),
        }
        for col in numeric_cols:
            if col in sub.columns:
                entry[col] = float(sub[col].mean())
        rows.append(entry)
    return np.stack(feats, axis=0), pd.DataFrame(rows)


def run_umap(features: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, max(2, features.shape[0] - 1)),
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(features)


def run_tsne(features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    perplexity = float(min(perplexity, max(5.0, (features.shape[0] - 1) / 4.0)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        metric="euclidean",
    )
    return tsne.fit_transform(features)


def scatter_categorical(
    ax: plt.Axes,
    coords: np.ndarray,
    categories: pd.Series,
    palette: dict[str, tuple],
    title: str,
    show_legend: bool,
) -> None:
    for name, color in palette.items():
        mask = (categories == name).to_numpy()
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            c=[color],
            edgecolors="white",
            linewidths=0.2,
            alpha=0.85,
            label=name,
        )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if show_legend:
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markersize=4, markeredgecolor="white", label=name)
            for name, color in palette.items()
        ]
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=6,
            handletextpad=0.4,
            borderaxespad=0.0,
        )


def scatter_continuous(
    ax: plt.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str,
    cbar_label: str,
    fig: plt.Figure,
) -> None:
    valid = np.isfinite(values)
    sc_inv = ax.scatter(
        coords[~valid, 0],
        coords[~valid, 1],
        s=6,
        c="lightgrey",
        edgecolors="none",
        alpha=0.4,
    )
    sc = ax.scatter(
        coords[valid, 0],
        coords[valid, 1],
        s=10,
        c=values[valid],
        cmap=cmap,
        edgecolors="white",
        linewidths=0.2,
        alpha=0.9,
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


def render_panel(
    coords: np.ndarray,
    metadata: pd.DataFrame,
    method: str,
    out_path: Path,
    top_families: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4))
    cation_series = collapse_minor_families(metadata["Cation_Family"].fillna("Unknown"), top_k=top_families)
    anion_series = collapse_minor_families(metadata["Anion_Family"].fillna("Unknown"), top_k=top_families)

    cation_palette = family_palette(cation_series.unique(), cmap="tab20")
    anion_palette = family_palette(anion_series.unique(), cmap="tab20b")

    scatter_categorical(axes[0, 0], coords, cation_series, cation_palette, f"{method}: cation family", show_legend=True)
    scatter_categorical(axes[0, 1], coords, anion_series, anion_palette, f"{method}: anion family", show_legend=True)
    scatter_continuous(
        axes[1, 0],
        coords,
        safe_log10(metadata["Density"].to_numpy(dtype=np.float64)),
        title=f"{method}: log10 density",
        cmap="viridis",
        cbar_label="log10 ρ (kg/m³)",
        fig=fig,
    )
    scatter_continuous(
        axes[1, 1],
        coords,
        safe_log10(metadata["Viscosity"].to_numpy(dtype=np.float64)),
        title=f"{method}: log10 viscosity",
        cmap="magma",
        cbar_label="log10 η (Pa·s)",
        fig=fig,
    )
    fig.suptitle(f"Latent organisation – {method}", fontsize=10, y=0.995)
    save_both(fig, out_path)


def main() -> None:
    args = parse_args()
    setup_style()

    embeddings_path, metadata_path = resolve_paths(args)
    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    if args.split.lower() != "all":
        mask = (metadata["split"] == args.split).to_numpy()
        if not mask.any():
            raise ValueError(f"No rows for split={args.split!r}")
        metadata = metadata.loc[mask].reset_index(drop=True)
        feature_full = gather_features(embeddings, args.feature)[mask]
    else:
        feature_full = gather_features(embeddings, args.feature)

    if args.per_il:
        features, summary = per_il_aggregate(feature_full, metadata)
        meta_for_plot = summary
        suffix = "_il"
    else:
        features = feature_full
        meta_for_plot = metadata
        suffix = "_row"

    print(f"[visualize_umap] feature={args.feature}, n={features.shape[0]}, dim={features.shape[1]}, split={args.split}")

    scaled = StandardScaler().fit_transform(features.astype(np.float32))

    print("[visualize_umap] computing UMAP ...")
    umap_coords = run_umap(scaled, args.n_neighbors, args.min_dist, args.seed)
    print("[visualize_umap] computing t-SNE ...")
    tsne_coords = run_tsne(scaled, args.tsne_perplexity, args.seed)

    out = output_dirs()
    coord_df = pd.DataFrame(
        {
            "IL_SMILES": meta_for_plot["IL_SMILES"].values if "IL_SMILES" in meta_for_plot else np.arange(features.shape[0]),
            "umap_1": umap_coords[:, 0],
            "umap_2": umap_coords[:, 1],
            "tsne_1": tsne_coords[:, 0],
            "tsne_2": tsne_coords[:, 1],
        }
    )
    extra_cols = [c for c in ["IL_Name", "Cation_Family", "Anion_Family", "Density", "Viscosity", "n_measurements", "split"] if c in meta_for_plot.columns]
    for col in extra_cols:
        coord_df[col] = meta_for_plot[col].values
    coord_df.to_csv(out["tables"] / f"latent_2d_coords_{args.feature}{suffix}.csv", index=False)

    render_panel(
        umap_coords,
        meta_for_plot,
        method="UMAP",
        out_path=out["figures"] / f"fig_umap_{args.feature}{suffix}",
        top_families=args.top_families,
    )
    render_panel(
        tsne_coords,
        meta_for_plot,
        method="t-SNE",
        out_path=out["figures"] / f"fig_tsne_{args.feature}{suffix}",
        top_families=args.top_families,
    )

    summary = {
        "feature": args.feature,
        "split": args.split,
        "n_points": int(features.shape[0]),
        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "tsne_perplexity": float(args.tsne_perplexity),
        "per_il": bool(args.per_il),
    }
    with (out["tables"] / f"latent_2d_summary_{args.feature}{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[visualize_umap] done.")


if __name__ == "__main__":
    main()
