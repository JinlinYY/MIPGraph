

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from _common import (
    LATENT_COLORS,
    LATENT_DISPLAY,
    LATENT_NAMES,
    PROPERTY_DISPLAY,
    PROPERTY_NAMES,
    output_dirs,
    safe_log10,
    save_both,
    setup_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Path to latent_embeddings.npz (defaults to exp7 output).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to latent_metadata.csv (defaults to exp7 output).",
    )
    parser.add_argument(
        "--split",
        default="all",
        help="Restrict analysis to a particular split (train/val/test/all).",
    )
    parser.add_argument(
        "--mi-subsample",
        type=int,
        default=4000,
        help="Sub-sample size for the slow mutual-information estimator.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    out = output_dirs()
    embeddings = args.embeddings or (out["embeddings"] / "latent_embeddings.npz")
    metadata = args.metadata or (out["embeddings"] / "latent_metadata.csv")
    return embeddings, metadata


def load_data(embeddings_path: Path, metadata_path: Path, split: str):
    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    if split.lower() != "all":
        mask = (metadata["split"] == split).to_numpy()
        if not mask.any():
            raise ValueError(f"No rows for split={split!r}")
        metadata = metadata.loc[mask].reset_index(drop=True)
    else:
        mask = np.ones(len(metadata), dtype=bool)

    factors: dict[str, np.ndarray] = {}
    for name in LATENT_NAMES:
        factors[name] = embeddings[name][mask]
    return factors, embeddings, metadata, mask


def reduce_to_pc1(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    pca = PCA(n_components=1)
    component = pca.fit_transform(matrix).reshape(-1)
    return component, float(pca.explained_variance_ratio_[0])


def cv_r2(features: np.ndarray, target: np.ndarray, n_splits: int = 5, seed: int = 0) -> float:
    valid = np.isfinite(target)
    if valid.sum() < 50:
        return float("nan")
    X = features[valid]
    y = target[valid]
    n_splits = min(n_splits, max(2, X.shape[0] // 50))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y)
    for tr, te in kf.split(X):
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 5:
        return float("nan")
    return float(pearsonr(a[valid], b[valid])[0])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 5:
        return float("nan")
    return float(spearmanr(a[valid], b[valid])[0])


def estimate_mi(factor_matrix: np.ndarray, target: np.ndarray, max_n: int) -> float:
    valid = np.isfinite(target)
    if valid.sum() < 50:
        return float("nan")
    X = factor_matrix[valid]
    y = target[valid]
    if X.shape[0] > max_n:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        X, y = X[idx], y[idx]
    pc1 = PCA(n_components=1).fit_transform(X).reshape(-1, 1)
    mi = float(mutual_info_regression(pc1, y, n_neighbors=5, random_state=0)[0])
    return max(mi, 0.0)


def compute_correlation_table(
    factors: dict[str, np.ndarray], property_logs: pd.DataFrame, mi_subsample: int
) -> pd.DataFrame:
    factor_pc1 = {name: reduce_to_pc1(mat)[0] for name, mat in factors.items()}
    rows: list[dict] = []
    for prop in PROPERTY_NAMES:
        target = property_logs[prop].to_numpy(dtype=np.float64)
        for name in LATENT_NAMES:
            pc1 = factor_pc1[name]
            entry = {
                "latent": name,
                "property": prop,
                "pearson_pc1": safe_pearson(pc1, target),
                "spearman_pc1": safe_spearman(pc1, target),
                "ridge_R2": cv_r2(factors[name], target),
                "mutual_info_pc1": estimate_mi(factors[name], target, mi_subsample),
            }
            rows.append(entry)
    return pd.DataFrame(rows)


def linear_cka(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    a = matrix_a - matrix_a.mean(axis=0, keepdims=True)
    b = matrix_b - matrix_b.mean(axis=0, keepdims=True)
    cross = float(np.linalg.norm(a.T @ b, ord="fro") ** 2)
    norm_a = float(np.linalg.norm(a.T @ a, ord="fro"))
    norm_b = float(np.linalg.norm(b.T @ b, ord="fro"))
    if norm_a == 0 or norm_b == 0:
        return float("nan")
    return cross / (norm_a * norm_b)


def disentanglement_matrices(factors: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pc1 = {name: reduce_to_pc1(mat)[0] for name, mat in factors.items()}
    pearson = pd.DataFrame(index=LATENT_NAMES, columns=LATENT_NAMES, dtype=float)
    cka = pd.DataFrame(index=LATENT_NAMES, columns=LATENT_NAMES, dtype=float)
    for a in LATENT_NAMES:
        for b in LATENT_NAMES:
            pearson.loc[a, b] = safe_pearson(pc1[a], pc1[b])
            cka.loc[a, b] = linear_cka(factors[a], factors[b])
    return pearson, cka


def heatmap(
    matrix: pd.DataFrame,
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    out_path: Path,
    fmt: str = ".2f",
    width: float = 4.4,
    height: float = 2.8,
    yticklabels: list[str] | None = None,
    xticklabels: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"shrink": 0.85},
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 7},
        ax=ax,
        xticklabels=xticklabels if xticklabels is not None else True,
        yticklabels=yticklabels if yticklabels is not None else True,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_both(fig, out_path)


def correlation_long_to_matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = df.pivot(index="latent", columns="property", values=value_col)
    pivot = pivot.reindex(LATENT_NAMES)[PROPERTY_NAMES]
    return pivot


def gate_matrix_from_embeddings(emb_path: Path) -> pd.DataFrame | None:
    data = np.load(emb_path)
    if "gates" not in data.files:
        return None
    gates = data["gates"]
    return pd.DataFrame(gates, index=PROPERTY_NAMES, columns=LATENT_NAMES)


def plot_gate_heatmap(gate: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    sns.heatmap(
        gate,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        vmin=0.0,
        vmax=float(gate.values.max()),
        cbar_kws={"shrink": 0.85, "label": "Gate weight"},
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 7},
        ax=ax,
        xticklabels=[LATENT_DISPLAY[n] for n in gate.columns],
        yticklabels=[PROPERTY_DISPLAY[p] for p in gate.index],
    )
    ax.set_title("Property → latent gating")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_both(fig, out_path)


def main() -> None:
    args = parse_args()
    setup_style()

    embeddings_path, metadata_path = resolve_paths(args)
    factors, raw_embeddings, metadata, _mask = load_data(embeddings_path, metadata_path, args.split)
    out = output_dirs()

    property_logs = pd.DataFrame(
        {prop: safe_log10(metadata[prop].to_numpy(dtype=np.float64)) for prop in PROPERTY_NAMES}
    )

    print(f"[latent_correlation] split={args.split}, samples={len(metadata)}")
    table = compute_correlation_table(factors, property_logs, args.mi_subsample)
    table.to_csv(out["tables"] / "latent_property_correlation.csv", index=False)

    pearson_mat = correlation_long_to_matrix(table, "pearson_pc1")
    spearman_mat = correlation_long_to_matrix(table, "spearman_pc1")
    r2_mat = correlation_long_to_matrix(table, "ridge_R2")
    mi_mat = correlation_long_to_matrix(table, "mutual_info_pc1")

    pearson_mat.to_csv(out["tables"] / "latent_pearson.csv")
    spearman_mat.to_csv(out["tables"] / "latent_spearman.csv")
    r2_mat.to_csv(out["tables"] / "latent_ridge_R2.csv")
    mi_mat.to_csv(out["tables"] / "latent_mutual_info.csv")

    prop_labels = [PROPERTY_DISPLAY[p] for p in PROPERTY_NAMES]
    latent_labels = [LATENT_DISPLAY[n] for n in LATENT_NAMES]

    heatmap(
        pearson_mat,
        title="Pearson(PC1, log10 property)",
        cmap="vlag",
        vmin=-1.0,
        vmax=1.0,
        out_path=out["figures"] / "fig_latent_pearson",
        xticklabels=prop_labels,
        yticklabels=latent_labels,
    )
    heatmap(
        spearman_mat,
        title="Spearman(PC1, log10 property)",
        cmap="vlag",
        vmin=-1.0,
        vmax=1.0,
        out_path=out["figures"] / "fig_latent_spearman",
        xticklabels=prop_labels,
        yticklabels=latent_labels,
    )
    heatmap(
        r2_mat,
        title=r"Ridge $R^2$ (latent → log10 property)",
        cmap="rocket_r",
        vmin=0.0,
        vmax=float(np.nanmax(r2_mat.values)),
        out_path=out["figures"] / "fig_latent_ridge_R2",
        xticklabels=prop_labels,
        yticklabels=latent_labels,
    )
    heatmap(
        mi_mat,
        title="Mutual information (PC1 vs log10 property)",
        cmap="rocket_r",
        vmin=0.0,
        vmax=float(np.nanmax(mi_mat.values)),
        out_path=out["figures"] / "fig_latent_mutual_info",
        xticklabels=prop_labels,
        yticklabels=latent_labels,
    )

    pearson_dis, cka_dis = disentanglement_matrices(factors)
    pearson_dis.to_csv(out["tables"] / "disentanglement_pearson_pc1.csv")
    cka_dis.to_csv(out["tables"] / "disentanglement_linear_cka.csv")

    heatmap(
        pearson_dis,
        title="Cross-factor PC1 Pearson",
        cmap="vlag",
        vmin=-1.0,
        vmax=1.0,
        out_path=out["figures"] / "fig_disentanglement_pc1",
        xticklabels=latent_labels,
        yticklabels=latent_labels,
        width=3.4,
        height=3.0,
    )
    heatmap(
        cka_dis,
        title="Cross-factor linear CKA",
        cmap="rocket_r",
        vmin=0.0,
        vmax=1.0,
        out_path=out["figures"] / "fig_disentanglement_cka",
        xticklabels=latent_labels,
        yticklabels=latent_labels,
        width=3.4,
        height=3.0,
    )

    gate_df = gate_matrix_from_embeddings(embeddings_path)
    if gate_df is not None:
        gate_df.to_csv(out["tables"] / "property_latent_gates.csv")
        plot_gate_heatmap(gate_df, out["figures"] / "fig_property_gating")

    summary = {
        "split": args.split,
        "n_samples": int(len(metadata)),
        "max_pearson_pc1": float(np.nanmax(np.abs(pearson_mat.values))),
        "mean_off_diagonal_pc1": float(
            np.nanmean(np.abs(pearson_dis.values)[np.eye(len(LATENT_NAMES)) == 0])
        ),
        "mean_off_diagonal_cka": float(
            np.nanmean(cka_dis.values[np.eye(len(LATENT_NAMES)) == 0])
        ),
        "table_rows": int(len(table)),
    }
    summary_path = out["tables"] / "latent_correlation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[latent_correlation] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
