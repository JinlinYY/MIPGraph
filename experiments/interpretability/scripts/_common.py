from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]
PROPERTY_DISPLAY = {
    "Density": r"$\rho$",
    "ElectricalConductivity": r"$\sigma$",
    "HeatCapacity": r"$c_p$",
    "SurfaceTension": r"$\gamma$",
    "ThermalConductivity": r"$\lambda$",
    "Viscosity": r"$\eta$",
}
LATENT_NAMES = ["packing", "cohesion", "transport", "thermal"]
LATENT_DISPLAY = {
    "packing": "Volumetric\npacking",
    "cohesion": "Cohesive\ninteraction",
    "transport": "Transport\nfriction",
    "thermal": "Thermal\nresponse",
}
LATENT_COLORS = {
    "packing": "#4c78a8",
    "cohesion": "#f58518",
    "transport": "#54a24b",
    "thermal": "#b279a2",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def il_project_root() -> Path:
    return repo_root() / "il_property_prediction"


def exp_root() -> Path:
    return repo_root() / "exp7_interpretability"


def output_dirs() -> dict[str, Path]:
    base = exp_root() / "outputs"
    dirs = {
        "embeddings": base / "embeddings",
        "figures": base / "figures",
        "tables": base / "tables",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def add_il_project_to_path() -> None:
    project = str(il_project_root())
    if project not in sys.path:
        sys.path.insert(0, project)
    exp1_scripts = repo_root() / "exp1_dataset_analysis" / "scripts"
    if exp1_scripts.exists() and str(exp1_scripts) not in sys.path:
        sys.path.insert(0, str(exp1_scripts))


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titlesize"] = 9
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["axes.linewidth"] = 0.6


def save_both(fig: plt.Figure, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def family_palette(values: Iterable[str], cmap: str = "tab20") -> dict[str, tuple]:
    uniq = sorted({v for v in values if isinstance(v, str)})
    cmap_obj = plt.get_cmap(cmap, max(len(uniq), 1))
    return {name: cmap_obj(i % cmap_obj.N) for i, name in enumerate(uniq)}


def collapse_minor_families(series: pd.Series, top_k: int = 8, fallback: str = "Other") -> pd.Series:
    counts = series.value_counts()
    keep = set(counts.head(top_k).index)
    return series.where(series.isin(keep), other=fallback)


def safe_log10(values: np.ndarray) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0)
    out[valid] = np.log10(values[valid])
    return out


def load_iptnet_checkpoint(checkpoint_path: Path):
    """Load checkpoint and rebuild ``IPTNet`` on CPU."""

    add_il_project_to_path()
    import torch

    from src.models.iptnet import IPTNet

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = IPTNet(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def default_checkpoint() -> Path:
    return (
        il_project_root()
        / "outputs"
        / "checkpoints"
        / "finetune_viscosity_from_weak_seed42"
        / "best_model.pt"
    )


def default_split_path(seed: int = 42) -> Path:
    return (
        il_project_root()
        / "data"
        / "processed"
        / "splits"
        / f"il_level_seed{seed}.json"
    )


def default_data_paths() -> dict[str, Path]:
    base = il_project_root() / "data" / "processed"
    return {
        "clean_csv": base / "il_multiprop_clean.csv",
        "arrays_path": base / "il_multiprop_arrays.npz",
        "graph_cache_path": base / "graph_cache.pt",
    }
