from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.chem.graph_featurizer import build_ion_pair_graph
from src.chem.smiles_utils import split_ion_pair
from src.utils.io import load_config


EDGE_TYPE_INDEX = 9
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "graphvis"


def _resolve_output_path(output: str | Path) -> Path:
    path = Path(output)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def _atom_label(atom: Chem.Atom) -> str:
    charge = atom.GetFormalCharge()
    if charge == 0:
        return atom.GetSymbol()
    sign = "+" if charge > 0 else "-"
    mag = "" if abs(charge) == 1 else str(abs(charge))
    return f"{atom.GetSymbol()}{mag}{sign}"


def _rdkit_2d_coords(mol: Chem.Mol) -> np.ndarray:
    mol = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    return np.asarray([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())], dtype=float)


def _layout_from_smiles(cation_smiles: str, anion_smiles: str, gap: float = 3.0) -> tuple[np.ndarray, list[str]]:
    cation = Chem.MolFromSmiles(cation_smiles)
    anion = Chem.MolFromSmiles(anion_smiles)
    if cation is None or anion is None:
        raise ValueError("Failed to parse cation or anion for 2D graph layout")
    c_coords = _rdkit_2d_coords(cation)
    a_coords = _rdkit_2d_coords(anion)
    c_coords = c_coords - c_coords.mean(axis=0, keepdims=True)
    a_coords = a_coords - a_coords.mean(axis=0, keepdims=True)
    c_right = float(c_coords[:, 0].max())
    a_left = float(a_coords[:, 0].min())
    a_coords[:, 0] += c_right - a_left + gap
    labels = [_atom_label(atom) for atom in cation.GetAtoms()] + [_atom_label(atom) for atom in anion.GetAtoms()]
    return np.vstack([c_coords, a_coords]), labels


def _layout_from_graph_pos(pos: np.ndarray) -> np.ndarray:
    coords = pos[:, :2].astype(float)
    if np.allclose(coords, 0.0):
        raise ValueError("Graph pos is all zero; use --layout rdkit2d")
    coords = coords - coords.mean(axis=0, keepdims=True)
    return coords


def _unique_edges(edge_index: np.ndarray, edge_attr: np.ndarray) -> list[tuple[int, int, int]]:
    seen = set()
    edges = []
    for k in range(edge_index.shape[1]):
        i = int(edge_index[0, k])
        j = int(edge_index[1, k])
        if i == j:
            continue
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        edge_type = int(round(float(edge_attr[k, EDGE_TYPE_INDEX]))) if edge_attr.shape[1] > EDGE_TYPE_INDEX else 0
        edges.append((key[0], key[1], edge_type))
    return edges


def _draw_graph(
    data,
    coords: np.ndarray,
    labels: list[str],
    output: Path,
    title: str,
    node_size: float,
    label_size: float,
    edge_width: float,
    cross_edge_width: float,
    padding: float,
) -> dict[str, int]:
    fragment_id = data.fragment_id.cpu().numpy()
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    edges = _unique_edges(edge_index, edge_attr)

    fig, ax = plt.subplots(figsize=(12, 8))
    covalent_count = 0
    cross_count = 0
    for i, j, edge_type in edges:
        x = [coords[i, 0], coords[j, 0]]
        y = [coords[i, 1], coords[j, 1]]
        if edge_type == 1:
            cross_count += 1
            ax.plot(x, y, color="#8a3ffc", lw=cross_edge_width, ls="--", alpha=0.9, zorder=1)
        else:
            covalent_count += 1
            ax.plot(x, y, color="#333333", lw=edge_width, alpha=0.95, zorder=2)

    c_mask = fragment_id == 0
    a_mask = fragment_id == 1
    ax.scatter(coords[c_mask, 0], coords[c_mask, 1], s=node_size, c="#ffb36b", edgecolors="#9a4d00", linewidths=1.4, zorder=3, label="cation atoms")
    ax.scatter(coords[a_mask, 0], coords[a_mask, 1], s=node_size, c="#9ec5ff", edgecolors="#2457a6", linewidths=1.4, zorder=3, label="anion atoms")

    for idx, label in enumerate(labels):
        ax.text(coords[idx, 0], coords[idx, 1], label, ha="center", va="center", fontsize=label_size, color="#111111", zorder=4)

    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    x_pad = max((x_max - x_min) * padding, 0.6)
    y_pad = max((y_max - y_min) * padding, 0.6)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.plot([], [], color="#333333", lw=edge_width, label="covalent edge")
    ax.plot([], [], color="#8a3ffc", lw=cross_edge_width, ls="--", label="cross-ion virtual edge")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "nodes": int(data.x.size(0)),
        "cation_nodes": int(c_mask.sum()),
        "anion_nodes": int(a_mask.sum()),
        "undirected_covalent_edges": covalent_count,
        "undirected_cross_ion_edges": cross_count,
        "directed_edges_in_pyg": int(data.edge_index.size(1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the actual PyG ion-pair graph used by MIPGraphNet.")
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "ion_pair_graph.png"))
    parser.add_argument("--layout", choices=["rdkit2d", "graph3d_xy"], default="rdkit2d")
    parser.add_argument("--cross-ion-mode", default=None, choices=[None, "deterministic_2d", "distance_3d"])
    parser.add_argument("--no-cross-edges", action="store_true")
    parser.add_argument("--node-size", type=float, default=620.0)
    parser.add_argument("--label-size", type=float, default=9.5)
    parser.add_argument("--edge-width", type=float, default=2.3)
    parser.add_argument("--cross-edge-width", type=float, default=1.8)
    parser.add_argument("--padding", type=float, default=0.08)
    args = parser.parse_args()

    cfg = load_config(args.config)
    chem_cfg = cfg.get("chem", {})
    cross_ion_mode = args.cross_ion_mode or chem_cfg.get("cross_ion_mode", "deterministic_2d")
    result = build_ion_pair_graph(
        args.smiles,
        use_3d=bool(chem_cfg.get("use_3d", True)),
        cutoff=float(chem_cfg.get("cross_ion_cutoff", 5.0)),
        seed=int(chem_cfg.get("seed", 42)),
        max_attempts=int(chem_cfg.get("max_conformer_attempts", 20)),
        optimize_method=chem_cfg.get("optimize_method", "UFF"),
        use_cross_edges=not args.no_cross_edges,
        cross_ion_mode=cross_ion_mode,
    )
    if result.data is None:
        raise RuntimeError(result.error)
    data = result.data
    parts = split_ion_pair(args.smiles)
    if args.layout == "graph3d_xy":
        coords = _layout_from_graph_pos(data.pos.cpu().numpy())
        cation = Chem.MolFromSmiles(data.cation_smiles)
        anion = Chem.MolFromSmiles(data.anion_smiles)
        labels = [_atom_label(atom) for atom in cation.GetAtoms()] + [_atom_label(atom) for atom in anion.GetAtoms()]
    else:
        coords, labels = _layout_from_smiles(data.cation_smiles, data.anion_smiles)

    output = _resolve_output_path(args.output)
    title = f"Ion-pair graph | cross_ion_mode={cross_ion_mode} | layout={args.layout}"
    summary = _draw_graph(
        data,
        coords,
        labels,
        output,
        title,
        args.node_size,
        args.label_size,
        args.edge_width,
        args.cross_edge_width,
        args.padding,
    )
    summary.update(
        {
            "output": str(output),
            "cation_smiles": data.cation_smiles,
            "anion_smiles": data.anion_smiles,
            "has_3d": int(data.has_3d.item()),
            "cross_ion_mode": data.cross_ion_mode,
            "conformer_note": data.conformer_note,
            "split_warnings": getattr(data, "split_warnings", ""),
        }
    )
    if parts.warnings:
        summary["input_split_warnings"] = "; ".join(parts.warnings)
    print(summary)


if __name__ == "__main__":
    main()
