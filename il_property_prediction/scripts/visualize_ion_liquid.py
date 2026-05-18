from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.chem.smiles_utils import canonicalize_smiles, formal_charge, split_ion_pair


DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "graphvis"


def _resolve_output_path(output: str | Path) -> Path:
    path = Path(output)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def _mol_from_smiles(smiles: str | None, label: str) -> Chem.Mol | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse {label} SMILES: {smiles}")
    AllChem.Compute2DCoords(mol)
    return mol


def _legend(label: str, smiles: str | None, mol: Chem.Mol | None) -> str:
    if mol is None:
        return f"{label}: missing"
    charge = formal_charge(mol)
    atom_count = mol.GetNumAtoms()
    return f"{label} | charge={charge:+d} | atoms={atom_count}"


def _atom_label(atom: Chem.Atom) -> str:
    charge = atom.GetFormalCharge()
    if charge == 0:
        return atom.GetSymbol()
    sign = "+" if charge > 0 else "-"
    mag = "" if abs(charge) == 1 else str(abs(charge))
    return f"{atom.GetSymbol()}{mag}{sign}"


def _draw_bond(ax, p0, p1, order: float, aromatic: bool) -> None:
    import numpy as np

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    vec = p1 - p0
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return
    perp = np.array([-vec[1], vec[0]]) / norm
    style = "--" if aromatic else "-"
    color = "#333333"
    if order >= 2.5:
        offsets = [-0.055, 0.0, 0.055]
    elif order >= 1.5:
        offsets = [-0.04, 0.04]
    else:
        offsets = [0.0]
    for off in offsets:
        delta = perp * off
        ax.plot([p0[0] + delta[0], p1[0] + delta[0]], [p0[1] + delta[1], p1[1] + delta[1]], style, color=color, lw=1.6)


def _draw_mol(ax, mol: Chem.Mol, legend: str) -> None:
    import numpy as np

    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())], dtype=float)

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        order = float(bond.GetBondTypeAsDouble())
        _draw_bond(ax, coords[begin], coords[end], order, bond.GetIsAromatic())

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        label = _atom_label(atom)
        color = "#c23b22" if atom.GetFormalCharge() > 0 else "#2359c4" if atom.GetFormalCharge() < 0 else "#111111"
        ax.text(
            coords[idx, 0],
            coords[idx, 1],
            label,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.85},
        )

    if len(coords):
        pad = 0.8
        ax.set_xlim(float(coords[:, 0].min() - pad), float(coords[:, 0].max() + pad))
        ax.set_ylim(float(coords[:, 1].min() - pad), float(coords[:, 1].max() + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(legend, fontsize=10)
    ax.axis("off")


def _draw_grid_matplotlib(mols: list[Chem.Mol], legends: list[str], output: Path) -> None:
    import matplotlib.pyplot as plt

    n = len(mols)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.8), squeeze=False)
    for ax, mol, legend in zip(axes[0], mols, legends):
        _draw_mol(ax, mol, legend)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_grid_rdkit(mols: list[Chem.Mol], legends: list[str], output: Path) -> None:
    from rdkit.Chem import Draw

    if output.suffix.lower() == ".svg":
        svg = Draw.MolsToGridImage(
            mols,
            molsPerRow=min(3, len(mols)),
            subImgSize=(440, 340),
            legends=legends,
            useSVG=True,
        )
        output.write_text(str(svg), encoding="utf-8")
        return
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(3, len(mols)),
        subImgSize=(440, 340),
        legends=legends,
        useSVG=False,
    )
    image.save(output)


def visualize_ion_liquid(
    smiles: str,
    output: str | Path,
    title: str | None = None,
    include_full: bool = True,
    drawer: str = "auto",
) -> Path:
    output = _resolve_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    canonical, err = canonicalize_smiles(smiles)
    if canonical is None:
        raise ValueError(f"Invalid IL SMILES: {err}")

    parts = split_ion_pair(canonical)
    cation = _mol_from_smiles(parts.cation_smiles, "cation")
    anion = _mol_from_smiles(parts.anion_smiles, "anion")
    full = _mol_from_smiles(canonical, "ion pair") if include_full else None

    mols: list[Chem.Mol] = []
    legends: list[str] = []
    if cation is not None:
        mols.append(cation)
        legends.append(_legend("Cation", parts.cation_smiles, cation))
    if anion is not None:
        mols.append(anion)
        legends.append(_legend("Anion", parts.anion_smiles, anion))
    if full is not None:
        mols.append(full)
        legends.append(title or "Ion pair")

    if not mols:
        raise ValueError("No drawable molecule was parsed from the input SMILES")

    if output.suffix.lower() not in {".svg", ".png"}:
        raise ValueError("Output file must end with .svg or .png")
    if drawer not in {"auto", "rdkit", "matplotlib"}:
        raise ValueError("drawer must be one of: auto, rdkit, matplotlib")
    if drawer in {"auto", "rdkit"}:
        try:
            _draw_grid_rdkit(mols, legends, output)
        except Exception as exc:  # noqa: BLE001
            if drawer == "rdkit":
                raise
            print({"warning": f"RDKit drawer failed, falling back to matplotlib: {type(exc).__name__}: {exc}"})
            _draw_grid_matplotlib(mols, legends, output)
    else:
        _draw_grid_matplotlib(mols, legends, output)

    if parts.warnings:
        print({"warnings": parts.warnings})
    print(
        {
            "output": str(output),
            "canonical_smiles": canonical,
            "cation_smiles": parts.cation_smiles,
            "anion_smiles": parts.anion_smiles,
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an ionic liquid SMILES as cation/anion 2D molecular structures.")
    parser.add_argument("--smiles", required=True, help="Ionic liquid SMILES, usually cation.anion")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "ion_pair_structure.svg"), help="Output .svg or .png path")
    parser.add_argument("--title", default=None, help="Legend for the combined ion-pair panel")
    parser.add_argument("--no-full", action="store_true", help="Only draw cation and anion panels, not the combined ion pair")
    parser.add_argument("--drawer", choices=["auto", "rdkit", "matplotlib"], default="auto", help="Molecule drawing backend")
    args = parser.parse_args()
    visualize_ion_liquid(args.smiles, args.output, args.title, include_full=not args.no_full, drawer=args.drawer)


if __name__ == "__main__":
    main()
