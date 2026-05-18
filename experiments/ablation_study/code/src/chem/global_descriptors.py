from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


ION_DESCRIPTOR_DIM = 24
PAIR_DESCRIPTOR_DIM = 8
GLOBAL_DESCRIPTOR_DIM = ION_DESCRIPTOR_DIM * 2 + PAIR_DESCRIPTOR_DIM


def _safe(fn, default: float = 0.0) -> float:
    try:
        value = fn()
        if value is None or not np.isfinite(float(value)):
            return default
        return float(value)
    except Exception:
        return default


def _shape_descriptors(pos: np.ndarray) -> tuple[float, float, float, float, float]:
    if pos is None or len(pos) < 2 or not np.isfinite(pos).all():
        return 0.0, 0.0, 0.0, 0.0, 0.0
    centered = pos - pos.mean(axis=0, keepdims=True)
    rg = float(np.sqrt((centered**2).sum(axis=1).mean()))
    cov = centered.T @ centered / max(len(pos), 1)
    eig = np.sort(np.linalg.eigvalsh(cov).clip(min=0.0))
    total = float(eig.sum()) + 1e-8
    asphericity = float(eig[-1] - 0.5 * (eig[0] + eig[1])) / total
    return rg, float(eig[0]), float(eig[1]), float(eig[2]), asphericity


def ion_descriptors(mol: Chem.Mol, pos: np.ndarray | None) -> list[float]:
    atoms = list(mol.GetAtoms())
    heavy = max(float(mol.GetNumHeavyAtoms()), 1.0)
    formal_charge = float(sum(atom.GetFormalCharge() for atom in atoms))
    ring_info = mol.GetRingInfo()
    aromatic_rings = sum(1 for ring in ring_info.AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
    rg, eig0, eig1, eig2, asph = _shape_descriptors(pos)
    counts = {z: sum(1 for atom in atoms if atom.GetAtomicNum() == z) / heavy for z in [6, 7, 8, 9, 15, 16, 17, 35, 53]}
    desc = [
        _safe(lambda: Descriptors.MolWt(mol)) / 1000.0,
        _safe(lambda: Descriptors.ExactMolWt(mol)) / 1000.0,
        heavy / 100.0,
        _safe(lambda: rdMolDescriptors.CalcTPSA(mol)) / 300.0,
        _safe(lambda: Lipinski.NumHDonors(mol)) / 10.0,
        _safe(lambda: Lipinski.NumHAcceptors(mol)) / 20.0,
        _safe(lambda: Lipinski.NumRotatableBonds(mol)) / 50.0,
        float(ring_info.NumRings()) / 20.0,
        float(aromatic_rings) / 20.0,
        sum(1 for atom in atoms if atom.GetIsAromatic()) / heavy,
        formal_charge / 5.0,
        _safe(lambda: Crippen.MolLogP(mol)) / 20.0,
        rg / 20.0,
        eig0 / 100.0,
        eig1 / 100.0,
        eig2 / 100.0,
        asph,
        counts[6],
        counts[7],
        counts[8],
        counts[9],
        counts[15] + counts[16],
        counts[17] + counts[35] + counts[53],
        sum(1 for atom in atoms if atom.GetFormalCharge() != 0) / heavy,
    ]
    return desc[:ION_DESCRIPTOR_DIM]


def ion_pair_descriptors(cation: Chem.Mol, anion: Chem.Mol, c_pos: np.ndarray | None, a_pos: np.ndarray | None) -> list[float]:
    c_desc = ion_descriptors(cation, c_pos)
    a_desc = ion_descriptors(anion, a_pos)
    c_heavy = max(float(cation.GetNumHeavyAtoms()), 1.0)
    a_heavy = max(float(anion.GetNumHeavyAtoms()), 1.0)
    c_charge = float(sum(atom.GetFormalCharge() for atom in cation.GetAtoms()))
    a_charge = float(sum(atom.GetFormalCharge() for atom in anion.GetAtoms()))
    pair = [
        (c_heavy + a_heavy) / 200.0,
        c_heavy / max(a_heavy, 1.0),
        a_heavy / max(c_heavy, 1.0),
        (c_desc[0] + a_desc[0]),
        (c_desc[3] + a_desc[3]),
        c_charge + a_charge,
        c_charge * a_charge / 25.0,
        abs(c_desc[12] - a_desc[12]),
    ]
    return c_desc + a_desc + pair[:PAIR_DESCRIPTOR_DIM]
