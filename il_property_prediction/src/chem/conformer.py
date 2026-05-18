from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_3d_conformer(mol: Chem.Mol, seed: int = 42, max_attempts: int = 20, optimize_method: str = "UFF"):
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = int(max_attempts)
    try:
        status = AllChem.EmbedMolecule(mol_h, params)
        if status != 0:
            return mol, np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32), False, "EmbedMolecule failed"
        try:
            if optimize_method.upper() == "MMFF" and AllChem.MMFFHasAllMoleculeParams(mol_h):
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            else:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        mol_no_h = Chem.RemoveHs(mol_h)
        conf = mol_no_h.GetConformer()
        pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol_no_h.GetNumAtoms())], dtype=np.float32)
        return mol_no_h, pos, True, None
    except Exception as exc:  # noqa: BLE001
        return mol, np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32), False, f"{type(exc).__name__}: {exc}"


def approximate_ion_pair_geometry(cation_pos: np.ndarray, anion_pos: np.ndarray, distance: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    c = cation_pos.copy()
    a = anion_pos.copy()
    if len(c) == 0 or len(a) == 0:
        return c, a
    c -= c.mean(axis=0, keepdims=True)
    a -= a.mean(axis=0, keepdims=True)
    c_extent = np.linalg.norm(c, axis=1).max() if len(c) else 0.0
    a_extent = np.linalg.norm(a, axis=1).max() if len(a) else 0.0
    shift = np.array([c_extent + a_extent + distance, 0.0, 0.0], dtype=np.float32)
    a += shift[None, :]
    return c.astype(np.float32), a.astype(np.float32)
