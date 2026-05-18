from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from .conformer import approximate_ion_pair_geometry, generate_3d_conformer
from .global_descriptors import GLOBAL_DESCRIPTOR_DIM, ion_pair_descriptors
from .smiles_utils import split_ion_pair


ATOM_FEATURE_DIM = 45
EDGE_FEATURE_DIM = 12


@dataclass
class GraphBuildResult:
    data: Data | None
    error: str | None


def _one_hot(value, choices):
    return [1.0 if value == c else 0.0 for c in choices]


def atom_features(atom: Chem.Atom, ion_type: int) -> list[float]:
    hyb = atom.GetHybridization()
    chiral = atom.GetChiralTag()
    feats = []
    feats += _one_hot(atom.GetAtomicNum(), [1, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 35, 53])
    feats += [atom.GetDegree() / 6.0, atom.GetFormalCharge() / 3.0]
    feats += _one_hot(hyb, [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3])
    feats += [float(atom.GetIsAromatic()), atom.GetTotalNumHs() / 4.0]
    feats += _one_hot(chiral, [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW])
    feats += [float(atom.IsInRing()), atom.GetMass() / 200.0, float(ion_type)]
    feats += [0.0] * (ATOM_FEATURE_DIM - len(feats))
    return feats[:ATOM_FEATURE_DIM]


def bond_features(bond: Chem.Bond, edge_type: int = 0, distance: float = 0.0, charge_product: float = 0.0, hbond: float = 0.0, aromatic_pair: float = 0.0, vdw: float = 0.0) -> list[float]:
    btype = bond.GetBondType() if bond is not None else None
    stereo = bond.GetStereo() if bond is not None else None
    feats = []
    feats += _one_hot(btype, [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])
    feats += [float(bond.GetIsConjugated()) if bond is not None else 0.0, float(bond.IsInRing()) if bond is not None else 0.0]
    feats += _one_hot(stereo, [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE])
    inv_d = 1.0 / max(distance, 1e-6) if distance > 0 else 0.0
    coulomb = charge_product * inv_d
    feats += [float(edge_type), float(distance), float(inv_d), float(charge_product), float(coulomb), float(hbond), float(aromatic_pair), float(vdw)]
    return feats[:EDGE_FEATURE_DIM]


def _donor_acceptor(atom: Chem.Atom) -> tuple[bool, bool]:
    z = atom.GetAtomicNum()
    donor = z in (7, 8) and atom.GetTotalNumHs() > 0
    acceptor = z in (7, 8, 9, 16) and atom.GetFormalCharge() <= 0
    return donor, acceptor


def _representative_atom(mol: Chem.Mol, prefer_positive: bool) -> int:
    best_idx = 0
    best_score = -1e9
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        score = abs(charge) * 10 + atom.GetDegree()
        if prefer_positive and charge > 0:
            score += 100
        if (not prefer_positive) and charge < 0:
            score += 100
        if atom.GetIsAromatic():
            score += 1
        if score > best_score:
            best_score = score
            best_idx = atom.GetIdx()
    return best_idx


def _deterministic_cross_pairs(cation: Chem.Mol, anion: Chem.Mol) -> list[tuple[int, int, dict[str, float]]]:
    pairs: dict[tuple[int, int], dict[str, float]] = {}
    for i, atom_i in enumerate(cation.GetAtoms()):
        qi = atom_i.GetFormalCharge()
        donor_i, acceptor_i = _donor_acceptor(atom_i)
        for j, atom_j in enumerate(anion.GetAtoms()):
            qj = atom_j.GetFormalCharge()
            donor_j, acceptor_j = _donor_acceptor(atom_j)
            charge_pair = qi > 0 and qj < 0
            hbond_pair = (donor_i and acceptor_j) or (donor_j and acceptor_i)
            aromatic_pair = atom_i.GetIsAromatic() and atom_j.GetIsAromatic()
            if not (charge_pair or hbond_pair or aromatic_pair):
                continue
            pairs[(i, j)] = {
                "charge_product": float(qi * qj),
                "hbond": float(hbond_pair),
                "aromatic": float(aromatic_pair),
            }
    if not pairs:
        i = _representative_atom(cation, prefer_positive=True)
        j = _representative_atom(anion, prefer_positive=False)
        qi = cation.GetAtomWithIdx(i).GetFormalCharge()
        qj = anion.GetAtomWithIdx(j).GetFormalCharge()
        pairs[(i, j)] = {
            "charge_product": float(qi * qj),
            "hbond": 0.0,
            "aromatic": float(cation.GetAtomWithIdx(i).GetIsAromatic() and anion.GetAtomWithIdx(j).GetIsAromatic()),
        }
    return [(i, j, feat) for (i, j), feat in pairs.items()]


def _combine_mols(cation: Chem.Mol, anion: Chem.Mol):
    return Chem.CombineMols(cation, anion)


def build_ion_pair_graph(
    smiles: str,
    use_3d: bool = True,
    cutoff: float = 5.0,
    seed: int = 42,
    max_attempts: int = 20,
    optimize_method: str = "UFF",
    use_cross_edges: bool = True,
    cross_ion_mode: str = "deterministic_2d",
) -> GraphBuildResult:
    parts = split_ion_pair(smiles)
    if not parts.cation_smiles:
        return GraphBuildResult(None, "missing cation fragment")
    cation = Chem.MolFromSmiles(parts.cation_smiles)
    anion = Chem.MolFromSmiles(parts.anion_smiles) if parts.anion_smiles else None
    if cation is None or anion is None:
        return GraphBuildResult(None, "RDKit failed to parse cation or anion")

    c_mol, c_pos, c_has3d, c_err = generate_3d_conformer(cation, seed, max_attempts, optimize_method) if use_3d else (cation, np.zeros((cation.GetNumAtoms(), 3), dtype=np.float32), False, None)
    a_mol, a_pos, a_has3d, a_err = generate_3d_conformer(anion, seed + 1, max_attempts, optimize_method) if use_3d else (anion, np.zeros((anion.GetNumAtoms(), 3), dtype=np.float32), False, None)
    has_3d = bool(c_has3d and a_has3d)
    if cross_ion_mode == "distance_3d" and has_3d:
        c_pos, a_pos = approximate_ion_pair_geometry(c_pos, a_pos)
    else:
        if not c_has3d:
            c_pos = np.zeros((c_mol.GetNumAtoms(), 3), dtype=np.float32)
        if not a_has3d:
            a_pos = np.zeros((a_mol.GetNumAtoms(), 3), dtype=np.float32)

    x = [atom_features(atom, 0) for atom in c_mol.GetAtoms()] + [atom_features(atom, 1) for atom in a_mol.GetAtoms()]
    pos = np.vstack([c_pos, a_pos]).astype(np.float32)
    n_c = c_mol.GetNumAtoms()
    n_a = a_mol.GetNumAtoms()
    fragment_id = [0] * n_c + [1] * n_a

    edge_index: list[list[int]] = []
    edge_attr: list[list[float]] = []
    for mol, offset in [(c_mol, 0), (a_mol, n_c)]:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx() + offset
            j = bond.GetEndAtomIdx() + offset
            d = float(np.linalg.norm(pos[i] - pos[j])) if has_3d else 0.0
            feat = bond_features(bond, edge_type=0, distance=d)
            edge_index += [[i, j], [j, i]]
            edge_attr += [feat, feat]

    if use_cross_edges and cross_ion_mode == "distance_3d":
        for i in range(n_c):
            atom_i = c_mol.GetAtomWithIdx(i)
            donor_i, acceptor_i = _donor_acceptor(atom_i)
            for j0 in range(n_a):
                j = n_c + j0
                atom_j = a_mol.GetAtomWithIdx(j0)
                donor_j, acceptor_j = _donor_acceptor(atom_j)
                if has_3d:
                    d = float(np.linalg.norm(pos[i] - pos[j]))
                    if d > cutoff:
                        continue
                else:
                    if i != 0 or j0 != 0:
                        continue
                    d = 0.0
                qi = atom_i.GetFormalCharge()
                qj = atom_j.GetFormalCharge()
                hbond = float((donor_i and acceptor_j) or (donor_j and acceptor_i))
                aromatic = float(atom_i.GetIsAromatic() and atom_j.GetIsAromatic())
                vdw = float(has_3d and d < 4.0)
                feat = bond_features(None, edge_type=1, distance=d, charge_product=qi * qj, hbond=hbond, aromatic_pair=aromatic, vdw=vdw)
                edge_index += [[i, j], [j, i]]
                edge_attr += [feat, feat]
    elif use_cross_edges:
        for i, j0, feat_dict in _deterministic_cross_pairs(c_mol, a_mol):
            j = n_c + j0
            feat = bond_features(
                None,
                edge_type=1,
                distance=0.0,
                charge_product=feat_dict["charge_product"],
                hbond=feat_dict["hbond"],
                aromatic_pair=feat_dict["aromatic"],
                vdw=0.0,
            )
            edge_index += [[i, j], [j, i]]
            edge_attr += [feat, feat]

    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr = [[0.0] * EDGE_FEATURE_DIM]

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        pos=torch.tensor(pos, dtype=torch.float32),
        fragment_id=torch.tensor(fragment_id, dtype=torch.long),
        has_3d=torch.tensor([1 if has_3d else 0], dtype=torch.long),
        valid_flag=torch.tensor([1], dtype=torch.long),
        global_desc=torch.tensor(ion_pair_descriptors(c_mol, a_mol, c_pos, a_pos), dtype=torch.float32).view(1, GLOBAL_DESCRIPTOR_DIM),
    )
    data.smiles = smiles
    data.cation_smiles = parts.cation_smiles
    data.anion_smiles = parts.anion_smiles or ""
    if cross_ion_mode == "distance_3d" and has_3d:
        data.conformer_note = "approximate 3D ion-pair conformer with distance-based cross-ion edges"
    else:
        data.conformer_note = "single-ion 3D conformers with deterministic 2D virtual cross-ion edges"
    data.cross_ion_mode = cross_ion_mode
    data.split_warnings = "; ".join(parts.warnings)
    return GraphBuildResult(data, None)
