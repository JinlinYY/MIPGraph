from __future__ import annotations

from collections import deque

import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski


ION_FUNCTIONAL_GROUP_NAMES = [
    "hbond_donor",
    "hbond_acceptor",
    "positive_atom_fraction",
    "negative_atom_fraction",
    "quaternary_ammonium",
    "imidazolium_like",
    "pyridinium_like",
    "phosphonium",
    "sulfonium",
    "ether",
    "hydroxyl",
    "nitrile",
    "carbonyl",
    "carboxylate",
    "sulfonyl",
    "sulfonate",
    "sulfate",
    "phosphate",
    "fluorinated_carbon",
    "borofluoride",
    "hexafluorophosphate",
    "halide_anion",
    "hetero_aromatic_atom",
    "aromatic_atom_fraction",
    "amide",
    "ester",
    "nitro",
    "thioether",
    "bis_sulfonimide",
    "charged_atom_fraction",
    "branched_aliphatic_carbon_fraction",
    "longest_aliphatic_carbon_chain",
]

PAIR_FUNCTIONAL_GROUP_NAMES = [
    "cation_positive_anion_negative",
    "anion_positive_cation_negative",
    "cation_hbd_anion_hba",
    "anion_hbd_cation_hba",
    "cation_aromatic_anion_aromatic",
    "total_fluorinated_carbon",
    "total_sulfonyl",
    "total_carboxylate",
    "alkyl_chain_length_difference",
    "alkyl_chain_length_sum",
    "cation_ether_anion_hba",
    "cation_hydroxyl_anion_hba",
    "imidazolium_sulfonyl_pair",
    "phosphonium_fluorinated_pair",
    "ammonium_halide_pair",
    "functional_group_dot",
]

ION_FUNCTIONAL_GROUP_DIM = len(ION_FUNCTIONAL_GROUP_NAMES)
PAIR_FUNCTIONAL_GROUP_DIM = len(PAIR_FUNCTIONAL_GROUP_NAMES)
FUNCTIONAL_GROUP_DESCRIPTOR_DIM = ION_FUNCTIONAL_GROUP_DIM * 2 + PAIR_FUNCTIONAL_GROUP_DIM


_SMARTS = {
    "quaternary_ammonium": "[NX4+]",
    "imidazolium_like": "[n+]1cc[n,c]c1",
    "pyridinium_like": "[n+]1ccccc1",
    "phosphonium": "[PX4+]",
    "sulfonium": "[SX3+]",
    "ether": "[OD2]([#6])[#6]",
    "hydroxyl": "[OX2H]",
    "nitrile": "[CX2]#N",
    "carbonyl": "[CX3]=[OX1]",
    "carboxylate": "[CX3](=O)[O-]",
    "sulfonyl": "[SX4](=[OX1])(=[OX1])",
    "sulfonate": "[SX4](=[OX1])(=[OX1])([O-])",
    "sulfate": "[SX4](=[OX1])(=[OX1])([O-,O])([O-,O])",
    "phosphate": "[PX4](=[OX1])([O-,O])([O-,O])([O-,O])",
    "fluorinated_carbon": "[#6]-[F]",
    "borofluoride": "[B-](F)(F)(F)F",
    "hexafluorophosphate": "[P-](F)(F)(F)(F)(F)F",
    "hetero_aromatic_atom": "[a;!#6]",
    "amide": "[NX3][CX3](=[OX1])",
    "ester": "[CX3](=O)[OX2][#6]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "thioether": "[SX2]([#6])[#6]",
    "bis_sulfonimide": "[NX3,NX2-]([SX4](=[OX1])(=[OX1]))[SX4](=[OX1])(=[OX1])",
}

_COMPILED_SMARTS = {
    name: Chem.MolFromSmarts(pattern)
    for name, pattern in _SMARTS.items()
}


def _count_pattern(mol: Chem.Mol, name: str) -> int:
    pattern = _COMPILED_SMARTS.get(name)
    if pattern is None:
        return 0
    try:
        return len(mol.GetSubstructMatches(pattern))
    except Exception:
        return 0


def _halide_anion_count(mol: Chem.Mol) -> int:
    halogens = {9, 17, 35, 53}
    return sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() in halogens and atom.GetFormalCharge() < 0
    )


def _longest_aliphatic_carbon_chain(mol: Chem.Mol) -> int:
    carbon_ids = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic()
    ]
    if not carbon_ids:
        return 0
    carbon_set = set(carbon_ids)
    adjacency = {idx: [] for idx in carbon_ids}
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in carbon_set and end in carbon_set:
            adjacency[begin].append(end)
            adjacency[end].append(begin)
    longest = 1
    for start in carbon_ids:
        visited = {start}
        queue: deque[tuple[int, int]] = deque([(start, 1)])
        while queue:
            node, length = queue.popleft()
            longest = max(longest, length)
            for neighbor in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, length + 1))
    return longest


def ion_functional_group_descriptors(mol: Chem.Mol) -> list[float]:
    atoms = list(mol.GetAtoms())
    heavy = max(float(mol.GetNumHeavyAtoms()), 1.0)
    positive_atoms = sum(1 for atom in atoms if atom.GetFormalCharge() > 0)
    negative_atoms = sum(1 for atom in atoms if atom.GetFormalCharge() < 0)
    charged_atoms = positive_atoms + negative_atoms
    aromatic_atoms = sum(1 for atom in atoms if atom.GetIsAromatic())
    branched_aliphatic_carbons = sum(
        1
        for atom in atoms
        if atom.GetAtomicNum() == 6
        and not atom.GetIsAromatic()
        and sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 6) >= 3
    )
    longest_chain = _longest_aliphatic_carbon_chain(mol)
    values = [
        float(Lipinski.NumHDonors(mol)) / 10.0,
        float(Lipinski.NumHAcceptors(mol)) / 20.0,
        positive_atoms / heavy,
        negative_atoms / heavy,
        _count_pattern(mol, "quaternary_ammonium") / heavy,
        _count_pattern(mol, "imidazolium_like") / heavy,
        _count_pattern(mol, "pyridinium_like") / heavy,
        _count_pattern(mol, "phosphonium") / heavy,
        _count_pattern(mol, "sulfonium") / heavy,
        _count_pattern(mol, "ether") / heavy,
        _count_pattern(mol, "hydroxyl") / heavy,
        _count_pattern(mol, "nitrile") / heavy,
        _count_pattern(mol, "carbonyl") / heavy,
        _count_pattern(mol, "carboxylate") / heavy,
        _count_pattern(mol, "sulfonyl") / heavy,
        _count_pattern(mol, "sulfonate") / heavy,
        _count_pattern(mol, "sulfate") / heavy,
        _count_pattern(mol, "phosphate") / heavy,
        _count_pattern(mol, "fluorinated_carbon") / heavy,
        _count_pattern(mol, "borofluoride") / heavy,
        _count_pattern(mol, "hexafluorophosphate") / heavy,
        _halide_anion_count(mol) / heavy,
        _count_pattern(mol, "hetero_aromatic_atom") / heavy,
        aromatic_atoms / heavy,
        _count_pattern(mol, "amide") / heavy,
        _count_pattern(mol, "ester") / heavy,
        _count_pattern(mol, "nitro") / heavy,
        _count_pattern(mol, "thioether") / heavy,
        _count_pattern(mol, "bis_sulfonimide") / heavy,
        charged_atoms / heavy,
        branched_aliphatic_carbons / heavy,
        longest_chain / 30.0,
    ]
    return [float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for value in values]


def ion_pair_functional_group_descriptors(cation: Chem.Mol, anion: Chem.Mol) -> list[float]:
    c_fg = ion_functional_group_descriptors(cation)
    a_fg = ion_functional_group_descriptors(anion)
    pair = [
        c_fg[2] * a_fg[3],
        a_fg[2] * c_fg[3],
        c_fg[0] * a_fg[1],
        a_fg[0] * c_fg[1],
        c_fg[23] * a_fg[23],
        c_fg[18] + a_fg[18],
        c_fg[14] + a_fg[14],
        c_fg[13] + a_fg[13],
        abs(c_fg[31] - a_fg[31]),
        c_fg[31] + a_fg[31],
        c_fg[9] * a_fg[1],
        c_fg[10] * a_fg[1],
        (c_fg[5] + c_fg[6]) * (a_fg[14] + a_fg[15]),
        c_fg[7] * (a_fg[18] + a_fg[20]),
        c_fg[4] * a_fg[21],
        float(np.dot(np.asarray(c_fg, dtype=np.float32), np.asarray(a_fg, dtype=np.float32)) / ION_FUNCTIONAL_GROUP_DIM),
    ]
    return c_fg + a_fg + [float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for value in pair]
