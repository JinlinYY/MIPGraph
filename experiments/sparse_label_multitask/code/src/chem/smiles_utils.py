from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass
class IonPairParts:
    cation_smiles: str | None
    anion_smiles: str | None
    warnings: list[str]


def canonicalize_smiles(smiles: str) -> tuple[str | None, str | None]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "RDKit MolFromSmiles returned None"
        return Chem.MolToSmiles(mol, canonical=True), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def formal_charge(mol: Chem.Mol) -> int:
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def split_ion_pair(smiles: str) -> IonPairParts:
    warnings: list[str] = []
    fragments = [frag for frag in str(smiles).split(".") if frag.strip()]
    if not fragments:
        return IonPairParts(None, None, ["empty SMILES"])
    parsed = []
    for frag in fragments:
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            warnings.append(f"failed fragment parse: {frag}")
            continue
        parsed.append((frag, mol, formal_charge(mol)))
    cations = [frag for frag, _, charge in parsed if charge > 0]
    anions = [frag for frag, _, charge in parsed if charge < 0]
    if cations and anions:
        return IonPairParts(".".join(cations), ".".join(anions), warnings)
    warnings.append("charge-based split ambiguous; using original fragment order")
    if len(fragments) >= 2:
        return IonPairParts(fragments[0], ".".join(fragments[1:]), warnings)
    return IonPairParts(fragments[0], None, warnings)
