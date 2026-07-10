from __future__ import annotations

from dataclasses import asdict

from rdkit import Chem

from .fragment_library import IonFragment


def _formal_charge(smiles: str) -> int | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def generate_candidates(
    cations: list[IonFragment],
    anions: list[IonFragment],
    max_candidates: int | None = None,
) -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for cation in cations:
        c_charge = _formal_charge(cation.smiles)
        c_smiles = _canonical_smiles(cation.smiles)
        if c_charge is None or c_smiles is None or c_charge <= 0 or c_charge != cation.charge:
            rejected.append({"fragment": cation.name, "smiles": cation.smiles, "reason": "invalid cation charge or SMILES"})
            continue
        for anion in anions:
            a_charge = _formal_charge(anion.smiles)
            a_smiles = _canonical_smiles(anion.smiles)
            if a_charge is None or a_smiles is None or a_charge >= 0 or a_charge != anion.charge:
                rejected.append({"fragment": anion.name, "smiles": anion.smiles, "reason": "invalid anion charge or SMILES"})
                continue
            if c_charge + a_charge != 0:
                rejected.append({"cation": cation.name, "anion": anion.name, "reason": "non-neutral ion-pair charge"})
                continue
            il_smiles = f"{c_smiles}.{a_smiles}"
            if il_smiles in seen:
                continue
            mol = Chem.MolFromSmiles(il_smiles)
            if mol is None:
                rejected.append({"cation": cation.name, "anion": anion.name, "smiles": il_smiles, "reason": "invalid ion-pair SMILES"})
                continue
            seen.add(il_smiles)
            candidates.append(
                {
                    "candidate_id": f"{cation.name}__{anion.name}",
                    "cation_name": cation.name,
                    "anion_name": anion.name,
                    "IL_SMILES": il_smiles,
                    "cation": asdict(cation),
                    "anion": asdict(anion),
                    "family": f"{cation.family}/{anion.family}",
                }
            )
            if max_candidates is not None and len(candidates) >= max_candidates:
                return candidates, rejected
    return candidates, rejected
