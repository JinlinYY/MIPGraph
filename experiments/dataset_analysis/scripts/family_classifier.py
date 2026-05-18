"""Cation / anion family classifier for ionic-liquid SMILES.

The classifier accepts either:

* a full ionic-liquid SMILES of the form ``cation.anion`` (one ``.`` separator),
* a single-component SMILES (e.g. just the cation or just the anion).

It returns a coarse chemical family label (e.g. ``Imidazolium``, ``NTf2``)
based on RDKit substructure matching.  An ordered priority list is used so
that more specific patterns are tested before more generic ones (e.g.
``Morpholinium`` is tested before generic ``Quaternary ammonium``).

The module is importable and also runnable as a CLI for quick inspection::

    python family_classifier.py --smiles "CCN1C=C[N+](C)=C1.F[B-](F)(F)F"
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Family pattern definitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FamilyPattern:
    """A single family rule defined by a SMARTS pattern."""

    name: str
    smarts: str
    description: str = ""

    def compile(self) -> Chem.Mol:
        mol = Chem.MolFromSmarts(self.smarts)
        if mol is None:
            raise ValueError(f"Invalid SMARTS for family {self.name!r}: {self.smarts}")
        return mol


# Order matters: more specific patterns first.
CATION_FAMILY_PATTERNS: Sequence[FamilyPattern] = (
    FamilyPattern("Imidazolium",   "[n+;R]1[c;R][n;R][c;R][c;R]1",
                  "1,3-disubstituted imidazolium ring"),
    FamilyPattern("Pyridinium",    "[n+;R]1[c;R][c;R][c;R][c;R][c;R]1"),
    FamilyPattern("Pyrrolidinium", "[N+;R;X4]1[C;R][C;R][C;R][C;R]1"),
    FamilyPattern("Piperidinium",  "[N+;R;X4]1[C;R][C;R][C;R][C;R][C;R]1"),
    FamilyPattern("Morpholinium",  "[N+;R;X4]1[C;R][C;R][O;R][C;R][C;R]1"),
    FamilyPattern("Pyrazolium",    "[n+;R]1[c,n;R][c;R][c;R][n;R]1"),
    FamilyPattern("Triazolium",    "[n+;R]1[c,n;R][n;R][n;R][c,n;R]1"),
    FamilyPattern("Thiazolium",    "[n+;R]1[c;R][s;R][c;R][c;R]1"),
    FamilyPattern("Quinolinium",   "[n+;R]1[c;R][c;R][c;R]c2ccccc12"),
    FamilyPattern("Phosphonium",   "[P+;X4]"),
    FamilyPattern("Sulfonium",     "[S+;X3;!R]"),
    FamilyPattern("Guanidinium",   "[N;X3]C(=[N+])[N;X3]"),
    FamilyPattern("Cholinium",     "[N+;X4](C)(C)(C)CCO"),
    FamilyPattern("Quaternary ammonium",
                  "[N+;X4;H0;!$([N+]=*)]",
                  "Acyclic / generic fully substituted quaternary ammonium"),
    FamilyPattern("Protic ammonium",
                  "[N+;H1,H2,H3,h1,h2,h3]",
                  "Protonated primary/secondary/tertiary amine"),
)

ANION_FAMILY_PATTERNS: Sequence[FamilyPattern] = (
    FamilyPattern("NTf2",
                  "[N-]([S](=O)(=O)C(F)(F)F)[S](=O)(=O)C(F)(F)F",
                  "Bis(trifluoromethylsulfonyl)imide"),
    FamilyPattern("FSI",
                  "[N-]([S](=O)(=O)F)[S](=O)(=O)F",
                  "Bis(fluorosulfonyl)imide"),
    FamilyPattern("Triflate",
                  "[O-][S](=O)(=O)C(F)(F)F"),
    FamilyPattern("Tosylate",
                  "[O-][S](=O)(=O)c1ccc(C)cc1"),
    FamilyPattern("BF4",  "[B-](F)(F)(F)F"),
    FamilyPattern("PF6",  "[P-](F)(F)(F)(F)(F)F"),
    FamilyPattern("SbF6", "[Sb-](F)(F)(F)(F)(F)F"),
    FamilyPattern("AsF6", "[As-](F)(F)(F)(F)(F)F"),
    FamilyPattern("Dicyanamide", "[N-](C#N)C#N"),
    FamilyPattern("Tricyanomethanide", "[C-](C#N)(C#N)C#N"),
    FamilyPattern("Tetracyanoborate", "[B-](C#N)(C#N)(C#N)C#N"),
    FamilyPattern("Thiocyanate", "[S-]C#N"),
    FamilyPattern("Cyanide", "[C-]#N"),
    FamilyPattern("Nitrate", "[O-][N+](=O)[O-]"),
    FamilyPattern("Nitrite", "[O-]N=O"),
    FamilyPattern("Perchlorate", "[O-][Cl](=O)(=O)=O"),
    FamilyPattern("Sulfate", "[O-]S(=O)(=O)[O-]"),
    FamilyPattern("Hydrogen sulfate", "[O-]S(=O)(=O)O"),
    FamilyPattern("Alkyl sulfate", "[O-]S(=O)(=O)O[#6]"),
    FamilyPattern("Alkyl sulfonate", "[O-]S(=O)(=O)[#6]"),
    FamilyPattern("Phosphate", "[O-]P(=O)([O-])[O-]"),
    FamilyPattern("Alkyl phosphate", "[O-]P(=O)(O[#6])O[#6]"),
    FamilyPattern("Borate (alkoxy / aryl)", "[B-]([O,#6])([O,#6])([O,#6])[O,#6]"),
    FamilyPattern("Amino acid",
                  "[NX3,NX4+;!H0][C;X4][C](=O)[O-]",
                  "Alpha-amino carboxylate"),
    FamilyPattern("Carboxylate", "[CX3](=O)[O-]"),
    FamilyPattern("Phenolate", "[O-]c1ccccc1"),
    FamilyPattern("Alkoxide",  "[O-][CX4]"),
    FamilyPattern("Halide",    "[F,Cl,Br,I;-]"),
    FamilyPattern("Hydroxide", "[OH-]"),
    FamilyPattern("Hydride",   "[H-]"),
)


# --------------------------------------------------------------------------- #
# Compiled cache
# --------------------------------------------------------------------------- #
def _compile_patterns(patterns: Sequence[FamilyPattern]) -> List[Tuple[str, Chem.Mol]]:
    return [(p.name, p.compile()) for p in patterns]


_CATION_COMPILED = _compile_patterns(CATION_FAMILY_PATTERNS)
_ANION_COMPILED = _compile_patterns(ANION_FAMILY_PATTERNS)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def split_il_smiles(il_smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """Split an ionic-liquid SMILES (``cation.anion``) into its components.

    Returns a tuple ``(cation_smiles, anion_smiles)``.  Components are
    distinguished by formal charge: the fragment containing a positive
    formal charge is the cation, the fragment containing the negative
    formal charge is the anion.  If only one fragment is present, the
    other tuple element is ``None``.
    """

    if il_smiles is None or not isinstance(il_smiles, str):
        return None, None

    mol = Chem.MolFromSmiles(il_smiles)
    if mol is None:
        return None, None

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

    cation_smi: Optional[str] = None
    anion_smi: Optional[str] = None
    neutral_smis: List[str] = []
    for frag in frags:
        charge = sum(a.GetFormalCharge() for a in frag.GetAtoms())
        smi = Chem.MolToSmiles(frag)
        if charge > 0 and cation_smi is None:
            cation_smi = smi
        elif charge < 0 and anion_smi is None:
            anion_smi = smi
        else:
            neutral_smis.append(smi)

    # If we still have ambiguity (e.g. zwitterion), fall back to dot split.
    if cation_smi is None and anion_smi is None and "." in il_smiles:
        parts = il_smiles.split(".")
        if len(parts) == 2:
            return parts[0], parts[1]

    return cation_smi, anion_smi


def _classify(smiles: Optional[str],
              compiled: Sequence[Tuple[str, Chem.Mol]]) -> str:
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return "Unknown"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Unknown"
    for name, patt in compiled:
        if mol.HasSubstructMatch(patt):
            return name
    return "Other"


def classify_cation(cation_smiles: Optional[str]) -> str:
    """Return the cation family for a single-cation SMILES."""

    return _classify(cation_smiles, _CATION_COMPILED)


def classify_anion(anion_smiles: Optional[str]) -> str:
    """Return the anion family for a single-anion SMILES."""

    return _classify(anion_smiles, _ANION_COMPILED)


def classify_il(il_smiles: str) -> Tuple[str, str]:
    """Classify a full ionic-liquid SMILES into ``(cation_family, anion_family)``."""

    cat, an = split_il_smiles(il_smiles)
    return classify_cation(cat), classify_anion(an)


def classify_dataframe(df, smiles_column: str = "IL_SMILES") -> "pandas.DataFrame":
    """Add ``Cation_SMILES``, ``Anion_SMILES``, ``Cation_Family``,
    ``Anion_Family`` columns to *df* (returned as a new DataFrame).
    """

    import pandas as pd

    out = df.copy()
    splits = out[smiles_column].apply(split_il_smiles)
    out["Cation_SMILES"] = splits.apply(lambda t: t[0])
    out["Anion_SMILES"] = splits.apply(lambda t: t[1])
    out["Cation_Family"] = out["Cation_SMILES"].apply(classify_cation)
    out["Anion_Family"] = out["Anion_SMILES"].apply(classify_anion)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify the cation / anion family of an ionic-liquid SMILES.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles", help="A single IL SMILES (cation.anion).")
    group.add_argument("--list-families", action="store_true",
                       help="List all supported family names and exit.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    if args.list_families:
        print("Cation families (priority order):")
        for p in CATION_FAMILY_PATTERNS:
            print(f"  - {p.name:<24s} {p.smarts}")
        print("\nAnion families (priority order):")
        for p in ANION_FAMILY_PATTERNS:
            print(f"  - {p.name:<24s} {p.smarts}")
        return

    cat_smi, an_smi = split_il_smiles(args.smiles)
    cat_fam = classify_cation(cat_smi)
    an_fam = classify_anion(an_smi)
    print(f"Input        : {args.smiles}")
    print(f"Cation SMILES: {cat_smi}")
    print(f"Anion  SMILES: {an_smi}")
    print(f"Cation family: {cat_fam}")
    print(f"Anion  family: {an_fam}")


if __name__ == "__main__":
    main()
