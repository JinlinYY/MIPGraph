from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class IonFragment:
    name: str
    smiles: str
    family: str
    charge: int
    notes: str = ""


def read_fragment_csv(path: str | Path) -> list[IonFragment]:
    df = pd.read_csv(path)
    required = {"name", "smiles", "family", "charge", "notes"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Fragment library {path} is missing columns: {sorted(missing)}")
    fragments = []
    for _, row in df.iterrows():
        fragments.append(
            IonFragment(
                name=str(row["name"]),
                smiles=str(row["smiles"]),
                family=str(row["family"]),
                charge=int(row["charge"]),
                notes="" if pd.isna(row["notes"]) else str(row["notes"]),
            )
        )
    return fragments


def load_fragment_library(cations_path: str | Path, anions_path: str | Path) -> tuple[list[IonFragment], list[IonFragment]]:
    return read_fragment_csv(cations_path), read_fragment_csv(anions_path)
