from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from rdkit import Chem
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.chem.functional_groups import FUNCTIONAL_GROUP_DESCRIPTOR_DIM, ion_pair_functional_group_descriptors


def safe_torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/processed/graph_cache.pt")
    parser.add_argument("--output", default="data/processed/graph_cache_fg.pt")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = (PROJECT_DIR / args.source).resolve()
    output = (PROJECT_DIR / args.output).resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output}. Pass --force to overwrite.")
    cache = safe_torch_load(source)
    if not isinstance(cache, dict) or not cache:
        raise ValueError(f"Invalid graph cache: {source}")

    augmented = {}
    failures = []
    for smiles, graph in tqdm(cache.items(), desc="Add functional groups"):
        cation_smiles = str(getattr(graph, "cation_smiles", ""))
        anion_smiles = str(getattr(graph, "anion_smiles", ""))
        cation = Chem.MolFromSmiles(cation_smiles)
        anion = Chem.MolFromSmiles(anion_smiles)
        if cation is None or anion is None:
            failures.append({"IL_SMILES": smiles, "cation": cation_smiles, "anion": anion_smiles})
            continue
        item = graph.clone()
        item.functional_group_desc = torch.tensor(
            ion_pair_functional_group_descriptors(cation, anion),
            dtype=torch.float32,
        ).view(1, FUNCTIONAL_GROUP_DESCRIPTOR_DIM)
        augmented[smiles] = item

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(augmented, output)
    print({"source": str(source), "output": str(output), "graphs": len(augmented), "failures": len(failures)})
    if failures:
        print({"failure_preview": failures[:5]})


if __name__ == "__main__":
    main()
