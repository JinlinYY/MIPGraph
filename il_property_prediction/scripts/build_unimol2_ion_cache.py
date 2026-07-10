from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataset import safe_torch_load


@lru_cache(maxsize=1)
def chemistry_feature_factory():
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    return ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))


def atom_chemistry(smiles: str):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse ion SMILES: {smiles}")
    molecule = Chem.RemoveHs(molecule)
    donors: set[int] = set()
    acceptors: set[int] = set()
    for feature in chemistry_feature_factory().GetFeaturesForMol(molecule):
        if feature.GetFamily() == "Donor":
            donors.update(int(index) for index in feature.GetAtomIds())
        elif feature.GetFamily() == "Acceptor":
            acceptors.update(int(index) for index in feature.GetAtomIds())
    rows = []
    for index, atom in enumerate(molecule.GetAtoms()):
        rows.append(
            [
                float(atom.GetFormalCharge()),
                float(index in donors),
                float(index in acceptors),
                float(atom.GetIsAromatic()),
                float(atom.GetAtomicNum()),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic Uni-Mol2 features for unique IL ions.")
    parser.add_argument("--graph-cache", default="data/processed/graph_cache.pt")
    parser.add_argument("--output", default="data/processed/unimol2_ion_features.pt")
    parser.add_argument("--max-atoms", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    graph_cache_path = (PROJECT_DIR / args.graph_cache).resolve()
    output_path = (PROJECT_DIR / args.output).resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --force to replace it.")

    graph_cache = safe_torch_load(graph_cache_path)
    ions: set[str] = set()
    for graph in graph_cache.values():
        ions.add(str(graph.cation_smiles))
        ions.add(str(graph.anion_smiles))

    from unimol_tools.data.conformer import UniMolV2Feature

    generator = UniMolV2Feature(
        seed=args.seed,
        max_atoms=args.max_atoms,
        remove_hs=True,
        multi_process=False,
        mode="fast",
    )
    features = {}
    chemistry = {}
    failures = {}
    for smiles in tqdm(sorted(ions), desc="Uni-Mol2 ion features"):
        try:
            feature, _ = generator.single_process(smiles)
            atom_annotations = atom_chemistry(smiles)
            source_tokens = np.asarray(feature["src_tokens"])
            if len(atom_annotations) != len(source_tokens):
                raise ValueError(
                    f"Atom alignment mismatch: chemistry={len(atom_annotations)}, "
                    f"Uni-Mol2={len(source_tokens)}"
                )
            if not (atom_annotations[:, 4].astype(source_tokens.dtype) == source_tokens).all():
                raise ValueError("Atomic-number order does not match the Uni-Mol2 feature order")
            features[smiles] = feature
            chemistry[smiles] = atom_annotations
        except Exception as exc:
            failures[smiles] = f"{type(exc).__name__}: {exc}"

    if failures:
        preview = list(failures.items())[:5]
        raise RuntimeError(f"Failed to build {len(failures)} ion features. Examples: {preview}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "unimol2_ion_features_v1",
            "source_graph_cache": str(graph_cache_path),
            "seed": args.seed,
            "max_atoms": args.max_atoms,
            "remove_hs": True,
            "ion_count": len(features),
            "features": features,
            "chemistry_columns": ["formal_charge", "donor", "acceptor", "aromatic", "atomic_number"],
            "chemistry": chemistry,
        },
        output_path,
    )
    print({"output": str(output_path), "ion_count": len(features), "failures": 0})


if __name__ == "__main__":
    main()
