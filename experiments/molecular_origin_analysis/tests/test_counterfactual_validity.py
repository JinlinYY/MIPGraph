"""Chemical-validity tests for real-SMILES counterfactual templates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem

from experiments.molecular_origin_analysis.src.counterfactual import CounterfactualGenerator
from experiments.molecular_origin_analysis.src.utils import load_config


def _charge(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def test_virtual_templates_generate_unique_charge_neutral_ion_pairs(config_path) -> None:
    config = load_config(config_path)
    generator = CounterfactualGenerator(config)
    library = generator.generate_virtual_library()

    assert not library.empty
    assert library["canonical_il_smiles"].is_unique
    assert set(library["validation_status"]) == {"valid"}
    for row in library.itertuples(index=False):
        assert Chem.MolFromSmiles(row.cation_smiles) is not None
        assert Chem.MolFromSmiles(row.anion_smiles) is not None
        assert _charge(row.cation_smiles) == 1
        assert _charge(row.anion_smiles) == -1
        assert _charge(row.canonical_il_smiles) == 0
        assert row.modification_type
        assert row.template_id


def test_invalid_or_duplicate_templates_are_reported_not_silently_used(
    config_path,
) -> None:
    config = load_config(config_path)
    generator = CounterfactualGenerator(config)
    valid, rejected = generator.validate_records(
        [
            {
                "template_id": "valid",
                "modification_type": "anion substitution",
                "cation_smiles": "C[N+](C)(C)C",
                "anion_smiles": "[Cl-]",
            },
            {
                "template_id": "invalid",
                "modification_type": "invalid structure",
                "cation_smiles": "not-a-smiles",
                "anion_smiles": "[Cl-]",
            },
            {
                "template_id": "duplicate",
                "modification_type": "duplicate",
                "cation_smiles": "C[N+](C)(C)C",
                "anion_smiles": "[Cl-]",
            },
        ]
    )

    assert len(valid) == 1
    assert set(rejected["rejection_reason"]) == {"rdkit_parse_failure", "duplicate_identity"}


def test_matched_pairs_preserve_observed_and_predicted_response_differences(
    config_path,
) -> None:
    config = load_config(config_path)
    frame = pd.DataFrame(
        {
            "sample_id": [1, 2],
            "cation_smiles": ["C[N+](C)(C)C", "CC[N+](C)(C)C"],
            "anion_smiles": ["[Cl-]", "[Cl-]"],
            "Temperature_K": [298.15, 298.25],
            "Pressure_kPa": [101.325, 101.325],
            "Density_ActualValue": [1000.0, 1100.0],
            "Density_mask": [1.0, 1.0],
            "prediction_Density": [1020.0, 1080.0],
        }
    )
    pairs = CounterfactualGenerator(config).matched_pairs(frame)

    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["fixed_role"] == "anion_fixed"
    assert bool(row["both_observed_Density"])
    assert np.isclose(
        row["observed_abs_log_difference_Density"],
        abs(np.log(1100.0) - np.log(1000.0)),
    )
    assert np.isclose(
        row["predicted_abs_log_difference_Density"],
        abs(np.log(1080.0) - np.log(1020.0)),
    )
