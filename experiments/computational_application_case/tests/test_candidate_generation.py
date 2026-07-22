from __future__ import annotations

import pandas as pd
import pytest

from experiments.computational_application_case.src import chemistry
from experiments.computational_application_case.src.chemistry import (
    CandidateGenerationSettings,
    build_candidate_tables,
    canonical_pair_key,
    parse_monovalent_pair,
)


CATION_A = "C[N+](C)(C)C"
CATION_B = "CC[N+](C)(C)C"
ANION_CL = "[Cl-]"
ANION_BR = "[Br-]"
IMIDAZOLIUM_RESONANCE_A = "CC[n+]1ccn(C)c1"
IMIDAZOLIUM_RESONANCE_B = "CCn1cc[n+](C)c1"
ANION_OTF = "O=S(=O)([O-])C(F)(F)F"


def test_charge_based_roles_are_identified_independent_of_fragment_order() -> None:
    pair = parse_monovalent_pair(f"{ANION_CL}.{CATION_A}")
    assert pair.cation_charge == 1
    assert pair.anion_charge == -1
    assert pair.cation_smiles == CATION_A
    assert pair.anion_smiles == ANION_CL


def test_non_monovalent_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="monovalent"):
        parse_monovalent_pair("[Mg+2].[Cl-]")


def test_canonical_pair_key_deduplicates_fragment_order_and_smiles_order() -> None:
    forward = canonical_pair_key(f"{CATION_A}.{ANION_CL}")
    reversed_pair = canonical_pair_key(f"{ANION_CL}.{CATION_A}")
    equivalent = canonical_pair_key("[Cl-].C[N+](C)(C)C")
    assert forward == reversed_pair == equivalent


def test_pair_identity_is_invariant_to_imidazolium_resonance_smiles() -> None:
    first = canonical_pair_key(f"{IMIDAZOLIUM_RESONANCE_A}.{ANION_OTF}")
    second = canonical_pair_key(f"{IMIDAZOLIUM_RESONANCE_B}.{ANION_OTF}")
    assert first == second


def test_generation_excludes_observed_pairs_and_records_support() -> None:
    frame = pd.DataFrame(
        {
            "IL_Name": ["a-cl", "a-cl duplicate", "a-br", "b-cl"],
            "IL_SMILES": [
                f"{CATION_A}.{ANION_CL}",
                f"{ANION_CL}.{CATION_A}",
                f"{CATION_A}.{ANION_BR}",
                f"{CATION_B}.{ANION_CL}",
            ],
        }
    )
    settings = CandidateGenerationSettings(
        min_cation_support=1,
        min_anion_support=1,
        max_cations=10,
        max_anions=10,
        max_candidates=10,
        max_observed_references=3,
        require_monovalent_1to1=True,
        exclude_observed_pairs=True,
        descriptor_prefilter_multiplier=2,
        random_seed=42,
    )
    result = build_candidate_tables(frame, list(frame.index), settings)
    assert set(result.observed_references["canonical_il_key"]) == {
        canonical_pair_key(f"{CATION_A}.{ANION_CL}"),
        canonical_pair_key(f"{CATION_A}.{ANION_BR}"),
        canonical_pair_key(f"{CATION_B}.{ANION_CL}"),
    }
    candidate = result.candidates.iloc[0]
    assert candidate["canonical_il_key"] == canonical_pair_key(
        f"{CATION_B}.{ANION_BR}"
    )
    assert candidate["candidate_type"] == "unseen_pair_recombination"
    assert not bool(candidate["pair_seen_in_benchmark"])
    assert candidate["cation_support_count"] == 1
    assert candidate["anion_support_count"] == 1


def test_deterministic_truncation_returns_the_same_candidate() -> None:
    rows = []
    cations = [CATION_A, CATION_B, "CCC[N+](C)(C)C"]
    anions = [ANION_CL, ANION_BR, "[I-]"]
    for index, (cation, anion) in enumerate(
        [
            (cations[0], anions[0]),
            (cations[0], anions[1]),
            (cations[1], anions[0]),
            (cations[2], anions[2]),
        ]
    ):
        rows.append({"IL_Name": str(index), "IL_SMILES": f"{cation}.{anion}"})
    frame = pd.DataFrame(rows)
    settings = CandidateGenerationSettings(
        min_cation_support=1,
        min_anion_support=1,
        max_cations=3,
        max_anions=3,
        max_candidates=1,
        max_observed_references=2,
        require_monovalent_1to1=True,
        exclude_observed_pairs=True,
        descriptor_prefilter_multiplier=2,
        random_seed=42,
    )
    first = build_candidate_tables(frame, list(frame.index), settings)
    second = build_candidate_tables(frame, list(reversed(frame.index)), settings)
    assert first.candidates["canonical_il_key"].tolist() == second.candidates[
        "canonical_il_key"
    ].tolist()
    assert len(first.candidates) == 1
    assert (first.trace["removal_reason"] == "deterministic_max_candidates").any()


def test_repeated_source_smiles_are_parsed_once_without_dropping_rows(monkeypatch) -> None:
    source = f"{CATION_A}.{ANION_CL}"
    frame = pd.DataFrame({"IL_Name": ["a", "b", "c"], "IL_SMILES": [source] * 3})
    original = chemistry.parse_monovalent_pair
    calls = []

    def counted(smiles: str, require_monovalent: bool = True):
        calls.append(smiles)
        return original(smiles, require_monovalent)

    monkeypatch.setattr(chemistry, "parse_monovalent_pair", counted)
    parsed, failures = chemistry._parse_frame(frame, require_monovalent=True)
    assert calls == [source]
    assert len(parsed) == 3
    assert failures.empty
