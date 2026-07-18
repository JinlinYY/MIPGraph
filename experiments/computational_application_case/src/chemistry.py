"""Charge-aware ion parsing and deterministic unseen-pair generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem

from il_property_prediction.src.chem.global_descriptors import ion_pair_descriptors
from il_property_prediction.src.chem.smiles_utils import (
    canonicalize_smiles,
    formal_charge,
    split_ion_pair,
)


@dataclass(frozen=True)
class ParsedIonPair:
    """Validated cation/anion assignment with source and canonical strings."""

    cation_smiles: str
    anion_smiles: str
    canonical_cation_smiles: str
    canonical_anion_smiles: str
    canonical_il_key: str
    cation_charge: int
    anion_charge: int
    warnings: tuple[str, ...]

    @property
    def il_smiles(self) -> str:
        """Return model-facing cation-first ion-pair SMILES."""

        return f"{self.cation_smiles}.{self.anion_smiles}"


@dataclass(frozen=True)
class CandidateGenerationSettings:
    """Configuration governing deterministic library construction."""

    min_cation_support: int
    min_anion_support: int
    max_cations: int
    max_anions: int
    max_candidates: int
    max_observed_references: int
    require_monovalent_1to1: bool = True
    exclude_observed_pairs: bool = True
    descriptor_prefilter_multiplier: int = 5
    random_seed: int = 42


@dataclass
class CandidateGenerationResult:
    """All candidate-generation tables, including failures and trace rows."""

    cations: pd.DataFrame
    anions: pd.DataFrame
    observed_references: pd.DataFrame
    candidates: pd.DataFrame
    trace: pd.DataFrame
    failures: pd.DataFrame


def _canonical_or_raise(smiles: str, role: str) -> str:
    canonical, error = canonicalize_smiles(smiles)
    if canonical is None:
        raise ValueError(f"Could not canonicalize {role}: {error}")
    return canonical


def parse_monovalent_pair(
    smiles: str,
    require_monovalent: bool = True,
) -> ParsedIonPair:
    """Parse a single 1:1 ion pair with the current charge-based splitter."""

    parts = split_ion_pair(str(smiles))
    if parts.cation_smiles is None or parts.anion_smiles is None:
        raise ValueError("Ion pair must contain one cation and one anion")
    if "." in parts.cation_smiles or "." in parts.anion_smiles:
        raise ValueError("Ion pair must contain exactly one cation and one anion fragment")
    cation = Chem.MolFromSmiles(parts.cation_smiles)
    anion = Chem.MolFromSmiles(parts.anion_smiles)
    if cation is None or anion is None:
        raise ValueError("RDKit could not parse the assigned ion fragments")
    cation_charge = int(formal_charge(cation))
    anion_charge = int(formal_charge(anion))
    if cation_charge <= 0 or anion_charge >= 0:
        raise ValueError(
            f"Charge-based ion roles are invalid: {cation_charge}, {anion_charge}"
        )
    if require_monovalent and (cation_charge != 1 or anion_charge != -1):
        raise ValueError(
            "The computational application case requires a monovalent +1/-1 ion pair"
        )
    canonical_cation = _canonical_or_raise(parts.cation_smiles, "cation")
    canonical_anion = _canonical_or_raise(parts.anion_smiles, "anion")
    return ParsedIonPair(
        cation_smiles=str(parts.cation_smiles),
        anion_smiles=str(parts.anion_smiles),
        canonical_cation_smiles=canonical_cation,
        canonical_anion_smiles=canonical_anion,
        canonical_il_key=f"{canonical_cation}||{canonical_anion}",
        cation_charge=cation_charge,
        anion_charge=anion_charge,
        warnings=tuple(parts.warnings),
    )


def canonical_pair_key(smiles: str) -> str:
    """Return the charge-role-aware canonical key for an ion-pair SMILES."""

    return parse_monovalent_pair(smiles).canonical_il_key


def _ion_family(smiles: str, role: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "unresolved"
    atomic_numbers = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
    if role == "anion":
        if molecule.GetNumAtoms() == 1 and atomic_numbers & {9, 17, 35, 53}:
            return "halide"
        text = Chem.MolToSmiles(molecule, canonical=True)
        if "[B-](F)(F)(F)F" in text:
            return "tetrafluoroborate"
        if "[P-](F)(F)(F)(F)(F)F" in text:
            return "hexafluorophosphate"
        if 16 in atomic_numbers:
            return "sulfur_containing"
        if 7 in atomic_numbers:
            return "nitrogen_containing"
        if 8 in atomic_numbers:
            return "oxygen_containing"
        return "other_anion"
    charged = [atom for atom in molecule.GetAtoms() if atom.GetFormalCharge() > 0]
    if any(atom.GetAtomicNum() == 15 for atom in charged):
        return "phosphonium"
    if any(atom.GetAtomicNum() == 16 for atom in charged):
        return "sulfonium"
    if any(atom.GetAtomicNum() == 7 and atom.GetIsAromatic() for atom in charged):
        return "aromatic_nitrogen_cation"
    if any(atom.GetAtomicNum() == 7 for atom in charged):
        return "ammonium"
    return "other_cation"


def _parse_frame(
    frame: pd.DataFrame,
    require_monovalent: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    parsed_cache: dict[str, ParsedIonPair] = {}
    failure_cache: dict[str, tuple[str, str]] = {}
    for position, (_, row) in enumerate(frame.iterrows()):
        source_smiles = str(row.get("IL_SMILES", ""))
        if source_smiles not in parsed_cache and source_smiles not in failure_cache:
            try:
                parsed_cache[source_smiles] = parse_monovalent_pair(
                    source_smiles, require_monovalent
                )
            except (TypeError, ValueError) as exc:
                failure_cache[source_smiles] = (type(exc).__name__, str(exc))
        if source_smiles in parsed_cache:
            pair = parsed_cache[source_smiles]
            rows.append(
                {
                    "row_position": position,
                    "IL_Name": str(row.get("IL_Name", "")),
                    "source_il_smiles": source_smiles,
                    **pair.__dict__,
                    "il_smiles": pair.il_smiles,
                }
            )
        else:
            exception_type, exception_message = failure_cache[source_smiles]
            failures.append(
                {
                    "row_position": position,
                    "IL_Name": str(row.get("IL_Name", "")),
                    "IL_SMILES": source_smiles,
                    "failed_stage": "candidate_parsing",
                    "exception_type": exception_type,
                    "exception_message": exception_message,
                    "excluded_from_analysis": True,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(failures)


def _library(
    unique_pairs: pd.DataFrame,
    role: str,
    support: pd.Series,
) -> pd.DataFrame:
    canonical_column = f"canonical_{role}_smiles"
    source_column = f"{role}_smiles"
    charge_column = f"{role}_charge"
    representatives = (
        unique_pairs.sort_values([canonical_column, source_column])
        .drop_duplicates(canonical_column)
        .set_index(canonical_column)
    )
    rows = []
    for canonical, count in support.items():
        representative = representatives.loc[canonical]
        rows.append(
            {
                f"canonical_{role}_smiles": canonical,
                f"{role}_smiles": representative[source_column],
                f"{role}_charge": int(representative[charge_column]),
                f"{role}_support_count": int(count),
                f"{role}_family": _ion_family(str(representative[source_column]), role),
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    return output.sort_values(
        [f"{role}_support_count", canonical_column], ascending=[False, True]
    ).reset_index(drop=True)


def _descriptor_vector(cation: str, anion: str) -> np.ndarray:
    cation_molecule = Chem.MolFromSmiles(cation)
    anion_molecule = Chem.MolFromSmiles(anion)
    if cation_molecule is None or anion_molecule is None:
        raise ValueError("RDKit descriptor parsing failed")
    return np.asarray(
        ion_pair_descriptors(cation_molecule, anion_molecule, None, None),
        dtype=float,
    )


def _deterministic_descriptor_selection(
    candidates: pd.DataFrame,
    maximum: int,
    prefilter_multiplier: int,
) -> pd.DataFrame:
    if len(candidates) <= maximum:
        return candidates.reset_index(drop=True)
    ranked = candidates.sort_values(
        ["minimum_ion_support", "combined_ion_support", "canonical_il_key"],
        ascending=[False, False, True],
    ).head(max(maximum, maximum * prefilter_multiplier))
    matrix = np.vstack(
        [
            _descriptor_vector(row.cation_smiles, row.anion_smiles)
            for row in ranked.itertuples(index=False)
        ]
    )
    scale = np.nanstd(matrix, axis=0)
    keep = np.isfinite(scale) & (scale > 1.0e-12)
    normalized = (
        (matrix[:, keep] - np.nanmean(matrix[:, keep], axis=0)) / scale[keep]
        if keep.any()
        else np.zeros((len(matrix), 1), dtype=float)
    )
    if not np.isfinite(normalized).all():
        raise ValueError("Descriptor coverage matrix contains NaN or infinity")
    selected = [0]
    minimum_distance = np.linalg.norm(normalized - normalized[0], axis=1)
    minimum_distance[0] = -np.inf
    while len(selected) < maximum:
        best_distance = float(np.max(minimum_distance))
        tied = np.flatnonzero(
            np.isclose(minimum_distance, best_distance, rtol=0.0, atol=1.0e-12)
        ).tolist()
        chosen = min(
            tied,
            key=lambda index: (
                -int(ranked.iloc[index]["minimum_ion_support"]),
                -int(ranked.iloc[index]["combined_ion_support"]),
                str(ranked.iloc[index]["canonical_il_key"]),
            ),
        )
        selected.append(chosen)
        new_distance = np.linalg.norm(normalized - normalized[chosen], axis=1)
        minimum_distance = np.minimum(minimum_distance, new_distance)
        minimum_distance[np.asarray(selected, dtype=int)] = -np.inf
    return ranked.iloc[selected].reset_index(drop=True)


def build_candidate_tables(
    benchmark: pd.DataFrame,
    training_indices: Sequence[int],
    settings: CandidateGenerationSettings,
) -> CandidateGenerationResult:
    """Build observed references and charge-valid unseen pair recombinations."""

    if "IL_SMILES" not in benchmark.columns:
        raise ValueError("Benchmark must contain IL_SMILES")
    parsed, failures = _parse_frame(benchmark, settings.require_monovalent_1to1)
    if parsed.empty:
        raise ValueError("No charge-valid ion pairs were parsed from the benchmark")
    unique_pairs = (
        parsed.sort_values(["canonical_il_key", "source_il_smiles"])
        .drop_duplicates("canonical_il_key")
        .reset_index(drop=True)
    )
    training_positions = {int(value) for value in training_indices}
    training_pairs = parsed[parsed["row_position"].isin(training_positions)].drop_duplicates(
        "canonical_il_key"
    )
    cation_support = training_pairs.groupby("canonical_cation_smiles").size()
    anion_support = training_pairs.groupby("canonical_anion_smiles").size()
    cations = _library(unique_pairs, "cation", cation_support)
    anions = _library(unique_pairs, "anion", anion_support)
    cations = cations[
        cations["cation_support_count"] >= settings.min_cation_support
    ].head(settings.max_cations).reset_index(drop=True)
    anions = anions[
        anions["anion_support_count"] >= settings.min_anion_support
    ].head(settings.max_anions).reset_index(drop=True)
    observed_keys = set(unique_pairs["canonical_il_key"])
    training_keys = set(training_pairs["canonical_il_key"])
    generated: list[dict[str, Any]] = []
    rejected_observed = 0
    for cation in cations.itertuples(index=False):
        for anion in anions.itertuples(index=False):
            key = f"{cation.canonical_cation_smiles}||{anion.canonical_anion_smiles}"
            seen_benchmark = key in observed_keys
            if settings.exclude_observed_pairs and seen_benchmark:
                rejected_observed += 1
                continue
            generated.append(
                {
                    "cation_smiles": cation.cation_smiles,
                    "anion_smiles": anion.anion_smiles,
                    "il_smiles": f"{cation.cation_smiles}.{anion.anion_smiles}",
                    "canonical_cation_smiles": cation.canonical_cation_smiles,
                    "canonical_anion_smiles": anion.canonical_anion_smiles,
                    "canonical_il_key": key,
                    "candidate_type": "unseen_pair_recombination",
                    "cation_charge": int(cation.cation_charge),
                    "anion_charge": int(anion.anion_charge),
                    "cation_support_count": int(cation.cation_support_count),
                    "anion_support_count": int(anion.anion_support_count),
                    "cation_family": cation.cation_family,
                    "anion_family": anion.anion_family,
                    "pair_seen_in_benchmark": seen_benchmark,
                    "pair_seen_in_training": key in training_keys,
                    "generation_status": "retained",
                    "minimum_ion_support": min(
                        int(cation.cation_support_count), int(anion.anion_support_count)
                    ),
                    "combined_ion_support": int(cation.cation_support_count)
                    + int(anion.anion_support_count),
                }
            )
    candidate_pool = pd.DataFrame(generated)
    selected = (
        _deterministic_descriptor_selection(
            candidate_pool,
            settings.max_candidates,
            settings.descriptor_prefilter_multiplier,
        )
        if not candidate_pool.empty
        else candidate_pool.copy()
    )
    if not selected.empty:
        selected.insert(
            0,
            "candidate_id",
            [f"UPR-{index:04d}" for index in range(1, len(selected) + 1)],
        )
    observed = unique_pairs.copy()
    observed["candidate_type"] = "observed_reference"
    observed["pair_seen_in_benchmark"] = True
    observed["pair_seen_in_training"] = observed["canonical_il_key"].isin(training_keys)
    observed["cation_support_count"] = observed["canonical_cation_smiles"].map(
        cation_support
    ).fillna(0).astype(int)
    observed["anion_support_count"] = observed["canonical_anion_smiles"].map(
        anion_support
    ).fillna(0).astype(int)
    observed["cation_family"] = observed["cation_smiles"].map(
        lambda value: _ion_family(str(value), "cation")
    )
    observed["anion_family"] = observed["anion_smiles"].map(
        lambda value: _ion_family(str(value), "anion")
    )
    observed["generation_status"] = "retained_reference"
    observed["combined_ion_support"] = (
        observed["cation_support_count"] + observed["anion_support_count"]
    )
    observed = observed.sort_values(
        ["combined_ion_support", "canonical_il_key"], ascending=[False, True]
    ).head(settings.max_observed_references).reset_index(drop=True)
    observed.insert(
        0,
        "candidate_id",
        [f"REF-{index:04d}" for index in range(1, len(observed) + 1)],
    )
    trace = pd.DataFrame(
        [
            {
                "step": "parse_benchmark",
                "input_count": len(benchmark),
                "retained_count": len(parsed),
                "removed_count": len(benchmark) - len(parsed),
                "removal_reason": "invalid_or_nonmonovalent_structure",
            },
            {
                "step": "canonical_pair_deduplication",
                "input_count": len(parsed),
                "retained_count": len(unique_pairs),
                "removed_count": len(parsed) - len(unique_pairs),
                "removal_reason": "duplicate_canonical_pair",
            },
            {
                "step": "exclude_observed_pairs",
                "input_count": len(cations) * len(anions),
                "retained_count": len(candidate_pool),
                "removed_count": rejected_observed,
                "removal_reason": "pair_seen_in_benchmark",
            },
            {
                "step": "deterministic_descriptor_coverage",
                "input_count": len(candidate_pool),
                "retained_count": len(selected),
                "removed_count": len(candidate_pool) - len(selected),
                "removal_reason": "deterministic_max_candidates",
            },
        ]
    )
    return CandidateGenerationResult(
        cations=cations,
        anions=anions,
        observed_references=observed,
        candidates=selected,
        trace=trace,
        failures=failures,
    )
