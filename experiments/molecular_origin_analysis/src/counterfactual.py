"""Chemically validated real-SMILES matched-pair and virtual counterfactuals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors

from .utils import MODULE_ROOT, resolve_path


def _formal_charge(mol: Chem.Mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _inchikey(mol: Chem.Mol) -> str | None:
    try:
        return Chem.MolToInchiKey(mol)
    except (RuntimeError, ValueError):
        return None


class CounterfactualGenerator:
    """Generate only parseable, charge-balanced ion-pair structures."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        template_value = config["counterfactual"].get(
            "template_path",
            "templates/counterfactual_templates.yaml",
        )
        self.template_path = resolve_path(template_value, MODULE_ROOT)

    def template_records(self) -> list[dict[str, Any]]:
        """Return the configured template records before chemistry validation."""

        with self.template_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        records = list(payload.get("records", []))
        for series in payload.get("series", []):
            for cation in series.get("cations", []):
                for anion in series.get("anions", []):
                    records.append(
                        {
                            "template_id": series["template_id"],
                            "modification_type": series["modification_type"],
                            "series_member": cation.get("label"),
                            "counterion_label": anion.get("label"),
                            "cation_smiles": cation["smiles"],
                            "anion_smiles": anion["smiles"],
                        }
                    )
        return records

    def validate_records(
        self,
        records: Iterable[dict[str, Any]],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        valid_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        identities: set[str] = set()
        for source in records:
            row = dict(source)
            cation = Chem.MolFromSmiles(str(row.get("cation_smiles", "")))
            anion = Chem.MolFromSmiles(str(row.get("anion_smiles", "")))
            if cation is None or anion is None:
                row["rejection_reason"] = "rdkit_parse_failure"
                rejected_rows.append(row)
                continue
            cation_charge = _formal_charge(cation)
            anion_charge = _formal_charge(anion)
            if cation_charge != 1 or anion_charge != -1 or cation_charge + anion_charge != 0:
                row["rejection_reason"] = "charge_or_stoichiometry_failure"
                row["cation_charge"] = cation_charge
                row["anion_charge"] = anion_charge
                rejected_rows.append(row)
                continue
            canonical_cation = Chem.MolToSmiles(cation, canonical=True)
            canonical_anion = Chem.MolToSmiles(anion, canonical=True)
            identity = f"{_inchikey(cation)}||{_inchikey(anion)}"
            if identity in identities:
                row["rejection_reason"] = "duplicate_identity"
                rejected_rows.append(row)
                continue
            identities.add(identity)
            pair = Chem.MolFromSmiles(f"{canonical_cation}.{canonical_anion}")
            if pair is None or _formal_charge(pair) != 0:
                row["rejection_reason"] = "combined_structure_failure"
                rejected_rows.append(row)
                continue
            row.update(
                {
                    "cation_smiles": canonical_cation,
                    "anion_smiles": canonical_anion,
                    "canonical_il_smiles": f"{canonical_cation}.{canonical_anion}",
                    "cation_inchikey": _inchikey(cation),
                    "anion_inchikey": _inchikey(anion),
                    "canonical_identity_key": identity,
                    "cation_charge": cation_charge,
                    "anion_charge": anion_charge,
                    "net_charge": 0,
                    "validation_status": "valid",
                }
            )
            valid_rows.append(row)
        return pd.DataFrame(valid_rows), pd.DataFrame(rejected_rows)

    def generate_virtual_library(self) -> pd.DataFrame:
        valid, _ = self.validate_records(self.template_records())
        return valid

    def matched_pairs(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Find observed pairs where one ion is fixed under matched conditions."""

        required = {
            "sample_id",
            "cation_smiles",
            "anion_smiles",
            "Temperature_K",
            "Pressure_kPa",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Matched-pair input lacks columns: {missing}")
        counterfactual_config = self.config.get("counterfactual", {})
        tolerance = float(
            counterfactual_config.get(
                "matched_temperature_tolerance_k",
                self.config["conditions"].get("temperature_tolerance_k", 2.0),
            )
        )
        pressure_tolerance = float(
            counterfactual_config.get("matched_pressure_tolerance_kpa", 5.0)
        )
        property_names = [
            column.removesuffix("_ActualValue")
            for column in frame.columns
            if column.endswith("_ActualValue")
        ]
        rows: list[dict[str, Any]] = []
        unique = frame.drop_duplicates(
            ["cation_smiles", "anion_smiles", "Temperature_K", "Pressure_kPa"]
        )
        for fixed_role, fixed_col, changed_col in [
            ("anion_fixed", "anion_smiles", "cation_smiles"),
            ("cation_fixed", "cation_smiles", "anion_smiles"),
        ]:
            for fixed_value, group in unique.groupby(fixed_col):
                values = list(group.itertuples(index=False))
                for left_index, left in enumerate(values):
                    for right in values[left_index + 1 :]:
                        comparison_left = left
                        comparison_right = right
                        if getattr(comparison_left, changed_col) == getattr(
                            comparison_right,
                            changed_col,
                        ):
                            continue
                        if (
                            abs(
                                float(comparison_left.Temperature_K)
                                - float(comparison_right.Temperature_K)
                            )
                            > tolerance
                        ):
                            continue
                        pressure_left = float(comparison_left.Pressure_kPa)
                        pressure_right = float(comparison_right.Pressure_kPa)
                        pressure_left_available = np.isfinite(pressure_left)
                        pressure_right_available = np.isfinite(pressure_right)
                        if pressure_left_available != pressure_right_available:
                            continue
                        if (
                            pressure_left_available
                            and abs(pressure_left - pressure_right)
                            > pressure_tolerance
                        ):
                            continue
                        left_changed = str(
                            getattr(comparison_left, changed_col)
                        )
                        right_changed = str(
                            getattr(comparison_right, changed_col)
                        )
                        if right_changed < left_changed:
                            comparison_left, comparison_right = (
                                comparison_right,
                                comparison_left,
                            )
                            pressure_left, pressure_right = (
                                pressure_right,
                                pressure_left,
                            )
                            left_changed, right_changed = (
                                right_changed,
                                left_changed,
                            )
                        pair_row = {
                            "fixed_role": fixed_role,
                            "fixed_ion_smiles": fixed_value,
                            "left_sample_id": int(comparison_left.sample_id),
                            "right_sample_id": int(comparison_right.sample_id),
                            "left_changed_ion_smiles": left_changed,
                            "right_changed_ion_smiles": right_changed,
                            "temperature_difference_K": abs(
                                float(comparison_left.Temperature_K)
                                - float(comparison_right.Temperature_K)
                            ),
                            "pressure_difference_kPa": (
                                abs(pressure_left - pressure_right)
                                if np.isfinite(pressure_left)
                                else np.nan
                            ),
                            "pressure_available": pressure_left_available,
                            "modification_type": f"{changed_col}_substitution",
                        }
                        for property_name in property_names:
                            left_value = float(
                                getattr(
                                    comparison_left,
                                    f"{property_name}_ActualValue",
                                )
                            )
                            right_value = float(
                                getattr(
                                    comparison_right,
                                    f"{property_name}_ActualValue",
                                )
                            )
                            left_mask = bool(
                                getattr(
                                    comparison_left,
                                    f"{property_name}_mask",
                                )
                            )
                            right_mask = bool(
                                getattr(
                                    comparison_right,
                                    f"{property_name}_mask",
                                )
                            )
                            observed_valid = (
                                left_mask
                                and right_mask
                                and np.isfinite(left_value)
                                and np.isfinite(right_value)
                                and left_value > 0
                                and right_value > 0
                            )
                            pair_row[f"both_observed_{property_name}"] = (
                                observed_valid
                            )
                            if observed_valid:
                                observed_delta = float(
                                    np.log(right_value) - np.log(left_value)
                                )
                                pair_row[
                                    f"observed_log_delta_{property_name}"
                                ] = observed_delta
                                pair_row[
                                    f"observed_abs_log_difference_{property_name}"
                                ] = abs(observed_delta)
                            else:
                                pair_row[
                                    f"observed_log_delta_{property_name}"
                                ] = np.nan
                                pair_row[
                                    f"observed_abs_log_difference_{property_name}"
                                ] = np.nan
                            prediction_column = f"prediction_{property_name}"
                            if prediction_column in frame.columns:
                                left_prediction = float(
                                    getattr(comparison_left, prediction_column)
                                )
                                right_prediction = float(
                                    getattr(comparison_right, prediction_column)
                                )
                                predicted_valid = (
                                    np.isfinite(left_prediction)
                                    and np.isfinite(right_prediction)
                                    and left_prediction > 0
                                    and right_prediction > 0
                                )
                                if predicted_valid:
                                    predicted_delta = float(
                                        np.log(right_prediction)
                                        - np.log(left_prediction)
                                    )
                                    pair_row[
                                        f"predicted_log_delta_{property_name}"
                                    ] = predicted_delta
                                    pair_row[
                                        f"predicted_abs_log_difference_{property_name}"
                                    ] = abs(predicted_delta)
                                else:
                                    pair_row[
                                        f"predicted_log_delta_{property_name}"
                                    ] = np.nan
                                    pair_row[
                                        f"predicted_abs_log_difference_{property_name}"
                                    ] = np.nan
                        rows.append(pair_row)
        output = pd.DataFrame(rows).drop_duplicates()
        if output.empty:
            return output
        output["substitution_pair_id"] = (
            output["fixed_role"].astype(str)
            + "||"
            + output["fixed_ion_smiles"].astype(str)
            + "||"
            + output["left_changed_ion_smiles"].astype(str)
            + "=>"
            + output["right_changed_ion_smiles"].astype(str)
        )
        output["condition_match_id"] = (
            output["substitution_pair_id"]
            + "||"
            + output["left_sample_id"].astype(str)
            + "||"
            + output["right_sample_id"].astype(str)
        )
        return output

    def summarize_matched_pairs(
        self,
        matched: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return raw condition matches, unique-pair effects, summaries, and signed descriptors."""

        if matched.empty:
            empty = pd.DataFrame()
            return empty, empty, empty, empty
        properties = [
            column.removeprefix("observed_log_delta_")
            for column in matched
            if column.startswith("observed_log_delta_")
        ]
        condition_parts: list[pd.DataFrame] = []
        for property_name in properties:
            delta = f"observed_log_delta_{property_name}"
            predicted = f"predicted_log_delta_{property_name}"
            columns = [
                "condition_match_id",
                "substitution_pair_id",
                "fixed_role",
                "fixed_ion_smiles",
                "left_changed_ion_smiles",
                "right_changed_ion_smiles",
                "left_sample_id",
                "right_sample_id",
                "temperature_difference_K",
                "pressure_difference_kPa",
                "pressure_available",
                delta,
            ]
            if predicted in matched:
                columns.append(predicted)
            part = matched[columns].copy()
            part = part.loc[np.isfinite(part[delta])].copy()
            part = part.rename(
                columns={
                    delta: "observed_log_delta",
                    predicted: "predicted_log_delta",
                }
            )
            part["property"] = property_name
            part["observed_abs_log_difference"] = part[
                "observed_log_delta"
            ].abs()
            if "predicted_log_delta" in part:
                part["predicted_abs_log_difference"] = part[
                    "predicted_log_delta"
                ].abs()
            condition_parts.append(part)
        condition_level = pd.concat(condition_parts, ignore_index=True)
        condition_level = condition_level.drop_duplicates(
            ["property", "condition_match_id"]
        )
        pair_level = (
            condition_level.groupby(
                [
                    "property",
                    "substitution_pair_id",
                    "fixed_role",
                    "fixed_ion_smiles",
                    "left_changed_ion_smiles",
                    "right_changed_ion_smiles",
                ],
                as_index=False,
            )
            .agg(
                n_condition_matches=("condition_match_id", "nunique"),
                observed_log_delta=("observed_log_delta", "median"),
                observed_abs_log_difference=(
                    "observed_abs_log_difference",
                    "median",
                ),
                predicted_log_delta=(
                    "predicted_log_delta",
                    "median",
                )
                if "predicted_log_delta" in condition_level
                else ("observed_log_delta", lambda values: np.nan),
            )
        )
        pair_level["predicted_abs_log_difference"] = pair_level[
            "predicted_log_delta"
        ].abs()
        pair_level["analysis_unit"] = "unique unordered ion-substitution pair"

        repeats = int(
            self.config.get("revision_analysis", {}).get(
                "identity_bootstrap_repeats",
                500,
            )
        )
        confidence = float(self.config["statistics"].get("confidence_level", 0.95))
        seed = int(self.config["statistics"].get("random_seed", 42))
        rng = np.random.default_rng(seed + 910000)
        summary_rows: list[dict[str, Any]] = []
        for (property_name, fixed_role), group in pair_level.groupby(
            ["property", "fixed_role"]
        ):
            values = group["observed_abs_log_difference"].to_numpy(dtype=float)
            bootstrap = np.asarray(
                [
                    np.median(rng.choice(values, len(values), replace=True))
                    for _ in range(repeats)
                ],
                dtype=float,
            )
            alpha = (1.0 - confidence) / 2.0
            raw_matches = condition_level.loc[
                (condition_level["property"] == property_name)
                & (condition_level["fixed_role"] == fixed_role)
            ]
            summary_rows.append(
                {
                    "property": property_name,
                    "fixed_role": fixed_role,
                    "n_unique_substitution_pairs": int(len(group)),
                    "n_raw_condition_matches": int(len(raw_matches)),
                    "n_unique_fixed_ions": int(group["fixed_ion_smiles"].nunique()),
                    "median_abs_log_difference": float(np.median(values)),
                    "bootstrap_ci_low": float(np.quantile(bootstrap, alpha)),
                    "bootstrap_ci_high": float(
                        np.quantile(bootstrap, 1.0 - alpha)
                    ),
                    "analysis_unit": "unique unordered ion-substitution pair",
                }
            )

        signed_rows: list[dict[str, Any]] = []
        for row in pair_level[
            [
                "substitution_pair_id",
                "fixed_role",
                "left_changed_ion_smiles",
                "right_changed_ion_smiles",
            ]
        ].drop_duplicates().itertuples(index=False):
            left_mol = Chem.MolFromSmiles(row.left_changed_ion_smiles)
            right_mol = Chem.MolFromSmiles(row.right_changed_ion_smiles)
            if left_mol is None or right_mol is None:
                continue
            def signature(molecule: Chem.Mol) -> dict[str, int | bool]:
                element_counts = {
                    symbol: sum(
                        atom.GetSymbol() == symbol
                        for atom in molecule.GetAtoms()
                    )
                    for symbol in ("B", "C", "N", "O", "F", "P", "S")
                }
                return {
                    **element_counts,
                    "hbond_donors": int(Descriptors.NumHDonors(molecule)),
                    "rings": int(Descriptors.RingCount(molecule)),
                    "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
                    "has_ether": bool(
                        molecule.HasSubstructMatch(
                            Chem.MolFromSmarts("[OD2]([#6])[#6]")
                        )
                    ),
                    "has_hydroxyl": bool(
                        molecule.HasSubstructMatch(
                            Chem.MolFromSmarts("[OX2H]")
                        )
                    ),
                    "has_carboxylate": bool(
                        molecule.HasSubstructMatch(
                            Chem.MolFromSmarts("[CX3](=O)[O-]")
                        )
                    ),
                }

            left_signature = signature(left_mol)
            right_signature = signature(right_mol)

            def anion_class(values: dict[str, int | bool]) -> str:
                if values["B"] == 1 and values["F"] == 4:
                    return "BF4"
                if values["P"] == 1 and values["F"] == 6:
                    return "PF6"
                if values["S"] == 2 and values["N"] == 1:
                    if values["F"] >= 6:
                        return "NTf2"
                    if values["F"] == 2:
                        return "FSI"
                return ""

            category = ""
            orientation = 1
            left_class = anion_class(left_signature)
            right_class = anion_class(right_signature)
            named_anion_changes = {
                ("BF4", "PF6"): "BF4 to PF6",
                ("BF4", "NTf2"): "BF4 to NTf2",
                ("FSI", "NTf2"): "FSI to NTf2",
            }
            if (left_class, right_class) in named_anion_changes:
                category = named_anion_changes[(left_class, right_class)]
            elif (right_class, left_class) in named_anion_changes:
                category = named_anion_changes[(right_class, left_class)]
                orientation = -1
            else:
                for direction, reference, modified in (
                    (1, left_signature, right_signature),
                    (-1, right_signature, left_signature),
                ):
                    carbon_delta = int(modified["C"]) - int(reference["C"])
                    oxygen_delta = int(modified["O"]) - int(reference["O"])
                    if (
                        bool(reference["has_carboxylate"])
                        and bool(modified["has_carboxylate"])
                        and carbon_delta > 0
                        and carbon_delta <= 2
                        and int(modified["heavy_atoms"])
                        - int(reference["heavy_atoms"])
                        == carbon_delta
                        and all(
                            int(modified[element])
                            == int(reference[element])
                            for element in ("B", "N", "O", "F", "P", "S")
                        )
                    ):
                        category = "carboxylate chain extension"
                    elif (
                        not bool(reference["has_hydroxyl"])
                        and bool(modified["has_hydroxyl"])
                        and oxygen_delta == 1
                        and int(modified["C"]) == int(reference["C"])
                        and int(modified["heavy_atoms"])
                        - int(reference["heavy_atoms"])
                        == 1
                        and int(modified["hbond_donors"])
                        > int(reference["hbond_donors"])
                    ):
                        category = "hydroxyl introduction"
                    elif (
                        not bool(reference["has_ether"])
                        and bool(modified["has_ether"])
                        and oxygen_delta == 1
                        and int(modified["C"]) == int(reference["C"])
                        and int(modified["heavy_atoms"])
                        - int(reference["heavy_atoms"])
                        == 1
                        and int(modified["hbond_donors"])
                        == int(reference["hbond_donors"])
                    ):
                        category = "ether introduction"
                    elif (
                        carbon_delta > 0
                        and carbon_delta <= 2
                        and int(modified["heavy_atoms"])
                        - int(reference["heavy_atoms"])
                        == carbon_delta
                        and int(modified["rings"]) == int(reference["rings"])
                        and all(
                            int(modified[element])
                            == int(reference[element])
                            for element in ("B", "N", "O", "F", "P", "S")
                        )
                    ):
                        category = "alkyl-chain extension"
                    if category:
                        orientation = direction
                        break
            signed_rows.append(
                {
                    "substitution_pair_id": row.substitution_pair_id,
                    "fixed_role": row.fixed_role,
                    "ordering_rule": "canonical-SMILES lexical order",
                    "recognized_transformation": category,
                    "recognized_orientation_multiplier": orientation,
                    "recognized_for_signed_analysis": bool(category),
                    "reference_changed_ion_smiles": (
                        row.left_changed_ion_smiles
                        if orientation == 1
                        else row.right_changed_ion_smiles
                    ),
                    "modified_changed_ion_smiles": (
                        row.right_changed_ion_smiles
                        if orientation == 1
                        else row.left_changed_ion_smiles
                    ),
                    "delta_heavy_atom_count": (
                        right_mol.GetNumHeavyAtoms()
                        - left_mol.GetNumHeavyAtoms()
                    ),
                    "delta_carbon_count": (
                        sum(a.GetAtomicNum() == 6 for a in right_mol.GetAtoms())
                        - sum(a.GetAtomicNum() == 6 for a in left_mol.GetAtoms())
                    ),
                    "delta_heteroatom_count": (
                        Descriptors.NumHeteroatoms(right_mol)
                        - Descriptors.NumHeteroatoms(left_mol)
                    ),
                    "delta_fluorine_count": (
                        sum(a.GetAtomicNum() == 9 for a in right_mol.GetAtoms())
                        - sum(a.GetAtomicNum() == 9 for a in left_mol.GetAtoms())
                    ),
                    "delta_molecular_weight_g_mol": (
                        Descriptors.MolWt(right_mol)
                        - Descriptors.MolWt(left_mol)
                    ),
                    "interpretation_scope": (
                        "descriptive signed structural difference; not a "
                        "single-edit causal intervention"
                    ),
                }
            )
        signed_descriptors = pd.DataFrame(signed_rows)
        signed_contrasts = pair_level.merge(
            signed_descriptors,
            on=["substitution_pair_id", "fixed_role"],
            how="left",
            validate="many_to_one",
        )
        signed_contrasts["signed_observed_log_contrast"] = (
            signed_contrasts["observed_log_delta"]
            * signed_contrasts["recognized_orientation_multiplier"]
        )
        signed_contrasts["signed_predicted_log_contrast"] = (
            signed_contrasts["predicted_log_delta"]
            * signed_contrasts["recognized_orientation_multiplier"]
        )
        signed_contrasts = signed_contrasts.loc[
            signed_contrasts["recognized_for_signed_analysis"].fillna(False)
        ].reset_index(drop=True)
        return (
            condition_level,
            pair_level,
            pd.DataFrame(summary_rows),
            signed_contrasts,
        )
