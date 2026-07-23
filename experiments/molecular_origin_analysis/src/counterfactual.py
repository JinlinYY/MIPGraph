"""Chemically validated real-SMILES matched-pair and virtual counterfactuals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem

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
        tolerance = float(self.config["conditions"].get("temperature_tolerance_k", 2.0))
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
                        if getattr(left, changed_col) == getattr(right, changed_col):
                            continue
                        if abs(float(left.Temperature_K) - float(right.Temperature_K)) > tolerance:
                            continue
                        pressure_left = float(left.Pressure_kPa)
                        pressure_right = float(right.Pressure_kPa)
                        if abs(pressure_left - pressure_right) > 5.0:
                            continue
                        pair_row = {
                            "fixed_role": fixed_role,
                            "fixed_ion_smiles": fixed_value,
                            "left_sample_id": int(left.sample_id),
                            "right_sample_id": int(right.sample_id),
                            "left_changed_ion_smiles": getattr(left, changed_col),
                            "right_changed_ion_smiles": getattr(right, changed_col),
                            "temperature_difference_K": abs(
                                float(left.Temperature_K) - float(right.Temperature_K)
                            ),
                            "pressure_difference_kPa": abs(
                                pressure_left - pressure_right
                            ),
                            "modification_type": f"{changed_col}_substitution",
                        }
                        for property_name in property_names:
                            left_value = float(
                                getattr(left, f"{property_name}_ActualValue")
                            )
                            right_value = float(
                                getattr(right, f"{property_name}_ActualValue")
                            )
                            left_mask = bool(
                                getattr(left, f"{property_name}_mask")
                            )
                            right_mask = bool(
                                getattr(right, f"{property_name}_mask")
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
                                    getattr(left, prediction_column)
                                )
                                right_prediction = float(
                                    getattr(right, prediction_column)
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
        return pd.DataFrame(rows).drop_duplicates()
