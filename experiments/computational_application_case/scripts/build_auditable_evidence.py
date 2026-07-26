"""Build the validation, identity-audit, sensitivity, and stability evidence.

This script consumes only persisted experimental outputs and the manuscript's
existing split-comparison source data.  It does not retrain a model or invent
uncertainty samples.
"""

from __future__ import annotations

import itertools
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CASE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.computational_application_case.src.chemistry import (  # noqa: E402
    canonical_pair_key,
    parse_monovalent_pair,
)
from experiments.computational_application_case.src.config import (  # noqa: E402
    load_case_config,
)
from experiments.computational_application_case.src.screening import (  # noqa: E402
    derive_reference_thresholds,
    prioritize_candidates,
    screen_candidates,
)


OUTPUT_ROOT = CASE_DIR / "outputs_primary_audited"
DATA_DIR = OUTPUT_ROOT / "data"
AUDIT_DIR = OUTPUT_ROOT / "audit"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
PAPER_DIR = PROJECT_ROOT / "LaTex-MIPGraph"
PAPER_FIG_DIR = PAPER_DIR / "Fig"
PERFORMANCE_SOURCE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "manuscript_figure_source_data"
    / "performance_results"
)

PROPERTIES = [
    "Density",
    "Viscosity",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
]
PROPERTY_SHORT = {
    "Density": "Density",
    "Viscosity": "Viscosity",
    "ElectricalConductivity": "Conductivity",
    "HeatCapacity": "Heat capacity",
    "SurfaceTension": "Surface tension",
    "ThermalConductivity": "Thermal cond.",
}
SPLIT_ORDER = ["Random point", "Random IL", "Balanced IL", "Ion-family"]
SPLIT_FILES = {
    "Random IL": PROJECT_ROOT
    / "il_property_prediction"
    / "data"
    / "processed"
    / "splits"
    / "il_level_seed42.json",
    "Balanced IL": PROJECT_ROOT
    / "il_property_prediction"
    / "data"
    / "processed"
    / "splits"
    / "il_level_property_balanced_seed42.json",
    "Ion-family": PROJECT_ROOT
    / "il_property_prediction"
    / "data"
    / "processed"
    / "splits"
    / "il_level_family_pair_seed42.json",
}


def _identity_key(smiles: object, cache: dict[str, str | None]) -> str | None:
    text = str(smiles)
    if text not in cache:
        try:
            cache[text] = canonical_pair_key(text)
        except (TypeError, ValueError):
            cache[text] = None
    return cache[text]


def build_split_identity_audit() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Audit train/test identity overlap after Standard-InChI normalization."""

    benchmark = pd.read_csv(
        PROJECT_ROOT
        / "il_property_prediction"
        / "data"
        / "processed"
        / "il_multiprop_clean.csv",
        low_memory=False,
    ).reset_index(drop=True)
    cache: dict[str, str | None] = {}
    identity = benchmark["IL_SMILES"].map(lambda value: _identity_key(value, cache))
    rows = []
    overlap_by_split: dict[str, set[str]] = {}
    for split_name, path in SPLIT_FILES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        train_index = [int(value) for value in payload["train"]]
        test_index = [int(value) for value in payload["test"]]
        train_keys = set(identity.iloc[train_index].dropna().astype(str))
        test_keys = set(identity.iloc[test_index].dropna().astype(str))
        overlap = train_keys & test_keys
        overlap_by_split[split_name] = overlap
        train_identity = identity.iloc[train_index]
        test_identity = identity.iloc[test_index]
        overlap_test_rows = int(test_identity.astype(str).isin(overlap).sum())
        unresolved_test_rows = int(test_identity.isna().sum())
        rows.append(
            {
                "split_strategy": split_name,
                "train_row_count": len(train_index),
                "test_row_count": len(test_index),
                "train_unique_identity_count": len(train_keys),
                "test_unique_identity_count_before_audit": len(test_keys),
                "train_test_identity_overlap_count": len(overlap),
                "train_identity_unresolved_row_count": int(train_identity.isna().sum()),
                "test_identity_unresolved_row_count": unresolved_test_rows,
                "test_overlap_rows_excluded": overlap_test_rows,
                "test_rows_fail_closed_total": overlap_test_rows
                + unresolved_test_rows,
                "test_unique_identity_count_after_audit": len(test_keys - overlap),
                "overlapping_identity_keys": " | ".join(sorted(overlap)),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(DATA_DIR / "whole_il_split_identity_audit.csv", index=False)
    return audit, overlap_by_split


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    mask = np.isfinite(left.to_numpy(dtype=float)) & np.isfinite(
        right.to_numpy(dtype=float)
    )
    if int(mask.sum()) < 3:
        return float("nan")
    result = spearmanr(
        left.to_numpy(dtype=float)[mask], right.to_numpy(dtype=float)[mask]
    )
    return float(result.statistic)


def _load_parity_source(
    overlap_by_split: dict[str, set[str]] | None,
) -> pd.DataFrame:
    frames = []
    for panel in "ABCDEF":
        path = PERFORMANCE_SOURCE_DIR / f"performance_results_source_data_{panel}.csv"
        frame = pd.read_csv(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    numeric = ["y_true", "y_pred", "Temperature_K", "Pressure_kPa"]
    for column in numeric:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined = combined[
        (combined["y_true"] > 0)
        & (combined["y_pred"] > 0)
        & combined["split_strategy"].isin(SPLIT_ORDER)
    ].copy()
    cache: dict[str, str | None] = {}
    combined["chemical_identity_key"] = combined["IL_SMILES"].map(
        lambda value: _identity_key(value, cache)
    )
    audit_enabled = overlap_by_split is not None
    overlap_lookup = overlap_by_split or {}
    combined["excluded_identity_overlap"] = [
        bool(pd.notna(key)) and str(key) in overlap_lookup.get(str(split), set())
        for split, key in zip(
            combined["split_strategy"], combined["chemical_identity_key"]
        )
    ]
    combined["excluded_identity_unresolved"] = (
        combined["chemical_identity_key"].isna() if audit_enabled else False
    )
    combined["identity_audit_exclusion_reason"] = np.select(
        [
            combined["excluded_identity_overlap"],
            combined["excluded_identity_unresolved"],
        ],
        ["train_test_identity_overlap", "identity_unresolved"],
        default="",
    )
    excluded_mask = (
        combined["excluded_identity_overlap"]
        | combined["excluded_identity_unresolved"]
    )
    excluded = combined[excluded_mask].copy()
    if audit_enabled:
        excluded.to_csv(
            DATA_DIR / "whole_il_identity_audit_excluded_rows.csv", index=False
        )
    combined = combined[~excluded_mask].copy()
    combined["log_true"] = np.log(combined["y_true"])
    combined["log_pred"] = np.log(combined["y_pred"])
    combined["absolute_percentage_error_pct"] = (
        100.0 * np.abs(combined["y_pred"] - combined["y_true"]) / combined["y_true"]
    )
    return combined


def build_whole_il_validation(
    overlap_by_split: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize identity-audited whole-IL point, curve, and ranking performance."""

    parity = _load_parity_source(overlap_by_split)
    original_parity = _load_parity_source(None)
    published_nmae = pd.read_csv(
        PERFORMANCE_SOURCE_DIR / "performance_results_source_data_I.csv"
    ).set_index(["split_strategy", "property"])["log_NMAE"]
    original_parity["absolute_log_error"] = np.abs(
        original_parity["log_pred"] - original_parity["log_true"]
    )
    original_log_mae = original_parity.groupby(
        ["split_strategy", "property"]
    )["absolute_log_error"].mean()
    rows = []
    curve_rows = []
    for (split, prop), group in parity.groupby(["split_strategy", "property"]):
        log_error = group["log_pred"] - group["log_true"]
        r2_denominator = float(np.sum((group["log_true"] - group["log_true"].mean()) ** 2))
        r2 = (
            1.0 - float(np.sum(log_error**2)) / r2_denominator
            if r2_denominator > 0
            else float("nan")
        )
        per_il = (
            group.groupby(["IL_Name", "IL_SMILES"], as_index=False)
            .agg(
                log_true=("log_true", "median"),
                log_pred=("log_pred", "median"),
                point_count=("y_true", "size"),
                mdape_pct=("absolute_percentage_error_pct", "median"),
            )
        )
        ambient = group[
            group["Temperature_K"].between(295.65, 300.65, inclusive="both")
            & group["Pressure_kPa"].between(90.0, 110.0, inclusive="both")
        ]
        ambient_il = (
            ambient.groupby(["IL_Name", "IL_SMILES"], as_index=False)
            .agg(log_true=("log_true", "median"), log_pred=("log_pred", "median"))
        )
        curve_agreement = []
        curve_mdape = []
        near_ambient_pressure = group[
            group["Pressure_kPa"].between(90.0, 110.0, inclusive="both")
        ]
        for (name, smiles), curve in near_ambient_pressure.groupby(
            ["IL_Name", "IL_SMILES"]
        ):
            curve = (
                curve.groupby("Temperature_K", as_index=False)
                .agg(
                    log_true=("log_true", "median"),
                    log_pred=("log_pred", "median"),
                    absolute_percentage_error_pct=(
                        "absolute_percentage_error_pct",
                        "median",
                    ),
                    pressure_min_kPa=("Pressure_kPa", "min"),
                    pressure_max_kPa=("Pressure_kPa", "max"),
                )
                .sort_values("Temperature_K")
            )
            if curve["Temperature_K"].nunique() < 3:
                continue
            true_slope = float(np.polyfit(curve["Temperature_K"], curve["log_true"], 1)[0])
            pred_slope = float(np.polyfit(curve["Temperature_K"], curve["log_pred"], 1)[0])
            curve_agreement.append(float(np.sign(true_slope) == np.sign(pred_slope)))
            curve_mdape.append(float(curve["absolute_percentage_error_pct"].median()))
            curve_rows.append(
                {
                    "split_strategy": split,
                    "property": prop,
                    "IL_Name": name,
                    "IL_SMILES": smiles,
                    "point_count": int(len(curve)),
                    "pressure_window_kPa": "90--110",
                    "experimental_log_slope_per_K": true_slope,
                    "predicted_log_slope_per_K": pred_slope,
                    "temperature_trend_sign_agreement": bool(
                        np.sign(true_slope) == np.sign(pred_slope)
                    ),
                    "curve_mdape_pct": float(
                        curve["absolute_percentage_error_pct"].median()
                    ),
                }
            )
        log_mae = float(np.mean(np.abs(log_error)))
        scale = float(
            original_log_mae.loc[(split, prop)]
            / published_nmae.loc[(split, prop)]
        )
        rows.append(
            {
                "split_strategy": split,
                "property": prop,
                "point_count": int(len(group)),
                "unique_IL_count": int(group["IL_SMILES"].nunique()),
                "log_MAE": log_mae,
                "log_RMSE": float(np.sqrt(np.mean(log_error**2))),
                "log_R2": r2,
                "log_NMAE": log_mae / scale,
                "point_MdAPE_pct": float(
                    group["absolute_percentage_error_pct"].median()
                ),
                "IL_median_rank_Spearman": _safe_spearman(
                    per_il["log_true"], per_il["log_pred"]
                ),
                "ambient_rank_Spearman": _safe_spearman(
                    ambient_il["log_true"], ambient_il["log_pred"]
                ),
                "ambient_IL_count": int(len(ambient_il)),
                "curve_count_ge3": int(len(curve_agreement)),
                "curve_pressure_scope": "finite 90--110 kPa only",
                "temperature_trend_sign_agreement_fraction": float(
                    np.mean(curve_agreement)
                )
                if curve_agreement
                else float("nan"),
                "median_curve_MdAPE_pct": float(np.median(curve_mdape))
                if curve_mdape
                else float("nan"),
            }
        )
    summary = pd.DataFrame(rows)
    curves = pd.DataFrame(curve_rows)
    summary.to_csv(DATA_DIR / "whole_il_validation_summary.csv", index=False)
    curves.to_csv(DATA_DIR / "whole_il_curve_validation.csv", index=False)
    return summary, curves


def build_identity_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Reclassify the old shortlist and verify the new shortlist by InChIKey."""

    benchmark = pd.read_csv(
        PROJECT_ROOT / "il_property_prediction" / "data" / "processed" / "il_multiprop_clean.csv",
        low_memory=False,
    )
    key_to_names: dict[str, set[str]] = {}
    parse_cache: dict[str, str | None] = {}
    for row in benchmark[["IL_Name", "IL_SMILES"]].dropna(subset=["IL_SMILES"]).itertuples(index=False):
        smiles = str(row.IL_SMILES)
        if smiles not in parse_cache:
            try:
                parse_cache[smiles] = canonical_pair_key(smiles)
            except ValueError:
                parse_cache[smiles] = None
        key = parse_cache[smiles]
        if key is not None:
            key_to_names.setdefault(key, set()).add(str(row.IL_Name))

    def audit(frame: pd.DataFrame, generation: str) -> pd.DataFrame:
        rows = []
        for row in frame.itertuples(index=False):
            smiles = str(row.il_smiles)
            try:
                key = canonical_pair_key(smiles)
                matched = sorted(key_to_names.get(key, set()))
                status = "known_pair_control" if matched else "genuinely_unseen_pair"
                error = ""
            except ValueError as exc:
                key = ""
                matched = []
                status = "identity_unresolved"
                error = str(exc)
            rows.append(
                {
                    "generation": generation,
                    "candidate_id": str(row.candidate_id),
                    "il_smiles": smiles,
                    "chemical_identity_key": key,
                    "identity_status": status,
                    "matched_benchmark_IL_names": " | ".join(matched),
                    "identity_error": error,
                }
            )
        return pd.DataFrame(rows)

    old_final = pd.read_csv(
        CASE_DIR / "outputs" / "data" / "final_prioritized_candidates.csv"
    )
    new_final = pd.read_csv(DATA_DIR / "final_prioritized_candidates.csv")
    old_audit = audit(old_final, "pre_audit_shortlist")
    new_audit = audit(new_final, "InChI_audited_shortlist")
    old_audit.to_csv(DATA_DIR / "chemical_identity_audit_old_shortlist.csv", index=False)
    new_audit.to_csv(DATA_DIR / "chemical_identity_audit_current_shortlist.csv", index=False)

    old_cations = pd.read_csv(CASE_DIR / "outputs" / "data" / "cation_library.csv")
    old_anions = pd.read_csv(CASE_DIR / "outputs" / "data" / "anion_library.csv")
    benchmark_old_keys: set[str] = set()
    for smiles in benchmark["IL_SMILES"].dropna().astype(str).unique():
        try:
            parsed = parse_monovalent_pair(smiles)
        except (TypeError, ValueError):
            continue
        benchmark_old_keys.add(
            f"{parsed.canonical_cation_smiles}||{parsed.canonical_anion_smiles}"
        )
    old_pool_rows = []
    for cation, anion in itertools.product(
        old_cations.itertuples(index=False), old_anions.itertuples(index=False)
    ):
        old_key = (
            f"{cation.canonical_cation_smiles}||{anion.canonical_anion_smiles}"
        )
        if old_key in benchmark_old_keys:
            continue
        ion_pair = f"{cation.cation_smiles}.{anion.anion_smiles}"
        try:
            identity_key = canonical_pair_key(ion_pair)
            matched = sorted(key_to_names.get(identity_key, set()))
            status = (
                "pseudo_unseen_known_identity"
                if matched
                else "genuinely_unseen_identity"
            )
            error = ""
        except (TypeError, ValueError) as exc:
            identity_key = ""
            matched = []
            status = "identity_unresolved"
            error = str(exc)
        old_pool_rows.append(
            {
                "old_canonical_pair_key": old_key,
                "il_smiles": ion_pair,
                "chemical_identity_key": identity_key,
                "identity_status": status,
                "matched_benchmark_IL_names": " | ".join(matched),
                "identity_error": error,
            }
        )
    old_pool_audit = pd.DataFrame(old_pool_rows)
    old_pool_audit.to_csv(
        DATA_DIR / "chemical_identity_audit_old_unseen_pool.csv", index=False
    )
    old_generation = json.loads(
        (CASE_DIR / "outputs" / "steps" / "candidate_generation.json").read_text(
            encoding="utf-8"
        )
    )
    new_generation = json.loads(
        (OUTPUT_ROOT / "steps" / "candidate_generation.json").read_text(
            encoding="utf-8"
        )
    )
    if len(old_pool_audit) != int(old_generation["valid_unseen_pool"]):
        raise RuntimeError(
            "Direct old-pool identity audit does not reproduce the persisted "
            f"SMILES-level unseen count: {len(old_pool_audit)} != "
            f"{old_generation['valid_unseen_pool']}"
        )
    counts = {
        "old_valid_unseen_pool": int(old_generation["valid_unseen_pool"]),
        "new_valid_unseen_pool": int(new_generation["valid_unseen_pool"]),
        "old_pool_directly_audited_entries": int(len(old_pool_audit)),
        "old_pool_pseudo_unseen_known_identity": int(
            old_pool_audit["identity_status"].eq("pseudo_unseen_known_identity").sum()
        ),
        "old_pool_genuinely_unseen_identity": int(
            old_pool_audit["identity_status"].eq("genuinely_unseen_identity").sum()
        ),
        "old_pool_identity_unresolved": int(
            old_pool_audit["identity_status"].eq("identity_unresolved").sum()
        ),
        "old_shortlist_known_pair_controls": int(
            old_audit["identity_status"].eq("known_pair_control").sum()
        ),
        "old_shortlist_genuinely_unseen": int(
            old_audit["identity_status"].eq("genuinely_unseen_pair").sum()
        ),
        "new_shortlist_known_pair_controls": int(
            new_audit["identity_status"].eq("known_pair_control").sum()
        ),
        "new_shortlist_genuinely_unseen": int(
            new_audit["identity_status"].eq("genuinely_unseen_pair").sum()
        ),
    }
    return old_audit, new_audit, counts


def build_threshold_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a predeclared 3x3x3x3 reference-quantile sensitivity grid."""

    config = load_case_config(
        CASE_DIR / "configs" / "auditable_virtual_screening.yaml"
    )
    robust = pd.read_csv(DATA_DIR / "candidate_robust_summary.csv", low_memory=False)
    ad = pd.read_csv(DATA_DIR / "applicability_domain.csv")
    library = pd.read_csv(DATA_DIR / "candidate_library.csv", low_memory=False)
    quantiles_low = [0.20, 0.25, 0.30]
    quantiles_high = [0.70, 0.75, 0.80]
    scenario_rows = []
    selection_rows = []
    nominal_set: set[str] | None = None
    scenario_payloads = []
    scenario_index = 0
    for q_cond, q_visc, q_heat, q_alpha in itertools.product(
        quantiles_low, quantiles_high, quantiles_low, quantiles_low
    ):
        scenario_index += 1
        screening_config = deepcopy(config["screening"])
        screening_config.update(
            {
                "conductivity_min_reference_quantile": q_cond,
                "viscosity_max_reference_quantile": q_visc,
                "volumetric_heat_capacity_min_reference_quantile": q_heat,
                "thermal_diffusivity_min_reference_quantile": q_alpha,
            }
        )
        thresholds = derive_reference_thresholds(robust, screening_config)
        trace = screen_candidates(
            robust, ad, library, thresholds, screening_config
        )
        ranked, final = prioritize_candidates(trace, config["pareto"])
        scenario_id = f"S{scenario_index:02d}"
        selected = set(final["candidate_id"].astype(str))
        is_nominal = bool(
            np.isclose(q_cond, 0.25)
            and np.isclose(q_visc, 0.75)
            and np.isclose(q_heat, 0.25)
            and np.isclose(q_alpha, 0.25)
        )
        if is_nominal:
            nominal_set = selected
        scenario_payloads.append((scenario_id, selected, is_nominal))
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "conductivity_quantile": q_cond,
                "viscosity_quantile": q_visc,
                "heat_capacity_quantile": q_heat,
                "thermal_diffusivity_quantile": q_alpha,
                "conductivity_threshold_S_m": thresholds["conductivity_min"],
                "viscosity_threshold_Pa_s": thresholds["viscosity_max"],
                "heat_capacity_threshold_J_m3_K": thresholds[
                    "volumetric_heat_capacity_min"
                ],
                "thermal_diffusivity_threshold_m2_s": thresholds[
                    "thermal_diffusivity_min"
                ],
                "hard_feasible_count": int(
                    trace[
                        trace["candidate_type"].eq("unseen_pair_recombination")
                        & trace["final_feasible"]
                    ]["candidate_id"].nunique()
                ),
                "pareto_rank_1_count": int(ranked["Pareto_rank"].eq(1).sum())
                if not ranked.empty
                else 0,
                "final_candidate_count": int(len(final)),
                "is_nominal": is_nominal,
            }
        )
        for candidate_id in selected:
            selection_rows.append(
                {"scenario_id": scenario_id, "candidate_id": candidate_id}
            )
    if nominal_set is None:
        raise RuntimeError("Nominal threshold scenario was not generated")
    scenario_table = pd.DataFrame(scenario_rows)
    for row_index, (_, selected, _) in enumerate(scenario_payloads):
        union = selected | nominal_set
        scenario_table.loc[row_index, "final_set_Jaccard_to_nominal"] = (
            len(selected & nominal_set) / len(union) if union else 1.0
        )
    selections = pd.DataFrame(selection_rows)
    stability = (
        selections.groupby("candidate_id", as_index=False)
        .agg(selection_count=("scenario_id", "nunique"))
        .sort_values(["selection_count", "candidate_id"], ascending=[False, True])
    )
    stability["selection_frequency"] = stability["selection_count"] / len(
        scenario_table
    )
    nominal_rank = pd.read_csv(DATA_DIR / "final_prioritized_candidates.csv")
    stability["selected_in_nominal"] = stability["candidate_id"].isin(
        nominal_rank["candidate_id"].astype(str)
    )
    scenario_table.to_csv(DATA_DIR / "threshold_sensitivity.csv", index=False)
    stability.to_csv(DATA_DIR / "candidate_selection_stability.csv", index=False)
    return scenario_table, stability


def build_priority_table() -> pd.DataFrame:
    """Select two leads plus decision-boundary and AD-stress controls."""

    library = pd.read_csv(DATA_DIR / "candidate_library.csv", low_memory=False)
    screening = pd.read_csv(DATA_DIR / "screening_trace.csv", low_memory=False)
    probability = pd.read_csv(DATA_DIR / "feasibility_probability.csv")
    final = pd.read_csv(DATA_DIR / "final_prioritized_candidates.csv", low_memory=False)
    merged = (
        library.merge(
            screening[
                [
                    "candidate_id",
                    "final_feasible",
                    "failure_reasons",
                    "AD_status",
                    "viscosity_worst",
                    "conductivity_worst",
                ]
            ],
            on="candidate_id",
            how="left",
        )
        .merge(probability, on="candidate_id", how="left")
    )
    final_ids = set(final["candidate_id"].astype(str))
    primary = merged[
        merged["candidate_id"].isin(final_ids)
        & merged["AD_status"].eq("in_domain")
    ].sort_values(
        ["constraint_pass_probability", "conductivity_worst"],
        ascending=[False, False],
    )
    selected_ids = primary.head(2)["candidate_id"].astype(str).tolist()
    boundary = merged[
        ~merged["final_feasible"].fillna(False)
        & merged["failure_reasons"].fillna("").str.contains("viscosity")
    ].sort_values(
        ["constraint_pass_probability", "viscosity_worst"],
        ascending=[False, True],
    )
    if not boundary.empty:
        selected_ids.append(str(boundary.iloc[0]["candidate_id"]))
    exploratory = merged[
        merged["candidate_id"].isin(final_ids)
        & merged["AD_status"].eq("borderline")
    ].sort_values(
        ["constraint_pass_probability", "conductivity_worst"],
        ascending=[False, False],
    )
    if not exploratory.empty:
        selected_ids.append(str(exploratory.iloc[0]["candidate_id"]))
    selected_ids = list(dict.fromkeys(selected_ids))[:4]
    priority = merged[merged["candidate_id"].isin(selected_ids)].copy()
    priority["priority_order"] = priority["candidate_id"].map(
        {candidate_id: index + 1 for index, candidate_id in enumerate(selected_ids)}
    )
    role_map = {
        selected_ids[0]: "primary_lead_A",
        selected_ids[1]: "primary_lead_B" if len(selected_ids) > 1 else "primary_lead_A",
    }
    if len(selected_ids) > 2:
        role_map[selected_ids[2]] = "decision_boundary_control"
    if len(selected_ids) > 3:
        role_map[selected_ids[3]] = "borderline_AD_stress_test"
    priority["validation_role"] = priority["candidate_id"].map(role_map)
    priority = priority.sort_values("priority_order")
    columns = [
        "priority_order",
        "candidate_id",
        "validation_role",
        "cation_smiles",
        "anion_smiles",
        "AD_status",
        "final_feasible",
        "failure_reasons",
        "constraint_pass_probability",
        "pareto_rank_1_probability",
        "conductivity_worst",
        "viscosity_worst",
    ]
    priority[columns].to_csv(DATA_DIR / "experimental_validation_priority.csv", index=False)
    return priority[columns]


def make_evidence_figure(
    validation: pd.DataFrame,
    identity_counts: dict[str, int],
    sensitivity: pd.DataFrame,
    stability: pd.DataFrame,
    priority: pd.DataFrame,
) -> list[Path]:
    """Create the six-panel validation figure from persisted numerical data."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.4,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.5,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "blue": "#0072B2",
        "sky": "#56B4E9",
        "green": "#009E73",
        "orange": "#E69F00",
        "vermillion": "#D55E00",
        "purple": "#CC79A7",
        "gray": "#8A8A8A",
        "light": "#E8ECEF",
    }
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.25), constrained_layout=True)
    axes = axes.ravel()

    def title(ax: mpl.axes.Axes, letter: str, text: str) -> None:
        ax.text(-0.16, 1.08, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)
        ax.set_title(text, loc="left", fontweight="bold", pad=8)

    # a | Identity-audited split-strategy generalization summary.
    macro = (
        validation.groupby("split_strategy", as_index=False)
        .agg(log_R2=("log_R2", "mean"), log_NMAE=("log_NMAE", "mean"))
    )
    macro["split_strategy"] = pd.Categorical(
        macro["split_strategy"], categories=SPLIT_ORDER, ordered=True
    )
    macro = macro.sort_values("split_strategy")
    x = np.arange(len(macro))
    ax = axes[0]
    title(ax, "a", "Identity-audited generalization")
    ax.plot(x, macro["log_R2"], marker="o", color=colors["blue"], label=r"macro log $R^2$")
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel(r"Macro log $R^2$")
    ax.set_xticks(x, ["Point", "Random\nIL", "Balanced\nIL", "Ion\nfamily"])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, macro["log_NMAE"], marker="s", color=colors["vermillion"], label="macro log NMAE")
    ax2.set_ylim(0.0, 0.36)
    ax2.set_ylabel("Macro log NMAE")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", frameon=False)

    # b | Random whole-IL rank and curve preservation.
    ax = axes[1]
    title(ax, "b", "Random-IL ranking and trends")
    random_il = validation[validation["split_strategy"].eq("Random IL")].copy()
    random_il["property"] = pd.Categorical(random_il["property"], PROPERTIES, ordered=True)
    random_il = random_il.sort_values("property")
    bx = np.arange(len(random_il))
    ax.bar(
        bx - 0.18,
        random_il["IL_median_rank_Spearman"],
        width=0.36,
        color=colors["blue"],
        label="IL-rank Spearman",
    )
    ax.bar(
        bx + 0.18,
        random_il["temperature_trend_sign_agreement_fraction"],
        width=0.36,
        color=colors["green"],
        label="trend-sign agreement",
    )
    for index, value in enumerate(
        random_il["temperature_trend_sign_agreement_fraction"].to_numpy(dtype=float)
    ):
        if not np.isfinite(value):
            ax.text(
                bx[index] + 0.18,
                0.50,
                "pressure\nmissing",
                ha="center",
                va="center",
                fontsize=5.7,
                color=colors["gray"],
            )
    ax.axhline(0.8, color=colors["gray"], ls="--", lw=0.8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fraction or correlation")
    ax.set_xticks(
        bx,
        [PROPERTY_SHORT[value] for value in random_il["property"].astype(str)],
        rotation=35,
        ha="right",
    )
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        loc="lower center",
        ncol=1,
        handlelength=1.3,
    )

    # c | Identity audit.
    ax = axes[2]
    title(ax, "c", "Chemical identity audit")
    values = [
        identity_counts["old_valid_unseen_pool"],
        identity_counts["new_valid_unseen_pool"],
        identity_counts["old_pool_pseudo_unseen_known_identity"],
    ]
    bars = ax.bar(
        [0, 1, 2],
        values,
        color=[colors["gray"], colors["blue"], colors["vermillion"]],
        width=0.65,
    )
    ax.set_xticks([0, 1, 2], ["SMILES\nnovel", "InChIKey\nnovel", "Pseudo-\nunseen"])
    ax.set_ylabel("Ion-pair count")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 9, str(value), ha="center", fontsize=8)
    ax.text(
        0.04,
        0.72,
        f"Old shortlist: {identity_counts['old_shortlist_known_pair_controls']}/8 known pairs\n"
        f"New shortlist: {identity_counts['new_shortlist_known_pair_controls']}/8 known pairs",
        transform=ax.transAxes,
        va="top",
        fontsize=7.1,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )

    # d | Threshold sensitivity.
    ax = axes[3]
    title(ax, "d", "Threshold sensitivity (81 settings)")
    jitter = np.linspace(-0.14, 0.14, len(sensitivity))
    ax.scatter(
        np.zeros(len(sensitivity)) + jitter,
        sensitivity["hard_feasible_count"],
        s=15,
        color=colors["sky"],
        alpha=0.7,
        edgecolor="none",
    )
    nominal = sensitivity[sensitivity["is_nominal"]].iloc[0]
    ax.scatter([0], [nominal["hard_feasible_count"]], s=58, marker="D", color=colors["vermillion"], zorder=4, label="nominal")
    ax.boxplot(
        sensitivity["hard_feasible_count"],
        positions=[0],
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "none", "edgecolor": colors["blue"]},
        medianprops={"color": colors["blue"], "linewidth": 1.4},
    )
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([0], ["Reference-quantile grid"])
    ax.set_ylabel("Hard-feasible candidates")
    ax.legend(frameon=False, loc="upper right")

    # e | Protocol-level feasibility/Pareto probability.
    ax = axes[4]
    title(ax, "e", "Protocol-level decision stability")
    probability = pd.read_csv(DATA_DIR / "feasibility_probability.csv")
    final_ids = set(
        pd.read_csv(DATA_DIR / "final_prioritized_candidates.csv")["candidate_id"].astype(str)
    )
    unseen = probability[probability["candidate_id"].str.startswith("UPR-")].copy()
    selected = unseen["candidate_id"].isin(final_ids)
    ax.scatter(
        unseen.loc[~selected, "constraint_pass_probability"],
        unseen.loc[~selected, "pareto_rank_1_probability"],
        s=14,
        color=colors["light"],
        edgecolor=colors["gray"],
        linewidth=0.35,
        alpha=0.75,
        label="screened population",
    )
    ax.scatter(
        unseen.loc[selected, "constraint_pass_probability"],
        unseen.loc[selected, "pareto_rank_1_probability"],
        s=34,
        color=colors["vermillion"],
        edgecolor="white",
        linewidth=0.6,
        label="nominal final eight",
        zorder=3,
    )
    ax.plot([0, 1], [0, 1], color=colors["gray"], lw=0.8, ls="--")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Hard-pass frequency")
    ax.set_ylabel("Pareto-1 frequency")
    ax.legend(frameon=False, loc="upper left")

    # f | Experimental priority set.
    ax = axes[5]
    title(ax, "f", "Measurement-priority set")
    plot_priority = priority.sort_values("priority_order", ascending=False)
    colors_priority = [
        colors["orange"] if status == "borderline" else colors["green"]
        for status in plot_priority["AD_status"]
    ]
    bars = ax.barh(
        plot_priority["candidate_id"],
        plot_priority["constraint_pass_probability"],
        color=colors_priority,
        height=0.58,
    )
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Hard-pass frequency")
    ax.axvline(2 / 3, color=colors["gray"], lw=0.8, ls="--")
    for bar, role in zip(bars, plot_priority["validation_role"]):
        label = role.replace("_", " ")
        ax.text(0.02, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=7.1)

    fig.suptitle(
        "Auditable virtual pre-screening: validation, identity control, and decision stability",
        fontsize=10.5,
        fontweight="bold",
    )
    base = FIGURE_DIR / "figure5_auditable_virtual_screening_validation"
    paths = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=600)
    plt.close(fig)
    shutil.copy2(paths[0], PAPER_FIG_DIR / paths[0].name)
    shutil.copy2(paths[1], PAPER_FIG_DIR / paths[1].name)
    return paths


def build_summary(
    validation: pd.DataFrame,
    split_identity_audit: pd.DataFrame,
    identity_counts: dict[str, int],
    sensitivity: pd.DataFrame,
    stability: pd.DataFrame,
    priority: pd.DataFrame,
) -> dict[str, object]:
    final = pd.read_csv(DATA_DIR / "final_prioritized_candidates.csv", low_memory=False)
    probability = pd.read_csv(DATA_DIR / "feasibility_probability.csv")
    final_probability = final[["candidate_id"]].merge(
        probability, on="candidate_id", how="left"
    )
    cell_ranges = {}
    for column in [
        "electrolyte_resistance_ohm_worst",
        "joule_heating_power_W_worst",
        "steady_state_temperature_rise_K_worst",
        "transient_temperature_rise_K_worst",
        "low_temperature_resistance_ratio_to_reference",
        "high_temperature_resistance_ratio_to_reference",
        "reference_cell_exceedance_index_worst",
    ]:
        cell_ranges[column] = {
            "min": float(final[column].min()),
            "max": float(final[column].max()),
        }
    random_il = validation[validation["split_strategy"].eq("Random IL")]
    identity_exclusions = pd.read_csv(
        DATA_DIR / "whole_il_identity_audit_excluded_rows.csv"
    )
    identity_exclusion_counts = (
        identity_exclusions.groupby(
            ["split_strategy", "identity_audit_exclusion_reason"], as_index=False
        )
        .size()
        .rename(columns={"size": "prediction_record_count"})
    )
    audited_macro = (
        validation.groupby("split_strategy", as_index=False)
        .agg(
            log_MAE=("log_MAE", "mean"),
            log_RMSE=("log_RMSE", "mean"),
            log_R2=("log_R2", "mean"),
            log_NMAE=("log_NMAE", "mean"),
        )
    )
    priority_records = (
        priority.astype(object)
        .where(pd.notna(priority), None)
        .to_dict(orient="records")
    )
    summary = {
        "identity_audit": identity_counts,
        "whole_il_validation": {
            "identity_audit": split_identity_audit.to_dict(orient="records"),
            "prediction_record_exclusions": identity_exclusion_counts.to_dict(
                orient="records"
            ),
            "random_IL_property_count": int(len(random_il)),
            "random_IL_point_MdAPE_pct_median_across_properties": float(
                random_il["point_MdAPE_pct"].median()
            ),
            "random_IL_IL_rank_Spearman_median_across_properties": float(
                random_il["IL_median_rank_Spearman"].median()
            ),
            "random_IL_trend_sign_agreement_median_across_properties": float(
                random_il["temperature_trend_sign_agreement_fraction"].median()
            ),
            "macro_split_metrics": audited_macro.to_dict(orient="records"),
        },
        "screening": {
            "hard_feasible_nominal": int(
                sensitivity.loc[sensitivity["is_nominal"], "hard_feasible_count"].iloc[0]
            ),
            "sensitivity_scenario_count": int(len(sensitivity)),
            "hard_feasible_min": int(sensitivity["hard_feasible_count"].min()),
            "hard_feasible_max": int(sensitivity["hard_feasible_count"].max()),
            "median_final_set_Jaccard_to_nominal": float(
                sensitivity["final_set_Jaccard_to_nominal"].median()
            ),
            "candidates_selected_in_all_scenarios": int(
                stability["selection_frequency"].eq(1.0).sum()
            ),
        },
        "protocol_stability": {
            "maximum_constraint_pass_probability": float(
                probability.loc[
                    probability["candidate_id"].str.startswith("UPR-"),
                    "constraint_pass_probability",
                ].max()
            ),
            "final_candidate_probability_rows": final_probability.to_dict(
                orient="records"
            ),
        },
        "reference_cell_final_eight_ranges": cell_ranges,
        "experimental_priority": priority_records,
    }
    (AUDIT_DIR / "auditable_chapter_evidence.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    for directory in [DATA_DIR, AUDIT_DIR, FIGURE_DIR, TABLE_DIR, PAPER_FIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    split_identity_audit, overlap_by_split = build_split_identity_audit()
    validation, curves = build_whole_il_validation(overlap_by_split)
    old_audit, new_audit, identity_counts = build_identity_audit()
    sensitivity, stability = build_threshold_sensitivity()
    priority = build_priority_table()
    figure_paths = make_evidence_figure(
        validation, identity_counts, sensitivity, stability, priority
    )
    summary = build_summary(
        validation,
        split_identity_audit,
        identity_counts,
        sensitivity,
        stability,
        priority,
    )
    for source_name in [
        "figure6_reference_cell_scenario.pdf",
        "figure6_reference_cell_scenario.png",
    ]:
        source = FIGURE_DIR / source_name
        suffix = Path(source_name).suffix
        target = PAPER_FIG_DIR / f"figure6_reference_cell_scenario_audited{suffix}"
        shutil.copy2(source, target)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Figures:", *(str(path) for path in figure_paths), sep="\n- ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
