"""Build the non-redundant application figures and manuscript evidence.

Formal candidate values come from one primary random-IL checkpoint.  The
balanced-IL and ion-family checkpoints are used only as independent protocol
sensitivity models; their predictions are never averaged.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


CASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CASE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import build_auditable_evidence as audit  # noqa: E402
import build_protocol_stability_outputs as protocol_inputs  # noqa: E402
from experiments.computational_application_case.src.config import (  # noqa: E402
    load_case_config,
    temperature_grid,
)
from experiments.computational_application_case.src.paths import (  # noqa: E402
    resolve_project_path,
)
from experiments.computational_application_case.src.proxies import (  # noqa: E402
    compute_application_proxies,
    summarize_whole_temperature_window,
)
from experiments.computational_application_case.src.screening import (  # noqa: E402
    audit_curve_quality,
    curve_counts,
    derive_reference_thresholds,
    prioritize_candidates,
    screen_candidates,
)


PRIMARY_ROOT = CASE_DIR / "outputs_primary_audited"
DATA_DIR = PRIMARY_ROOT / "data"
AUDIT_DIR = PRIMARY_ROOT / "audit"
FIGURE_DIR = PRIMARY_ROOT / "figures"
TABLE_DIR = PRIMARY_ROOT / "tables"
PAPER_DIR = PROJECT_ROOT / "LaTex-MIPGraph"
PAPER_FIG_DIR = PAPER_DIR / "Fig"
PAPER_SOURCE_DIR = PAPER_FIG_DIR / "source_data"

# Files produced by superseded application-case workflows.  They must never
# coexist with the submission source-data bundle because several contain an
# obsolete three-checkpoint ensemble or obsolete candidate lists.
LEGACY_APPLICATION_SOURCE_NAMES = {
    "auditable_chapter_evidence.json",
    "experimental_validation_priority.csv",
    "feasibility_probability.csv",
    "final_candidate_table.csv",
    "reference_cell_output_audit.json",
    "reference_cell_scenario_parameters.csv",
    "whole_il_validation_summary.csv",
    "whole_il_curve_validation.csv",
    "whole_il_split_identity_audit.csv",
    "whole_il_identity_audit_excluded_rows.csv",
}

MAIN_T_MIN = 298.15
MAIN_T_MAX = 353.15
STRESS_ENDPOINTS = (278.15, 373.15)
PROTOCOL_SPECS = (
    (
        "Random-IL primary",
        PRIMARY_ROOT,
        CASE_DIR / "configs" / "auditable_virtual_screening.yaml",
        True,
    ),
    (
        "Balanced-IL sensitivity",
        CASE_DIR / "outputs_protocol_balanced",
        CASE_DIR / "configs" / "protocol_stability_balanced.yaml",
        False,
    ),
    (
        "Ion-family sensitivity",
        CASE_DIR / "outputs_protocol_ion_family",
        CASE_DIR / "configs" / "protocol_stability_ion_family.yaml",
        False,
    ),
)
PROTOCOL_ORDER = [spec[0] for spec in PROTOCOL_SPECS]

COLORS = {
    "navy": "#17324D",
    "blue": "#2878B5",
    "sky": "#76B7D8",
    "teal": "#138A8A",
    "green": "#2A9D68",
    "gold": "#E3A424",
    "orange": "#D97732",
    "red": "#C84A3A",
    "purple": "#7A5AA6",
    "ink": "#25313C",
    "gray": "#8E9AA3",
    "light": "#E7ECEF",
    "pale": "#F5F7F8",
}
LEAD_COLORS = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 7.7,
            "axes.labelsize": 7.1,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def panel_title(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=9.0, fontweight="bold")
    ax.set_title(title, loc="left", pad=5, fontweight="bold")


def panel_title_wide(ax: plt.Axes, label: str, title: str) -> None:
    """Place a panel label immediately beside a full-width panel title."""

    ax.text(-0.018, 1.06, label, transform=ax.transAxes, fontsize=9.0, fontweight="bold")
    ax.set_title(title, loc="left", pad=5, fontweight="bold")


def panel_title_compact(ax: plt.Axes, label: str, title: str) -> None:
    """Use compact typography for five-column figure rows."""

    ax.text(-0.13, 1.055, label, transform=ax.transAxes, fontsize=8.0, fontweight="bold")
    ax.set_title(title, loc="left", pad=5, fontweight="bold", fontsize=6.6)


def _load_protocol_source(
    protocol: str,
    output_root: Path,
    config_path: Path,
    primary_library: pd.DataFrame,
    require_manifest: bool,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, str]:
    """Load one independently generated checkpoint output with fail-closed provenance checks."""

    config = load_case_config(config_path)
    source_library = pd.read_csv(output_root / "data" / "candidate_library.csv", low_memory=False)
    primary_digest = protocol_inputs.identity_sha256(primary_library)
    if protocol_inputs.identity_sha256(source_library) != primary_digest:
        raise RuntimeError(f"{protocol} does not use the frozen primary candidate identities")

    inference_path = output_root / "audit" / "inference_pipeline.json"
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    expected_checkpoint = resolve_project_path(PROJECT_ROOT, config["model"]["checkpoint_path"])
    expected_split = resolve_project_path(PROJECT_ROOT, config["data"]["split_path"]).resolve()
    repository_audit = json.loads(
        (output_root / "audit" / "repository_audit.json").read_text(encoding="utf-8")
    )
    if Path(str(repository_audit["split"])).resolve() != expected_split:
        raise RuntimeError(f"{protocol} repository audit has a stale training split")
    actual_checkpoint = Path(str(inference["checkpoint_path"])).resolve()
    if actual_checkpoint != expected_checkpoint.resolve():
        raise RuntimeError(
            f"{protocol} checkpoint mismatch: expected {expected_checkpoint}, got {actual_checkpoint}"
        )
    if require_manifest:
        manifest_path = output_root / "audit" / "protocol_stability_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"Missing {manifest_path}; run build_protocol_stability_outputs.py first"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_full_grid = np.unique(
            np.concatenate(
                (
                    temperature_grid(config["conditions"]),
                    temperature_grid(config["conditions"], extended=True),
                )
            )
        )
        try:
            protocol_inputs.validate_protocol_manifest(
                manifest,
                config_path=config_path,
                training_split_path=expected_split,
                candidate_identity_digest=primary_digest,
                checkpoint_digest=protocol_inputs.file_sha256(expected_checkpoint),
                candidate_count=len(primary_library),
                temperature_grid_K=expected_full_grid,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{protocol}: {exc}") from exc

    predictions = pd.read_csv(
        output_root / "data" / "property_predictions_long.csv", low_memory=False
    )
    predicted_identity = predictions[list(protocol_inputs.IDENTITY_COLUMNS)].drop_duplicates()
    if protocol_inputs.identity_sha256(predicted_identity) != primary_digest:
        raise RuntimeError(f"{protocol} predictions do not preserve the frozen ID--InChI mapping")
    expected_temperatures = temperature_grid(config["conditions"])
    member = predictions[predictions["analysis_window"].eq("main")].copy()
    actual_temperatures = np.sort(member["temperature_K"].unique())
    if not np.array_equal(actual_temperatures, expected_temperatures):
        raise RuntimeError(
            f"{protocol} main-window temperatures are incomplete: {actual_temperatures.tolist()}"
        )
    expected_rows = len(primary_library) * len(expected_temperatures)
    if len(member) != expected_rows:
        raise RuntimeError(
            f"{protocol} main-window coverage is incomplete: expected {expected_rows}, got {len(member)}"
        )
    ad_table = pd.read_csv(output_root / "data" / "applicability_domain.csv")
    if set(ad_table["candidate_id"].astype(str)) != set(primary_library["candidate_id"].astype(str)):
        raise RuntimeError(f"{protocol} applicability-domain table has stale candidate identities")
    split_path = str(expected_split)
    return config, member, ad_table, split_path


def _robust_objective_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Return q05--q95 clipped scores, with 1 denoting the preferred direction."""

    definitions = {
        "conductivity_worst": "maximize",
        "viscosity_worst": "minimize",
        "volumetric_heat_capacity_worst": "maximize",
        "thermal_diffusivity_worst": "maximize",
    }
    scores = pd.DataFrame(index=frame.index)
    for column, sense in definitions.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        low, high = [float(value) for value in values.quantile([0.05, 0.95])]
        if np.isclose(low, high, rtol=0.0, atol=1.0e-15):
            scores[column] = 1.0
        elif sense == "maximize":
            scores[column] = np.clip((values - low) / (high - low), 0.0, 1.0)
        else:
            scores[column] = np.clip((high - values) / (high - low), 0.0, 1.0)
    return scores


def build_cross_protocol_decisions(primary_final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-evaluate checkpoints on fixed candidates and fixed primary property thresholds."""

    primary_config = load_case_config(CASE_DIR / "configs" / "auditable_virtual_screening.yaml")
    library = pd.read_csv(DATA_DIR / "candidate_library.csv", low_memory=False)
    thresholds = json.loads((DATA_DIR / "reference_thresholds.json").read_text(encoding="utf-8"))
    benchmark = pd.read_csv(
        PROJECT_ROOT
        / "il_property_prediction"
        / "data"
        / "processed"
        / "il_multiprop_clean.csv",
        low_memory=False,
    )
    primary_ids = primary_final["candidate_id"].astype(str).tolist()
    decision_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    protocol_detail_dir = DATA_DIR / "cross_protocol_details"
    protocol_detail_dir.mkdir(parents=True, exist_ok=True)
    (protocol_detail_dir / "fixed_primary_thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )

    for protocol, output_root, config_path, formal_primary in PROTOCOL_SPECS:
        _, member, ad_table, split_path = _load_protocol_source(
            protocol,
            output_root,
            config_path,
            library,
            require_manifest=not formal_primary,
        )
        proxies = compute_application_proxies(member, primary_config["proxies"])
        robust = summarize_whole_temperature_window(proxies)
        flags = audit_curve_quality(member, benchmark, primary_config["curve_quality"])
        counts = curve_counts(flags)
        robust = robust.drop(
            columns=["curve_warning_count", "severe_curve_failure_count"],
            errors="ignore",
        )
        robust = robust.merge(counts, on="candidate_id", how="left")
        for column in ["curve_warning_count", "severe_curve_failure_count"]:
            robust[column] = robust[column].fillna(0).astype(int)
        trace = screen_candidates(
            robust,
            ad_table,
            library,
            thresholds,
            primary_config["screening"],
        )
        ranked, selected = prioritize_candidates(trace, primary_config["pareto"])
        rank_lookup = ranked.set_index("candidate_id")["Pareto_rank"].to_dict()
        protocol_scores = _robust_objective_scores(ranked)
        ranked["protocol_decision_distance"] = np.sqrt(
            np.mean((1.0 - protocol_scores.to_numpy(dtype=float)) ** 2, axis=1)
        )
        ranked = ranked.sort_values(
            ["Pareto_rank", "protocol_decision_distance", "candidate_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        ranked["protocol_rank_order"] = np.arange(1, len(ranked) + 1, dtype=int)
        order_lookup = ranked.set_index("candidate_id")["protocol_rank_order"].to_dict()
        distance_lookup = ranked.set_index("candidate_id")["protocol_decision_distance"].to_dict()
        selected_ids = set(selected["candidate_id"].astype(str))
        if formal_primary and selected_ids != set(primary_ids):
            missing = sorted(set(primary_ids) - selected_ids)
            unexpected = sorted(selected_ids - set(primary_ids))
            raise RuntimeError(
                "Recomputed primary protocol does not reproduce the formal candidate set: "
                f"missing={missing}, unexpected={unexpected}"
            )
        trace_lookup = trace.set_index("candidate_id")
        for candidate_id in primary_ids:
            row = trace_lookup.loc[candidate_id]
            feasible = bool(row["final_feasible"])
            rank = rank_lookup.get(candidate_id, np.nan)
            chosen = candidate_id in selected_ids
            if str(row.get("AD_status", "")) == "out_of_domain" or not bool(
                row.get("pass_AD", True)
            ):
                code, label = 0, "out of\ndomain"
            elif not bool(row.get("pass_curve_quality", True)):
                code, label = 1, "severe\ncurve"
            elif not feasible:
                code, label = 2, "hard\nfail"
            elif np.isfinite(rank) and int(rank) == 1:
                code, label = 4, "P1"
            else:
                code, label = 3, "hard pass\n(dominated)"
            decision_rows.append(
                {
                    "candidate_id": candidate_id,
                    "protocol": protocol,
                    "formal_primary_protocol": formal_primary,
                    "hard_feasible": feasible,
                    "pareto_rank": rank,
                    "protocol_rank_order": order_lookup.get(candidate_id, np.nan),
                    "protocol_decision_distance": distance_lookup.get(candidate_id, np.nan),
                    "protocol_top8": chosen,
                    "decision_code": code,
                    "decision_label": label,
                    "failure_reasons": str(row.get("failure_reasons", "")),
                    "property_threshold_source": "formal primary reference thresholds",
                    "applicability_domain_split": split_path,
                }
            )
        summary_rows.append(
            {
                "protocol": protocol,
                "formal_primary_protocol": formal_primary,
                "hard_feasible_count": int(
                    (
                        trace["candidate_type"].eq("unseen_pair_recombination")
                        & trace["final_feasible"]
                    ).sum()
                ),
                "pareto_rank_1_count": int(ranked["Pareto_rank"].eq(1).sum()),
                "reported_top8_count": int(len(selected)),
                "prediction_aggregation": "none; independent checkpoint",
                "property_threshold_source": "formal primary reference thresholds",
                "applicability_domain_split": split_path,
            }
        )
        slug = protocol.lower().replace("-", "_").replace(" ", "_")
        trace.to_csv(protocol_detail_dir / f"{slug}_screening_trace.csv", index=False)
        ranked.to_csv(protocol_detail_dir / f"{slug}_pareto.csv", index=False)

    matrix = pd.DataFrame(decision_rows)
    summary = pd.DataFrame(summary_rows)
    matrix.to_csv(DATA_DIR / "cross_protocol_decision_matrix.csv", index=False)
    summary.to_csv(DATA_DIR / "cross_protocol_summary.csv", index=False)
    return matrix, summary


def build_downstream_priorities(
    final: pd.DataFrame, protocol_matrix: pd.DataFrame
) -> pd.DataFrame:
    """Apply the four prespecified qualification-role rules without manual substitution."""

    frame = final.copy().reset_index(drop=True)
    scores = _robust_objective_scores(frame)
    score_columns = [
        "conductivity_worst",
        "viscosity_worst",
        "volumetric_heat_capacity_worst",
        "thermal_diffusivity_worst",
    ]
    frame["balanced_distance"] = np.sqrt(
        np.mean((1.0 - scores[score_columns].to_numpy(dtype=float)) ** 2, axis=1)
    )
    frame["transport_distance"] = np.sqrt(
        np.mean(
            (1.0 - scores[["conductivity_worst", "viscosity_worst"]].to_numpy(dtype=float)) ** 2,
            axis=1,
        )
    )
    frame["thermal_management_distance"] = np.sqrt(
        np.mean(
            (
                1.0
                - scores[
                    ["volumetric_heat_capacity_worst", "thermal_diffusivity_worst"]
                ].to_numpy(dtype=float)
            )
            ** 2,
            axis=1,
        )
    )
    protocol = protocol_matrix[
        protocol_matrix["protocol"].isin(["Random-IL primary", "Balanced-IL sensitivity"])
    ].copy()
    protocol["eligible_in_protocol"] = protocol["hard_feasible"] | protocol["pareto_rank"].eq(1)
    eligibility = protocol.pivot(index="candidate_id", columns="protocol", values="eligible_in_protocol")
    ranks = protocol.pivot(index="candidate_id", columns="protocol", values="protocol_rank_order")
    primary_col, balanced_col = "Random-IL primary", "Balanced-IL sensitivity"
    cross_eligible = eligibility.get(primary_col, pd.Series(dtype=bool)).fillna(False) & eligibility.get(
        balanced_col, pd.Series(dtype=bool)
    ).fillna(False)
    rank_change = (ranks.get(primary_col, pd.Series(dtype=float)) - ranks.get(
        balanced_col, pd.Series(dtype=float)
    )).abs()
    frame["cross_protocol_eligible"] = frame["candidate_id"].map(cross_eligible).fillna(False)
    frame["cross_protocol_rank_change"] = frame["candidate_id"].map(rank_change)

    role_specs = [
        ("balanced lead", frame["AD_status"].eq("in_domain"), "balanced_distance"),
        ("transport-focused lead", pd.Series(True, index=frame.index), "transport_distance"),
        ("thermal-management lead", pd.Series(True, index=frame.index), "thermal_management_distance"),
        ("cross-protocol robust lead", frame["cross_protocol_eligible"], "cross_protocol_rank_change"),
    ]
    selected_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for role_order, (role, eligible, criterion) in enumerate(role_specs, start=1):
        candidates = frame[eligible & frame[criterion].notna()].sort_values(
            [criterion, "candidate_id"], kind="mergesort"
        )
        selected_id = str(candidates.iloc[0]["candidate_id"]) if not candidates.empty else None
        for index, row in frame.iterrows():
            criterion_value = pd.to_numeric(row[criterion], errors="coerce")
            audit_rows.append(
                {
                    "qualification_role": role,
                    "candidate_id": str(row["candidate_id"]),
                    "eligible": bool(eligible.loc[index]),
                    "criterion": criterion,
                    "criterion_value": float(criterion_value) if np.isfinite(criterion_value) else np.nan,
                    "selected": str(row["candidate_id"]) == selected_id,
                    "deterministic_tie_breaker": "candidate_id ascending",
                }
            )
        if selected_id is not None:
            chosen = frame[frame["candidate_id"].astype(str).eq(selected_id)].iloc[0].to_dict()
            chosen["priority_order"] = role_order
            chosen["qualification_role"] = role
            chosen["selection_criterion"] = criterion
            chosen["selection_criterion_value"] = float(chosen[criterion])
            selected_rows.append(chosen)
    priority = pd.DataFrame(selected_rows)
    pd.DataFrame(audit_rows).to_csv(DATA_DIR / "qualification_role_selection_audit.csv", index=False)
    keep = [
        "priority_order",
        "candidate_id",
        "qualification_role",
        "cation_smiles",
        "anion_smiles",
        "AD_status",
        "Pareto_rank",
        "conductivity_worst",
        "viscosity_worst",
        "volumetric_heat_capacity_worst",
        "thermal_diffusivity_worst",
        "selection_criterion",
        "selection_criterion_value",
        "balanced_distance",
        "transport_distance",
        "thermal_management_distance",
        "cross_protocol_eligible",
        "cross_protocol_rank_change",
    ]
    available = [column for column in keep if column in priority]
    priority[available].to_csv(DATA_DIR / "downstream_qualification_priorities.csv", index=False)
    return priority[available]


def constraint_margins(final: pd.DataFrame) -> pd.DataFrame:
    thresholds = json.loads((DATA_DIR / "reference_thresholds.json").read_text(encoding="utf-8"))
    definitions = [
        ("conductivity_worst", "conductivity_min", "lower", r"$\sigma_{\min}$"),
        ("viscosity_worst", "viscosity_max", "upper", r"$\eta_{\max}$"),
        (
            "volumetric_heat_capacity_worst",
            "volumetric_heat_capacity_min",
            "lower",
            r"$C_{v,\min}$",
        ),
        (
            "thermal_diffusivity_worst",
            "thermal_diffusivity_min",
            "lower",
            r"$\alpha_{\min}$",
        ),
        (
            "surface_tension_reference_envelope_deviation_worst",
            "surface_tension_reference_envelope_deviation_max",
            "upper",
            "ST-envelope",
        ),
    ]
    rows = []
    for row in final.itertuples(index=False):
        for metric, threshold_key, direction, label in definitions:
            value = float(getattr(row, metric))
            threshold = float(thresholds[threshold_key])
            if value <= 0.0 or threshold <= 0.0:
                margin = float("inf") if direction == "upper" and value == 0.0 else float("nan")
            elif direction == "lower":
                margin = float(np.log2(value / threshold))
            else:
                margin = float(np.log2(threshold / value))
            rows.append(
                {
                    "candidate_id": str(row.candidate_id),
                    "constraint": label,
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "constraint_direction": direction,
                    "log2_margin": margin,
                    "plot_log2_margin": float(np.clip(margin, -4.0, 4.0)) if np.isfinite(margin) else 4.0,
                    "margin_definition": "log2(value/threshold)" if direction == "lower" else "log2(threshold/value)",
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(DATA_DIR / "final_candidate_constraint_margins.csv", index=False)
    return table


def build_reference_bootstrap_stability() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bootstrap observed references and replay hard screening, Pareto, and Top-8."""

    config = load_case_config(CASE_DIR / "configs" / "auditable_virtual_screening.yaml")
    repetitions = int(config["screening"].get("reference_bootstrap_replicates", 500))
    seed = int(config["screening"].get("reference_bootstrap_seed", 42))
    rng = np.random.default_rng(seed)
    robust = pd.read_csv(DATA_DIR / "candidate_robust_summary.csv", low_memory=False)
    trace_nominal = pd.read_csv(DATA_DIR / "screening_trace.csv", low_memory=False)
    predictions = pd.read_csv(DATA_DIR / "property_predictions_long.csv", low_memory=False)
    predictions = predictions[predictions["analysis_window"].eq("main")].copy()
    unseen = trace_nominal[
        trace_nominal["candidate_type"].eq("unseen_pair_recombination")
    ].copy().reset_index(drop=True)
    references = robust[robust["candidate_type"].eq("observed_reference")].copy()
    reference_ids = references["candidate_id"].astype(str).to_numpy()
    reference_lookup = references.set_index("candidate_id")
    nominal_top8 = set(
        pd.read_csv(DATA_DIR / "final_prioritized_candidates.csv")["candidate_id"].astype(str)
    )
    nominal_pareto = pd.read_csv(DATA_DIR / "pareto_candidates.csv")
    nominal_rank1 = set(
        nominal_pareto.loc[nominal_pareto["Pareto_rank"].eq(1), "candidate_id"].astype(str)
    )
    candidate_ids = unseen["candidate_id"].astype(str).tolist()
    hard_counts = dict.fromkeys(candidate_ids, 0)
    rank1_counts = dict.fromkeys(candidate_ids, 0)
    selection_counts = dict.fromkeys(candidate_ids, 0)
    iteration_rows: list[dict[str, object]] = []
    fixed_gates = ["pass_structure", "pass_inference", "pass_curve_quality", "pass_AD"]
    if not bool(config["screening"].get("exclude_out_of_domain", True)):
        fixed_gates.remove("pass_AD")
    if not bool(config["screening"].get("exclude_severe_curve_failures", True)):
        fixed_gates.remove("pass_curve_quality")
    surface = predictions[
        ["candidate_id", "candidate_type", "temperature_K", "SurfaceTension"]
    ].copy()
    surface_unseen = surface[surface["candidate_type"].eq("unseen_pair_recombination")]
    surface_reference = surface[surface["candidate_type"].eq("observed_reference")]
    epsilon = float(config["proxies"]["numerical_epsilon"])
    gamma_low_q = float(config["proxies"]["reference_gamma_low_quantile"])
    gamma_high_q = float(config["proxies"]["reference_gamma_high_quantile"])

    for bootstrap_index in range(repetitions):
        sampled_ids = rng.choice(reference_ids, size=len(reference_ids), replace=True)
        sampled_reference = reference_lookup.loc[sampled_ids].reset_index()
        threshold_input = pd.concat(
            [sampled_reference, unseen.iloc[0:0]], ignore_index=True, sort=False
        )
        thresholds = derive_reference_thresholds(threshold_input, config["screening"])
        envelope_max: dict[str, float] = dict.fromkeys(candidate_ids, -np.inf)
        for temperature in sorted(surface_unseen["temperature_K"].unique()):
            ref_t = surface_reference[np.isclose(surface_reference["temperature_K"], temperature)]
            ref_values = ref_t.set_index("candidate_id").loc[sampled_ids, "SurfaceTension"].to_numpy(dtype=float)
            gamma_low, gamma_high = np.quantile(ref_values, [gamma_low_q, gamma_high_q])
            gamma_q25, gamma_q75 = np.quantile(ref_values, [0.25, 0.75])
            denominator = max(float(gamma_q75 - gamma_q25), epsilon)
            candidate_t = surface_unseen[np.isclose(surface_unseen["temperature_K"], temperature)]
            values = candidate_t["SurfaceTension"].to_numpy(dtype=float)
            deviation = np.maximum(
                np.maximum(gamma_low - values, values - gamma_high), 0.0
            ) / denominator
            for candidate_id, value in zip(candidate_t["candidate_id"].astype(str), deviation):
                envelope_max[candidate_id] = max(envelope_max[candidate_id], float(value))

        replay = unseen.copy()
        replay["surface_tension_reference_envelope_deviation_worst"] = replay["candidate_id"].astype(str).map(envelope_max)
        replay["pass_conductivity"] = replay["conductivity_worst"] >= thresholds["conductivity_min"]
        replay["pass_viscosity"] = replay["viscosity_worst"] <= thresholds["viscosity_max"]
        replay["pass_heat_capacity"] = replay["volumetric_heat_capacity_worst"] >= thresholds["volumetric_heat_capacity_min"]
        replay["pass_thermal_diffusivity"] = replay["thermal_diffusivity_worst"] >= thresholds["thermal_diffusivity_min"]
        replay["pass_surface_tension_reference_envelope"] = replay[
            "surface_tension_reference_envelope_deviation_worst"
        ] <= thresholds["surface_tension_reference_envelope_deviation_max"]
        all_gates = fixed_gates + [
            "pass_conductivity",
            "pass_viscosity",
            "pass_heat_capacity",
            "pass_thermal_diffusivity",
            "pass_surface_tension_reference_envelope",
        ]
        replay["final_feasible"] = replay[all_gates].all(axis=1)
        ranked, selected = prioritize_candidates(replay, config["pareto"])
        hard_ids = set(replay.loc[replay["final_feasible"], "candidate_id"].astype(str))
        rank1_ids = set(ranked.loc[ranked["Pareto_rank"].eq(1), "candidate_id"].astype(str))
        selected_ids = set(selected["candidate_id"].astype(str))
        for candidate_id in hard_ids:
            hard_counts[candidate_id] += 1
        for candidate_id in rank1_ids:
            rank1_counts[candidate_id] += 1
        for candidate_id in selected_ids:
            selection_counts[candidate_id] += 1
        union = nominal_top8 | selected_ids
        jaccard = len(nominal_top8 & selected_ids) / len(union) if union else 1.0
        iteration_rows.append(
            {
                "bootstrap_iteration": bootstrap_index + 1,
                "hard_feasible_count": len(hard_ids),
                "pareto_rank_1_count": len(rank1_ids),
                "formal_shortlist_count": len(selected_ids),
                "top8_jaccard_to_nominal": jaccard,
                "formal_shortlist_ids": ";".join(sorted(selected_ids)),
                "conductivity_min": thresholds["conductivity_min"],
                "viscosity_max": thresholds["viscosity_max"],
                "volumetric_heat_capacity_min": thresholds["volumetric_heat_capacity_min"],
                "thermal_diffusivity_min": thresholds["thermal_diffusivity_min"],
            }
        )

    iterations = pd.DataFrame(iteration_rows)
    candidates = unseen[["candidate_id", "canonical_il_key"]].copy()
    candidates["nominal_pareto_rank_1"] = candidates["candidate_id"].astype(str).isin(nominal_rank1)
    candidates["nominal_formal_shortlist"] = candidates["candidate_id"].astype(str).isin(nominal_top8)
    candidates["hard_feasible_frequency"] = candidates["candidate_id"].astype(str).map(hard_counts) / repetitions
    candidates["pareto_rank_1_frequency"] = candidates["candidate_id"].astype(str).map(rank1_counts) / repetitions
    candidates["selection_frequency"] = candidates["candidate_id"].astype(str).map(selection_counts) / repetitions
    candidates = candidates.sort_values(
        ["selection_frequency", "pareto_rank_1_frequency", "candidate_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    jaccard = iterations["top8_jaccard_to_nominal"].to_numpy(dtype=float)
    summary = pd.DataFrame(
        [
            {
                "bootstrap_replicates": repetitions,
                "random_seed": seed,
                "reference_candidate_count": len(reference_ids),
                "top8_jaccard_median": float(np.median(jaccard)),
                "top8_jaccard_95pct_lower": float(np.quantile(jaccard, 0.025)),
                "top8_jaccard_95pct_upper": float(np.quantile(jaccard, 0.975)),
                "hard_feasible_count_median": float(iterations["hard_feasible_count"].median()),
                "pareto_rank_1_count_median": float(iterations["pareto_rank_1_count"].median()),
                "formal_shortlist_count_median": float(iterations["formal_shortlist_count"].median()),
            }
        ]
    )
    iterations.to_csv(DATA_DIR / "reference_bootstrap_iterations.csv", index=False)
    candidates.to_csv(DATA_DIR / "reference_bootstrap_candidate_selection.csv", index=False)
    summary.to_csv(DATA_DIR / "reference_bootstrap_summary.csv", index=False)
    summary.to_csv(TABLE_DIR / "reference_bootstrap_summary.csv", index=False)
    (TABLE_DIR / "reference_bootstrap_summary.tex").write_text(
        summary.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.3g}"),
        encoding="utf-8",
    )
    return iterations, candidates, summary


def build_identity_and_reference_audits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write the complete identity, reference-selection, and Pareto-rank-one audits."""

    library = pd.read_csv(DATA_DIR / "candidate_library.csv", low_memory=False)
    predictions = pd.read_csv(DATA_DIR / "property_predictions_long.csv", low_memory=False)
    ad = pd.read_csv(DATA_DIR / "applicability_domain.csv", low_memory=False)

    unseen = library[library["candidate_type"].eq("unseen_pair_recombination")].copy()
    unseen["charge_balance_pass"] = (
        pd.to_numeric(unseen["cation_charge"], errors="coerce")
        + pd.to_numeric(unseen["anion_charge"], errors="coerce")
    ).eq(0)
    unseen["standard_inchikey_parse_pass"] = unseen[
        ["cation_identity_key", "anion_identity_key", "canonical_il_key"]
    ].notna().all(axis=1)
    unseen["identity_new_to_benchmark"] = ~unseen["pair_seen_in_benchmark"].astype(bool)
    unseen["identity_new_to_primary_training_split"] = ~unseen[
        "pair_seen_in_training"
    ].astype(bool)
    unseen["identity_audit_pass"] = unseen[
        [
            "charge_balance_pass",
            "standard_inchikey_parse_pass",
            "identity_new_to_benchmark",
            "identity_new_to_primary_training_split",
        ]
    ].all(axis=1)
    identity_columns = [
        "candidate_id",
        "cation_smiles",
        "anion_smiles",
        "canonical_cation_smiles",
        "canonical_anion_smiles",
        "cation_identity_key",
        "anion_identity_key",
        "canonical_il_key",
        "cation_charge",
        "anion_charge",
        "cation_support_count",
        "anion_support_count",
        "charge_balance_pass",
        "standard_inchikey_parse_pass",
        "identity_new_to_benchmark",
        "identity_new_to_primary_training_split",
        "identity_audit_pass",
        "generation_status",
        "warnings",
    ]
    identity = unseen[identity_columns].sort_values("candidate_id", kind="mergesort")
    if len(identity) != 608 or not bool(identity["identity_audit_pass"].all()):
        raise RuntimeError("The Standard-InChIKey audit must contain 608 passing identities")
    identity.to_csv(DATA_DIR / "standard_inchikey_identity_audit_608.csv", index=False)

    references = library[library["candidate_type"].eq("observed_reference")].copy()
    main = predictions[predictions["analysis_window"].eq("main")].copy()
    coverage = main.groupby("candidate_id").agg(
        main_window_prediction_rows=("temperature_K", "size"),
        main_window_temperature_count=("temperature_K", "nunique"),
        inference_success_count=("inference_status", lambda values: int(values.eq("success").sum())),
    )
    reference_audit = references.merge(ad, on="candidate_id", how="left", suffixes=("", "_ad"))
    reference_audit = reference_audit.merge(coverage, on="candidate_id", how="left")
    reference_audit["family_stratified_selection"] = False
    reference_audit["selection_rule"] = (
        "highest combined cation-plus-anion support; canonical_il_key ascending tie-break"
    )
    reference_audit["benchmark_identity_present"] = reference_audit[
        "pair_seen_in_benchmark"
    ].astype(bool)
    reference_audit["complete_main_window_prediction"] = reference_audit[
        "main_window_temperature_count"
    ].eq(12) & reference_audit["inference_success_count"].eq(12)
    reference_audit["fixed_before_candidate_ranking"] = True
    reference_columns = [
        "candidate_id",
        "IL_Name",
        "canonical_il_key",
        "cation_family",
        "anion_family",
        "cation_support_count",
        "anion_support_count",
        "combined_ion_support",
        "AD_status",
        "descriptor_knn_distance",
        "benchmark_identity_present",
        "complete_main_window_prediction",
        "main_window_temperature_count",
        "family_stratified_selection",
        "fixed_before_candidate_ranking",
        "selection_rule",
    ]
    reference_audit = reference_audit[reference_columns].sort_values(
        ["combined_ion_support", "canonical_il_key"],
        ascending=[False, True],
        kind="mergesort",
    )
    if len(reference_audit) != 30:
        raise RuntimeError("The threshold-reference audit must contain exactly 30 liquids")
    reference_audit.to_csv(DATA_DIR / "observed_reference_selection_audit.csv", index=False)

    pareto = pd.read_csv(DATA_DIR / "pareto_candidates.csv", low_memory=False)
    rank_one = pareto[pareto["Pareto_rank"].eq(1)].copy()
    rank_one = rank_one.sort_values("utopia_rank_one_order", kind="mergesort")
    if len(rank_one) != 12:
        raise RuntimeError("The nominal primary result must contain 12 Pareto-rank-one candidates")
    rank_one.to_csv(DATA_DIR / "pareto_rank1_all_candidates.csv", index=False)
    return identity, reference_audit, rank_one


def build_reference_cell_heat_resistance_audit(
    final: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit conduction and convection shares in the post hoc thermal resistance."""

    metrics = pd.read_csv(DATA_DIR / "reference_cell_metrics_temperature.csv", low_memory=False)
    required = {
        "thermal_resistance_conduction_fraction",
        "thermal_resistance_convection_fraction",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise RuntimeError(
            "Reference-cell metrics must be regenerated before the heat-resistance audit: "
            + ", ".join(sorted(missing))
        )
    formal_ids = set(final["candidate_id"].astype(str))
    audit_table = metrics[
        metrics["candidate_id"].astype(str).isin(formal_ids)
        & metrics["analysis_window"].eq("main")
    ][
        [
            "candidate_id",
            "temperature_K",
            "internal_thermal_conduction_resistance_K_per_W",
            "convective_thermal_resistance_K_per_W",
            "thermal_resistance_K_per_W",
            "thermal_resistance_conduction_fraction",
            "thermal_resistance_convection_fraction",
            "electrolyte_resistance_ohm",
            "transient_temperature_rise_K",
        ]
    ].copy()
    if len(audit_table) != 8 * 12:
        raise RuntimeError("Heat-resistance audit requires 12 main-window rows for each Top-8 liquid")
    fraction_sum = (
        audit_table["thermal_resistance_conduction_fraction"]
        + audit_table["thermal_resistance_convection_fraction"]
    )
    if not np.allclose(fraction_sum, 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Thermal-resistance fractions do not sum to one")
    audit_table.to_csv(
        DATA_DIR / "reference_cell_heat_resistance_contribution_audit.csv", index=False
    )

    summary = audit_table.groupby("candidate_id", as_index=False).agg(
        conduction_fraction_min=("thermal_resistance_conduction_fraction", "min"),
        conduction_fraction_median=("thermal_resistance_conduction_fraction", "median"),
        conduction_fraction_max=("thermal_resistance_conduction_fraction", "max"),
        convection_fraction_min=("thermal_resistance_convection_fraction", "min"),
        convection_fraction_median=("thermal_resistance_convection_fraction", "median"),
        convection_fraction_max=("thermal_resistance_convection_fraction", "max"),
        electrolyte_resistance_ohm_max=("electrolyte_resistance_ohm", "max"),
        conditional_temperature_rise_K_max=("transient_temperature_rise_K", "max"),
    )
    summary.to_csv(
        DATA_DIR / "reference_cell_heat_resistance_contribution_summary.csv", index=False
    )
    summary.to_csv(
        TABLE_DIR / "reference_cell_heat_resistance_contribution_summary.csv", index=False
    )
    (TABLE_DIR / "reference_cell_heat_resistance_contribution_summary.tex").write_text(
        summary.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.4g}"),
        encoding="utf-8",
    )

    log_resistance = np.log(audit_table["electrolyte_resistance_ohm"].to_numpy(dtype=float))
    log_rise = np.log(audit_table["transient_temperature_rise_K"].to_numpy(dtype=float))
    correlation = float(np.corrcoef(log_resistance, log_rise)[0, 1])
    population = pd.DataFrame(
        [
            {
                "formal_candidate_count": len(formal_ids),
                "temperature_row_count": len(audit_table),
                "conduction_fraction_min": float(
                    audit_table["thermal_resistance_conduction_fraction"].min()
                ),
                "conduction_fraction_median": float(
                    audit_table["thermal_resistance_conduction_fraction"].median()
                ),
                "conduction_fraction_max": float(
                    audit_table["thermal_resistance_conduction_fraction"].max()
                ),
                "convection_fraction_min": float(
                    audit_table["thermal_resistance_convection_fraction"].min()
                ),
                "convection_fraction_median": float(
                    audit_table["thermal_resistance_convection_fraction"].median()
                ),
                "convection_fraction_max": float(
                    audit_table["thermal_resistance_convection_fraction"].max()
                ),
                "pearson_log_resistance_vs_log_temperature_rise": correlation,
                "interpretation": (
                    "conditional temperature-rise differences are dominated by electrolyte "
                    "resistance under the selected boundary condition; thermal-conductivity "
                    "differences provide a secondary contribution"
                ),
            }
        ]
    )
    population.to_csv(
        DATA_DIR / "reference_cell_heat_resistance_population_summary.csv", index=False
    )
    return audit_table, summary, population


def build_extreme_property_audit(final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit extreme shortlist predictions against descriptor-nearest training liquids."""

    features = pd.read_csv(DATA_DIR / "model_features.csv", low_memory=False)
    reference = pd.read_csv(
        DATA_DIR / "training_domain_descriptor_reference.csv", low_memory=False
    )
    descriptor_columns = [
        column
        for column in features.columns
        if column.startswith("global_desc_") or column.startswith("fg_desc_")
    ]
    if len(descriptor_columns) != 136:
        raise RuntimeError(f"Expected 136 descriptor columns, found {len(descriptor_columns)}")
    reference_matrix = reference[descriptor_columns].to_numpy(dtype=float)
    candidate_features = features[
        features["candidate_id"].astype(str).isin(final["candidate_id"].astype(str))
    ].copy()
    candidate_features = candidate_features.set_index("candidate_id").loc[
        final["candidate_id"].astype(str)
    ].reset_index()
    variance = np.var(reference_matrix, axis=0)
    keep = np.isfinite(variance) & (variance > 1.0e-12)
    scaler = StandardScaler().fit(reference_matrix[:, keep])
    reference_scaled = scaler.transform(reference_matrix[:, keep])
    candidate_scaled = scaler.transform(
        candidate_features[descriptor_columns].to_numpy(dtype=float)[:, keep]
    )
    neighbors = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(reference_scaled)
    neighbor_distances, neighbor_indices = neighbors.kneighbors(candidate_scaled)

    benchmark = pd.read_csv(
        PROJECT_ROOT
        / "il_property_prediction"
        / "data"
        / "processed"
        / "il_multiprop_clean.csv",
        low_memory=False,
    )
    split_path = (
        PROJECT_ROOT
        / "il_property_prediction"
        / "data"
        / "processed"
        / "splits"
        / "il_level_seed42.json"
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    training = benchmark.iloc[split["train"]].copy()
    training = training[
        training["Temperature_K"].between(MAIN_T_MIN, MAIN_T_MAX, inclusive="both")
        & np.isclose(training["Pressure_kPa"], 101.325, rtol=0.0, atol=0.5)
    ].copy()
    viscosity_values = pd.to_numeric(
        training["Viscosity_ActualValue"], errors="coerce"
    ).dropna()
    conductivity_values = pd.to_numeric(
        training["ElectricalConductivity_ActualValue"], errors="coerce"
    ).dropna()
    if viscosity_values.empty or conductivity_values.empty:
        raise RuntimeError("Training observations are unavailable for extreme-property auditing")
    viscosity_bounds = np.quantile(viscosity_values, [0.005, 0.995])
    conductivity_bounds = np.quantile(conductivity_values, [0.005, 0.995])

    unit_audit = json.loads((AUDIT_DIR / "unit_audit.json").read_text(encoding="utf-8"))
    inference_audit = json.loads(
        (AUDIT_DIR / "inference_pipeline.json").read_text(encoding="utf-8")
    )
    inverse_expression = str(inference_audit.get("target_inverse_transform", ""))
    expected_inverse_expression = "exp(y_scaled * target_std + target_mean) - 1e-8"
    inverse_passed = inverse_expression == expected_inverse_expression
    unit_passed = bool(unit_audit.get("passed", False))
    final_lookup = final.set_index("candidate_id")
    neighbor_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for row_index, candidate_id in enumerate(candidate_features["candidate_id"].astype(str)):
        candidate = final_lookup.loc[candidate_id]
        candidate_neighbor_rows: list[pd.DataFrame] = []
        for rank, (distance, reference_index) in enumerate(
            zip(neighbor_distances[row_index], neighbor_indices[row_index]), start=1
        ):
            reference_id = str(reference.iloc[int(reference_index)]["reference_id"])
            observed = training[training["IL_SMILES"].astype(str).eq(reference_id)].copy()
            candidate_neighbor_rows.append(observed)
            viscosity = pd.to_numeric(observed["Viscosity_ActualValue"], errors="coerce").dropna()
            conductivity = pd.to_numeric(
                observed["ElectricalConductivity_ActualValue"], errors="coerce"
            ).dropna()
            neighbor_rows.append(
                {
                    "candidate_id": candidate_id,
                    "neighbor_rank": rank,
                    "training_reference_id": reference_id,
                    "training_IL_name": (
                        str(observed["IL_Name"].dropna().iloc[0])
                        if not observed["IL_Name"].dropna().empty
                        else "not available in filtered rows"
                    ),
                    "descriptor_distance": float(distance),
                    "observed_viscosity_min_Pa_s": float(viscosity.min())
                    if not viscosity.empty
                    else np.nan,
                    "observed_viscosity_max_Pa_s": float(viscosity.max())
                    if not viscosity.empty
                    else np.nan,
                    "observed_conductivity_min_S_m": float(conductivity.min())
                    if not conductivity.empty
                    else np.nan,
                    "observed_conductivity_max_S_m": float(conductivity.max())
                    if not conductivity.empty
                    else np.nan,
                    "observed_rows_at_ambient_pressure_in_main_window": int(len(observed)),
                }
            )
        observed_neighbors = pd.concat(candidate_neighbor_rows, ignore_index=True)
        neighbor_viscosity = pd.to_numeric(
            observed_neighbors["Viscosity_ActualValue"], errors="coerce"
        ).dropna()
        neighbor_conductivity = pd.to_numeric(
            observed_neighbors["ElectricalConductivity_ActualValue"], errors="coerce"
        ).dropna()
        predicted_viscosity = float(candidate["viscosity_worst"])
        predicted_conductivity = float(candidate["conductivity_worst"])
        viscosity_outside = bool(
            predicted_viscosity < viscosity_bounds[0]
            or predicted_viscosity > viscosity_bounds[1]
        )
        conductivity_outside = bool(
            predicted_conductivity < conductivity_bounds[0]
            or predicted_conductivity > conductivity_bounds[1]
        )
        extrapolative_flag = viscosity_outside or conductivity_outside
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "AD_status": candidate["AD_status"],
                "descriptor_knn_distance": candidate.get("descriptor_knn_distance", np.nan),
                "cation_support_count": int(candidate["cation_support_count"]),
                "anion_support_count": int(candidate["anion_support_count"]),
                "observed_neighbor_viscosity_min_Pa_s": float(neighbor_viscosity.min())
                if not neighbor_viscosity.empty
                else np.nan,
                "observed_neighbor_viscosity_max_Pa_s": float(neighbor_viscosity.max())
                if not neighbor_viscosity.empty
                else np.nan,
                "observed_neighbor_conductivity_min_S_m": float(neighbor_conductivity.min())
                if not neighbor_conductivity.empty
                else np.nan,
                "observed_neighbor_conductivity_max_S_m": float(neighbor_conductivity.max())
                if not neighbor_conductivity.empty
                else np.nan,
                "candidate_viscosity_worst_Pa_s": predicted_viscosity,
                "candidate_conductivity_worst_S_m": predicted_conductivity,
                "candidate_viscosity_training_observation_percentile": float(
                    100.0 * np.mean(viscosity_values.to_numpy(dtype=float) <= predicted_viscosity)
                ),
                "candidate_conductivity_training_observation_percentile": float(
                    100.0
                    * np.mean(
                        conductivity_values.to_numpy(dtype=float) <= predicted_conductivity
                    )
                ),
                "training_viscosity_q005_Pa_s": float(viscosity_bounds[0]),
                "training_viscosity_q995_Pa_s": float(viscosity_bounds[1]),
                "training_conductivity_q005_S_m": float(conductivity_bounds[0]),
                "training_conductivity_q995_S_m": float(conductivity_bounds[1]),
                "viscosity_outside_training_q005_q995": viscosity_outside,
                "conductivity_outside_training_q005_q995": conductivity_outside,
                "extrapolative_property_flag": extrapolative_flag,
                "unit_audit_passed": unit_passed,
                "inverse_transform_audit_passed": inverse_passed,
                "inverse_transform_expression": inverse_expression,
                "qualification_measurement_priority": (
                    "high-value measurement target"
                    if extrapolative_flag
                    else "qualification measurement target"
                ),
                "comparison_population": (
                    "primary-training observations at 101.325+/-0.5 kPa and 298.15--353.15 K"
                ),
            }
        )
    neighbors_table = pd.DataFrame(neighbor_rows)
    summary = pd.DataFrame(summary_rows)
    neighbors_table.to_csv(DATA_DIR / "extreme_property_nearest_neighbors.csv", index=False)
    summary.to_csv(DATA_DIR / "extreme_property_audit.csv", index=False)
    summary.to_csv(TABLE_DIR / "extreme_property_audit.csv", index=False)
    table_columns = [
        "candidate_id",
        "AD_status",
        "descriptor_knn_distance",
        "candidate_viscosity_worst_Pa_s",
        "candidate_viscosity_training_observation_percentile",
        "candidate_conductivity_worst_S_m",
        "candidate_conductivity_training_observation_percentile",
        "extrapolative_property_flag",
        "unit_audit_passed",
        "inverse_transform_audit_passed",
    ]
    (TABLE_DIR / "extreme_property_audit.tex").write_text(
        summary[table_columns].to_latex(
            index=False, escape=True, float_format=lambda value: f"{value:.4g}"
        ),
        encoding="utf-8",
    )
    return neighbors_table, summary


def attach_extreme_audit_to_priorities(
    priorities: pd.DataFrame, extreme: pd.DataFrame
) -> pd.DataFrame:
    """Attach measurement-target flags without altering any qualification-role decision."""

    audit_columns = [
        "candidate_id",
        "extrapolative_property_flag",
        "qualification_measurement_priority",
    ]
    updated = priorities.merge(extreme[audit_columns], on="candidate_id", how="left")
    updated.to_csv(DATA_DIR / "downstream_qualification_priorities.csv", index=False)
    return updated


def _draw_flow_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    color: str,
    width: float = 0.25,
    height: float = 0.21,
    fontsize: float = 5.3,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=color,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=COLORS["ink"], linespacing=0.92)


ION_PAIR_DISPLAY_NAMES = {
    ("CN1C=[N+](C=C1)C", "CCC(=O)[O-]"): "1,3-dimethylimidazolium propionate",
    ("CC[N+](C)(C)CCOC", "N#C[B-](C#N)(C#N)C#N"): "ethyl(2-methoxyethyl)dimethylammonium tetracyanoborate",
    ("CCC[n+]1ccn(C)c1", "O=C([O-])C(F)(F)F"): "1-propyl-3-methylimidazolium trifluoroacetate",
}


def _draw_ion_pair(
    ax: plt.Axes,
    cation: str,
    anion: str,
    candidate_id: str,
    role: str,
    ad_status: str,
    protocol_status: str,
) -> None:
    molecules = [Chem.MolFromSmiles(cation), Chem.MolFromSmiles(anion)]
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=2,
        subImgSize=(180, 112),
        legends=["cation", "anion"],
        useSVG=False,
    )
    ax.axis("off")
    image_ax = ax.inset_axes([0.00, 0.02, 0.58, 0.82])
    image_ax.imshow(image)
    image_ax.axis("off")
    name = ION_PAIR_DISPLAY_NAMES.get((cation, anion), "identity-audited ion pair")
    ax.text(0.00, 0.98, candidate_id, transform=ax.transAxes, ha="left", va="top", fontsize=5.6, fontweight="bold")
    ax.text(0.60, 0.92, role, transform=ax.transAxes, ha="left", va="top", fontsize=4.7, fontweight="bold", color=COLORS["red"], wrap=True)
    ax.text(0.60, 0.64, name, transform=ax.transAxes, ha="left", va="top", fontsize=4.2, color=COLORS["ink"], wrap=True)
    ax.text(0.60, 0.34, f"AD: {ad_status}", transform=ax.transAxes, ha="left", va="top", fontsize=4.1, color=COLORS["gray"])
    ax.text(0.60, 0.19, protocol_status, transform=ax.transAxes, ha="left", va="top", fontsize=3.8, color=COLORS["gray"], linespacing=0.9)


def make_figure5(
    identity_counts: dict[str, int],
    bootstrap_candidates: pd.DataFrame,
    final: pd.DataFrame,
    margins: pd.DataFrame,
    protocol_matrix: pd.DataFrame,
    priorities: pd.DataFrame,
) -> list[Path]:
    """Create Figure 5 with the requested a--f application-only panels."""

    configure_style()
    fig = plt.figure(figsize=(7.25, 3.75))
    gs = fig.add_gridspec(
        2,
        5,
        height_ratios=[0.65, 2.05],
        width_ratios=[1.15, 1.00, 1.00, 1.05, 2.05],
        hspace=0.42,
        wspace=0.58,
        left=0.045,
        right=0.992,
        top=0.965,
        bottom=0.085,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b, ax_c, ax_d, ax_e, ax_f = [fig.add_subplot(gs[1, i]) for i in range(5)]

    # a | identity-controlled candidate-space funnel.
    panel_title_wide(ax_a, "a", "Identity-controlled candidate-space funnel")
    generation = json.loads(
        (PRIMARY_ROOT / "steps" / "candidate_generation.json").read_text(encoding="utf-8")
    )
    screening = json.loads((PRIMARY_ROOT / "steps" / "screening.json").read_text(encoding="utf-8"))
    pareto = json.loads((PRIMARY_ROOT / "steps" / "pareto.json").read_text(encoding="utf-8"))
    stages = [
        ("Combinatorial\npairs", int(generation["theoretical_combinations"])),
        ("InChI identity-\nnew pool", int(generation["valid_unseen_pool"])),
        ("Evaluated by\nprimary model", int(generation["unseen_candidates"])),
        ("Hard-feasible", int(screening["hard_constraint_pass"])),
        ("Pareto rank 1", int(pareto["pareto_rank_1"])),
        ("Formal\nshortlist", int(pareto["final_recommendations"])),
    ]
    palette = [COLORS["navy"], COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["orange"], COLORS["red"]]
    width_weights = np.array([1.05, 1.20, 1.25, 0.95, 0.95, 1.05], dtype=float)
    left, right, gap = 0.015, 0.985, 0.018
    stage_widths = width_weights / width_weights.sum() * (right - left - gap * (len(stages) - 1))
    x_left = left
    box_y, box_height = 0.34, 0.36
    for index, ((label, value), width, color) in enumerate(zip(stages, stage_widths, palette)):
        ax_a.add_patch(
            FancyBboxPatch(
                (x_left, box_y),
                width,
                box_height,
                boxstyle="round,pad=0.004,rounding_size=0.010",
                facecolor=color,
                edgecolor="white",
                linewidth=0.75,
            )
        )
        center = x_left + width / 2
        ax_a.text(center, box_y + 0.235, f"{value}", ha="center", va="center", fontsize=6.6, color="white", fontweight="bold")
        ax_a.text(center, box_y + 0.115, label, ha="center", va="center", fontsize=5.25, color="white", fontweight="bold", linespacing=0.92)
        if index < len(stages) - 1:
            ax_a.add_patch(
                FancyArrowPatch(
                    (x_left + width + 0.0025, box_y + box_height / 2),
                    (x_left + width + gap - 0.0025, box_y + box_height / 2),
                    arrowstyle="-|>",
                    mutation_scale=6.5,
                    linewidth=0.75,
                    color=COLORS["gray"],
                )
            )
        x_left += width + gap
    ax_a.text(0.50, 0.17, "Selection window: 298.15–353.15 K  |  primary model: random-IL checkpoint", transform=ax_a.transAxes, ha="center", va="center", fontsize=5.8, color=COLORS["gray"])
    ax_a.text(0.985, 0.02, "Formal shortlist fixed  →  post hoc reference-cell mapping", transform=ax_a.transAxes, ha="right", va="bottom", fontsize=5.5, color=COLORS["red"], fontweight="bold")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")

    # b | two independent identity flows; deliberately no shared quantitative axis.
    panel_title_compact(ax_b, "b", "Identity audit")
    ax_b.axvline(0.50, color=COLORS["light"], lw=0.8)
    ax_b.text(0.25, 0.965, "old SMILES", ha="center", va="top", fontsize=4.8, color=COLORS["gray"])
    ax_b.text(0.75, 0.965, "current InChI", ha="center", va="top", fontsize=4.8, color=COLORS["gray"])
    _draw_flow_box(ax_b, (0.25, 0.80), "SMILES-novel\n649", COLORS["light"], 0.38, 0.15, 5.0)
    _draw_flow_box(ax_b, (0.25, 0.58), "InChI\nre-audit", "#DDE7EE", 0.34, 0.14, 4.9)
    _draw_flow_box(ax_b, (0.25, 0.36), "genuine 522", "#D8EFE5", 0.38, 0.13, 4.8)
    _draw_flow_box(ax_b, (0.25, 0.17), "known identity 127", "#F4D8D2", 0.38, 0.13, 4.5)
    ax_b.annotate("", (0.25, 0.66), (0.25, 0.72), arrowprops=dict(arrowstyle="->", lw=0.7, color=COLORS["gray"]))
    ax_b.annotate("", (0.25, 0.45), (0.25, 0.50), arrowprops=dict(arrowstyle="->", lw=0.7, color=COLORS["gray"]))
    ax_b.annotate("", (0.25, 0.245), (0.25, 0.29), arrowprops=dict(arrowstyle="->", lw=0.7, color=COLORS["gray"]))
    _draw_flow_box(ax_b, (0.75, 0.80), "30 cations ×\n30 anions", "#DDE7EE", 0.38, 0.15, 4.9)
    _draw_flow_box(ax_b, (0.75, 0.58), "900 nominal\npairs", "#DDE7EE", 0.38, 0.14, 4.9)
    _draw_flow_box(ax_b, (0.75, 0.36), "charge-aware\nInChI exclusion", "#D8EFE5", 0.38, 0.15, 4.6)
    _draw_flow_box(ax_b, (0.75, 0.14), f"identity-new\n{identity_counts['new_valid_unseen_pool']}", "#CFE8F3", 0.38, 0.15, 4.9)
    for upper, lower in [(0.72, 0.66), (0.50, 0.44), (0.275, 0.22)]:
        ax_b.annotate("", (0.75, lower), (0.75, upper), arrowprops=dict(arrowstyle="->", lw=0.7, color=COLORS["gray"]))
    ax_b.text(0.50, 0.02, "independent identity audits", fontsize=4.8, color=COLORS["gray"], ha="center", va="bottom")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")

    # c | margins relative to the five quantitative hard thresholds.
    panel_title_compact(ax_c, "c", "Constraint margins")
    matrix = margins.pivot(index="candidate_id", columns="constraint", values="plot_log2_margin")
    matrix = matrix.reindex(final["candidate_id"].astype(str))
    columns = [r"$\sigma_{\min}$", r"$\eta_{\max}$", r"$C_{v,\min}$", r"$\alpha_{\min}$", "ST-envelope"]
    matrix = matrix[columns]
    vmax = max(1.0, min(4.0, float(np.nanquantile(np.abs(matrix.to_numpy()), 0.95))))
    cmap = LinearSegmentedColormap.from_list("margin", ["#C84A3A", "#FAFAFA", "#27866D"])
    ax_c.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax_c.set_xticks(range(len(columns)), [r"$\sigma$", r"$\eta$", r"$C_v$", r"$\alpha$", "ST\nenvelope"], rotation=0)
    ax_c.set_yticks(range(len(matrix)), [value.replace("UPR-", "") for value in matrix.index])
    ax_c.set_ylabel("UPR candidate")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            raw = margins[
                margins["candidate_id"].eq(matrix.index[i])
                & margins["constraint"].eq(columns[j])
            ]["log2_margin"].iloc[0]
            label = r"$\geq4$" if not np.isfinite(raw) or raw >= 4.0 else f"{raw:.1f}"
            ax_c.text(j, i, label, ha="center", va="center", fontsize=4.5, color=COLORS["ink"])
    ax_c.text(0.5, -0.19, r"log$_2$ constraint margin", transform=ax_c.transAxes, ha="center", fontsize=5.4)
    ax_c.tick_params(length=0)
    for spine in ax_c.spines.values():
        spine.set_visible(False)

    # d | observed-reference bootstrap selection frequency.
    panel_title_compact(ax_d, "d", "Bootstrap stability")
    shown = bootstrap_candidates[
        bootstrap_candidates["nominal_pareto_rank_1"]
        | bootstrap_candidates["nominal_formal_shortlist"]
    ].sort_values(["selection_frequency", "candidate_id"], ascending=[True, True])
    colors = np.where(shown["nominal_formal_shortlist"], COLORS["blue"], COLORS["gray"])
    ax_d.barh(
        [value.replace("UPR-", "") for value in shown["candidate_id"].astype(str)],
        shown["selection_frequency"],
        color=colors,
        height=0.68,
    )
    ax_d.axvline(0.5, color=COLORS["gray"], lw=0.6, ls="--")
    ax_d.set_xlabel("Top-8 selection frequency")
    ax_d.set_xlim(0, 1.02)
    ax_d.grid(axis="x", alpha=0.16, lw=0.5)

    # e | categorical checkpoint decisions, not uncertainty probabilities.
    panel_title_compact(ax_e, "e", "Protocol sensitivity")
    decision = protocol_matrix.pivot(index="candidate_id", columns="protocol", values="decision_code")
    decision = decision.reindex(index=final["candidate_id"].astype(str), columns=PROTOCOL_ORDER)
    labels = protocol_matrix.pivot(index="candidate_id", columns="protocol", values="decision_label").reindex(index=decision.index, columns=decision.columns)
    cmap_decision = mpl.colors.ListedColormap(
        ["#7A6F76", "#B2182B", "#D9DDE0", "#8FC7B5", "#D97732"]
    )
    ax_e.imshow(
        decision.to_numpy(),
        aspect="auto",
        cmap=cmap_decision,
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap_decision.N),
    )
    ax_e.set_xticks(range(3), ["Primary", "Balanced", "Family"], rotation=35, ha="right")
    ax_e.set_yticks(range(len(decision)), [value.replace("UPR-", "") for value in decision.index])
    for i in range(decision.shape[0]):
        for j in range(decision.shape[1]):
            ax_e.text(j, i, str(labels.iloc[i,j]), ha="center", va="center", fontsize=3.7, linespacing=0.85)
    ax_e.text(
        0.98,
        -0.20,
        "Family = stricter family-transfer stress test",
        transform=ax_e.transAxes,
        ha="right",
        va="top",
        fontsize=4.5,
        color=COLORS["gray"],
    )
    ax_e.tick_params(length=0)
    for spine in ax_e.spines.values():
        spine.set_visible(False)

    # f | four structures and their qualification roles.
    panel_title_compact(ax_f, "f", "Qualification roles")
    ax_f.axis("off")
    status_lookup = protocol_matrix.pivot(
        index="candidate_id", columns="protocol", values="decision_label"
    )
    for index, row in enumerate(priorities.sort_values("priority_order").itertuples(index=False)):
        inset = ax_f.inset_axes([0.00, 0.75 - index * 0.25, 1.00, 0.238])
        candidate_status = status_lookup.loc[str(row.candidate_id)]
        protocol_status = (
            f"P/B/F: {str(candidate_status.get('Random-IL primary', 'n/a')).replace(chr(10), ' ')} / "
            f"{str(candidate_status.get('Balanced-IL sensitivity', 'n/a')).replace(chr(10), ' ')} / "
            f"{str(candidate_status.get('Ion-family sensitivity', 'n/a')).replace(chr(10), ' ')}"
        )
        _draw_ion_pair(
            inset,
            str(row.cation_smiles),
            str(row.anion_smiles),
            str(row.candidate_id),
            str(row.qualification_role),
            str(row.AD_status),
            protocol_status,
        )

    base = FIGURE_DIR / "figure5_auditable_virtual_screening_validation"
    paths = [
        base.with_suffix(".pdf"),
        base.with_suffix(".png"),
        base.with_suffix(".svg"),
    ]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=600)
    fig.savefig(paths[2])
    plt.close(fig)
    return paths


def _cell_schematic(ax: plt.Axes, panel_label: str = "a") -> None:
    panel_title_wide(ax, panel_label, "60-s constant-current thermal stress scenario")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.25,0.22), 4.6,1.5, boxstyle="round,pad=0.06", facecolor="#F4F6F7", edgecolor=COLORS["navy"], linewidth=0.8))
    ax.add_patch(Rectangle((0.55,0.42),0.72,1.08,facecolor="#263746",edgecolor="none"))
    ax.add_patch(Rectangle((3.82,0.42),0.72,1.08,facecolor="#263746",edgecolor="none"))
    ax.add_patch(Rectangle((1.28,0.42),0.32,1.08,facecolor="#88B7C9",edgecolor="none"))
    ax.add_patch(Rectangle((3.50,0.42),0.32,1.08,facecolor="#88B7C9",edgecolor="none"))
    ax.add_patch(Rectangle((2.34,0.34),0.42,1.24,facecolor="#E9C46A",edgecolor="white",linewidth=0.6,hatch="////"))
    ax.add_patch(Rectangle((1.60,0.42),0.74,1.08,facecolor="#B8DDE8",alpha=0.75,edgecolor="none"))
    ax.add_patch(Rectangle((2.76,0.42),0.74,1.08,facecolor="#B8DDE8",alpha=0.75,edgecolor="none"))
    ax.text(2.55,0.98,"separator\n100 µm",ha="center",va="center",fontsize=5.8)
    ax.text(0.91,0.96,"porous\ncarbon",ha="center",va="center",fontsize=5.6,color="white")
    ax.text(4.18,0.96,"porous\ncarbon",ha="center",va="center",fontsize=5.6,color="white")
    ax.annotate("I = 2 A", xy=(3.8,1.92), xytext=(1.3,1.92), arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.0), ha="center", va="center", fontsize=6.2, color=COLORS["red"])
    ax.text(5.25,1.67,r"$A=100$ cm$^2$   $L=100$ μm   $V=1.0$ mL   $I=2$ A",fontsize=6.3,fontweight="bold",color=COLORS["ink"])
    ax.text(5.25,1.17,r"$h=10$ W m$^{-2}$ K$^{-1}$   $N_f=2$   $t=60$ s",fontsize=6.2,color=COLORS["ink"])
    ax.text(5.25,0.67,"Formal shortlist fixed → post hoc electrolyte-path interpretation.",fontsize=5.9,color=COLORS["gray"])
    ax.text(5.25,0.28,"No capacitance, safety, failure, or device-performance prediction.",fontsize=5.9,color=COLORS["red"])


def _curve_with_stress(ax: plt.Axes, frame: pd.DataFrame, column: str, priorities: pd.DataFrame, ylabel: str, logy: bool = False) -> None:
    priority_ids = priorities.sort_values("priority_order").drop_duplicates("candidate_id")["candidate_id"].astype(str).tolist()
    final_ids = set(pd.read_csv(DATA_DIR / "final_prioritized_candidates.csv")["candidate_id"].astype(str))
    for candidate_id in final_ids - set(priority_ids):
        group = frame[frame["candidate_id"].eq(candidate_id)].sort_values("temperature_K")
        main = group[group["analysis_window"].eq("main")]
        ax.plot(main["temperature_K"], main[column], color="#C9D0D5", lw=0.7, alpha=0.9, zorder=1)
    for color, candidate_id in zip(LEAD_COLORS, priority_ids):
        group = frame[frame["candidate_id"].eq(candidate_id)].sort_values("temperature_K")
        main = group[group["analysis_window"].eq("main")]
        stress = group[group["temperature_K"].isin(STRESS_ENDPOINTS)]
        ax.plot(main["temperature_K"], main[column], color=color, lw=1.25, label=candidate_id, zorder=3)
        ax.scatter(stress["temperature_K"], stress[column], s=18, facecolor="white", edgecolor=color, linewidth=0.8, zorder=4)
    ax.axvspan(STRESS_ENDPOINTS[0], MAIN_T_MIN, color="#F2F4F5", zorder=0)
    ax.axvspan(MAIN_T_MAX, STRESS_ENDPOINTS[1], color="#F2F4F5", zorder=0)
    ax.axvline(MAIN_T_MIN, color=COLORS["gray"], lw=0.55, ls="--")
    ax.axvline(MAIN_T_MAX, color=COLORS["gray"], lw=0.55, ls="--")
    ax.set_xlim(STRESS_ENDPOINTS)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.14, lw=0.45)


def make_figure6(final: pd.DataFrame, priorities: pd.DataFrame) -> list[Path]:
    """Create Figure 6 as a strictly post hoc engineering interpretation."""

    configure_style()
    metrics = pd.read_csv(DATA_DIR / "reference_cell_metrics_temperature.csv", low_memory=False)
    robust = pd.read_csv(DATA_DIR / "candidate_robust_summary.csv", low_memory=False)
    summary = pd.read_csv(DATA_DIR / "reference_cell_candidate_summary.csv")
    trace = pd.read_csv(DATA_DIR / "screening_trace.csv", low_memory=False)
    thresholds = json.loads((DATA_DIR / "reference_thresholds.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(7.25, 3.15))
    gs = fig.add_gridspec(2,5,height_ratios=[0.72,1.55],width_ratios=[1.24,1.02,1.02,1.02,1.18],hspace=0.45,wspace=0.62,left=0.055,right=0.992,top=0.96,bottom=0.10)
    ax_a = fig.add_subplot(gs[0,:])
    axes = [fig.add_subplot(gs[1,i]) for i in range(5)]
    _cell_schematic(ax_a, "a")

    ax_b, ax_c, ax_d, ax_e, ax_f = axes
    panel_title_compact(ax_b, "b", "Resistance")
    _curve_with_stress(ax_b, metrics, "electrolyte_resistance_ohm", priorities, r"$R_{el}$ (Ω)", logy=True)
    ax_b.legend(frameon=False, loc="upper right", handlelength=1.0, fontsize=4.8)

    panel_title_compact(ax_c, "c", "Thermal-property map")
    unseen = trace[trace["candidate_type"].eq("unseen_pair_recombination")]
    ax_c.scatter(unseen["volumetric_heat_capacity_worst"] / 1e6, unseen["thermal_diffusivity_worst"] * 1e8, s=9, color="#D6DBDE", alpha=0.65, edgecolors="none")
    feasible = unseen[unseen["final_feasible"]]
    ax_c.scatter(feasible["volumetric_heat_capacity_worst"] / 1e6, feasible["thermal_diffusivity_worst"] * 1e8, s=13, color=COLORS["sky"], alpha=0.8, edgecolors="none")
    priority_lookup = priorities.drop_duplicates("candidate_id").set_index("candidate_id")
    for color, candidate_id in zip(LEAD_COLORS, priority_lookup.index):
        row = priority_lookup.loc[candidate_id]
        ax_c.scatter(float(row["volumetric_heat_capacity_worst"])/1e6, float(row["thermal_diffusivity_worst"])*1e8, s=30, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax_c.axvline(float(thresholds["volumetric_heat_capacity_min"])/1e6, color=COLORS["gray"], lw=0.6, ls="--")
    ax_c.axhline(float(thresholds["thermal_diffusivity_min"])*1e8, color=COLORS["gray"], lw=0.6, ls="--")
    ax_c.set_xlabel(r"$C_{v,\mathrm{worst}}$ (MJ m$^{-3}$ K$^{-1}$)")
    ax_c.set_ylabel(r"$\alpha_{\mathrm{worst}}$ ($10^{-8}$ m$^2$ s$^{-1}$)")
    ax_c.grid(alpha=0.14,lw=0.45)

    panel_title_compact(ax_d, "d", r"Conditional $\Delta T_{60}$")
    _curve_with_stress(ax_d, metrics, "transient_temperature_rise_K", priorities, r"$\Delta T_{60}$ (K)", logy=False)

    panel_title_compact(ax_e, "e", "Endpoint trade-off")
    final_metrics = metrics[metrics["candidate_id"].isin(final["candidate_id"].astype(str))]
    rows = []
    for candidate_id, group in final_metrics.groupby("candidate_id"):
        values = group.set_index("temperature_K")["electrolyte_resistance_ohm"]
        if all(any(np.isclose(values.index.astype(float), t)) for t in [278.15,298.15,373.15]):
            get = lambda t: float(values.iloc[np.argmin(np.abs(values.index.astype(float)-t))])
            rows.append({"candidate_id":candidate_id,"cold":get(278.15)/get(298.15),"hot":get(373.15)/get(298.15)})
    trade = pd.DataFrame(rows)
    for row in trade.itertuples(index=False):
        if row.candidate_id in set(priorities["candidate_id"]):
            unique_priority_ids = priorities.sort_values("priority_order").drop_duplicates("candidate_id")["candidate_id"].astype(str).tolist()
            color = LEAD_COLORS[unique_priority_ids.index(row.candidate_id)]
            size = 30
        else:
            color, size = COLORS["gray"], 18
        ax_e.scatter(row.cold,row.hot,s=size,color=color,edgecolor="white",linewidth=0.45)
        ax_e.text(row.cold,row.hot,row.candidate_id.replace("UPR-",""),fontsize=4.7,ha="left",va="bottom")
    ax_e.set_xlabel(r"$R_{278}/R_{298}$")
    ax_e.set_ylabel(r"$R_{373}/R_{298}$")
    ax_e.text(0.04,0.04,"extended endpoints only",transform=ax_e.transAxes,ha="left",va="bottom",fontsize=5.0,color=COLORS["red"])
    ax_e.grid(alpha=0.14,lw=0.45)

    panel_title_compact(ax_f, "f", "Exceedance context")
    population = summary[summary["candidate_type"].eq("unseen_pair_recombination")]
    ax_f.scatter(population["reference_cell_exceedance_index_worst_temperature_K"], population["reference_cell_exceedance_index_worst"], s=9, color="#D5DADD", alpha=0.6, edgecolors="none")
    selected_summary = population[population["candidate_id"].isin(final["candidate_id"].astype(str))]
    ax_f.scatter(selected_summary["reference_cell_exceedance_index_worst_temperature_K"], selected_summary["reference_cell_exceedance_index_worst"], s=25, color=COLORS["blue"], marker="D", edgecolor="white", linewidth=0.45)
    ax_f.axhline(1.0,color=COLORS["red"],lw=0.65,ls="--")
    ax_f.text(MAIN_T_MIN+1,1.02,r"$\Xi_{\max}=1$",fontsize=5.2,color=COLORS["red"],va="bottom")
    ax_f.set_xlim(MAIN_T_MIN,MAIN_T_MAX)
    ax_f.set_yscale("log")
    ax_f.set_xlabel(r"$T$ at $\Xi_{\max}$ (K)")
    ax_f.set_ylabel(r"$\Xi_{\max}$")
    ax_f.grid(alpha=0.14,lw=0.45)

    base = FIGURE_DIR / "figure6_reference_cell_scenario_audited"
    paths = [
        base.with_suffix(".pdf"),
        base.with_suffix(".png"),
        base.with_suffix(".svg"),
    ]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=600)
    fig.savefig(paths[2])
    plt.close(fig)
    trade.to_csv(DATA_DIR / "extended_endpoint_transport_tradeoff.csv", index=False)
    return paths


def make_si_figures(
    priorities: pd.DataFrame,
    sensitivity: pd.DataFrame,
    bootstrap_iterations: pd.DataFrame,
    bootstrap_candidates: pd.DataFrame,
) -> list[Path]:
    configure_style()
    metrics = pd.read_csv(DATA_DIR / "reference_cell_metrics_temperature.csv", low_memory=False)
    priority_ids = priorities.sort_values("priority_order").drop_duplicates("candidate_id")["candidate_id"].astype(str).tolist()
    fig, axes = plt.subplots(1,2,figsize=(7.15,2.30),constrained_layout=True)
    specs = [
        ("joule_heating_power_W", r"$\dot Q_J=I^2R_{el}$ (W)", "a", "Constant-current Joule term"),
        ("steady_state_temperature_rise_K", r"$\Delta T_{ss}=\dot Q_JR_{th}$ (K)", "b", "Conditional steady rise"),
    ]
    for ax, (column,ylabel,label,title) in zip(axes,specs):
        panel_title(ax,label,title)
        for color,candidate_id in zip(LEAD_COLORS,priority_ids):
            group = metrics[metrics["candidate_id"].eq(candidate_id)].sort_values("temperature_K")
            main = group[group["analysis_window"].eq("main")]
            stress = group[group["temperature_K"].isin(STRESS_ENDPOINTS)]
            ax.plot(main["temperature_K"],main[column],color=color,lw=1.2,label=candidate_id,zorder=2)
            ax.scatter(stress["temperature_K"],stress[column],s=17,facecolor="white",edgecolor=color,linewidth=0.7,zorder=4)
        ax.axvspan(278.15,298.15,color="#F2F4F5",zorder=0)
        ax.axvspan(353.15,373.15,color="#F2F4F5",zorder=0)
        ax.set_xlim(276.0,375.0)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.14,lw=0.45)
    axes[0].legend(frameon=False,fontsize=5.2)
    base = FIGURE_DIR / "figureS_application_derived_metrics"
    paths = [
        base.with_suffix(".pdf"),
        base.with_suffix(".png"),
        base.with_suffix(".svg"),
    ]
    fig.savefig(paths[0])
    fig.savefig(paths[1],dpi=600)
    fig.savefig(paths[2])
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), constrained_layout=True)
    panel_title(axes[0], "a", "Prespecified threshold grid")
    nominal = sensitivity[sensitivity["is_nominal"]].iloc[0]
    scatter = axes[0].scatter(
        sensitivity["hard_feasible_count"],
        sensitivity["final_set_Jaccard_to_nominal"],
        c=sensitivity["pareto_rank_1_count"],
        cmap="viridis",
        s=18,
        alpha=0.75,
        edgecolors="none",
    )
    axes[0].scatter(
        nominal["hard_feasible_count"], nominal["final_set_Jaccard_to_nominal"],
        marker="*", s=85, color=COLORS["red"], edgecolor="white", linewidth=0.5,
    )
    axes[0].set_xlabel("Hard-feasible count")
    axes[0].set_ylabel("Top-8 Jaccard to nominal")
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].text(0.98, 0.04, "colour: Pareto-rank-1 count", transform=axes[0].transAxes, ha="right", fontsize=5.2, color=COLORS["gray"])

    panel_title(axes[1], "b", "Bootstrap Top-8 overlap")
    axes[1].hist(
        bootstrap_iterations["top8_jaccard_to_nominal"], bins=np.linspace(0, 1, 17),
        color=COLORS["blue"], edgecolor="white", linewidth=0.5,
    )
    median = float(bootstrap_iterations["top8_jaccard_to_nominal"].median())
    lower, upper = bootstrap_iterations["top8_jaccard_to_nominal"].quantile([0.025, 0.975])
    axes[1].axvline(median, color=COLORS["red"], lw=0.8, ls="--")
    axes[1].text(0.03, 0.95, f"median={median:.2f}\n95% interval={lower:.2f}–{upper:.2f}", transform=axes[1].transAxes, va="top", fontsize=5.6)
    axes[1].set_xlabel("Jaccard to nominal Top-8")
    axes[1].set_ylabel("Bootstrap replicates")

    panel_title(axes[2], "c", "Candidate selection frequency")
    shown = bootstrap_candidates.head(20).sort_values("selection_frequency")
    axes[2].barh(
        [value.replace("UPR-", "") for value in shown["candidate_id"].astype(str)],
        shown["selection_frequency"],
        color=np.where(shown["nominal_formal_shortlist"], COLORS["blue"], COLORS["gray"]),
        height=0.65,
    )
    axes[2].set_xlabel("Top-8 selection frequency")
    axes[2].set_xlim(0, 1.02)
    for ax in axes:
        ax.grid(alpha=0.14, lw=0.45)
    stability_base = FIGURE_DIR / "figureS_application_decision_stability"
    stability_paths = [
        stability_base.with_suffix(".pdf"),
        stability_base.with_suffix(".png"),
        stability_base.with_suffix(".svg"),
    ]
    fig.savefig(stability_paths[0])
    fig.savefig(stability_paths[1], dpi=600)
    fig.savefig(stability_paths[2])
    plt.close(fig)
    paths += stability_paths

    for path in paths:
        shutil.copy2(path,PAPER_FIG_DIR/path.name)
    return paths


def copy_source_data(paths: list[Path]) -> None:
    PAPER_SOURCE_DIR.mkdir(parents=True,exist_ok=True)
    legacy_paths = [PAPER_SOURCE_DIR / name for name in LEGACY_APPLICATION_SOURCE_NAMES]
    legacy_paths.extend(PAPER_SOURCE_DIR.glob("figure_application_case*.csv"))
    for legacy_path in legacy_paths:
        if legacy_path.exists():
            legacy_path.unlink()

    copied: list[Path] = []
    for path in paths:
        if path.exists():
            destination = PAPER_SOURCE_DIR / path.name
            shutil.copy2(path,destination)
            copied.append(destination)

    remaining_legacy = [
        str(path) for path in legacy_paths if path.exists()
    ]
    if remaining_legacy:
        raise RuntimeError(
            "Legacy application source data remain in the submission bundle: "
            + ", ".join(remaining_legacy)
        )

    manifest = {
        "schema_version": 1,
        "scope": "split application Figures 5 and 6 plus application-case Supporting Information",
        "formal_prediction_source": "single primary random-IL checkpoint",
        "cross_protocol_prediction_aggregation": "none",
        "source_root": str(PRIMARY_ROOT),
        "files": [
            {
                "name": path.name,
                "sha256": protocol_inputs.file_sha256(path),
            }
            for path in sorted(copied, key=lambda item: item.name)
        ],
    }
    (PAPER_SOURCE_DIR / "application_source_data_manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8"
    )


def main() -> int:
    for directory in [DATA_DIR,AUDIT_DIR,FIGURE_DIR,TABLE_DIR,PAPER_FIG_DIR,PAPER_SOURCE_DIR]:
        directory.mkdir(parents=True,exist_ok=True)
    # Redirect the reusable audit module to the primary-output namespace.
    audit.OUTPUT_ROOT=PRIMARY_ROOT
    audit.DATA_DIR=DATA_DIR
    audit.AUDIT_DIR=AUDIT_DIR
    audit.FIGURE_DIR=FIGURE_DIR
    audit.TABLE_DIR=TABLE_DIR
    _,_,identity_counts=audit.build_identity_audit()
    sensitivity,stability=audit.build_threshold_sensitivity()
    final=pd.read_csv(DATA_DIR/"final_prioritized_candidates.csv",low_memory=False)
    full_trace = pd.read_csv(DATA_DIR / "screening_trace.csv", low_memory=False)
    full_trace[full_trace["candidate_type"].eq("unseen_pair_recombination")].to_csv(
        DATA_DIR / "candidate_screening_trajectory_608.csv", index=False
    )
    protocol_matrix,protocol_summary=build_cross_protocol_decisions(final)
    priorities=build_downstream_priorities(final,protocol_matrix)
    margins=constraint_margins(final)
    bootstrap_iterations,bootstrap_candidates,bootstrap_summary=build_reference_bootstrap_stability()
    identity_audit,reference_audit,rank_one=build_identity_and_reference_audits()
    heat_audit,heat_summary,heat_population=build_reference_cell_heat_resistance_audit(final)
    extreme_neighbors,extreme_summary=build_extreme_property_audit(final)
    priorities=attach_extreme_audit_to_priorities(priorities,extreme_summary)
    primary_library=pd.read_csv(DATA_DIR/"candidate_library.csv",low_memory=False)
    identity_digest=protocol_inputs.identity_sha256(primary_library)
    protocol_provenance=[]
    for protocol,output_root,config_path,formal_primary in PROTOCOL_SPECS:
        protocol_config=load_case_config(config_path)
        checkpoint=resolve_project_path(PROJECT_ROOT,protocol_config["model"]["checkpoint_path"])
        provenance={
            "protocol":protocol,
            "formal_primary_protocol":formal_primary,
            "checkpoint_path":str(checkpoint.resolve()),
            "checkpoint_sha256":protocol_inputs.file_sha256(checkpoint),
            "training_split_path":str(resolve_project_path(PROJECT_ROOT,protocol_config["data"]["split_path"]).resolve()),
            "candidate_identity_sha256":identity_digest,
            "prediction_aggregation":"none; single checkpoint",
        }
        manifest_path=output_root/"audit"/"protocol_stability_manifest.json"
        if manifest_path.exists():
            provenance["manifest_path"]=str(manifest_path.resolve())
        protocol_provenance.append(provenance)
    figure5=make_figure5(identity_counts,bootstrap_candidates,final,margins,protocol_matrix,priorities)
    figure6=make_figure6(final,priorities)
    for path in figure5 + figure6:
        shutil.copy2(path, PAPER_FIG_DIR / path.name)
    for suffix in [".pdf", ".png", ".svg"]:
        obsolete = PAPER_FIG_DIR / f"figure5_computational_application_case_combined{suffix}"
        if obsolete.exists():
            obsolete.unlink()
    si_figures=make_si_figures(priorities,sensitivity,bootstrap_iterations,bootstrap_candidates)
    evidence={
        "formal_prediction_checkpoint":"random-IL whole-ion-holdout checkpoint",
        "formal_prediction_aggregation":"none; single checkpoint",
        "main_temperature_window_K":[MAIN_T_MIN,MAIN_T_MAX],
        "extended_stress_test_endpoints_K":list(STRESS_ENDPOINTS),
        "extended_endpoints_enter_screening":False,
        "pareto_objectives":{
            "maximize":["conductivity_worst","volumetric_heat_capacity_worst","thermal_diffusivity_worst"],
            "minimize":["viscosity_worst"],
        },
        "posthoc_only":["electrolyte_resistance_ohm","transient_temperature_rise_K","reference_cell_exceedance_index_worst"],
        "hard_constraints":["surface_tension_reference_envelope","applicability_domain","curve_quality","conductivity","viscosity","volumetric_heat_capacity","thermal_diffusivity"],
        "identity_counts":identity_counts,
        "protocol_summary":protocol_summary.to_dict(orient="records"),
        "protocol_provenance":protocol_provenance,
        "formal_final_candidates":final["candidate_id"].astype(str).tolist(),
        "figures":[str(path) for path in figure5+figure6+si_figures],
        "xi_definition":"max over 298.15--353.15 K of max(R_el/q75_reference_R_el, DeltaT_60/q75_reference_DeltaT_60); no weighted sum",
        "bootstrap_summary":bootstrap_summary.to_dict(orient="records"),
        "reference_selection_audit":{
            "reference_count":int(len(reference_audit)),
            "selection_rule":"highest combined cation-plus-anion support; canonical identity tie-break",
            "family_stratified":False,
            "AD_status_counts":reference_audit["AD_status"].value_counts().to_dict(),
            "complete_main_window_predictions":int(reference_audit["complete_main_window_prediction"].sum()),
            "benchmark_identity_present":int(reference_audit["benchmark_identity_present"].sum()),
            "fixed_before_candidate_ranking":True,
        },
        "heat_resistance_contribution_summary":heat_population.to_dict(orient="records"),
        "extreme_property_flagged_candidates":extreme_summary.loc[
            extreme_summary["extrapolative_property_flag"],"candidate_id"
        ].astype(str).tolist(),
        "claim_boundary":"identity-controlled prioritization of seen-ion/unseen-pair recombinations; no wide-temperature operation, liquid-range, safety, or device-performance claim",
    }
    (AUDIT_DIR/"refactored_application_evidence.json").write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding="utf-8")
    copy_source_data([
        DATA_DIR/"chemical_identity_audit_old_unseen_pool.csv",
        DATA_DIR/"chemical_identity_audit_old_shortlist.csv",
        DATA_DIR/"chemical_identity_audit_current_shortlist.csv",
        DATA_DIR/"standard_inchikey_identity_audit_608.csv",
        DATA_DIR/"observed_reference_selection_audit.csv",
        DATA_DIR/"threshold_sensitivity.csv",
        DATA_DIR/"candidate_selection_stability.csv",
        DATA_DIR/"reference_bootstrap_iterations.csv",
        DATA_DIR/"reference_bootstrap_candidate_selection.csv",
        DATA_DIR/"reference_bootstrap_summary.csv",
        DATA_DIR/"cross_protocol_decision_matrix.csv",
        DATA_DIR/"cross_protocol_summary.csv",
        DATA_DIR/"downstream_qualification_priorities.csv",
        DATA_DIR/"qualification_role_selection_audit.csv",
        DATA_DIR/"pareto_rank1_all_candidates.csv",
        DATA_DIR/"pareto_rank1_top8_selection.csv",
        DATA_DIR/"final_candidate_constraint_margins.csv",
        DATA_DIR/"candidate_screening_trajectory_608.csv",
        DATA_DIR/"extended_endpoint_transport_tradeoff.csv",
        DATA_DIR/"final_prioritized_candidates.csv",
        DATA_DIR/"reference_cell_candidate_summary.csv",
        DATA_DIR/"reference_cell_metrics_temperature.csv",
        DATA_DIR/"reference_cell_heat_resistance_contribution_audit.csv",
        DATA_DIR/"reference_cell_heat_resistance_contribution_summary.csv",
        DATA_DIR/"reference_cell_heat_resistance_population_summary.csv",
        DATA_DIR/"extreme_property_nearest_neighbors.csv",
        DATA_DIR/"extreme_property_audit.csv",
        AUDIT_DIR/"reference_cell_scenario.json",
        AUDIT_DIR/"final_output_audit.json",
        AUDIT_DIR/"refactored_application_evidence.json",
        AUDIT_DIR/"protected_manuscript_scope_hashes.json",
        TABLE_DIR/"application_figure_captions_bilingual.tex",
        CASE_DIR/"APPLICATION_CHAPTER_CHANGELOG.md",
    ])
    print(json.dumps(evidence,indent=2,ensure_ascii=False))
    print("Modified manuscript scope: application chapter only")
    print("Other manuscript sections modified: No")
    print("Method section modified: No")
    print("Application protocol placed at chapter opening: Yes")
    print("Formal candidate model: random-IL checkpoint")
    print("Secondary checkpoints averaged: No")
    print("Reference-cell mapping used for selection: No")
    print("Experimental validation claimed: No")
    print(
        "Other manuscript sections may require later consistency updates, "
        "but they were not modified in this task."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
