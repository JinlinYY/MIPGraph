"""Semantic field definitions for manuscript figure source-data catalogs."""

from __future__ import annotations

Definition = tuple[str, str]


COMMON: dict[str, Definition] = {
    "panel": ("Figure panel identifier.", "not applicable / categorical"),
    "panel_title": ("Display title of the figure panel.", "not applicable / text"),
    "property": ("Thermophysical property represented by the record.", "not applicable / categorical"),
    "property_short": ("Abbreviated thermophysical-property label.", "not applicable / categorical"),
    "value": ("Numeric value represented by the record.", "defined by the companion unit field"),
    "unit": ("Physical unit associated with the numeric value.", "not applicable / unit label"),
    "count": ("Number of records or identities in the named category.", "count"),
    "selected": ("Whether the record is selected by the stated rule.", "boolean"),
    "eligible": ("Whether the record satisfies the stated eligibility rule.", "boolean"),
    "criterion": ("Name of the evaluated decision criterion.", "not applicable / categorical"),
    "criterion_value": ("Numeric value of the evaluated decision criterion.", "criterion-dependent"),
    "metric": ("Name of the reported evaluation metric.", "not applicable / categorical"),
    "measure": ("Name of the reported measure.", "not applicable / categorical"),
    "source": ("Origin of the record or derived quantity.", "not applicable / categorical"),
    "provenance": ("Upstream table or rule supporting the record.", "not applicable / text"),
    "warnings": ("Semicolon-delimited warnings produced by the workflow.", "not applicable / text"),
    "row_position": ("One-based display position of the record.", "ordinal index"),
}


INTRO: dict[str, Definition] = {
    "content": ("Content type represented by the schematic panel.", "not applicable / categorical"),
    "quantitative_source_data": (
        "Whether the schematic panel contains plotted quantitative source data.",
        "boolean",
    ),
}


PERFORMANCE: dict[str, Definition] = {
    "sample_id": ("Stable identifier of the evaluated condition record.", "not applicable / identifier"),
    "IL_Name": ("Reported ionic-liquid name.", "not applicable / text"),
    "IL_SMILES": ("SMILES representation of the ionic-liquid ion pair.", "not applicable / molecular string"),
    "Temperature_K": ("Measurement temperature.", "K"),
    "Pressure_kPa": ("Measurement pressure.", "kPa"),
    "split": ("Full name of the evaluation split protocol.", "not applicable / categorical"),
    "split_short": ("Abbreviated evaluation-split label.", "not applicable / categorical"),
    "split_strategy": ("Rule used to construct the evaluation split.", "not applicable / categorical"),
    "y_true": ("Experimental target value.", "property-dependent; see property"),
    "y_pred": ("Model-predicted target value.", "property-dependent; see property"),
    "absolute_error": ("Absolute difference between prediction and experiment.", "property-dependent; see property"),
    "plot_included": ("Whether the point is included in the plotted panel.", "boolean"),
    "log_R2": ("Coefficient of determination evaluated in natural-log target space.", "dimensionless"),
    "log_MAE": ("Mean absolute error evaluated in natural-log target space.", "dimensionless log error"),
    "log_RMSE": ("Root-mean-square error evaluated in natural-log target space.", "dimensionless log error"),
    "log_NMAE": ("Range-normalized mean absolute error in natural-log target space.", "dimensionless"),
}


DATASET: dict[str, Definition] = {
    "dataset_row_index": ("Zero-based row index in the curated condition-record table.", "index"),
    "property_row_index": ("Zero-based row index within the named property subset.", "index"),
    "il_rank": ("Display rank of the ionic liquid in the coverage matrix.", "ordinal index"),
    "il_name": ("Reported ionic-liquid name.", "not applicable / text"),
    "il_smiles": ("SMILES representation of the ionic-liquid ion pair.", "not applicable / molecular string"),
    "n_labels": ("Number of available property labels for the ionic liquid.", "count"),
    "pct_il": ("Percentage of ionic liquids in the named label-count category.", "%"),
    "source_category": ("Observed or label-expansion source category.", "not applicable / categorical"),
    "raw_value": ("Curated property value before plotting transformation.", "property-dependent; see unit"),
    "transform": ("Transformation applied for cross-property display.", "not applicable / categorical"),
    "transformed_value": ("Property value after the stated plotting transformation.", "transformation-dependent"),
    "temperature_k": ("Measurement temperature.", "K"),
    "pressure_kpa": ("Measurement pressure.", "kPa"),
    "pressure_category": ("Ambient, non-ambient, or missing-pressure category.", "not applicable / categorical"),
    "pressure_min_kpa": ("Minimum reported pressure in the category.", "kPa"),
    "pressure_max_kpa": ("Maximum reported pressure in the category.", "kPa"),
    "ion_type": ("Cation or anion classification.", "not applicable / categorical"),
    "ion_family": ("Curated ion-family label.", "not applicable / categorical"),
    "bin_index": ("Zero-based temperature-bin index.", "index"),
    "bin_left": ("Lower temperature-bin boundary.", "K"),
    "bin_right": ("Upper temperature-bin boundary.", "K"),
    "bin_center": ("Temperature-bin midpoint.", "K"),
    "q01": ("First percentile of the named property.", "property-dependent; see unit"),
    "q99": ("Ninety-ninth percentile of the named property.", "property-dependent; see unit"),
    "figure": ("Figure identifier supported by the record.", "not applicable / identifier"),
}


INTERPRETABILITY: dict[str, Definition] = {
    "sample_id": ("Stable identifier of the evaluated condition record.", "not applicable / identifier"),
    "IL_Name": ("Reported ionic-liquid name.", "not applicable / text"),
    "IL_SMILES": ("SMILES representation of the ionic-liquid ion pair.", "not applicable / molecular string"),
    "Cation_Family": ("Curated cation-family label.", "not applicable / categorical"),
    "Anion_Family": ("Curated anion-family label.", "not applicable / categorical"),
    "Temperature_K": ("Measurement temperature.", "K"),
    "Pressure_kPa": ("Measurement pressure.", "kPa"),
    "checkpoint": ("Model checkpoint used to produce the result.", "not applicable / identifier"),
    "PC1": ("First principal-component score of the response representation.", "dimensionless standardized score"),
    "PC2": ("Second principal-component score of the response representation.", "dimensionless standardized score"),
    "domain": ("Cation, anion, or ion-pair structural domain.", "not applicable / categorical"),
    "group": ("Analysis group represented by the record.", "not applicable / categorical"),
    "motif": ("Curated molecular motif.", "not applicable / categorical"),
    "feature": ("Machine-readable descriptor name.", "not applicable / identifier"),
    "feature_pretty": ("Human-readable descriptor label.", "not applicable / text"),
    "feature_group": ("Descriptor-family grouping.", "not applicable / categorical"),
    "feature_type": ("Descriptor or attribution type.", "not applicable / categorical"),
    "importance": ("Raw feature-importance value.", "model-dependent attribution scale"),
    "normalized_importance": ("Feature importance normalized within its analysis.", "dimensionless"),
    "mean_normalized_importance": ("Mean normalized feature importance across records.", "dimensionless"),
    "mean_bar_color": ("Hexadecimal color assigned to the mean-importance bar.", "not applicable / color"),
    "color": ("Hexadecimal plotting color.", "not applicable / color"),
    "marker": ("Plotting-marker code.", "not applicable / categorical"),
    "plot_y": ("Final vertical plotting coordinate.", "display coordinate"),
    "plot_y_base": ("Base vertical plotting coordinate before offset.", "display coordinate"),
    "plot_y_offset": ("Vertical display offset applied to the record.", "display coordinate"),
    "plot_order_top_to_bottom": ("Top-to-bottom display order.", "ordinal index"),
    "plot_order_bottom_to_top": ("Bottom-to-top display order.", "ordinal index"),
    "temperature_bin": ("Temperature interval used for error aggregation.", "not applicable / categorical"),
    "display_cation": ("Display label for the cation family.", "not applicable / text"),
    "display_anion": ("Display label for the anion family.", "not applicable / text"),
    "display_family": ("Combined display label for an ion family.", "not applicable / text"),
    "label_count": ("Number of available labels represented by the record.", "count"),
    "n_labels": ("Number of evaluated labels.", "count"),
    "n_present": ("Number of records in which the motif is present.", "count"),
    "n_absent": ("Number of records in which the motif is absent.", "count"),
    "present_median_abs_log_error": ("Median absolute natural-log error when the motif is present.", "dimensionless log error"),
    "absent_median_abs_log_error": ("Median absolute natural-log error when the motif is absent.", "dimensionless log error"),
    "delta_median_abs_log_error": ("Present-minus-absent median absolute natural-log error.", "dimensionless log error"),
    "abs_log_error": ("Absolute prediction error in natural-log target space.", "dimensionless log error"),
    "positive_prediction": ("Whether the model prediction is positive on the original scale.", "boolean"),
    "val_score": ("Validation score stored with the checkpoint.", "model-metric dependent"),
    "raw_R2": ("Coefficient of determination on the original target scale.", "dimensionless"),
    "raw_MAE": ("Mean absolute error on the original target scale.", "property-dependent"),
    "raw_RMSE": ("Root-mean-square error on the original target scale.", "property-dependent"),
    "raw_NMAE": ("Range-normalized mean absolute error on the original target scale.", "dimensionless"),
    "log_R2": ("Coefficient of determination in natural-log target space.", "dimensionless"),
    "log_MAE": ("Mean absolute error in natural-log target space.", "dimensionless log error"),
    "log_RMSE": ("Root-mean-square error in natural-log target space.", "dimensionless log error"),
    "log_NMAE": ("Range-normalized mean absolute error in natural-log target space.", "dimensionless"),
}


PROPERTY_FIELDS: dict[str, Definition] = {
    "Density": ("Model-predicted density.", "kg m^-3"),
    "Viscosity": ("Model-predicted dynamic viscosity.", "Pa s"),
    "ElectricalConductivity": ("Model-predicted electrical conductivity.", "S m^-1"),
    "HeatCapacity": ("Model-predicted molar heat capacity.", "J K^-1 mol^-1"),
    "SurfaceTension": ("Model-predicted surface tension.", "N m^-1"),
    "ThermalConductivity": ("Model-predicted thermal conductivity.", "W m^-1 K^-1"),
}


APPLICATION_EXACT: dict[str, Definition] = {
    "candidate_id": ("Stable identifier assigned to a candidate ion pair.", "not applicable / identifier"),
    "candidate_type": ("Observed-reference or unseen-recombination candidate class.", "not applicable / categorical"),
    "generation_status": ("Outcome of candidate construction and charge validation.", "not applicable / categorical"),
    "cation_charge": ("Formal charge of the cation component.", "elementary-charge units"),
    "anion_charge": ("Formal charge of the anion component.", "elementary-charge units"),
    "cation_smiles": ("Input cation SMILES.", "not applicable / molecular string"),
    "anion_smiles": ("Input anion SMILES.", "not applicable / molecular string"),
    "il_smiles": ("Combined ion-pair SMILES.", "not applicable / molecular string"),
    "model_il_smiles": ("Ion-pair SMILES supplied to the prediction model.", "not applicable / molecular string"),
    "source_il_smiles": ("Ion-pair SMILES in the source reference record.", "not applicable / molecular string"),
    "canonical_cation_smiles": ("Canonicalized cation SMILES.", "not applicable / molecular string"),
    "canonical_anion_smiles": ("Canonicalized anion SMILES.", "not applicable / molecular string"),
    "canonical_il_key": ("Canonical cation-anion identity key.", "not applicable / identifier"),
    "old_canonical_pair_key": ("Legacy SMILES-derived ion-pair key.", "not applicable / identifier"),
    "chemical_identity_key": ("Charge-aware standard-InChIKey pair identity.", "not applicable / identifier"),
    "cation_identity_key": ("Standard-InChIKey identity of the cation.", "not applicable / identifier"),
    "anion_identity_key": ("Standard-InChIKey identity of the anion.", "not applicable / identifier"),
    "standard_inchikey_parse_pass": ("Whether standard-InChIKey generation succeeded.", "boolean"),
    "identity_status": ("Outcome of the chemical-identity audit.", "not applicable / categorical"),
    "identity_error": ("Identity-parsing error, empty when parsing succeeds.", "not applicable / text"),
    "matched_benchmark_IL_names": ("Benchmark ionic-liquid names sharing the audited identity.", "not applicable / text"),
    "identity_audit_pass": ("Whether the candidate passes the identity audit.", "boolean"),
    "identity_new_to_benchmark": ("Whether the ion pair is absent from the benchmark by audited identity.", "boolean"),
    "identity_new_to_primary_training_split": ("Whether the ion pair is absent from the primary training split.", "boolean"),
    "pair_seen_in_benchmark": ("Whether the audited ion-pair identity occurs in the benchmark.", "boolean"),
    "pair_seen_in_training": ("Whether the audited ion-pair identity occurs in the training split.", "boolean"),
    "charge_balance_pass": ("Whether the ion pair satisfies the configured charge-balance rule.", "boolean"),
    "cation_family": ("Curated cation-family label.", "not applicable / categorical"),
    "anion_family": ("Curated anion-family label.", "not applicable / categorical"),
    "cation_support_count": ("Number of primary-training records supporting the cation identity.", "count"),
    "anion_support_count": ("Number of primary-training records supporting the anion identity.", "count"),
    "combined_ion_support": ("Sum of cation and anion support counts.", "count"),
    "minimum_ion_support": ("Smaller of the cation and anion support counts.", "count"),
    "IL_Name": ("Reported ionic-liquid name.", "not applicable / text"),
    "checkpoint_name": ("Model checkpoint label.", "not applicable / identifier"),
    "protocol": ("Evaluation or screening protocol.", "not applicable / categorical"),
    "formal_primary_protocol": ("Whether the protocol supplies the formal candidate decision.", "boolean"),
    "applicability_domain_split": ("Split used to construct the applicability domain.", "not applicable / categorical"),
    "AD_status": ("Applicability-domain classification.", "not applicable / categorical"),
    "AD_reason": ("Reason for the applicability-domain classification.", "not applicable / text"),
    "descriptor_knn_distance": ("Nearest-neighbour distance in standardized descriptor space.", "dimensionless distance"),
    "descriptor_distance_percentile": ("Reference percentile of the descriptor-space distance.", "fraction in [0,1]"),
    "temperature_K": ("Evaluation temperature.", "K"),
    "temperature_grid_complete": ("Whether all prescribed temperature points are present.", "boolean"),
    "temperature_point_count": ("Number of evaluated temperature points.", "count"),
    "temperature_point_count_x": ("Temperature-point count before table merge.", "count"),
    "temperature_point_count_y": ("Temperature-point count after table merge.", "count"),
    "expected_temperature_point_count": ("Prespecified number of temperature points.", "count"),
    "low_temperature_K": ("Lower extended stress-test endpoint.", "K"),
    "reference_temperature_K": ("Reference temperature used for ratios.", "K"),
    "high_temperature_K": ("Upper extended stress-test endpoint.", "K"),
    "analysis_window": ("Temperature-window designation for the record.", "not applicable / categorical"),
    "temperature_domain_flag": ("Primary-window or extended-endpoint designation.", "not applicable / categorical"),
    "cold": ("Cold-endpoint resistance ratio.", "dimensionless"),
    "hot": ("Hot-endpoint resistance ratio.", "dimensionless"),
    "molar_mass_kg_per_mol": ("RDKit-derived molar mass of the ion pair.", "kg mol^-1"),
    "cp_mass_J_kg-1_K-1": ("Mass-specific heat capacity converted from molar heat capacity.", "J kg^-1 K^-1"),
    "volumetric_heat_capacity": ("Density multiplied by mass-specific heat capacity.", "J m^-3 K^-1"),
    "thermal_diffusivity": ("Thermal conductivity divided by volumetric heat capacity.", "m^2 s^-1"),
    "simplified_thermal_diffusion_timescale": ("Configured squared length divided by thermal diffusivity.", "s"),
    "thermal_effusivity": ("Square root of thermal conductivity times volumetric heat capacity.", "J m^-2 K^-1 s^-1/2"),
    "electrolyte_mass_kg": ("Electrolyte mass in the standardized scenario.", "kg"),
    "z_conductivity": ("Reference-median/IQR standardized log10 conductivity.", "dimensionless standardized score"),
    "z_viscosity": ("Reference-median/IQR standardized log10 viscosity.", "dimensionless standardized score"),
    "transport_favorability": ("Transport proxy z_conductivity minus z_viscosity.", "dimensionless standardized score"),
    "surface_tension_reference_envelope_deviation": ("Distance outside the temperature-matched reference surface-tension envelope.", "reference-IQR units"),
    "proxy_warnings": ("Semicolon-delimited proxy-calculation warnings.", "not applicable / text"),
    "curve_warning_count": ("Number of curve-quality warnings over the main window.", "count"),
    "severe_curve_failure_count": ("Number of fail-closed severe curve failures.", "count"),
    "density_range": ("Maximum minus minimum predicted density over the main temperature window.", "kg m^-3"),
    "inference_status": ("Candidate-inference completion status.", "not applicable / categorical"),
    "failure_reasons": ("Semicolon-delimited hard-screening failure reasons.", "not applicable / text"),
    "final_feasible": ("Whether all hard constraints are satisfied.", "boolean"),
    "hard_feasible": ("Whether all hard constraints are satisfied.", "boolean"),
    "pareto_rank": ("Non-dominated Pareto rank under the four prespecified objectives.", "ordinal index"),
    "Pareto_rank": ("Non-dominated Pareto rank under the four prespecified objectives.", "ordinal index"),
    "nominal_pareto_rank_1": ("Whether the candidate is Pareto rank 1 under nominal thresholds.", "boolean"),
    "formal_shortlist_selected": ("Whether the candidate enters the deterministic formal Top-8.", "boolean"),
    "formal_shortlist_order": ("Deterministic order within the formal Top-8.", "ordinal index"),
    "formal_shortlist_rule": ("Rule used to select the formal shortlist.", "not applicable / text"),
    "formal_shortlist_ids": ("Delimited candidate identifiers in the formal shortlist.", "not applicable / identifiers"),
    "nominal_formal_shortlist": ("Whether the candidate belongs to the nominal formal shortlist.", "boolean"),
    "utopia_distance": ("Euclidean distance to the robust-normalized four-objective ideal point.", "dimensionless"),
    "utopia_rank_one_order": ("Order of Pareto-rank-1 candidates by utopia distance and tie-breaker.", "ordinal index"),
    "deterministic_tie_breaker": ("Canonical deterministic tie-break value.", "not applicable / identifier"),
    "domination_count": ("Number of candidates dominating this candidate.", "count"),
    "dominated_set_size": ("Number of candidates dominated by this candidate.", "count"),
    "protocol_top8": ("Whether the candidate enters a protocol-specific Top-8.", "boolean"),
    "decision_code": ("Compact protocol-decision code.", "not applicable / categorical"),
    "decision_label": ("Human-readable protocol-decision label.", "not applicable / categorical"),
    "protocol_rank_order": ("Candidate rank within the stated protocol.", "ordinal index"),
    "protocol_decision_distance": ("Protocol-specific normalized decision distance.", "dimensionless"),
    "cross_protocol_eligible": ("Whether the candidate satisfies the cross-protocol role rule.", "boolean"),
    "cross_protocol_rank_change": ("Absolute rank change between primary and balanced protocols.", "rank positions"),
    "selection_frequency": ("Fraction of resamples in which the candidate enters the Top-8.", "fraction in [0,1]"),
    "hard_feasible_frequency": ("Fraction of resamples in which all hard constraints pass.", "fraction in [0,1]"),
    "pareto_rank_1_frequency": ("Fraction of resamples in which the candidate is Pareto rank 1.", "fraction in [0,1]"),
    "top8_jaccard_to_nominal": ("Jaccard similarity between the resampled and nominal Top-8.", "fraction in [0,1]"),
    "final_set_Jaccard_to_nominal": ("Jaccard similarity between the threshold-grid and nominal final sets.", "fraction in [0,1]"),
    "bootstrap_iteration": ("Bootstrap replicate index.", "index"),
    "is_nominal": ("Whether the record is the prespecified nominal setting.", "boolean"),
    "qualification_role": ("Deterministically defined downstream qualification role.", "not applicable / categorical"),
    "downstream_priority": ("Downstream qualification-priority label.", "not applicable / categorical"),
    "qualification_measurement_priority": ("Recommended order of downstream measurements.", "not applicable / text"),
    "recommendation_class": ("Candidate qualification recommendation class.", "not applicable / categorical"),
    "main_advantage": ("Primary predicted advantage motivating qualification.", "not applicable / text"),
    "main_limitation": ("Primary predicted limitation requiring qualification.", "not applicable / text"),
    "selection_criterion": ("Deterministic rule used for the qualification role.", "not applicable / text"),
    "selection_criterion_value": ("Value of the role-selection objective.", "criterion-dependent"),
    "priority_order": ("Display and downstream-qualification priority order.", "ordinal index"),
    "uncertainty_status": ("Decision-stability qualification attached to the candidate.", "not applicable / categorical"),
    "extrapolative_property_flag": ("Whether a predicted property is flagged as extrapolative.", "boolean"),
    "transport_distance": ("Distance to the ideal point in the two-objective transport space.", "dimensionless"),
    "thermal_management_distance": ("Distance to the ideal point in the two-objective thermal space.", "dimensionless"),
    "balanced_distance": ("Distance to the ideal point in the four-objective balanced space.", "dimensionless"),
    "stage_order": ("Display order of the candidate-space funnel stage.", "ordinal index"),
    "stage": ("Machine-readable candidate-space funnel stage.", "not applicable / categorical"),
    "parameter": ("Name of a standardized reference-cell scenario parameter.", "not applicable / categorical"),
    "symbol": ("Mathematical symbol used for the scenario parameter.", "not applicable / text"),
    "scenario_id": ("Stable identifier of a sensitivity or reference-cell scenario.", "not applicable / identifier"),
    "scenario_interpretation": ("Scope statement for the reference-cell calculation.", "not applicable / text"),
    "scenario_electrode_area_m2": ("Electrode area fixed by the reference-cell scenario.", "m^2"),
    "scenario_separator_thickness_m": ("Separator thickness fixed by the reference-cell scenario.", "m"),
    "scenario_electrolyte_volume_m3": ("Electrolyte volume fixed by the reference-cell scenario.", "m^3"),
    "scenario_heat_transfer_area_m2": ("Exposed heat-transfer area fixed by the scenario.", "m^2"),
    "electrolyte_resistance_ohm": ("Ideal separator-path electrolyte resistance.", "ohm"),
    "electrolyte_resistance_ohm_at_low_temperature": ("Ideal electrolyte resistance at the lower stress-test endpoint.", "ohm"),
    "electrolyte_resistance_ohm_at_reference_temperature": ("Ideal electrolyte resistance at the reference temperature.", "ohm"),
    "electrolyte_resistance_ohm_at_high_temperature": ("Ideal electrolyte resistance at the upper stress-test endpoint.", "ohm"),
    "electrolyte_resistance_ohm_worst": ("Maximum ideal electrolyte resistance over the evaluated temperatures.", "ohm"),
    "relative_electrolyte_resistance": ("Electrolyte resistance divided by its reference-temperature value.", "dimensionless"),
    "reference_temperature_resistance_ohm": ("Electrolyte resistance at the reference temperature.", "ohm"),
    "joule_heating_power_W": ("Constant-current Joule term I^2 R.", "W"),
    "joule_heating_power_W_worst": ("Maximum constant-current Joule term over the evaluated temperatures.", "W"),
    "initial_adiabatic_temperature_rise_rate_K_per_s": ("Initial adiabatic temperature-rise rate from the Joule term.", "K s^-1"),
    "convective_thermal_resistance_K_per_W": ("Convective component of thermal resistance.", "K W^-1"),
    "internal_thermal_conduction_resistance_K_per_W": ("Separator-path conduction component of thermal resistance.", "K W^-1"),
    "thermal_resistance_K_per_W": ("Series conduction-plus-convection thermal resistance.", "K W^-1"),
    "thermal_resistance_conduction_fraction": ("Fraction of total thermal resistance from internal conduction.", "fraction in [0,1]"),
    "thermal_resistance_convection_fraction": ("Fraction of total thermal resistance from convection.", "fraction in [0,1]"),
    "electrolyte_thermal_capacitance_J_per_K": ("Lumped electrolyte thermal capacitance.", "J K^-1"),
    "lumped_thermal_time_constant_s": ("Lumped thermal resistance-capacitance time constant.", "s"),
    "steady_state_temperature_rise_K": ("Conditional steady temperature rise under the reference scenario.", "K"),
    "steady_state_temperature_rise_K_worst": ("Maximum conditional steady temperature rise over the evaluated temperatures.", "K"),
    "transient_temperature_rise_K": ("Conditional 60-s lumped temperature rise.", "K"),
    "transient_temperature_rise_K_worst": ("Maximum conditional 60-s lumped temperature rise over the evaluated temperatures.", "K"),
    "reference_electrolyte_resistance_ohm_q75": ("Temperature-matched observed-reference 75th percentile resistance.", "ohm"),
    "reference_electrolyte_resistance_ohm_q95": ("Temperature-matched observed-reference 95th percentile resistance.", "ohm"),
    "reference_transient_temperature_rise_K_q75": ("Temperature-matched observed-reference 75th percentile 60-s rise.", "K"),
    "reference_transient_temperature_rise_K_q95": ("Temperature-matched observed-reference 95th percentile 60-s rise.", "K"),
    "electrical_exceedance_ratio_to_reference_q75": ("Resistance divided by the temperature-matched reference 75th percentile.", "dimensionless"),
    "thermal_exceedance_ratio_to_reference_q75": ("60-s rise divided by the temperature-matched reference 75th percentile.", "dimensionless"),
    "reference_cell_exceedance_index": ("Maximum of the electrical and thermal reference-q75 exceedance ratios.", "dimensionless"),
    "reference_cell_exceedance_index_worst": ("Largest reference-q75 exceedance index over the evaluated temperatures.", "dimensionless"),
    "reference_cell_exceedance_index_at_band_worst": ("Exceedance-index value at the temperature defining the worst exceedance band.", "dimensionless"),
    "reference_cell_exceedance_band": ("Reference-population exceedance band.", "not applicable / categorical"),
    "reference_cell_exceedance_band_worst": ("Most severe reference-population exceedance band over temperature.", "not applicable / categorical"),
    "reference_cell_exceedance_component": ("Component controlling the exceedance index.", "not applicable / categorical"),
    "reference_cell_exceedance_component_worst": ("Component controlling the largest exceedance index.", "not applicable / categorical"),
    "reference_cell_exceedance_band_component_worst": ("Component controlling the most severe exceedance band.", "not applicable / categorical"),
    "pressure_kPa": ("Pressure associated with the predicted condition.", "kPa"),
    "property_threshold_source": ("Origin of the hard-screening threshold.", "not applicable / text"),
    "constraint": ("Hard constraint represented by the record.", "not applicable / categorical"),
    "constraint_direction": ("Whether a larger or smaller value is favorable.", "not applicable / categorical"),
    "threshold": ("Numerical hard-constraint threshold.", "constraint-dependent"),
    "conductivity_threshold_S_m": ("Reference-derived minimum conductivity threshold.", "S m^-1"),
    "viscosity_threshold_Pa_s": ("Reference-derived maximum viscosity threshold.", "Pa s"),
    "heat_capacity_threshold_J_m3_K": ("Reference-derived minimum volumetric heat-capacity threshold.", "J m^-3 K^-1"),
    "thermal_diffusivity_threshold_m2_s": ("Reference-derived minimum thermal-diffusivity threshold.", "m^2 s^-1"),
    "heat_capacity_quantile": ("Reference quantile used for the volumetric heat-capacity threshold.", "fraction in [0,1]"),
    "margin_definition": ("Formula used to calculate the constraint margin.", "not applicable / text"),
    "log2_margin": ("Base-2 logarithm of the favorable constraint ratio.", "dimensionless"),
    "plot_log2_margin": ("Clipped base-2 margin used only for plotting.", "dimensionless"),
}


BASE_METRICS: dict[str, Definition] = {
    "conductivity": ("electrical conductivity", "S m^-1"),
    "viscosity": ("dynamic viscosity", "Pa s"),
    "volumetric_heat_capacity": ("volumetric heat capacity", "J m^-3 K^-1"),
    "thermal_diffusivity": ("thermal diffusivity", "m^2 s^-1"),
    "transport_favorability": ("standardized transport-favorability score", "dimensionless standardized score"),
    "simplified_thermal_diffusion_timescale": ("simplified thermal-diffusion timescale", "s"),
    "thermal_timescale": ("simplified thermal-diffusion timescale", "s"),
    "surface_tension_reference_envelope_deviation": ("surface-tension reference-envelope deviation", "reference-IQR units"),
    "density": ("density", "kg m^-3"),
}


def _property_statistic(column: str) -> Definition | None:
    for prefix, (description, unit) in PROPERTY_FIELDS.items():
        if column == prefix:
            return description, unit
        if not column.startswith(prefix + "_"):
            continue
        suffix = column[len(prefix) + 1 :]
        lower = description.removeprefix("Model-predicted ").rstrip(".")
        if suffix == "mean":
            return f"Mean predicted {lower} over the main temperature window.", unit
        if suffix == "slope":
            return f"Linear temperature slope of predicted {lower}.", f"{unit} K^-1"
        if suffix == "relative_change":
            return f"Full-window relative change in predicted {lower}.", "dimensionless"
        if suffix == "coefficient_of_variation":
            return f"Coefficient of variation of predicted {lower} over temperature.", "dimensionless"
    return None


def _derived_statistic(column: str) -> Definition | None:
    suffixes = (
        "_worst_temperature_K",
        "_coefficient_of_variation",
        "_relative_change",
        "_temperature_K",
        "_worst",
        "_mean",
        "_slope",
        "_min",
        "_max",
        "_quantile",
    )
    for suffix in suffixes:
        if not column.endswith(suffix):
            continue
        base = column[: -len(suffix)]
        if base not in BASE_METRICS:
            continue
        name, unit = BASE_METRICS[base]
        if suffix in {"_temperature_K", "_worst_temperature_K"}:
            return f"Temperature at which the reported {name} extreme occurs.", "K"
        if suffix == "_worst":
            return f"Least favorable {name} over the main temperature window.", unit
        if suffix == "_mean":
            return f"Mean {name} over the main temperature window.", unit
        if suffix == "_slope":
            return f"Linear temperature slope of {name}.", f"{unit} K^-1"
        if suffix == "_relative_change":
            return f"Full-window relative change in {name}.", "dimensionless"
        if suffix == "_coefficient_of_variation":
            return f"Coefficient of variation of {name} over temperature.", "dimensionless"
        if suffix == "_min":
            return f"Minimum reference value of {name}.", unit
        if suffix == "_max":
            return f"Maximum allowed value of {name}.", unit
        if suffix == "_quantile":
            return f"Reference quantile used to set the {name} threshold.", "fraction in [0,1]"
    return None


def _application_pattern(column: str) -> Definition | None:
    property_definition = _property_statistic(column)
    if property_definition is not None:
        return property_definition
    derived_definition = _derived_statistic(column)
    if derived_definition is not None:
        return derived_definition

    if column.startswith("pass_"):
        label = column.removeprefix("pass_").replace("_", " ")
        return f"Whether the candidate passes the {label} hard gate.", "boolean"
    if column.startswith("utopia_normalized_"):
        label = column.removeprefix("utopia_normalized_").replace("_", " ")
        return f"Robust-normalized objective value for {label}.", "dimensionless"
    if column.endswith("_count"):
        label = column[: -len("_count")].replace("_", " ")
        return f"Number of {label}.", "count"
    if column.endswith("_frequency"):
        label = column[: -len("_frequency")].replace("_", " ")
        return f"Fraction of bootstrap replicates with {label}.", "fraction in [0,1]"
    if column.endswith("_pct"):
        label = column[: -len("_pct")].replace("_", " ")
        return f"{label.capitalize()}, expressed as a percentage.", "%"
    if column.endswith("_ratio") or "_ratio_to_" in column:
        label = column.replace("_", " ")
        return f"Dimensionless {label}.", "dimensionless"
    if column.endswith("_temperature_K"):
        label = column[: -len("_temperature_K")].replace("_", " ")
        return f"Temperature associated with {label}.", "K"
    if column.endswith("_ohm"):
        return column.replace("_", " ").capitalize() + ".", "ohm"
    if column.endswith("_W"):
        return column.replace("_", " ").capitalize() + ".", "W"
    if column.endswith("_K"):
        return column.replace("_", " ").capitalize() + ".", "K"
    return None


def definition_for(bundle: str, column: str) -> tuple[str, str, str]:
    """Return description, unit/scale, and definition provenance."""

    if column in COMMON:
        return *COMMON[column], "curated-common"
    if bundle == "Intro-method" and column in INTRO:
        return *INTRO[column], "curated-bundle"
    if bundle == "performance_results" and column in PERFORMANCE:
        return *PERFORMANCE[column], "curated-bundle"
    if bundle == "dataset_statistics" and column in DATASET:
        return *DATASET[column], "curated-bundle"
    if bundle == "interpretability_feature_importance_4x3":
        if column in INTERPRETABILITY:
            return *INTERPRETABILITY[column], "curated-bundle"
        if column in {"Density", "Visc.", "EC", "HC", "ST", "TC"}:
            return (
                f"Property-specific error or attribution value for {column}.",
                "analysis-dependent",
                "curated-bundle",
            )
    if bundle == "computational_application_case":
        if column in PROPERTY_FIELDS:
            return *PROPERTY_FIELDS[column], "curated-bundle"
        if column in APPLICATION_EXACT:
            return *APPLICATION_EXACT[column], "curated-bundle"
        patterned = _application_pattern(column)
        if patterned is not None:
            return *patterned, "schema-rule"
    raise KeyError(
        f"No semantic field definition for bundle={bundle!r}, column={column!r}"
    )
