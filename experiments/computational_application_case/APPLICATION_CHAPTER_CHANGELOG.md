# Computational application chapter changelog

Date: 2026-07-20

## Scope guard

- Modified manuscript scope: application chapter only
- Other manuscript sections modified: No
- Method section modified: No
- Application protocol placed at chapter opening: Yes
- Formal candidate model: random-IL checkpoint
- Secondary checkpoints averaged: No
- Reference-cell mapping used for selection: No
- Experimental validation claimed: No

Other manuscript sections may require later consistency updates, but they were not modified in this task.

## Application protocol and results

- Formal application window: 298.15--353.15 K at 101.325 kPa.
- Extended stress-test endpoints: 278.15 and 373.15 K; excluded from screening, Pareto sorting, Top-8 selection, and formal feasibility claims.
- Candidate construction: 30 supported cations x 30 supported anions = 900 nominal pairs.
- Charge-aware Standard-InChIKey audit: 608 benchmark-new ion-pair recombinations.
- Primary inference: all 608 candidates evaluated; no 608-to-600 property-dependent truncation.
- Nominal primary decision: 26 hard-feasible, 12 Pareto rank one, 8 formal candidates.
- Formal Pareto objectives: maximize worst-window conductivity, minimize worst-window viscosity, maximize worst-window volumetric heat capacity, and maximize worst-window thermal diffusivity.
- Formal Top-8 rule: robust q05--q95 normalization, ascending utopia distance, then descriptor AD distance, descending cation support, descending anion support, and canonical ion-pair identity.
- Reference-set bootstrap: 500 replicates with a median Top-8 Jaccard similarity of 0.333 and percentile 95% interval 0.067--1.000.
- Cross-protocol sensitivity: primary 26/12, balanced 16/5, and family-transfer stress test 12/1 for hard-feasible/Pareto-rank-one counts.
- Reference-cell mapping was performed only after the formal shortlist and qualification roles were fixed.
- Thermal-resistance audit: conduction fraction 1.000%--1.323%, convection fraction 98.677%--99.000%, and Pearson correlation between log electrolyte resistance and log conditional temperature rise 0.999996.
- Extreme-property audit: five nearest training-domain liquids per formal candidate; no Top-8 candidate exceeded the observed-training 0.5%--99.5% property range; unit and inverse-transform audits passed.

## Principal deliverables

- `LaTex-MIPGraph/section_auditable_virtual_screening_bilingual.tex`
- `LaTex-MIPGraph/Fig/figure5_auditable_virtual_screening_validation.pdf`
- `LaTex-MIPGraph/Fig/figure5_auditable_virtual_screening_validation.png`
- `LaTex-MIPGraph/Fig/figure6_reference_cell_scenario_audited.pdf`
- `LaTex-MIPGraph/Fig/figure6_reference_cell_scenario_audited.png`
- `LaTex-MIPGraph/section_refactored_application_SI.tex`
- `outputs_primary_audited/data/candidate_screening_trajectory_608.csv`
- `outputs_primary_audited/data/standard_inchikey_identity_audit_608.csv`
- `outputs_primary_audited/data/pareto_rank1_all_candidates.csv`
- `outputs_primary_audited/data/pareto_rank1_top8_selection.csv`
- `outputs_primary_audited/data/qualification_role_selection_audit.csv`
- `outputs_primary_audited/data/threshold_sensitivity.csv`
- `outputs_primary_audited/data/reference_bootstrap_iterations.csv`
- `outputs_primary_audited/data/reference_bootstrap_candidate_selection.csv`
- `outputs_primary_audited/data/cross_protocol_decision_matrix.csv`
- `outputs_primary_audited/data/reference_cell_heat_resistance_contribution_audit.csv`
- `outputs_primary_audited/data/extreme_property_audit.csv`
- `outputs_primary_audited/data/extreme_property_nearest_neighbors.csv`

## Integrity statement

No threshold, candidate identity, model output, or qualification-role definition was changed to obtain a more stable or favorable result. The formal candidate identities and order were unchanged after adding explicit deterministic tie-break fields.
