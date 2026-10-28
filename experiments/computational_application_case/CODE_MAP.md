# Computational screening workflow

This workflow corresponds to the ionic-liquid screening and reference-cell
analysis in the manuscript and Supporting Information.

## Calculation stages

1. Define candidate cation–anion combinations from the declared training domain.
2. Standardize identities and distinguish observed pairs from unseen combinations.
3. Predict the six properties across the formal temperature window.
4. Apply whole-window constraints and four-objective Pareto selection.
5. Evaluate threshold, bootstrap, and cross-protocol decision stability.
6. Map selected candidates to the specified reference-cell scenario.

The primary random IL-level checkpoint defines the formal shortlist. Other
partition-specific checkpoints are evaluated independently for sensitivity.

## Code locations

| Location | Research function |
|---|---|
| `run_all.py` | Screening entry point |
| `configs/auditable_virtual_screening.yaml` | Primary screening configuration |
| `configs/protocol_stability_balanced.yaml` | Property-balanced partition sensitivity |
| `configs/protocol_stability_ion_family.yaml` | Family-combination partition sensitivity |
| `src/` | Candidate identities, predictions, constraints, Pareto ranking and reference-cell quantities |
| `scripts/build_protocol_stability_outputs.py` | Independent secondary-protocol calculations |
| `scripts/build_refactored_application_case.py` | Figure and numerical-table generation |
| `scripts/render_screening_figure.py` | Render the released screening figure |
| `scripts/render_reference_cell_figure.py` | Render the reference-cell figure |
| `scripts/render_derived_metrics_figure.py` | Render Joule and steady-rise curves |

Run commands from the repository root using `python`. See `README.md` for the
released-figure commands. Numerical sources are stored in
`experiments/result_analysis/figures/manuscript_figure_source_data/computational_application_case/`;
final figures and their source links are in `experiments/result_analysis/`.
