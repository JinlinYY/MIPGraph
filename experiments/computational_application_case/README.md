# Ionic-liquid screening and reference-cell analysis

This workflow implements the computational screening study in the manuscript.
It combines six-property predictions with identity checks, whole-temperature-window
constraints, four-objective Pareto ranking, and reference-cell calculations.

## Candidate selection

Candidate ion pairs are assembled from cations and anions represented in the
declared training domain. Standardized charge-aware identities separate observed
reference pairs from benchmark-unseen combinations. The primary random IL-level
checkpoint defines the formal shortlist. Property-balanced and family-pair
checkpoints provide independent protocol-sensitivity analyses.

Formal screening uses 298.15–353.15 K at 101.325 kPa. The extended 278.15 and
373.15 K endpoints provide temperature-sensitivity context. Whole-window
conductivity, viscosity, volumetric heat capacity and thermal diffusivity
determine feasibility and Pareto ranking.

Four candidates illustrate complementary thermophysical roles:

- UPR-0581: multi-objective lead.
- UPR-0024: transport-focused lead.
- UPR-0343: thermal-management lead.
- UPR-0558: property-balanced Pareto-retained candidate.

The formal shortlist contains eight candidates. Display roles and curve colours
are recorded separately from the screening ranks.

## Reference-cell interpretation

Predicted density, molar heat capacity, conductivity and thermal conductivity are
mapped to volumetric heat capacity, diffusivity, electrolyte resistance and
conditional temperature rise under the specified reference-cell assumptions.
The reported Joule and temperature-rise quantities are computational estimates
for that defined scenario.

## Inputs and commands

Model-driven screening loads a six-output checkpoint with its own target and
condition scales. Configuration files are under `configs/`; the entry point is:

```bash
python experiments/computational_application_case/run_all.py --help
```

Recreate the released figures from the numerical tables:

```bash
python experiments/computational_application_case/scripts/render_screening_figure.py
python experiments/computational_application_case/scripts/render_reference_cell_figure.py
python experiments/computational_application_case/scripts/render_derived_metrics_figure.py
```

The source tables are stored once in
`experiments/result_analysis/figures/manuscript_figure_source_data/computational_application_case/`.
Final PNG/PDF/SVG/TIFF figures are under `experiments/result_analysis/figures/`.
The screening tables retain identities, constraints, applicability-domain
information, protocol-specific decisions, and bootstrap selection frequencies.
