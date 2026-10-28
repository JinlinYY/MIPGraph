# Molecular interpretation of property-specific thermophysical behavior

This module contains the structure–property association analysis presented in the manuscript
and Supporting Information. It relates experimental observations and MIPGraph
representations to chemical features of cations, anions and their combinations.

## Analyses

The consolidated main-text figure combines:

1. Structural associations and cross-ion attention contrasts for six properties.
2. Leading associations after thermodynamic-condition adjustment and IL-identity
   balancing, with family and descriptor-redundancy controls.
3. Property-wise response shapes for selected non-redundant factors, shown as
   IL-level quintile medians and bootstrap confidence intervals.
4. Absolute property differences between condition-matched pairs differing in
   cation or anion identity.

The SI analysis examines the heat-capacity association after molecular-size
control. Statistical associations, model attention, and matched-pair differences
are reported with their property-specific coverage and uncertainty.

## Data and model inputs

The analyses use the curated IL identities, experimental property observations,
temperature and pressure records, molecular descriptors, and the configured
MIPGraph checkpoint. Model-based stages load the checkpoint together with its
matching architecture, target scales, and condition scales.

Settings are in [`config/analysis_config.yaml`](config/analysis_config.yaml).
Use `model.checkpoint` to specify a compatible checkpoint and `data.max_samples: 0`
for the complete analysis dataset. The configuration also controls reference
conditions, identity-level bootstrap repeats, descriptor filtering, and output
formats.

## Entry points

Run from the repository root:

```bash
python experiments/molecular_origin_analysis/run_all.py   --config experiments/molecular_origin_analysis/config/analysis_config.yaml
```

The `--stage` option selects association, attribution, attention, applicability,
or figure-generation stages. For example:

```bash
python experiments/molecular_origin_analysis/run_all.py --stage association
python experiments/molecular_origin_analysis/run_all.py --stage figures
```

Install the additional analysis dependencies when needed:

```bash
python -m pip install -r experiments/molecular_origin_analysis/requirements_extra.txt
```

## Figure data

The final numerical panel tables are stored once in
`experiments/manuscript_figure_source_data/molecular_origin_analysis/source_data/`.
Final figure exports are in `experiments/result_analysis/figures/` under
`molecular_origin_analysis` and `figureS_heat_capacity_size_control`.

The source-table fields, units, coverage, and interval definitions are given in
the companion data dictionary. To export newly calculated panel tables:

```bash
python experiments/molecular_origin_analysis/package_source_data.py
```
