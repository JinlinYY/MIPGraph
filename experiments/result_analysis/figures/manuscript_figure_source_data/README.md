# Manuscript figure source data

This directory stores one canonical copy of the numerical tables underlying the
manuscript and Supporting Information. Final image exports are stored separately
in the parent [`figures/`](../) directory.

| Data group | Contents |
|---|---|
| `dataset_statistics/` | Property distributions, condition ranges and label coverage |
| `performance_results/` | Three-seed test predictions and log-space performance summaries |
| `molecular_interpretation/` | Four main-figure panels and molecular-size-controlled heat-capacity analysis |
| `interpretability_feature_importance_4x3/` | Atom, edge, descriptor and functional-group attribution |
| `computational_application_case/` | Candidate identities, constraints, Pareto selection and reference-cell quantities |
| `Intro-method/` | Panel inventory for the conceptual illustration |

`manifest.csv` lists the data tables and checksums. `column_dictionary.csv`
defines field meanings, units, and storage types. The mapping in
`../../figure_source_map.csv` refers directly to these CSVs, so no
second copy is needed under the figure directory.

Performance annotations are arithmetic means of metrics calculated independently
for training seeds 0, 42, and 123. Panels a–f include each run's predictions;
panel-g error bars show the sample standard deviation across seeds. Test labels
are original experimental observations. The numerical NMAE denominator is the
property scale fitted to original observations in the training partition.

The reference-cell display list is recorded in
`computational_application_case/reference_cell_highlights.csv`. Its order and
colours are display metadata, separate from candidate screening ranks.

Rebuild the catalog and direct figure links:

```bash
python experiments/result_analysis/figures/manuscript_figure_source_data/rebuild_manifest.py
python experiments/result_analysis/scripts/package_manuscript_results.py --source-data-only
```
