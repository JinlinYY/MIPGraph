# MIPGraph research workflows

These modules accompany *Ion-Pair Representation Learning for Sparse Multi-Property
Prediction and Ionic-Liquid Screening*.

| Module | Research content |
|---|---|
| `baseline_comparison/` | Descriptor, molecular-language, and graph baselines |
| `dataset_analysis/` | Property distributions, label availability, and chemical coverage |
| `performance_results/` | Four-protocol, three-seed predictive performance figure |
| `interpretability/` | Feature, atom, and cross-ion attribution analyses |
| `molecular_interpretation/` | Condition-adjusted structure–property associations and response shapes |
| `computational_application_case/` | Ion-pair screening and reference-cell calculations |
| `manuscript_figure_source_data/` | Single canonical copy of numerical figure data |
| `result_analysis/` | Final figure files and links to their source tables |

Run commands from the repository root using the scientific environment described
in the main README. Each module README identifies its inputs, methods and entry
points. Numerical tables are stored once in `manuscript_figure_source_data/`;
`result_analysis/figure_source_map.csv` connects figure panels to those tables.

To recreate the performance figure from its released CSVs:

```bash
python -m experiments.performance_results.plot_performance_results
```

To update the numerical source catalog and figure links:

```bash
python experiments/result_analysis/figures/manuscript_figure_source_data/rebuild_manifest.py
python experiments/result_analysis/scripts/package_manuscript_results.py --source-data-only
```
