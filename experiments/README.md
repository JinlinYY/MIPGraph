# Experiments

This directory keeps lightweight experiment code for manuscript figures and
baseline comparisons. Generated checkpoints, logs, metrics, predictions,
caches, and one-off experiment outputs should stay outside version control
unless explicitly released.

Use a Python environment with the project scientific stack installed:
`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `rdkit`, `torch`,
`torch-geometric`, and `joblib`. `xgboost`, `lightgbm`, and `transformers` are
needed only for their specific baselines.

## Layout

- `baseline_comparison/`: tabular, SMILES, and graph baseline runners plus
  summary-table scripts.
- `dataset_analysis/`: dataset statistics and source-data export scripts.
- `performance_results/`: audited nine-panel MIPGraph performance figure.
- `interpretability/`: interpretability and feature-importance figure scripts.
- `computational_application_case/`: audited virtual-screening application.
- `molecular_origin_analysis/`: molecular-structure--property analysis.
- `manuscript_figure_source_data/`: sole authoritative panel-level CSV tree.
- `result_analysis/`: submission figures and figure-to-source mapping only.

## Smoke Test

From the repository root, all scripts should at least expose a working help
message:

```bash
python experiments/baseline_comparison/scripts/run_baseline_comparison.py --help
python experiments/dataset_analysis/scripts/plot_dataset_statistics_nature.py --help
python -m experiments.performance_results.plot_performance_results --help
python experiments/interpretability/scripts/plot_feature_importance_summary.py --help
python experiments/computational_application_case/scripts/build_refactored_application_case.py --help
```

Baseline training can be checked without launching full model fits:

```bash
python experiments/baseline_comparison/scripts/run_baseline_comparison.py --models rf --dry-run
python experiments/baseline_comparison/scripts/run_split_baseline_comparison.py --cases random_il_level --models rf --dry-run
```

## Regenerate Figures

Dataset overview:

```bash
python experiments/dataset_analysis/scripts/plot_dataset_statistics_nature.py
python experiments/dataset_analysis/scripts/export_dataset_statistics_source_data.py
```

Performance figure:

```bash
python -m experiments.performance_results.plot_performance_results
```

If staged figure inputs are not under the default training-output path:

```bash
python -m experiments.performance_results.plot_performance_results --input-root path/to/figure_inputs
```

Interpretability and feature-importance plate:

```bash
python experiments/interpretability/scripts/plot_interpretability_results_current.py
python experiments/interpretability/scripts/plot_feature_importance_summary.py --panel-labels j,k,l --color-mode panel
python experiments/interpretability/scripts/compose_interpretability_four_by_three.py
```

Audited application-case figures:

```bash
python experiments/computational_application_case/scripts/build_refactored_application_case.py
```

Rebuild the authoritative source-data catalogs after refreshing panel CSVs:

```bash
python experiments/manuscript_figure_source_data/rebuild_manifest.py
```
