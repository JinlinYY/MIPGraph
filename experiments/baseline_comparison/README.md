# Baseline Comparison Experiments

This folder contains scripts for rerunning baseline models after the source-aware
dataset update. Large artifacts are written to
`il_property_prediction/outputs/baseline_comparison_random_point_seed42` by
default.

## Evaluation Protocol

- Data: `il_property_prediction/data/processed_ilthermo_interpolated`
- Split: `splits/row_level_seed42.json`
- Training labels: expanded `mask`
- Test labels: original-label `evaluation_mask`
- Label weights: `label_weight` multiplied into the training loss or sample weight
- Target space: standardized natural-log targets during training, log-space
  metrics for reporting

This matches the source-aware sparse-label setting described in the current
method section: augmented labels can supervise training, while validation and
test metrics remain restricted to experimentally reported labels.

## Models

The runner supports the following model names:

- `rf`
- `xgboost`
- `lgbm`
- `chemberta`
- `mpnn_concat`
- `gcn`
- `gat`
- `graphsage`
- `gin`

Tree models use fixed molecular descriptors from the graph cache, optional
functional-group descriptors when available, and thermodynamic condition
features. Graph neural network baselines use the cached ion-pair graphs. The
ChemBERTa baseline requires a local HuggingFace cache unless `--allow-download`
is passed.

## Commands

Dry run:

```powershell
E:\anaconda\envs\ggnn39\python.exe experiments\baseline_comparison\scripts\run_baseline_comparison.py --dry-run
```

Run all locally available non-ChemBERTa baselines:

```powershell
E:\anaconda\envs\ggnn39\python.exe experiments\baseline_comparison\scripts\run_baseline_comparison.py --models rf,xgboost,lgbm,mpnn_concat,gcn,gat,graphsage,gin
```

Run ChemBERTa only, using local cached weights:

```powershell
E:\anaconda\envs\ggnn39\python.exe experiments\baseline_comparison\scripts\run_baseline_comparison.py --models chemberta
```

Summarize results and add the current MIPGraph random-point metrics:

```powershell
E:\anaconda\envs\ggnn39\python.exe experiments\baseline_comparison\scripts\summarize_baseline_results.py
```

Run the same baseline suite across split strategies:

```powershell
E:\anaconda\envs\ggnn39\python.exe experiments\baseline_comparison\scripts\run_split_baseline_comparison.py --cases random_il_level,property_balanced_il_level,ion_family --models all --output-root il_property_prediction\outputs\baseline_comparison_by_split_seed42 --skip-existing
```

Summarize the split-strategy comparison table:

```powershell
E:\anaconda\envs\ggnn39\python.exe experiments\baseline_comparison\scripts\summarize_split_baseline_results.py --output-root il_property_prediction\outputs\baseline_comparison_by_split_seed42 --random-point-root il_property_prediction\outputs\baseline_comparison_random_point_seed42
```

## Main Outputs

- `metrics/<model>/test_metrics_log.csv`
- `metrics/<model>/run_manifest.json`
- `predictions/<model>/test_predictions.csv`
- `checkpoints/<model>/best_model.pt` for neural baselines
- `baseline_metrics_long.csv`
- `baseline_metrics_summary.csv`
- `baseline_comparison_table.tex`
- `split_baseline_macro_long.csv`
- `split_baseline_macro_wide.csv`
- `split_baseline_macro_table.tex`
- `split_baseline_property_long.csv`
- `split_baseline_property_wide.csv`
- `split_baseline_property_table.tex`
