# MIPGraphNet Model Workflow

This document describes the model-side workflow for training, fine-tuning, evaluation, and graph visualization.

Run all commands from the repository root:

```powershell
cd D:\GGNN\IL-model\Sparse-Label-Prediction
```

## 1. Model Preprocessing

Before training, convert the checked Excel workbook into model-ready CSV/NPZ files.

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\preprocess_data.py `
  --config configs/default.yaml
```

Main outputs:

```text
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\data\processed\il_multiprop_clean.csv
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\data\processed\il_multiprop_arrays.npz
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\data\processed\preprocessing_report.json
```

The preprocessing step:

- Reads the Excel workbook configured in `configs/default.yaml`.
- Builds sparse multi-property target arrays.
- Builds property masks.
- Keeps `ErrorValue` arrays for optional error-aware loss weighting.
- Generates an IL-level split file if needed.

## 2. Build Graph Cache

The first time you train the model, build the graph cache.

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\build_graph_cache.py `
  --config configs/default.yaml
```

Output:

```text
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\data\processed\graph_cache.pt
```

This cache stores the PyTorch Geometric graph for each unique `IL_SMILES`.

The model uses this file during training so that SMILES parsing, conformer generation, and graph construction are not repeated every epoch.

Rebuild `graph_cache.pt` when any of the following changes:

- `src/chem/graph_featurizer.py`
- `src/chem/conformer.py`
- graph feature dimensions
- cross-ion edge construction mode
- global descriptor construction

## 3. Train MIPGraphNet

Train a single model with seed 42:

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\train_mipgraphnet.py `
  --config configs/default.yaml `
  --seed 42 `
  --run-name mipgraphnet_seed42
```

Main outputs:

```text
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\outputs\checkpoints\mipgraphnet_seed42\best_model.pt
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\outputs\checkpoints\mipgraphnet_seed42\last_model.pt
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\outputs\logs\mipgraphnet_seed42\train_log.csv
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\outputs\metrics\mipgraphnet_seed42\val_metrics.json
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\outputs\metrics\mipgraphnet_seed42\test_metrics.json
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\outputs\metrics\mipgraphnet_seed42\test_metrics.csv
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\outputs\metrics\mipgraphnet_seed42\test_metrics_log.csv
```

`test_metrics.csv` reports raw-unit metrics.

`test_metrics_log.csv` reports log-space metrics.

For properties spanning multiple orders of magnitude, especially `ElectricalConductivity` and `Viscosity`, log-space metrics are usually more informative.

## 4. Multi-Seed Training

For a more robust comparison, train multiple IL-level splits:

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\run_multi_seed.py `
  --config configs/default.yaml `
  --seeds 42,43,44,45,46 `
  --run-prefix mipgraphnet
```

Each seed writes to its own run directory:

```text
outputs/checkpoints/mipgraphnet_42/
outputs/checkpoints/mipgraphnet_43/
...
```

Summary output:

```text
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\outputs\metrics\mipgraphnet_multi_seed_summary.csv
```

## 5. Property-Specific Fine-Tuning

After the initial model is trained, inspect:

```text
outputs/metrics/<run_name>/test_metrics.csv
outputs/metrics/<run_name>/test_metrics_log.csv
```

If one or more properties have weak performance, continue training from the best checkpoint with property-specific weighting.

### 5.1 Fine-Tune SurfaceTension

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\fine_tune_properties.py `
  --config configs/default.yaml `
  --checkpoint outputs\checkpoints\mipgraphnet_seed42\best_model.pt `
  --properties SurfaceTension `
  --seed 42 `
  --run-name finetune_surface_mipgraphnet_seed42 `
  --epochs 100 `
  --lr 0.00005 `
  --focus-weight 8.0 `
  --background-weight 0.1 `
  --monitor-space log `
  --freeze-mode graph_frozen
```

Outputs:

```text
outputs/checkpoints/finetune_surface_mipgraphnet_seed42/best_model.pt
outputs/metrics/finetune_surface_mipgraphnet_seed42/test_metrics.csv
outputs/metrics/finetune_surface_mipgraphnet_seed42/test_metrics_log.csv
```

### 5.2 Fine-Tune ElectricalConductivity

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\fine_tune_properties.py `
  --config configs/default.yaml `
  --checkpoint outputs\checkpoints\mipgraphnet_seed42\best_model.pt `
  --properties ElectricalConductivity `
  --seed 42 `
  --run-name finetune_electrical_mipgraphnet_seed42 `
  --epochs 100 `
  --lr 0.00005 `
  --focus-weight 8.0 `
  --background-weight 0.1 `
  --monitor-space log `
  --freeze-mode graph_frozen
```

### 5.3 Fine-Tune Viscosity

For `Viscosity`, the raw-unit R2 can be dominated by high-viscosity tail samples. Fine-tuning can optionally upweight high-value labels.

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\fine_tune_properties.py `
  --config configs/default.yaml `
  --checkpoint outputs\checkpoints\mipgraphnet_seed42\best_model.pt `
  --properties Viscosity `
  --seed 42 `
  --run-name finetune_viscosity_mipgraphnet_seed42 `
  --epochs 100 `
  --lr 0.00005 `
  --focus-weight 8.0 `
  --background-weight 0.1 `
  --monitor-space log `
  --tail-property Viscosity `
  --tail-threshold-scaled 1.0 `
  --tail-multiplier 3.0 `
  --freeze-mode graph_frozen
```

Use `--monitor-space raw` if the target is raw-unit R2.

Use `--monitor-space log` if the target is log-space R2.

### 5.4 Fine-Tune Multiple Properties Together

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\fine_tune_properties.py `
  --config configs/default.yaml `
  --checkpoint outputs\checkpoints\mipgraphnet_seed42\best_model.pt `
  --properties ElectricalConductivity,SurfaceTension,Viscosity `
  --seed 42 `
  --run-name finetune_electrical_surface_viscosity_seed42 `
  --epochs 100 `
  --lr 0.00005 `
  --focus-weight 8.0 `
  --background-weight 0.1 `
  --monitor-space log `
  --freeze-mode graph_frozen
```

## 6. Evaluate a Model

Evaluate a trained checkpoint on the test split:

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\evaluate_model.py `
  --config configs/default.yaml `
  --checkpoint outputs\checkpoints\mipgraphnet_seed42\best_model.pt `
  --wide-as-main
```

Evaluation outputs:

```text
outputs/metrics/<run_name>/test_metrics.json
outputs/metrics/<run_name>/test_metrics.csv
outputs/metrics/<run_name>/test_metrics_log.csv
outputs/predictions/<run_name>/test_predictions.csv
outputs/predictions/<run_name>/test_predictions_wide.csv
outputs/predictions/<run_name>/test_predictions_long.csv
outputs/figures/<run_name>/parity_*.png
outputs/figures/<run_name>/parity_log_*.png
outputs/figures/<run_name>/residual_*.png
```

If you evaluate a checkpoint whose saved config already contains output directories, the outputs are written according to the checkpoint config.

## 7. Evaluate the Current Recommended Checkpoint

The current reproduced checkpoint with strong log-space metrics is:

```text
outputs/checkpoints/finetune_ElectricalConductivity_SurfaceTension_Viscosity_seed42/best_model.pt
```

Run:

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\evaluate_model.py `
  --config configs/default.yaml `
  --checkpoint outputs\checkpoints\finetune_ElectricalConductivity_SurfaceTension_Viscosity_seed42\best_model.pt `
  --wide-as-main
```

A reproduced evaluation has been exported to:

```text
outputs/predictions/finetune_ElectricalConductivity_SurfaceTension_Viscosity_seed42_eval/
```

Files:

```text
test_predictions.csv
test_predictions_wide.csv
test_predictions_long.csv
test_metrics.json
test_metrics.csv
test_metrics_log.csv
```

## 8. Graph Visualization

### 8.1 Visualize Ionic-Liquid Molecular Structure

This script draws the cation, anion, and complete ion pair.

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\visualize_ion_liquid.py `
  --smiles "C=CC(=O)NC(C)(C)CS(=O)(=O)[O-].COC[P+](c1ccccc1)(c1ccccc1)c1ccccc1" `
  --drawer rdkit
```

Default output:

```text
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\outputs\graphvis\ion_pair_structure.svg
```

### 8.2 Visualize the Actual PyG Graph Used by the Model

This script visualizes the graph constructed by `src/chem/graph_featurizer.py`.

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\visualize_ion_pair_graph.py `
  --smiles "C=CC(=O)NC(C)(C)CS(=O)(=O)[O-].COC[P+](c1ccccc1)(c1ccccc1)c1ccccc1"
```

Default output:

```text
D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\outputs\graphvis\ion_pair_graph.png
```

Graph legend:

- Orange nodes: cation atoms.
- Blue nodes: anion atoms.
- Black solid lines: covalent edges.
- Purple dashed lines: virtual cross-ion edges.

You can adjust the graph appearance:

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\visualize_ion_pair_graph.py `
  --smiles "C=CC(=O)NC(C)(C)CS(=O)(=O)[O-].COC[P+](c1ccccc1)(c1ccccc1)c1ccccc1" `
  --node-size 720 `
  --label-size 10 `
  --edge-width 2.5 `
  --cross-edge-width 2.0 `
  --padding 0.06
```

