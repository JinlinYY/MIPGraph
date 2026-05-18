# MIPGraph

Implementation for the project accompanying the manuscript:

```text
A Mechanism-Factorized Ion-Pair Graph Learning Framework for Thermophysical Property Prediction of Ionic Liquids
```

This repository provides a complete research codebase for sparse-label multi-property prediction of ionic liquids. The main model, **MIPGraph**, predicts six thermophysical properties from ionic-liquid SMILES, temperature, and pressure:

- Density
- ElectricalConductivity
- HeatCapacity
- SurfaceTension
- ThermalConductivity
- Viscosity


![image](https://github.com/JinlinYY/MIPGraph/blob/main/method.png).

## Overview

MIPGraphNet is designed for ionic-liquid datasets where each sample may contain only a subset of property labels. The training objective uses masked multi-task learning, so missing labels do not contribute to the loss.

The model builds an ion-pair graph from ionic-liquid SMILES. Cation and anion fragments are represented with molecular graph features and approximate single-ion 3D conformer information. Cross-ion virtual interaction edges are constructed from deterministic chemical rules, including formal charge, atom type, hydrogen-bond donor/acceptor compatibility, aromaticity, and cohesive interaction cues. Global ion-pair descriptors are also included to represent molecular size, packing, polarity, flexibility, and shape-related effects.

The learned ion-pair representation is factorized into four mechanism-related latent spaces:

- volumetric-packing
- cohesive-interaction
- transport-friction
- thermal-response

The final decoder uses thermodynamics-inspired structured readout heads plus neural residual corrections. These readouts are physics-inspired response functions, not hard-coded exact physical equations.

## Repository Layout

```text
Sparse-Label-Prediction/
├── README.md
├── data/
│   ├── README.md
│   └── scripts/
└── il_property_prediction/
    ├── README.md
    ├── requirements.txt
    ├── configs/
    ├── scripts/
    └── src/
```

- `data/`: data preparation notes and scripts for merging raw property files, completing SMILES, and optionally filling missing properties from ILThermoPy.
- `il_property_prediction/`: model preprocessing, graph construction, training, fine-tuning, evaluation, and visualization.



## Installation

```powershell
cd D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction
pip install -r requirements.txt
```

For PyTorch Geometric, install the wheel that matches your PyTorch and CUDA version if the default installation does not resolve correctly.

## Data Processing

See the data workflow document:

```text
data/README.md
```

Model-side preprocessing:

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\preprocess_data.py `
  --config configs/default.yaml
```

Build the graph cache:

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\build_graph_cache.py `
  --config configs/default.yaml
```

The graph cache is saved to:

```text
il_property_prediction/data/processed/graph_cache.pt
```

## Training

Train MIPGraphNet:

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\train_mipgraphnet.py `
  --config configs/default.yaml `
  --seed 42 `
  --run-name mipgraphnet_seed42
```

Run multiple IL-level split seeds:

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\run_multi_seed.py `
  --config configs/default.yaml `
  --seeds 42,43,44,45,46 `
  --run-prefix mipgraphnet
```

## Property-Specific Fine-Tuning

For properties with weaker performance, fine-tune from a trained checkpoint:

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\fine_tune_properties.py `
  --config configs/default.yaml `
  --checkpoint outputs\checkpoints\mipgraphnet_seed42\best_model.pt `
  --properties Viscosity `
  --seed 42 `
  --run-name finetune_viscosity `
  --epochs 100 `
  --lr 0.00005 `
  --focus-weight 8.0 `
  --background-weight 0.1 `
  --freeze-mode graph_frozen
```

## Evaluation

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\evaluate_model.py `
  --config configs/default.yaml `
  --checkpoint outputs\checkpoints\mipgraphnet_seed42\best_model.pt `
  --wide-as-main
```

Evaluation outputs include metrics, predictions, and parity plots. Metrics are computed after inverse-transforming predictions back to original physical units; log-space metrics are also exported when enabled.

## Visualization

Visualize an ionic-liquid structure:

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\visualize_ion_liquid.py `
  --smiles "C=CC(=O)NC(C)(C)CS(=O)(=O)[O-].COC[P+](c1ccccc1)(c1ccccc1)c1ccccc1"
```

Visualize the constructed ion-pair graph:

```powershell
python D:\GGNN\IL-model\Sparse-Label-Prediction\il_property_prediction\scripts\visualize_ion_pair_graph.py `
  --smiles "C=CC(=O)NC(C)(C)CS(=O)(=O)[O-].COC[P+](c1ccccc1)(c1ccccc1)c1ccccc1"
```

Graph visualizations are saved under:

```text
il_property_prediction/outputs/graphvis
```

