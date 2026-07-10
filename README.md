# MIPGraph

MIPGraph is a condition-aware ion-pair graph learning framework for sparse multi-property prediction of ionic-liquid thermophysical properties. This repository accompanies the manuscript:

```text
An Ion-Pair Graph Learning Framework with Multiphysics Mixture-of-Experts
for Sparse Multi-Property Prediction of Ionic-Liquid Thermophysical Properties
```

The model predicts six properties from ionic-liquid structure and thermodynamic conditions:

- Density
- Viscosity
- Electrical conductivity
- Heat capacity
- Surface tension
- Thermal conductivity



![MIPGraph framework](https://github.com/JinlinYY/MIPGraph/blob/main/Intro-method.png)


## Repository Layout

```text
.
|-- data/                         # Data curation scripts and processed IL tables
|-- experiments/                  # Baseline, dataset, performance, and figure scripts
|-- il_property_prediction/       # Model package, configs, training and evaluation code
|-- result_fig/                   # Manuscript figures and panel-level source CSV files
|-- scripts/                      # Release and maintenance helper scripts
|-- README.md
```

Large generated training outputs, scratch files, and local caches are ignored by git. Result figures and binary data are stored with Git LFS when committed.

## Installation

Clone with Git LFS enabled:

```bash
git lfs install
git clone https://github.com/JinlinYY/MIPGraph.git
cd MIPGraph
git lfs pull
```

Create an environment and install the model package requirements:

```bash
cd il_property_prediction
pip install -r requirements.txt
```

Install PyTorch and PyTorch Geometric wheels that match your CUDA/Python environment if the generic entries in `requirements.txt` do not resolve a compatible build. The experiment scripts also use `matplotlib`, `seaborn`, `scikit-learn`, `rdkit`, and `joblib`; `xgboost`, `lightgbm`, and `transformers` are only needed for those baselines.

## Data

The main processed workbook is:

```text
data/processed/ionic_liquid_6_properties_values_errors_ilthermo_strict_v2_interpolated.xlsx
```

Model-ready files are under:

```text
il_property_prediction/data/processed/
il_property_prediction/data/processed_ilthermo_interpolated/
```

The raw and processed data workflow is documented in:

```text
data/README.md
```

## Preprocessing

From `il_property_prediction/`:

```bash
python scripts/preprocess_data.py --config configs/default.yaml
python scripts/build_graph_cache.py --config configs/default.yaml
python scripts/add_functional_groups_to_graph_cache.py
python scripts/build_unimol2_ion_cache.py --config configs/physics_moe_fg_transformer.yaml
```

The Uni-Mol2 encoder is treated as a frozen feature extractor in the reported experiments. If the pretrained Uni-Mol2 checkpoint is not included in your clone, download it from the original Uni-Mol2 source and place it at the path expected by the config.

## Training

Random-point MIPGraph training:

```bash
cd il_property_prediction
python scripts/train_mipgraphnet.py \
  --config configs/physics_moe_fg_transformer.yaml \
  --seed 42 \
  --run-name mipgraph_random_point_seed42
```

Four-split retraining:

```bash
python scripts/run_mipgraph_four_split_retraining.py
```

Property-specialist and adapter workflows are available in:

```text
il_property_prediction/scripts/run_property_specialists.py
il_property_prediction/scripts/run_property_adapter_specialists.py
il_property_prediction/scripts/merge_property_specialists.py
```

## Evaluation

```bash
cd il_property_prediction
python scripts/evaluate_model.py \
  --config configs/physics_moe_fg_transformer.yaml \
  --checkpoint outputs/checkpoints/mipgraph_random_point_seed42/best_model.pt \
  --wide-as-main
```

Evaluation exports metrics, predictions, and optional parity plots under `il_property_prediction/outputs/`.

## Baselines

Baseline scripts are in:

```text
experiments/baseline_comparison/
```

Dry run:

```bash
python experiments/baseline_comparison/scripts/run_baseline_comparison.py --dry-run
```

Run selected baselines:

```bash
python experiments/baseline_comparison/scripts/run_baseline_comparison.py \
  --models rf,xgboost,lgbm,mpnn_concat,gcn,gat,graphsage,gin
```

Summarize split-strategy results:

```bash
python experiments/baseline_comparison/scripts/summarize_split_baseline_results.py
```

## Reproducing Manuscript Figures

The final figures and CSV source data are stored in:

```text
result_fig/
```

The figure-generation scripts are retained under `experiments/`:

```bash
python experiments/dataset_analysis/scripts/plot_dataset_statistics_nature.py
python experiments/result_analysis/scripts/plot_performance_results_with_splits.py
python experiments/interpretability/scripts/plot_interpretability_results_current.py
python experiments/interpretability/scripts/plot_feature_importance_summary.py --panel-labels j,k,l --color-mode panel
python experiments/interpretability/scripts/compose_interpretability_four_by_three.py
python experiments/application_case/scripts/plot_application_case_agent_merged.py
```

For smoke tests or alternative output locations, use each script's `--help`.

## Application Case

The design-agent example is implemented in:

```text
il_property_prediction/scripts/run_design_agent.py
il_property_prediction/src/agent/
experiments/application_case/scripts/plot_application_case_agent_merged.py
```

It converts predicted thermophysical properties for candidate IL-condition records into response atlases, conductivity-viscosity trade-offs, constraint maps, Pareto screening, and ranked operating recommendations.

