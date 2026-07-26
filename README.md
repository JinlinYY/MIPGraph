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
|-- experiments/                  # Experiment, analysis, and figure-generation code
|   |-- manuscript_figure_source_data/
|   |                               # Sole authoritative panel-level CSV source-data tree
|   `-- result_analysis/           # Submission figures and figure-to-source-data mapping
|-- il_property_prediction/       # Model package, configs, training and evaluation code
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

Submission figures and their source data have deliberately separate ownership:

```text
experiments/
|-- manuscript_figure_source_data/           # Sole authoritative CSV source-data tree
|   |-- manifest.csv                         # File inventory, dimensions, producer, SHA-256
|   |-- column_dictionary.csv                # Field definitions, units, types, provenance
|   |-- Intro-method/
|   |-- computational_application_case/
|   |-- dataset_statistics/
|   |-- interpretability_feature_importance_4x3/
|   |-- molecular_origin_analysis/
|   `-- performance_results/
`-- result_analysis/                         # Figures and figure metadata only
    |-- figures/<figure_id>/                 # PNG, SVG, PDF, and TIFF deliverables
    |-- manifest.csv                         # Figure-format inventory
    `-- figure_source_map.csv                # Panel-to-canonical-CSV mapping
```

`experiments/manuscript_figure_source_data/` is the only authoritative
location for manuscript panel CSVs. Do not copy source tables into
`experiments/result_analysis/`, `result_fig/`, or a LaTeX figure directory.
`experiments/result_analysis/figure_source_map.csv` links every manuscript
panel to its canonical CSV and records the table shape and SHA-256 digest.

The root source-data catalog can be rebuilt and validated with:

```bash
python experiments/manuscript_figure_source_data/rebuild_manifest.py
```

This command rejects figure files in the source-data tree, byte-identical
duplicate CSVs, missing field definitions, invalid producer paths, and
non-canonical source-data links.

The principal figure and source-data producers are:

```bash
python experiments/dataset_analysis/scripts/plot_dataset_statistics_nature.py
python experiments/dataset_analysis/scripts/export_dataset_statistics_source_data.py
python -m experiments.performance_results.plot_performance_results
python experiments/interpretability/scripts/plot_interpretability_results_current.py
python il_property_prediction/scripts/compute_feature_importance_heatmap.py
python experiments/interpretability/scripts/plot_feature_importance_summary.py --panel-labels j,k,l --color-mode panel
python experiments/interpretability/scripts/compose_interpretability_four_by_three.py
python experiments/computational_application_case/scripts/build_refactored_application_case.py
python experiments/molecular_origin_analysis/run_all.py --stage all
python experiments/molecular_origin_analysis/package_source_data.py
python experiments/manuscript_figure_source_data/rebuild_manifest.py
```

The computational application producer publishes only the 18 panel tables
actually used by Figures 5 and 6 and their application-case Supplementary
Information. The molecular-origin pipeline keeps its panel CSVs under the
same canonical root.

To package a final manuscript figure directory without duplicating CSVs:

```bash
python experiments/result_analysis/scripts/package_manuscript_results.py \
  --manuscript-figure-dir "<path-to-final-manuscript-Fig>"
```

The packager does not retrain a model or change reported statistics. It
collects the approved figure formats, verifies canonical CSV references, and
refreshes the figure manifest and panel mapping.

For smoke tests or alternative output locations, use each script's `--help`.

## Notes On Large Files

This repository uses Git LFS for model checkpoints, binary graph caches, workbooks, PNG/TIFF figures, and other large artifacts. If a large file appears as a small pointer file after cloning, run:

```bash
git lfs pull
```

Training outputs under `outputs/`, local caches, scratch directories, and external pretrained checkpoints are not intended for source control unless explicitly released. Validation-selected MIPGraph checkpoints can be published separately through Git LFS or GitHub Release assets.
