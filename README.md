# MIPGraph

**Deciphering Molecular Origins of Ionic-Liquid Thermophysical Properties through Ion-Pair Representation Learning**

MIPGraph is a condition-aware graph-learning framework for predicting multiple thermophysical properties of ionic liquids (ILs) from molecular structure, temperature, and pressure. It represents an IL as an interacting cation–anion pair and learns shared information across sparsely observed properties while retaining property-specific decoding.

This repository contains the data-processing workflow, model implementation, training and evaluation scripts, baseline studies, interpretation analyses, and computational application used in the accompanying manuscript:

> **Deciphering Molecular Origins of Ionic-Liquid Thermophysical Properties through Ion-Pair Representation Learning**

![MIPGraph framework](Intro-method.png)

## Scientific Motivation

Ionic-liquid design involves a large combinatorial space of cations, anions, substituents, and operating conditions. Experimental thermophysical data are therefore sparse and strongly imbalanced:

- different properties are available for different ILs;
- measurements are reported at heterogeneous temperatures and pressures;
- some ion families are much better represented than others;
- optimizing one property can degrade another.

Conventional single-property regressors do not fully exploit information shared across properties, while representations that treat the two ions independently can miss chemically relevant cation–anion interactions. MIPGraph addresses these limitations by combining ion-pair molecular representation, thermodynamic-condition modulation, and sparse multi-property learning in one framework.

## What MIPGraph Predicts

Given cation and anion structures together with temperature and pressure, MIPGraph predicts six properties in one forward pass:

| Property | Symbol | Model output unit |
|---|---:|---:|
| Density | ρ | kg m⁻³ |
| Viscosity | η | Pa s |
| Electrical conductivity | σ | S m⁻¹ |
| Molar heat capacity | Cₚ,ₘ | J mol⁻¹ K⁻¹ |
| Surface tension | γ | N m⁻¹ |
| Thermal conductivity | k | W m⁻¹ K⁻¹ |

The manuscript benchmark contains **37,556 condition records** covering **1,125 unique ionic liquids**. Missing property labels are handled directly rather than requiring a complete six-property table for every record.

## Model Overview

MIPGraph combines four modeling components:

1. **Chemistry-resolved ion inputs**

   Cation and anion molecular graphs, global RDKit descriptors, and functional-group descriptors preserve ion-specific structural information.

2. **Explicit cross-ion interaction modeling**

   Cross-ion attention and chemically typed interaction features allow the representation to depend on the complete ion pair rather than on two independently pooled fragments.

3. **Condition-aware representation learning**

   Temperature and pressure embeddings modulate the shared molecular representation so that predictions describe property responses under specified thermodynamic conditions.

4. **Mechanism-factorized multi-property decoding**

   Shared latent information is routed through property-specific experts associated with packing, cohesion, transport, and thermal response. Sparse-label optimization updates only the properties available for each training record.

The implementation uses PyTorch, PyTorch Geometric, RDKit, and frozen Uni-Mol2 ion representations.

## Manuscript Evaluation

The repository supports the evaluation chain reported in the paper:

- **random-point split** for condition-level interpolation;
- **whole-IL split** for unseen ionic-liquid evaluation;
- **property-balanced whole-IL split** for label-balanced chemical generalization;
- **ion-family split** for the stricter family-transfer setting;
- baseline comparisons, ablation studies, sparse multi-task analysis, and molecular-interpretation experiments.

These protocols are intentionally reported separately. Performance under a random-point split should not be interpreted as performance on unseen ionic liquids or unseen ion families.

## Molecular Interpretation

The manuscript links learned model behavior to molecular structure using:

- condition-adjusted structure–property associations;
- atom-node and cross-ion interaction attributions;
- functional-group importance;
- ion-substitution sensitivity;
- response-shape analysis across structural-factor quantiles.

These analyses identify statistical and model-supported associations. They are not presented as causal physical laws, and exploratory results are explicitly labeled where data coverage is limited.

## Computational Application

The application study demonstrates how MIPGraph predictions can be converted into an auditable candidate-prioritization workflow for neat ionic-liquid electrolytes.

The prospective candidate space is restricted to:

```text
seen cation + seen anion -> previously unseen ion-pair recombination
```

Candidates are identity-audited, checked against the model applicability domain, evaluated over a defined temperature window, filtered by prespecified multi-property constraints, and ranked with deterministic Pareto and stability analyses. A fixed 60 s reference-cell-conditioned scenario is applied only after the formal shortlist has been selected.

This case illustrates **thermophysical pre-screening and experiment prioritization**. It does not establish liquid-range persistence, electrochemical stability, capacitance, energy or power density, cycle life, electrode compatibility, or device safety.

## Repository Structure

```text
MIPGraph/
|-- data/                                  # Data curation and processed benchmark tables
|-- il_property_prediction/
|   |-- configs/                           # Model and experiment configurations
|   |-- scripts/                           # Preprocessing, training, and evaluation entry points
|   `-- src/                               # MIPGraph model implementation
|-- experiments/
|   |-- baseline_comparison/               # Conventional and graph-learning baselines
|   |-- dataset_analysis/                  # Dataset statistics
|   |-- performance_results/               # Formal split-protocol performance figure
|   |-- interpretability/                  # Model interpretation analyses
|   |-- molecular_origin_analysis/         # Structure–property evidence analysis
|   |-- computational_application_case/    # Auditable electrolyte pre-screening case
|   |-- manuscript_figure_source_data/     # Authoritative panel-level source data
|   `-- result_analysis/                   # Manuscript figures and panel-to-data map
`-- Intro-method.png                       # Model and workflow overview
```

## Installation

Clone the repository with Git LFS enabled:

```bash
git lfs install
git clone https://github.com/JinlinYY/MIPGraph.git
cd MIPGraph
git lfs pull
```

Create a Python environment and install the core dependencies:

```bash
cd il_property_prediction
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric must match the local Python and CUDA versions. The main workflows also require RDKit, pandas, NumPy, scikit-learn, Matplotlib, seaborn, and joblib. Some baseline models require optional packages such as XGBoost, LightGBM, or Transformers.

## Data Preparation

The curated multi-property workbook is located at:

```text
data/processed/ionic_liquid_6_properties_values_errors_ilthermo_strict_v2_interpolated.xlsx
```

From `il_property_prediction/`, prepare the model inputs with:

```bash
python scripts/preprocess_data.py --config configs/default.yaml
python scripts/build_graph_cache.py --config configs/default.yaml
python scripts/add_functional_groups_to_graph_cache.py
python scripts/build_unimol2_ion_cache.py --config configs/physics_moe_fg_transformer.yaml
```

The complete data workflow and curation notes are documented in [`data/README.md`](data/README.md).

## Training

Train the random-point MIPGraph configuration:

```bash
cd il_property_prediction
python scripts/train_mipgraphnet.py \
  --config configs/physics_moe_fg_transformer.yaml \
  --seed 42 \
  --run-name mipgraph_random_point_seed42
```

Run the four manuscript split protocols with:

```bash
python scripts/run_mipgraph_four_split_retraining.py
```

The Uni-Mol2 encoder is used as a frozen feature extractor. A compatible pretrained Uni-Mol2 checkpoint must be placed at the path declared by the selected configuration.

Evaluation and application workflows likewise require a compatible trained MIPGraph checkpoint. If a released checkpoint is not present in the clone, train the corresponding split locally and update the checkpoint path in the selected configuration.

## Evaluation

Evaluate a trained checkpoint:

```bash
cd il_property_prediction
python scripts/evaluate_model.py \
  --config configs/physics_moe_fg_transformer.yaml \
  --checkpoint outputs/checkpoints/mipgraph_random_point_seed42/best_model.pt \
  --wide-as-main
```

Evaluation outputs include property-level metrics, predictions, and optional parity plots.

## Reproducing Manuscript Analyses

Run these commands from the repository root unless noted otherwise.

Dataset overview:

```bash
python experiments/dataset_analysis/scripts/plot_dataset_statistics_nature.py
python experiments/dataset_analysis/scripts/export_dataset_statistics_source_data.py
```

Formal performance figure:

```bash
python -m experiments.performance_results.plot_performance_results
```

Interpretability and molecular-origin analyses:

```bash
python experiments/interpretability/scripts/plot_interpretability_results_current.py
python il_property_prediction/scripts/compute_feature_importance_heatmap.py
python experiments/interpretability/scripts/plot_feature_importance_summary.py --panel-labels j,k,l --color-mode panel
python experiments/interpretability/scripts/compose_interpretability_four_by_three.py
python experiments/molecular_origin_analysis/run_all.py --stage all
python experiments/molecular_origin_analysis/package_source_data.py
```

Computational application:

```bash
python experiments/computational_application_case/run_all.py \
  --config experiments/computational_application_case/configs/auditable_virtual_screening.yaml
python experiments/computational_application_case/scripts/build_protocol_stability_outputs.py --force
python experiments/computational_application_case/scripts/build_refactored_application_case.py
```

The performance, molecular-origin, and computational-application directories contain dedicated READMEs describing their assumptions, inputs, outputs, and scientific boundaries.

## Figure Source Data

Authoritative panel-level CSV files for the manuscript figures are stored under:

```text
experiments/manuscript_figure_source_data/
```

Submission figures and the panel-to-source-data map are stored under:

```text
experiments/result_analysis/
```

The source-data manifest and field dictionary can be validated with:

```bash
python experiments/manuscript_figure_source_data/rebuild_manifest.py
```

## Citation

If MIPGraph is useful in your research, please cite the accompanying manuscript:

> Jinlin Ye *et al.* **Mechanism-Factorized Ion-Pair Graph Learning for Multi-Property Prediction of Ionic-Liquid Thermophysical Properties.** Manuscript in preparation.

A DOI and final BibTeX entry will be added after publication.

## Scope and Responsible Use

MIPGraph is a research model for thermophysical-property prediction and hypothesis prioritization within its represented chemical and thermodynamic domains. Predictions should be accompanied by identity checks, applicability-domain assessment, and direct experimental validation before material, process, electrochemical, or safety claims are made.
