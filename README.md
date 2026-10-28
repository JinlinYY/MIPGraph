# MIPGraph

**Ion-Pair Representation Learning for Sparse Multi-Property Prediction and Ionic-Liquid Screening**

MIPGraph predicts ionic-liquid thermophysical properties from cation–anion structures, temperature, and pressure. It combines ion-pair representation learning with sparse multi-property training to connect molecular structure, thermodynamic conditions, and property prediction.

![MIPGraph framework](Intro-method.png)

## Predicted properties

| Property | Output unit |
|---|---|
| Density | kg m⁻³ |
| Viscosity | Pa s |
| Electrical conductivity | S m⁻¹ |
| Molar heat capacity | J mol⁻¹ K⁻¹ |
| Surface tension | N m⁻¹ |
| Thermal conductivity | W m⁻¹ K⁻¹ |

The benchmark contains 37,556 thermodynamic-condition records covering 1,125 ionic liquids. Each record can contain any subset of the six measured properties.

## Method

- A shared frozen Uni-Mol2 encoder represents cation and anion structures.
- Cross-ion attention, global descriptors, and functional-group descriptors characterize the complete ion pair.
- A Transformer fuses the molecular representations, while temperature and pressure modulate them through FiLM.
- Property-specific routing combines packing, cohesion, transport, and thermal-response experts to predict six properties.
- Masked training uses available labels and source-reliability weights for copied and interpolated training observations.

## Data and evaluation

The evaluation uses four fixed partitions: random-point, random IL-level, property-balanced IL-level, and family-pair-disjoint. Training is repeated with seeds 0, 42, and 123.

Target-standardization statistics are fitted only to original experimental observations in the training partition. They are shared across the four label-completion configurations. Copied and interpolated labels contribute to training; validation and test metrics use original experimental observations.

For each property, NMAE is the MAE in natural-log target space divided by its training-set standardization scale. Macro NMAE is the unweighted mean over the six properties. Detailed experimental results and statistical analyses are reported in the manuscript and Supporting Information.

## Installation

```bash
git lfs install
git clone https://github.com/JinlinYY/MIPGraph.git
cd MIPGraph
git lfs pull
python -m pip install -r il_property_prediction/requirements.txt
```

Use PyTorch and PyTorch Geometric builds compatible with your Python and CUDA environment. Uni-Mol2 pretrained model setup and optional baseline dependencies are documented with the corresponding workflows.

## Research workflows

The codebase provides the components for:

1. Preparing ionic-liquid identities, condition records, property masks, and fixed partitions.
2. Training and evaluating MIPGraph and the eight baseline models.
3. Reproducing label-completion sensitivity and architectural ablation studies.
4. Computing repeated-run statistics and test-set coverage.
5. Performing molecular interpretation and ionic-liquid candidate screening.
6. Recreating manuscript figures from their panel-level source data.

## Repository guide

| Location | Contents |
|---|---|
| `il_property_prediction/src/` | Molecular representation, model, data transforms, training, and metrics |
| `il_property_prediction/configs/` | Model and experiment configurations |
| `il_property_prediction/scripts/` | Preprocessing, training, and prediction entry points |
| `data/` | Data-processing workflow and property tables |
| `experiments/baseline_comparison/` | Descriptor, molecular-language, and graph baselines |
| `experiments/molecular_origin_analysis/` | Structure–property and attribution analyses |
| `experiments/computational_application_case/` | Candidate screening and reference-cell interpretation |
| `experiments/manuscript_figure_source_data/` | Panel-level numerical source data |
| `experiments/result_analysis/` | Manuscript figures and per-figure data |

## Figures and source data

Each quantitative figure is accompanied by the numerical tables used to draw its panels. Performance figures use the three-seed results reported in the manuscript. The application figures include the candidate identities, temperature-resolved predictions, and derived quantities used for screening and reference-cell interpretation.

The framework illustration is a conceptual schematic with a panel inventory rather than a numerical dataset.

## Citation

Please cite the accompanying manuscript, **Ion-Pair Representation Learning for Sparse Multi-Property Prediction and Ionic-Liquid Screening**, when using MIPGraph or its associated research data. Bibliographic details will be updated with the publication record.
