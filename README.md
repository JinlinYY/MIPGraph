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

## Installation

```bash
git lfs install
git clone https://github.com/JinlinYY/MIPGraph.git
cd MIPGraph
git lfs pull
python -m pip install -r il_property_prediction/requirements.txt
```

Use PyTorch and PyTorch Geometric builds compatible with your Python and CUDA environment. Uni-Mol2 pretrained model setup and optional baseline dependencies are documented with the corresponding workflows.

## Repository guide

| Location | Contents |
|---|---|
| `il_property_prediction/src/` | Molecular representation, model, data transforms, training, and metrics |
| `il_property_prediction/configs/` | Model and experiment configurations |
| `il_property_prediction/scripts/` | Preprocessing, training, and prediction entry points |
| `data/` | Data-processing workflow and property tables |
| `experiments/baseline_comparison/` | Descriptor, molecular-language, and graph baselines |
| `experiments/molecular_interpretation/` | Structure–property and attribution analyses |
| `experiments/computational_application_case/` | Candidate screening and reference-cell interpretation |
| `experiments/result_analysis/figures/manuscript_figure_source_data/` | Panel-level numerical source data |
| `experiments/result_analysis/` | Manuscript figures and links to source tables |
