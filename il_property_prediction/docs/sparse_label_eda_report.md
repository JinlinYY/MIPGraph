# Sparse-Label Dataset Audit

Date: 2026-06-28

## Dataset

- Source: `data/processed/il_multiprop_clean.csv`
- Rows: 37,556
- Unique ionic liquids: 1,125
- Observed property labels: 50,177 of 225,336
- Label-matrix density: 22.27%
- Validation protocol: fixed test set; augmentation and model selection must occur inside each IL-level training fold only.

## Property Coverage

| Property | All labels | All ILs | Train labels | Train ILs | Safe train interpolation intervals (<=40 K) |
|---|---:|---:|---:|---:|---:|
| Density | 17,855 | 980 | 12,530 | 682 | 11,511 |
| ElectricalConductivity | 4,842 | 404 | 3,660 | 291 | 3,284 |
| HeatCapacity | 9,794 | 263 | 5,959 | 178 | 5,740 |
| SurfaceTension | 5,154 | 402 | 3,697 | 293 | 3,404 |
| ThermalConductivity | 812 | 79 | 626 | 55 | 518 |
| Viscosity | 11,720 | 750 | 8,566 | 512 | 7,828 |

## Findings

1. Temperature interpolation can densify measured curves but does not add chemical diversity.
2. ThermalConductivity is structure-sparse: only 55 training ILs are labeled. Interpolation alone cannot solve unseen-IL generalization.
3. SurfaceTension has no row-level co-labels with the other five properties, so ordinary row-wise matrix completion cannot infer it from observed companion properties.
4. Viscosity has many temperature intervals but remains limited by IL-level distribution shift; duplicating curve points will overweight well-sampled ILs unless sampling is group-balanced.
5. Synthetic and pseudo labels must never be included when computing validation or test metrics.

## Recommended Experiments

1. Physics-constrained, within-curve interpolation for SurfaceTension and ThermalConductivity. Generate points only inside the observed temperature range of the same IL and pressure group. Give synthetic points lower loss weight than measurements.
2. Group-balanced sampling so each IL contributes a bounded total loss weight regardless of the number of measured or synthetic temperatures.
3. Self-supervised molecular encoder pretraining, followed by independent property-head fine-tuning. This uses molecular structures without inventing target values.
4. OOF teacher pseudo-labeling only as a separately flagged semi-supervised experiment. Accept predictions only when independently trained teachers agree, and assign low loss weight.
5. Add new ThermalConductivity ILs from experiments or a clearly marked low-fidelity source. Use multi-fidelity pretraining/fine-tuning instead of merging calculated values with experimental labels as if they had equal fidelity.

## Evaluation Guardrails

- Fit interpolation and pseudo-label generators on the training portion of each fold only.
- Keep original validation and fixed test rows unchanged.
- Select hyperparameters using five-fold IL-level CV.
- Report real-label-only metrics as the primary result.
- Run ablations for interpolation, group balancing, pretraining, and pseudo-labeling separately.
