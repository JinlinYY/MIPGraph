# Repository audit for the computational application case

Audit date: 2026-07-18

## Deliberate exclusion

web/ is a deprecated implementation and was deliberately excluded from model loading, preprocessing, inference, and result generation.

No file below that directory was used to infer an API, model shape, preprocessing rule, descriptor, graph, cache, scaler, checkpoint, or result. The implementation described here uses only the current `il_property_prediction/`, `data/`, and `experiments/` trees.

## Current model artefacts

- Source workbook recorded by the selected checkpoint: `data/processed/ionic_liquid_6_properties_values_errors.xlsx`, sheet `Merged`.
- Model-ready benchmark: `il_property_prediction/data/processed/il_multiprop_clean.csv` (37,556 condition rows; 1,125 unique ionic liquids).
- Model arrays: `il_property_prediction/data/processed/il_multiprop_arrays.npz`.
- Training/validation/test split: `il_property_prediction/data/processed/splits/row_level_seed42.json`.
- Current external configuration: `il_property_prediction/configs/physics_moe_fg_transformer.yaml`.
- Selected six-property checkpoint: `il_property_prediction/outputs/fg_transformer_random_point_seed42/checkpoints/unimol2_fg_transformer_random_point_seed42/best_model_pid68732_epoch012.pt`.
- Graph cache: `il_property_prediction/data/processed/graph_cache_fg.pt` (1,125 ion-pair graphs).
- Uni-Mol2 ion cache: `il_property_prediction/data/processed/unimol2_ion_features.pt` (640 ions, including atom-chemistry annotations).
- Uni-Mol2 pretrained weights: `il_property_prediction/data/pretrained/unimol2/modelzoo/84M/checkpoint.pt`.

The local release tree also contains property-specialist checkpoints selected separately for different split protocols. They are not a single compatible ensemble and are not silently merged. The checkpoint above is the only checkpoint under the current `il_property_prediction/outputs/` tree and stores one jointly trained six-output model, its complete model configuration, both scalers, and the six-property order. It is therefore selected explicitly in this case configuration. Any override must be supplied explicitly with `--checkpoint` or `--checkpoints`.

## Current call chain

1. `il_property_prediction/src/chem/smiles_utils.py::split_ion_pair` splits dot-separated fragments by formal charge. When charge assignment is ambiguous it falls back to source fragment order and records a warning.
2. `canonicalize_smiles` uses RDKit `MolFromSmiles` followed by canonical `MolToSmiles`.
3. `il_property_prediction/src/chem/graph_featurizer.py::build_ion_pair_graph` builds the molecular graph with 45-dimensional atom features, 12-dimensional edge features, single-ion conformers, and deterministic 2D virtual cross-ion edges for the selected checkpoint.
4. The graph builder calls `ion_pair_descriptors` for 56 global descriptors and `ion_pair_functional_group_descriptors` for 80 functional-group descriptors.
5. `il_property_prediction/src/models/unimol2_ion_encoder.py::UniMol2IonEncoder` retrieves cached, canonical ion-level Uni-Mol2 inputs and atom-chemistry annotations, then applies the frozen 84M Uni-Mol2 backbone.
6. `il_property_prediction/src/models/factory.py::build_model` constructs `il_property_prediction/src/models/mipgraph.py::MIPGraph`.
7. `MIPGraph.forward` performs ion encoding, atom-level cross-attention, transformer ion-pair fusion, condition encoding, physics-MoE routing, and independent six-property decoding.
8. `il_property_prediction/src/data/scaler.py::ConditionScaler.transform` standardizes temperature and pressure. Missing pressure is filled with the checkpoint training median (101.32499694824219 kPa).
9. The checkpoint `model_state_dict` is loaded and checked for missing and unexpected keys before inference.
10. `TargetScaler.inverse_transform` maps scaled outputs to physical units as `exp(y_scaled * std + mean) - eps`, with `eps = 1e-8`.

The checkpoint schema contains `model_state_dict`, `config`, `condition_scaler`, `target_scaler`, `property_names`, `target_means`, `target_stds`, `epoch`, and `metrics`. Its condition scaler is:

- temperature mean: 325.61469629213644 K;
- temperature standard deviation: 41.291560770432476 K;
- pressure median: 101.32499694824219 kPa;
- pressure mean: 248.34947607927077 kPa;
- pressure standard deviation: 2474.111183622406 kPa.

## Property order, transforms, and physical units

The stored property order is:

1. `Density` — kg m^-3;
2. `ElectricalConductivity` — S m^-1;
3. `HeatCapacity` — J mol^-1 K^-1;
4. `SurfaceTension` — N m^-1;
5. `ThermalConductivity` — W m^-1 K^-1;
6. `Viscosity` — Pa s.

All six targets are transformed with the natural logarithm before standardization. Unit metadata is corroborated by `experiments/dataset_analysis/scripts/export_dataset_statistics_source_data.py`, the source-table magnitudes, and the current application figure labels. Heat capacity is molar and must be divided by RDKit molecular mass in kg mol^-1 exactly once before computing volumetric heat capacity.

## New-pair inference feasibility

Unseen cation-anion recombinations are feasible without changing the core model when both component ions already occur in the current Uni-Mol2 ion cache. The application adapter dynamically calls the current graph builder for the new pair, while using the original ion strings as the Uni-Mol2 cache keys. It does not substitute a nearest-neighbour graph or property value.

A truly unseen ion is not supported by the current runtime cache. The formal generator is `il_property_prediction/scripts/build_unimol2_ion_cache.py`, but that script builds from a graph cache and refuses to overwrite its output without `--force`. This application does not overwrite the training cache. One-ion extrapolation therefore remains disabled unless an explicit external ion library and an application-local cache generation run are supplied.

## Adapter responsibilities

The application-local adapter must:

- load and validate the checkpoint-embedded configuration and scalers;
- rebase only stale absolute artefact paths to the audited current project paths;
- call the current model factory and current graph builder;
- preserve graph fields, tensor shapes, dtypes, ion cache keys, raw conditions, and property order;
- save only application-local graph/cache outputs;
- inverse-transform predictions with the stored target scaler;
- expose global/functional-group descriptors and, when available without core modification, the model `h_structure` representation;
- record every failed candidate and stage without replacement predictions.

## Potential issues found

- The selected historical transformer-fusion checkpoint retains dormant `interaction.*` and `interaction_fusion.*` tensors from the older constructor. Current official evaluation scripts load checkpoints with `strict=False`. This adapter permits only those audited surplus prefixes for this transformer configuration, still requires every current model key, rejects every other unexpected key, and records the exact ignored list.
- The repository contains multiple release property-specialist checkpoints with different validation selections; they must not be treated as an automatic ensemble.
- The selected checkpoint embeds absolute Windows paths. The adapter validates them and uses audited project-relative fallbacks when the embedded path does not exist.
- The preprocessing table contains one suspicious 101.325 K temperature record, so raw min/max alone is not a robust temperature-domain diagnostic. The case preserves it in audits and uses the configured screening window without rewriting source data.
- Thermal-conductivity labels are sparse (812 labels in the selected non-interpolated preprocessing report), so downstream conclusions must retain a coverage limitation.
- The current dataset loader only accepts graph-cache keys. Dynamic pair graphs must therefore be assembled by the application adapter rather than by changing the training dataset.
- Full reference-domain embedding extraction is not precomputed. Descriptor-distance AD remains mandatory; embedding distance is reported unavailable unless it can be obtained for the actual reference set during the run.

## Implementation decision

The case is implemented as an isolated adapter and analysis pipeline under `experiments/computational_application_case/`. It reuses current chemistry, graph, descriptor, model factory, scaler, checkpoint, and Uni-Mol2 code. It generates observed references and charge-valid unseen recombinations from the audited training/benchmark records, predicts all requested temperatures, performs unit and curve audits before proxy screening, calibrates descriptor kNN AD thresholds only on the configured reference domain, applies whole-window constraints fixed by observed-reference quantiles, performs non-dominated sorting, and generates data-driven figures/tables/reports. No existing project file needs modification.
