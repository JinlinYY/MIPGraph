# MIPGraph molecular-origin analysis

This directory is a non-invasive post-processing module for connecting frozen
MIPGraph outputs with auditable molecular structure–property hypotheses. It
loads the original data, split, model class, and checkpoint through adapters;
it does not retrain MIPGraph or modify the original project.

The workflow keeps three evidence classes separate:

1. **Observed-data association**: condition-controlled and family-stratified
   relations between measured properties and molecular descriptors.
2. **Frozen-model sensitivity**: gradient×input and descriptor-surrogate
   permutation analyses of MIPGraph responses.
3. **Counterfactual prediction**: MIPGraph responses for RDKit-valid,
   charge-balanced ion-pair structures, qualified by an applicability-domain
   check.

Agreement among these classes can support a prioritization hypothesis, but it
does not establish causality, interaction energy, liquid-state persistence, or
electrochemical performance.

## Relationship to the original project

All new code and outputs are below
`experiments/molecular_origin_analysis/`. `ProjectAdapter`, `DataAdapter`, and
`ModelAdapter` import and call the original implementation read-only. The
module does not patch the original model, data, configuration, checkpoints,
screening results, manuscript, figures, or Supporting Information.

The formal run found only one compatible checkpoint:
`random_point_seed42`. The requested random-IL, property-balanced, and
ion-family checkpoints were not present. The executed results therefore use
the random-point checkpoint as an explicitly recorded fallback; multi-
checkpoint stability is reported as not assessable.

## Directory layout

```text
molecular_origin_analysis/
├── config/                  # analysis configuration
├── scripts/                 # stage-specific command-line entrypoints
├── src/                     # adapters and analysis implementations
├── templates/               # real-SMILES counterfactual definitions
├── tests/                   # adapter, cache, chemistry, and smoke tests
├── results/
│   ├── cache/               # aligned predictions and intermediate tensors
│   ├── tables/              # analysis tables and figure source data
│   ├── figures/             # PNG, PDF, and SVG figures
│   ├── logs/                # timestamped pipeline log
│   ├── reports/             # inspection, provenance, and audit reports
│   └── manuscript/          # independent writing suggestions
├── requirements_extra.txt
└── run_all.py
```

## Environment

Use the environment that can already import the original MIPGraph project.
The formal run used Python 3.9, PyTorch, RDKit, NumPy, Pandas,
scikit-learn, SciPy, Matplotlib, PyYAML, and PyArrow. Install only the
additional serialization/analysis dependencies if missing:

```powershell
python -m pip install -r experiments\molecular_origin_analysis\requirements_extra.txt
```

No package installation is performed automatically.

## Configuration

The default configuration is
[`config/analysis_config.yaml`](config/analysis_config.yaml). Important fields
include:

- `project.root` and `project.output_root`;
- `model.checkpoint`, `model.primary_checkpoint_type`, `model.device`, and
  `model.batch_size`;
- `data.split`, `data.include_train_for_reference`, and `data.max_samples`;
- reference temperatures, statistical thresholds, attribution settings,
  applicability-domain quantiles, and output formats.

Use `model.checkpoint: auto` for audited discovery, or set an absolute
checkpoint path. To switch checkpoints without editing YAML:

```powershell
python experiments\molecular_origin_analysis\run_all.py `
  --set model.checkpoint="D:/path/to/checkpoint.pt" --force
```

The matching model configuration and split must remain compatible with that
checkpoint. Selection decisions and limitations are written to
`results/reports/project_inspection.md`.

The data locations are auto-discovered from the original project. They can be
changed through dotted overrides when the corresponding configuration key is
available:

```powershell
python experiments\molecular_origin_analysis\run_all.py `
  --set data.split=test --set data.max_samples=32
```

`--set SECTION.KEY=VALUE` may be repeated and is supported by the unified and
stage-specific commands.

## Running on Windows

From the project root in PowerShell or CMD:

```powershell
python experiments\molecular_origin_analysis\run_all.py `
  --config experiments\molecular_origin_analysis\config\analysis_config.yaml
```

Run a single stage:

```powershell
python experiments\molecular_origin_analysis\run_all.py --stage inspect
python experiments\molecular_origin_analysis\run_all.py --stage extract
python experiments\molecular_origin_analysis\run_all.py --stage association
python experiments\molecular_origin_analysis\run_all.py --stage attribution
python experiments\molecular_origin_analysis\run_all.py --stage attention
python experiments\molecular_origin_analysis\run_all.py --stage counterfactual
python experiments\molecular_origin_analysis\run_all.py --stage applicability
python experiments\molecular_origin_analysis\run_all.py --stage rules
python experiments\molecular_origin_analysis\run_all.py --stage figures
python experiments\molecular_origin_analysis\run_all.py --stage manuscript
python experiments\molecular_origin_analysis\run_all.py --stage validate
```

Equivalent wrappers are available in `scripts/`, for example:

```powershell
python experiments\molecular_origin_analysis\scripts\extract_model_outputs.py `
  --config experiments\molecular_origin_analysis\config\analysis_config.yaml
```

Use `--force` after changing a checkpoint or an upstream analysis setting.
For a lightweight smoke run, add
`--max-samples 32 --set statistics.bootstrap_repeats=20`.

## Cache and reproducibility

Extraction cache metadata records the configuration and checkpoint hashes.
The formal manifest also records the Git commit, data split, software versions,
random seed, and stage status:

`results/reports/reproducibility_manifest.json`.

The main cached outputs are:

- `model_outputs.parquet`: aligned identities, conditions, labels, masks, and
  inverse-transformed predictions;
- `descriptor_matrix.parquet`: named global, functional-group, and pair
  descriptors;
- `latent_representations.npz`: batched cation, anion, pair, condition, and
  expert representations;
- `router_weights.parquet`;
- `cross_ion_attention/cross_ion_attention_summary.parquet`.

Do not manually combine caches generated from different checkpoints.

## Results

### Manuscript-facing consolidated figure

The main-text result is the integrated six-panel figure:

```text
results/figures/figure_main_molecular_origin_analysis.png
results/figures/figure_main_molecular_origin_analysis.pdf
results/figures/figure_main_molecular_origin_analysis.svg
results/figures/figure_main_molecular_origin_analysis.tiff
```

Panels a–f respectively report the strongest evidence-gated association per
property, the condition-controlled association matrix, observed response
curves, matched-pair and counterfactual evidence, shared-attention diagnostics,
and the post hoc connection to the unchanged Top-8. The original standalone
Figures A–F remain available as auditable component views; they are not
required as six separate main-text figures.

Each composite panel has an independent source-data CSV under
`results/tables/figure_source_data/`. The figure contract and export audit are
stored in `results/reports/composite_figure_contract.md` and
`results/reports/composite_figure_qa.md`.

The manuscript-ready analysis and captions are:

```text
results/manuscript/molecular_origin_results_section_en.tex
results/manuscript/molecular_origin_results_section_zh.tex
results/manuscript/composite_figure_caption_en.tex
results/manuscript/composite_figure_caption_zh.tex
```

Key tables:

- `feature_property_associations.csv`;
- `partial_correlations.csv`;
- `family_stratified_associations.csv`;
- `robust_structure_property_factors.csv`;
- `property_feature_importance.csv`;
- `attribution_method_agreement.csv`;
- `matched_molecular_pairs.csv`;
- `virtual_counterfactual_library.csv`;
- `counterfactual_predictions.csv`;
- `observed_test_applicability_domain.csv`;
- `design_rule_summary.csv`;
- `unsupported_hypotheses.csv`;
- `candidate_structural_profiles.csv`;
- `candidate_rule_consistency.csv`.

Standalone Figures A–F are exported as 600-dpi PNG and vector PDF/SVG. The
consolidated figure additionally includes a 600-dpi TIFF. Every quantitative
panel has a CSV source table in `results/tables/figure_source_data/`. Reproduce
only the figures, without rerunning inference, with:

```powershell
python experiments\molecular_origin_analysis\run_all.py --stage figures --force
```

The independent manuscript materials and captions are in
`results/manuscript/`; no existing LaTeX or manuscript file is edited.

## Applicability-domain interpretation

The descriptor-space AD is fitted only on the configured training reference
set. Standardized k-nearest-neighbour distance thresholds are determined from
reference quantiles:

- at or below q90: `in_domain`;
- above q90 and at or below q95: `borderline`;
- above q95: `out_of_domain`.

These labels are a distance-based diagnostic, not proof that a structure is
chemically synthesizable or that an ionic liquid remains liquid. Candidate and
counterfactual tables retain the status and nearest-reference information.

## Counterfactual templates

Templates are defined in
[`templates/counterfactual_templates.yaml`](templates/counterfactual_templates.yaml).
To add a structure:

1. use explicit cation and anion SMILES;
2. give each ion a net formal charge of +1 and −1, respectively;
3. specify a unique `template_id`, `modification_type`, and series labels;
4. run `pytest` and the `counterfactual` stage.

RDKit parsing, canonicalization, InChIKey deduplication, charge balance, and
combined-pair neutrality are checked before inference. Rejected and
checkpoint-incompatible structures are preserved in audit tables rather than
silently discarded.

## Memory and GPU usage

- lower `model.batch_size` for GPU memory pressure;
- set `--device cpu` when CUDA is unavailable;
- use `--max-samples` for development only;
- lower `attribution.maximum_samples` and
  `attribution.permutation_repeats` to reduce attribution cost;
- keep `model.num_workers: 0` on Windows if DataLoader multiprocessing causes
  errors.

The formal tables should be regenerated with `data.max_samples: 0`.

## Tests

From the project root:

```powershell
pytest experiments\molecular_origin_analysis\tests -v
```

Or from this module:

```powershell
cd experiments\molecular_origin_analysis
pytest tests -v
```

The real-checkpoint smoke test may allocate substantial memory; the routine
unit suite uses bounded seams and temporary directories.

## Common errors

- **No checkpoint found**: set `model.checkpoint` explicitly and confirm the
  original model configuration is present.
- **State-dict mismatch**: do not force partial loading; select the checkpoint's
  matching architecture/configuration.
- **Parquet import error**: install PyArrow from `requirements_extra.txt`.
- **CUDA out of memory**: reduce batch/attribution sample sizes or use CPU.
- **SMILES rejection**: inspect `counterfactual_rejections.csv`.
- **Counterfactual inference failure**: the ion may not exist in the frozen
  Uni-Mol cache; inspect `counterfactual_inference_failures.csv`.
- **No attention result**: consult `cross_ion_interpretation.md`; missing or
  semantically unaudited tensors are never fabricated.
- **Empty plot input**: run the required upstream stage and inspect the log.

## Scientific boundaries

- Association tables describe conditional statistical relations in the
  available observed data.
- Attribution tables describe sensitivity of the frozen checkpoint or an
  explicitly labelled descriptor surrogate.
- Attention tables describe shared model focus patterns, not quantum-chemical
  interaction energies.
- Counterfactual tables describe conditional model responses for valid
  structures within or near the descriptor reference domain.
- Candidate analyses are post hoc. They do not alter hard constraints,
  Pareto ranks, the formal Top-8, or qualification roles.
- No output validates capacitance, energy density, cycling, electrochemical
  stability, safety, or wide-temperature operation.

## Verifying scope integrity

Record the repository state before and after execution:

```powershell
git status --short
git diff -- experiments\molecular_origin_analysis
git diff --check
```

Only this module should be new or changed by the workflow. Pre-existing dirty
files elsewhere in the repository are listed in
`results/reports/implementation_report.md` and must not be attributed to this
module.
