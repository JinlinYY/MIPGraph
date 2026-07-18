# MIPGraph computational application case

This directory implements **MIPGraph-Guided Wide-Temperature Thermophysical Pre-screening of Neat Ionic Liquids with a Conditional Supercapacitor Reference-Cell Scenario** as a self-contained, reproducible computational application case. It does not alter model training, evaluation, source data, checkpoints, or their caches.

## 1. Scientific purpose

The case asks whether the current six-output MIPGraph model can convert joint, temperature-dependent thermophysical predictions into an auditable application-oriented shortlist of previously unobserved cation–anion pairings. Neat ionic liquids are considered because their low volatility, ion-only composition, structural diversity, and potentially wide liquid-temperature range make them plausible electrolyte design spaces. Those features do not by themselves establish supercapacitor performance.

This workflow can demonstrate scalable six-property inference, temperature-dependent trade-offs, property-to-application proxy mapping, detection of unseen pair recombinations within a supported ion domain, and transparent multi-objective prioritization. It cannot establish capacitance, energy or power density, cycle life, electrochemical stability, liquid-state persistence, synthesis feasibility, purity tolerance, safety, or device superiority.

The work is computational, so it is not an empirical confirmation study. The phrase “experimental validation” is inappropriate for its results: no new material was synthesized or measured, and no device was assembled. The correct terms are *prospective virtual screening*, *thermophysical pre-screening*, and *experimental prioritization*.

## 2. Strict repository boundary

`web/` is a deprecated archive. It is excluded from model loading, preprocessing, inference, cache access, and result generation. The adapter imports the current implementation under `il_property_prediction/src`, and it writes only inside this case directory.

The runtime chain is:

```text
current cleaned benchmark + declared training split
  -> current cation/anion parser and canonicalization
  -> current molecular-graph builder and descriptor functions
  -> current cached per-ion Uni-Mol2 features
  -> current model factory
  -> explicitly configured six-output checkpoint
  -> checkpoint-owned condition and target scalers
  -> inverse-transformed physical-unit predictions
```

No second MIPGraph implementation is copied here. [repository_audit.md](repository_audit.md) records the exact current files, classes, state format, scalers, property order, and compatibility findings.

## 3. Directory layout

```text
computational_application_case/
├── configs/                 explicit full and smoke configurations
├── scripts/                 independently runnable step wrappers
├── src/                     case-only adapters and analysis modules
├── tests/                   offline unit tests
├── outputs/                 ignored runtime artefacts and schemas
├── repository_audit.md      static repository and model audit
└── run_all.py               ordered, resumable command-line entry point
```

## 4. Candidate-space definitions

Candidate generation uses only structures present in the declared training split.

- `observed_reference`: a complete canonical ion pair observed in the training split. These rows calibrate proxy distributions and hard thresholds; they are never counted as new candidates.
- `unseen_pair_recombination`: both canonical ions occur in the training domain, but their canonical complete pair does not. The default requires one +1 cation and one −1 anion, minimum component support, valid RDKit parsing, current-cache compatibility, and deterministic de-duplication.
- `one_ion_extrapolation`: disabled. The repository has no explicit compatible external-ion library, and inventing molecules or Uni-Mol2 features is prohibited.

When the combinatorial space exceeds the cap, candidates are ordered deterministically by component support and descriptor-space coverage. Every retained, excluded, and failed structure is recorded. A reversed SMILES fragment order cannot create a false unseen pair.

## 5. Six model properties and units

The checkpoint fixes the following order and native output units:

| Order | Property | Unit |
|---:|---|---|
| 1 | Density | kg m^-3 |
| 2 | ElectricalConductivity | S m^-1 |
| 3 | HeatCapacity | J mol^-1 K^-1 |
| 4 | SurfaceTension | N m^-1 |
| 5 | ThermalConductivity | W m^-1 K^-1 |
| 6 | Viscosity | Pa s |

Targets use the checkpoint's natural-log transformation. The adapter applies its stored target mean and standard deviation, exponentiates once, and subtracts the stored epsilon. Temperature and pressure are transformed once with checkpoint-owned statistics. The unit audit fails on non-positive finite source values because all six modeled quantities are positive before logging.

## 6. Application proxies

For pair molar mass \(M\) in kg mol^-1, molar heat capacity \(C_{p,m}\), density \(\rho\), and thermal conductivity \(k\):

\[
C_{p,\mathrm{mass}}=C_{p,m}/M,
\quad C_\mathrm{vol}=\rho C_{p,\mathrm{mass}},
\quad \alpha=k/C_\mathrm{vol},
\quad \tau_\mathrm{thermal}=L^2/\alpha.
\]

The complete cation-plus-anion RDKit molar mass is used once. For electrolyte volume \(V\), mass is \(m=\rho V\), with mL converted once to m^3. Thermal effusivity is \(e=\sqrt{kC_\mathrm{vol}}\).

At each temperature, positive conductivity and viscosity are log10-transformed and standardized by the observed-reference median and IQR:

\[
z_x=\frac{\log_{10}x-\operatorname{median}(\log_{10}x_\mathrm{ref})}
{\max(\operatorname{IQR}(\log_{10}x_\mathrm{ref}),\epsilon)},
\quad f_\mathrm{transport}=z_\sigma-z_\eta.
\]

Surface tension is used only as an **interfacial-property-window proxy**. Deviation is zero inside the configured observed-reference quantile window and equals distance outside the window divided by its IQR. It is not a contact-angle model and does not establish electrode wetting or an electrochemical stability window.

## 7. Conditional reference-cell scenario

The application now maps the temperature-resolved thermophysical predictions into one explicit comparison geometry. The default fixed assumptions are 100 cm2 electrode area, 100 micrometre separator thickness, 1 mL electrolyte, 10 F nominal capacitance, 2 A constant current, two exposed faces, 10 W m-2 K-1 convection, and a 60 s transient horizon. These values define a normalized scenario; they are not claimed to be an industry standard or a measured cell.

For conductivity \(\sigma\), separator thickness \(L\), electrode area \(A\), current \(I\), and imposed capacitance \(C\):

\[
R_\mathrm{elyte}=L/(\sigma A),
\quad \tau_\mathrm{RC}=R_\mathrm{elyte}C,
\quad P_\mathrm{Joule}=I^2R_\mathrm{elyte}.
\]

The thermal model uses a series electrolyte-conduction and external-convection resistance, electrolyte thermal capacitance \(C_\mathrm{th}=C_\mathrm{vol}V\), and the first-order constant-property response

\[
\Delta T(t)=P_\mathrm{Joule}R_\mathrm{th}
\left[1-\exp\left(-t/(R_\mathrm{th}C_\mathrm{th})\right)\right].
\]

Every candidate receives absolute and reference-temperature-normalized resistance, electrolyte-only RC contribution, Joule power, steady and transient temperature rise, explicit low/high-temperature resistance retention, the corresponding conductivity retention, and a worst-temperature comparative risk. Risk bands are calibrated at each temperature from observed-reference q75/q95 resistance and transient-rise distributions. The maximum numeric q75-relative index and the most severe categorical q95-tail band are stored with their own temperatures because they need not occur at the same condition. These are prioritization measures, not thermal-safety thresholds.

The model does not include electrode/contact resistance, current collectors, packaging, leakage, reaction heat, electrochemical stability, or a predicted capacitance. Liquid phase is conditionally assumed throughout the configured window and must be established independently.

## 8. Full-window robust metrics and curve audit

Hard decisions use all configured temperatures, never a favorable single point:

- minimum conductivity, transport favorability, volumetric heat capacity, and thermal diffusivity;
- maximum viscosity, simplified diffusion timescale, and interfacial-window deviation;
- density range, per-metric mean, temperature slope, relative change, and coefficient of variation.

Every property curve is checked for non-finite and non-positive values, excursions beyond benchmark property ranges, benchmark-temperature extrapolation, and adjacent jumps above an observed-reference quantile. Severe failures can be excluded; warnings remain in the trace.

The primary robust summary and every screening/Pareto decision use only the configured main window (278.15–373.15 K, or 5–100 degrees Celsius, by default). This is a modeled comparison window, not evidence that every candidate remains liquid across it. If `run_extended_sensitivity` is enabled, additional configured rows are tagged `extended_sensitivity`, audited, and retained separately; they do not alter the primary thresholds or ranks.

## 9. Applicability domain

The descriptor AD uses the current 56 global and 80 functional-group descriptors. Constant columns are removed, `StandardScaler` is fitted on unique training-domain references only, and leave-one-out mean k-nearest-neighbor distances calibrate the q90 and q95 boundaries. Candidates are classified as `in_domain`, `borderline`, or `out_of_domain`; unseen ion components, weak ion or ion-family support, and insufficient temperature coverage can only worsen that status.

Embedding AD is reported as unavailable because no complete reference embedding bank generated by the identical checkpoint/preprocessing path exists. The code does not substitute descriptor distances or random values under an embedding label.

## 10. Uncertainty mode

The default checkpoint is a single deterministic point-prediction model. Without at least three explicitly configured compatible checkpoints or held-out residual calibration, property intervals, proxy intervals, feasibility probabilities, and Pareto probabilities are marked `not_available`. With three or more paths, the pipeline runs every member, verifies complete member coverage, saves per-checkpoint rows, computes physical-unit property and proxy means/standard deviations, and propagates every member through frozen thresholds, curve checks, hard constraints, and Pareto sorting.

## 11. Frozen screening thresholds

Before inspecting unseen-candidate ranks, observed-reference whole-window summaries define:

- conductivity minimum: reference q25;
- viscosity maximum: reference q75;
- volumetric heat-capacity minimum: reference q25;
- thermal-diffusivity minimum: reference q25;
- interfacial-window deviation maximum: 1 reference IQR.

Separate gates enforce valid 1:1 charge, complete finite inference, no severe curve failures, allowed AD status, and every thermophysical threshold. The frozen numerical thresholds and every pass/fail bit are persisted.

## 12. Pareto objectives and recommendation classes

Non-dominated sorting maximizes worst-window conductivity, volumetric heat capacity, and thermal diffusivity while minimizing worst-window viscosity, interfacial deviation, and conditional reference-cell risk. Utopia distance orders otherwise transparent trade-offs.

- `balanced`: in-domain Pareto-rank-1 lead nearest the normalized utopia point;
- `high_transport`: transport-side normalized score dominates the thermal-side score;
- `thermal_robust`: thermal-side normalized score dominates the transport-side score;
- `exploratory`: a hard-feasible borderline-AD lead requiring AD-focused qualification.

An out-of-domain pair cannot enter the default final set. If no pair passes, the correct result is an empty candidate table—not relaxed thresholds.

## 13. Running the case

Use the Python environment already capable of importing PyTorch, PyTorch Geometric, RDKit, pandas, scikit-learn, Matplotlib, and PyYAML.

Smoke test from the project root:

```powershell
python experiments\computational_application_case\run_all.py `
  --config experiments\computational_application_case\configs\smoke_test.yaml `
  --smoke-test
```

Full run:

```powershell
python experiments\computational_application_case\run_all.py `
  --config experiments\computational_application_case\configs\default.yaml
```

Linux/macOS uses the same arguments with `/` separators. `--resume` skips successful step markers; `--force` explicitly replaces case outputs for selected steps. `--only-step STEP` is intended for rerunning a step after its dependencies exist. Other overrides are `--checkpoint`, `--checkpoints`, `--device`, `--batch-size`, `--output-dir`, `--skip-figures`, `--skip-report`, and `--verbose`.

The default YAML contains the audited checkpoint explicitly. Override it only with a structurally compatible six-output checkpoint:

```powershell
python experiments\computational_application_case\run_all.py --config ... --checkpoint path\to\checkpoint.pt
```

For an actual ensemble, pass at least three structurally compatible six-output checkpoints. They must share property order and preprocessing semantics:

```powershell
python experiments\computational_application_case\run_all.py --config ... `
  --checkpoints member1.pt member2.pt member3.pt
```

The primary screening prediction is the physical-unit ensemble mean. Incomplete candidate-condition member sets are excluded from that mean and fail the complete-window gate.

## 14. Outputs and Figures 5–6 reproduction

`outputs/data/` contains candidate libraries, raw predictions, proxies, reference-cell temperature/summary tables, robust summaries, AD, screening, Pareto, and counterfactual tables. `outputs/audit/` contains units, inference, conditional-cell equations/assumptions, AD, and uncertainty metadata. `outputs/steps/` contains resumable markers. `outputs/tables/` contains CSV and directly reusable LaTeX tables. `outputs/report/` contains Markdown, LaTeX, JSON, and blocking reports. Large runtime files are ignored by Git.

Figure 5 is generated solely from persisted screening outputs. Figure 6 separately shows the fixed cell assumptions, resistance, RC contribution, Joule power, steady/transient temperature rise, high/low-temperature retention, and worst-temperature comparative risk. The full config writes 600-dpi PNG and vector PDF files plus individual panels. Reproduce only the figures after a successful pipeline with:

```powershell
python experiments\computational_application_case\scripts\make_figures.py `
  --config experiments\computational_application_case\configs\default.yaml `
  --force
```

## 15. Common failures

- **Checkpoint or scaler mismatch:** select a checkpoint created by the current model factory with all six properties and strict state keys. Inspect `outputs/audit/inference_pipeline.json`.
- **Uni-Mol2 cache miss:** unseen *pairs* are supported only when both component ions have current-cache entries. Use the current formal cache-generation script for real ions; never copy or invent tensors. Application graph caching is separate and local.
- **Dynamic pair inference failure:** inspect `candidate_generation_failures.csv`, `inference_failures.csv`, and `blocking_report.md`. Fix the case adapter or current cache compatibility; do not use archived code or synthetic predictions.
- **Existing output marker:** use `--resume` only when configuration, checkpoint identities, case source fingerprint, and all expected artifacts match the marker. Otherwise use `--force`.
- **No feasible or Pareto candidates:** this may be the valid result under frozen constraints. Audit failure reasons instead of changing thresholds after seeing ranks.
- **CUDA unavailable:** `device: auto` falls back to CPU; the smoke config is explicitly CPU-safe.

## 16. Required downstream evidence

Before any material or device claim, follow-up work must determine synthesis and purification feasibility, water content, melting/glass-transition behavior and liquid range, thermal stability, ion size–pore matching, electrode-specific wetting/contact behavior, electrochemical stability, conductivity and viscosity by independent measurement, capacitance and impedance, rate capability, self-discharge, cycling stability, cell packaging, and safety. Family-level extrapolation and the sparse thermal-conductivity label coverage should be treated as explicit risk factors when selecting those measurements.
