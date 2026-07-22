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

Surface tension is used only through a **surface-tension reference-envelope constraint**. \(D_{\gamma,\mathrm{ref}}\) is zero inside the configured observed-reference quantile envelope and equals the distance outside that envelope divided by its IQR. It is not a contact-angle, wetting, or interfacial-compatibility prediction and does not establish an electrochemical stability window.

## 7. Post hoc 60-s constant-current thermal stress scenario

Only after the formal shortlist is fixed, the application maps the temperature-resolved thermophysical predictions into one explicit comparison geometry. The fixed assumptions are 100 cm2 electrode area, 100 micrometre separator thickness, 1 mL electrolyte, 2 A constant current, two exposed faces, 10 W m-2 K-1 convection, and a 60 s pulse. These values define a conditional comparison scenario; they are not claimed to be an industry standard or a measured cell.

For conductivity \(\sigma\), separator thickness \(L\), electrode area \(A\), and current \(I\):

\[
R_\mathrm{elyte}=L/(\sigma A),
\quad P_\mathrm{Joule}=I^2R_\mathrm{elyte}.
\]

The thermal model uses a series electrolyte-conduction and external-convection resistance, electrolyte thermal capacitance \(C_\mathrm{th}=C_\mathrm{vol}V\), and the first-order constant-property response

\[
\Delta T(t)=P_\mathrm{Joule}R_\mathrm{th}
\left[1-\exp\left(-t/(R_\mathrm{th}C_\mathrm{th})\right)\right].
\]

Every candidate receives absolute and reference-temperature-normalized resistance, Joule power, steady and 60-s transient temperature rise, explicit endpoint resistance ratios, and the corresponding conductivity ratios. The primary-window reference-population exceedance index is

\[
\Xi_{\max}=\max_{T\in[298.15,353.15]}\max\left[
\frac{R_\mathrm{elyte}(T)}{q_{75}^{\mathrm{ref}}[R_\mathrm{elyte}(T)]},
\frac{\Delta T_{60}(T)}{q_{75}^{\mathrm{ref}}[\Delta T_{60}(T)]}
\right].
\]

It has no fitted weights. It records only the worst component exceedance relative to the temperature-matched observed-reference q75; it is not a thermal-runaway, safety, failure, or lifetime risk index. Joule-power and steady-state curves are retained in SI because they are derivative or nearly redundant under the fixed scenario.

The model does not include electrode/contact resistance, current collectors, packaging, leakage, reaction heat, electrochemical stability, capacitance, or device performance. Liquid phase is conditionally assumed throughout the configured window and must be established independently.

## 8. Full-window robust metrics and curve audit

Hard decisions use all configured temperatures, never a favorable single point:

- minimum conductivity, transport favorability, volumetric heat capacity, and thermal diffusivity;
- maximum viscosity, simplified diffusion timescale, and surface-tension reference-envelope deviation;
- density range, per-metric mean, temperature slope, relative change, and coefficient of variation.

Every property curve is checked for non-finite and non-positive values, excursions beyond benchmark property ranges, benchmark-temperature extrapolation, and adjacent jumps above an observed-reference quantile. Severe failures can be excluded; warnings remain in the trace.

The primary robust summary and every screening/Pareto decision use only the configured 298.15--353.15 K main window. The outer-grid rows are tagged `extended_sensitivity`; intermediate outer-grid values are retained only for curve-continuity auditing, while the main text reports 278.15 and 373.15 K as the two open-ended stress-test endpoints. No outer-grid row alters primary thresholds or ranks or establishes that a candidate remains liquid there.

## 9. Applicability domain

The descriptor AD uses the current 56 global and 80 functional-group descriptors. Constant columns are removed, `StandardScaler` is fitted on unique training-domain references only, and leave-one-out mean k-nearest-neighbor distances calibrate the q90 and q95 boundaries. Candidates are classified as `in_domain`, `borderline`, or `out_of_domain`; unseen ion components, weak ion or ion-family support, and insufficient temperature coverage can only worsen that status.

Embedding AD is reported as unavailable because no complete reference embedding bank generated by the identical checkpoint/preprocessing path exists. The code does not substitute descriptor distances or random values under an embedding label.

## 10. Cross-protocol decision stability

The formal application result uses one deterministic random-IL checkpoint. The property-balanced IL and ion-family checkpoints are run independently on the frozen primary candidate space, main temperature window, primary-model property thresholds, and four Pareto objectives. Applicability-domain status is recomputed from the training split aligned with each checkpoint. No checkpoint average, posterior probability, calibration interval, or ensemble uncertainty is used for the formal shortlist.

## 11. Frozen screening thresholds

Before inspecting unseen-candidate ranks, observed-reference whole-window summaries define:

- conductivity minimum: reference q25;
- viscosity maximum: reference q75;
- volumetric heat-capacity minimum: reference q25;
- thermal-diffusivity minimum: reference q25;
- surface-tension reference-envelope deviation maximum: 1 reference IQR.

Separate gates enforce valid 1:1 charge, complete finite inference, no severe curve failures, allowed AD status, and every thermophysical threshold. The frozen numerical thresholds and every pass/fail bit are persisted.

## 12. Pareto objectives and recommendation classes

Non-dominated sorting uses exactly four objectives: it maximizes worst-window conductivity, volumetric heat capacity, and thermal diffusivity while minimizing worst-window viscosity. The surface-tension reference envelope, applicability domain, and curve quality are hard constraints. Reference-cell resistance, temperature rise, and \(\Xi_{\max}\) are post hoc engineering-context statistics and do not enter hard screening, Pareto sorting, Top-8 selection, or qualification-role selection. Pareto-rank-1 candidates are robust-normalized with clipped q05--q95 scaling, then ordered deterministically by Euclidean distance to the four-objective utopia point with candidate ID as the final tie-breaker.

- `balanced lead`: in-domain candidate nearest the four-objective utopia point;
- `transport-focused lead`: candidate nearest the two-objective conductivity--viscosity utopia point;
- `thermal-management lead`: candidate nearest the two-objective volumetric-heat-capacity--thermal-diffusivity utopia point;
- `cross-protocol robust lead`: candidate passing the hard constraints or reaching Pareto rank one in both primary and balanced protocols, with the smallest absolute protocol-order change.

If a role has no eligible candidate, the role is left unassigned. Different roles may select the same candidate; alternatives are not inserted manually.

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

Do not pass the three split-protocol checkpoints through `--checkpoints` for this application: they represent different evaluation protocols and must not be averaged into a formal prediction.

## 14. Outputs and Figures 5–6 reproduction

`outputs/data/` contains candidate libraries, raw predictions, proxies, reference-cell temperature/summary tables, robust summaries, AD, screening, Pareto, and counterfactual tables. `outputs/audit/` contains units, inference, conditional-cell equations/assumptions, AD, and uncertainty metadata. `outputs/steps/` contains resumable markers. `outputs/tables/` contains CSV and directly reusable LaTeX tables. `outputs/report/` contains Markdown, LaTeX, JSON, and blocking reports. Large runtime files are ignored by Git.

Figure 5 contains the identity audit, full 608-candidate funnel, constraint margins, reference-bootstrap selection frequencies, protocol sensitivity, and automatically assigned qualification roles. Figure 6 begins only after the formal shortlist has been fixed and reports the 60-s scenario, resistance, the volumetric-heat-capacity--thermal-diffusivity map, conditional temperature rise, endpoint transport trade-off, and reference-population exceedance context. Joule-power and steady-state temperature-rise curves are generated only for the Supporting Information. The builder writes separate 600-dpi PNG and vector PDF files plus machine-readable source data. After a successful primary pipeline, first build the two independently audited protocol-sensitivity outputs on the frozen primary candidate identities and then reproduce the figures and 500-replicate bootstrap:

```powershell
python experiments\computational_application_case\scripts\build_protocol_stability_outputs.py --force
python experiments\computational_application_case\scripts\build_refactored_application_case.py
```

The first command runs each sensitivity checkpoint separately, aligns its applicability domain to its own training split, verifies the candidate ID--InChI mapping and temperature coverage, and records checkpoint and candidate-identity SHA-256 digests. The figure builder then reads only those verified CSV/JSON outputs and never averages predictions.

## 15. Common failures

- **Checkpoint or scaler mismatch:** select a checkpoint created by the current model factory with all six properties and strict state keys. Inspect `outputs/audit/inference_pipeline.json`.
- **Uni-Mol2 cache miss:** unseen *pairs* are supported only when both component ions have current-cache entries. Use the current formal cache-generation script for real ions; never copy or invent tensors. Application graph caching is separate and local.
- **Dynamic pair inference failure:** inspect `candidate_generation_failures.csv`, `inference_failures.csv`, and `blocking_report.md`. Fix the case adapter or current cache compatibility; do not use archived code or synthetic predictions.
- **Existing output marker:** use `--resume` only when configuration, checkpoint identities, case source fingerprint, and all expected artifacts match the marker. Otherwise use `--force`.
- **No feasible or Pareto candidates:** this may be the valid result under frozen constraints. Audit failure reasons instead of changing thresholds after seeing ranks.
- **CUDA unavailable:** `device: auto` falls back to CPU; the smoke config is explicitly CPU-safe.

## 16. Required downstream evidence

Before any material or device claim, follow-up work must determine synthesis and purification feasibility, water content, melting/glass-transition behavior and liquid range, thermal stability, ion size–pore matching, electrode-specific wetting/contact behavior, electrochemical stability, conductivity and viscosity by independent measurement, capacitance and impedance, rate capability, self-discharge, cycling stability, cell packaging, and safety. Family-level extrapolation and the sparse thermal-conductivity label coverage should be treated as explicit risk factors when selecting those measurements.

## 17. Auditable bilingual chapter rerun

The application-facing rerun uses resonance-invariant, ion-level Standard
InChIKeys for chemical identity control. The random-IL whole-ion-holdout
checkpoint is the sole primary model for formal candidate values and
decisions. The balanced-IL and ion-family checkpoints are evaluated
independently for cross-protocol decision stability and are never averaged.
Run the primary calculation and evidence builder from the project root with:

```powershell
E:\anaconda\envs\ggnn39\python.exe experiments\computational_application_case\run_all.py `
  --config experiments\computational_application_case\configs\auditable_virtual_screening.yaml `
  --force --skip-figures --skip-report
E:\anaconda\envs\ggnn39\python.exe `
  experiments\computational_application_case\scripts\build_protocol_stability_outputs.py --force
E:\anaconda\envs\ggnn39\python.exe `
  experiments\computational_application_case\scripts\build_refactored_application_case.py
```

The first command writes the formal single-checkpoint result tree under
`outputs_primary_audited/`. The second command independently evaluates the two
sensitivity checkpoints on the frozen primary candidate identities and writes
checkpoint, split, identity, and temperature-coverage manifests. The third
command applies the frozen primary property thresholds, performs the 81-setting
threshold sensitivity analysis, generates the application-only Figures 5 and 6
plus the SI figures, and stages their source data for the bilingual chapter; it
does not update Figure 3 or other preceding manuscript results. Run
`scripts/package_chapter_artifacts.py` to refresh the application-only
`chapter_results/` release bundle. Compile the main manuscript with
XeLaTeX:

The evidence builder directly audits all 649 entries in the old SMILES-novel
pool instead of inferring identity corrections from before/after pool sizes.
It also removes the one Standard-InChI train--test identity overlap found in
each of the random-IL and property-balanced IL evaluations and estimates
temperature-trend agreement only from records with finite pressures of
90--110 kPa.  Evaluation rows with unresolved chemical identity or missing
pressure for the curve analysis are handled fail-closed.  These exclusions are
written to auditable CSV files.  The released checkpoints predate this audit,
so unresolved training rows remain a disclosed limitation unless the models
are retrained from newly identity-grouped splits.

```powershell
cd LaTex-MIPGraph
latexmk -xelatex -interaction=nonstopmode -halt-on-error acs-latex-template.tex
```

The three models differ by split protocol rather than by random seed. Their
categorical hard-pass, Pareto-rank, and top-eight decisions quantify protocol
sensitivity; they are not ensemble uncertainty, calibrated posterior
probabilities, or formal prediction intervals. The formal screening window is
298.15--353.15 K. Predictions at 278.15 and 373.15 K are stress-test endpoints
only and do not enter hard screening or Pareto sorting.
