# Performance-results figure

This module reproduces the manuscript performance figure containing:

- panels a–f: property-wise experimental-versus-predicted parity plots for
  point, whole-IL, property-balanced whole-IL and ion-family split protocols;
- panel g: macro log-space performance across protocols;
- panels h–i: property-wise log-space \(R^2\) and NMAE matrices.

The plotting code does not retrain MIPGraph and does not alter any metric.

## Files

```text
performance_results/
|-- plot_performance_results.py  # single formal plotting entrypoint
|-- prepare_inputs.py            # optional conversion from raw split outputs
|-- tests/                       # source-data and rendering tests
`-- README.md
```

## Reproduce the current audited figure

From the repository root:

```powershell
python -m experiments.performance_results.plot_performance_results
```

The default source tables are the tracked CSVs under
`experiments/manuscript_figure_source_data/performance_results/`. Outputs are
written to
`experiments/result_analysis/figures/performance_results/` at 600 dpi.

Alternative audited source-data directory:

```powershell
python -m experiments.performance_results.plot_performance_results `
  --source-data-dir path\to\performance_results_source_data `
  --output-dir path\to\figures
```

## Refresh source data from model outputs

Only use this route when the four split-protocol evaluations have been
regenerated:

```powershell
python -m experiments.performance_results.prepare_inputs
python -m experiments.performance_results.plot_performance_results `
  --input-root il_property_prediction\outputs\mps_weak_merged_validation\figure_inputs
```

`prepare_inputs.py` expects the split folders `random_point`,
`random_il_level`, `property_balanced_il_level` and `ion_family`. The plotting
script then exports updated panel-level A–I CSVs alongside the figure.

## Scientific boundary

The panels report held-out predictive performance under four split protocols.
They do not constitute prospective candidate screening, uncertainty
calibration or evidence of electrochemical device performance.
