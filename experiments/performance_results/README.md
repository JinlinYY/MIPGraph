# Predictive performance across split protocols

This module creates the nine-panel performance figure in the current manuscript.

- Panels a–f show test predictions for density, viscosity, electrical conductivity,
  molar heat capacity, surface tension, and thermal conductivity.
- Panel g shows mean macro log-space R², sample-SD error bars across training
  seeds, and mean macro NMAE.
- Panels h and i show the mean property-wise log-space R² and NMAE.

The four partitions are random-point, random IL-level, property-balanced IL-level,
and family-pair-disjoint. Training seeds are 0, 42 and 123 under fixed partitions.
Parity plots contain the predictions from all three runs; annotations average
the per-run metrics. Each property-wise NMAE uses the original-training-observation
standardization scale. Metrics use originally observed test labels.

From the repository root:

```bash
python -m experiments.performance_results.plot_performance_results
```

Default inputs are the A–I CSVs in
`experiments/manuscript_figure_source_data/performance_results/`. Output files are
written to `experiments/result_analysis/figures/performance_results/` as PNG,
PDF, SVG, and TIFF. The raster output uses 600 dpi.

Custom paths:

```bash
python -m experiments.performance_results.plot_performance_results   --source-data-dir path/to/performance_results   --output-dir path/to/figures
```

`prepare_inputs.py` provides conversion helpers for evaluated split outputs.
Keep the CSV source catalog and figure-source map synchronized when refreshing
the released numerical results.
