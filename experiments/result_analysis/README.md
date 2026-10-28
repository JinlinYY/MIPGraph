# Manuscript figures

This directory contains the final figures for the current manuscript and
Supporting Information. Numerical source tables are stored once, in
[`figures/manuscript_figure_source_data/`](figures/manuscript_figure_source_data/).

## Figure files

Each `figures/<figure_id>/` directory contains PNG, PDF, SVG, and TIFF exports.
`manifest.csv` records the file formats, image dimensions, and checksums.
`figure_source_map.csv` links each figure panel directly to its canonical CSV.
The source tables sit beside the figure folders under `figures/manuscript_figure_source_data/`.

| Figure identifier | Content |
|---|---|
| `Intro-method` | Framework illustration |
| `dataset_statistics` | Experimental data coverage and distributions |
| `performance_results` | Predictive performance under four fixed partitions |
| `molecular_interpretation` | Structure–property associations and response shapes |
| `interpretability` | Molecular and feature attribution |
| `auditable_virtual_screening_validation` | Screening funnel, constraints and candidate roles |
| `reference_cell_scenario_audited` | Reference-cell interpretation of selected candidates |
| `figureS_application_derived_metrics` | Joule heating and conditional steady temperature rise |
| `figureS_application_decision_stability` | Threshold and bootstrap selection stability |
| `figureS_heat_capacity_size_control` | Heat-capacity association with molecular-size control |

The performance figure uses seeds 0, 42 and 123. The derived-metric curves include
UPR-0343, UPR-0024, UPR-0558, and UPR-0581. Framework PDF/SVG exports embed the
matching PNG when no native vector illustration is supplied.

Refresh the links without copying numerical tables:

```bash
python experiments/result_analysis/scripts/package_manuscript_results.py --source-data-only
```

The underlying CSV schemas and field definitions are documented in the canonical
source-data directory.
