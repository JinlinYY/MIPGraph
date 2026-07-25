# Molecular-origin analysis: manuscript figure source data

## Scope

This bundle contains the exact manuscript-facing figures and panel-level CSV
tables exported by `experiments/molecular_origin_analysis/`.  The tables
support inspection and redrawing of the plotted results; they are processed
figure source data, not the raw thermophysical-property database.

## Figure-to-data map

### Figures

- `figures/figure_main_molecular_origin_analysis_final.pdf` — Main Figure
- `figures/figure_main_molecular_origin_analysis_final.png` — Main Figure
- `figures/figure_main_molecular_origin_analysis_final.svg` — Main Figure
- `figures/figure_main_molecular_origin_analysis_final.tiff` — Main Figure
- `figures/figure_main_molecular_origin_analysis_final_17p8cm.png` — Main Figure
- `figures/figure_si_heat_capacity_size_control.pdf` — Supplementary Figure
- `figures/figure_si_heat_capacity_size_control.png` — Supplementary Figure
- `figures/figure_si_heat_capacity_size_control.svg` — Supplementary Figure

### Source-data tables

- `source_data/figure_main_molecular_origin_analysis_final_panel_a_source_data.csv` — Main Figure, panel a
- `source_data/figure_main_molecular_origin_analysis_final_panel_b_source_data.csv` — Main Figure, panel b
- `source_data/figure_main_molecular_origin_analysis_final_panel_c_source_data.csv` — Main Figure, panel c
- `source_data/figure_main_molecular_origin_analysis_final_panel_d_source_data.csv` — Main Figure, panel d
- `source_data/figure_si_heat_capacity_size_control_source_data.csv` — Supplementary Figure

## Interpretation boundaries

The association, attention-contrast and matched-substitution results are
non-causal.  Thermal-conductivity results marked `exploratory=True` retain that
qualification.  SMILES strings are included only where they define the
matched-substitution analysis unit.

## Provenance and integrity

`manifest.csv` records the source path, manuscript mapping, file size, table
shape and SHA-256 checksum.  `column_dictionary.csv` defines the exported
fields and units.  Rebuild the analysis outputs first, then refresh this bundle:

```powershell
python experiments\molecular_origin_analysis\run_all.py --stage all
python experiments\molecular_origin_analysis\package_source_data.py
```
