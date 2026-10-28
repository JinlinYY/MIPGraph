# Ionic-liquid property data

The benchmark used in *Ion-Pair Representation Learning for Sparse Multi-Property
Prediction and Ionic-Liquid Screening* contains 37,556 thermodynamic-condition
records covering 1,125 ionic liquids and six thermophysical properties.

## Records and identities

A record links an ionic-liquid ion pair to a temperature, pressure, and the
available measured property values. Missing properties remain explicitly masked.
Standardized, charge-aware cation–anion identities distinguish unique ILs from
repeated measurements and alternative names or SMILES representations.

The six outputs are density (kg m⁻³), viscosity (Pa s), electrical conductivity
(S m⁻¹), molar heat capacity (J mol⁻¹ K⁻¹), surface tension (N m⁻¹), and thermal
conductivity (W m⁻¹ K⁻¹). Model input arrays retain their declared property order.

## Training and evaluation data

The study uses random-point, random IL-level, property-balanced IL-level, and
family-pair-disjoint partitions. The latter separates ion-family combinations.
Training seeds 0, 42, and 123 vary model initialization under fixed partitions.

Original experimental training observations determine the natural-log target
standardization statistics. Those statistics are fixed across the four
label-completion settings. Copied and interpolated training labels have separate
source information; validation and test evaluation use original observations.

## Locations

- `processed/`: curated property workbooks and identity/condition tables.
- `scripts/`: data merging, SMILES processing, and label preparation.
- `design_fragments/`: small example ion libraries for optional design examples.
- `../experiments/manuscript_figure_source_data/`: numerical data underlying
  the manuscript figures, including property coverage and prediction results.

Preprocessing entry points are under `il_property_prediction/scripts/`.
Paths and sheets are supplied through YAML configurations. Keep all temperature,
pressure, property-unit, and missing-value conventions consistent when supplying
additional data.
