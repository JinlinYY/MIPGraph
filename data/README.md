# Data Processing Pipeline

This directory contains the raw ionic-liquid property spreadsheets, intermediate processed files, ILThermoPy resources, and data-processing scripts used before training MIPGraphNet.

The final workbook used by the modeling project is:

```text
data/processed/ionic_liquid_6_properties_values_errors.xlsx
```

An optional ILThermoPy-filled version is:

```text
data/processed/ionic_liquid_6_properties_values_errors_ilthermo_strict.xlsx
```

## Directory Layout

```text
data/
  raw/
  processed/
  cache/
  ILThermoPy-main/
  scripts/
    merge_property_datasets.py
    add_il_smiles_and_export_csv.py
    apply_manual_smiles_and_drop_missing.py
    fill_missing_properties_from_ilthermopy_strict.py
```

## Expected Raw Data

The merge script expects the original property spreadsheets under:

```text
data/raw/Supporting Information_2/data set/
```

Expected files:

```text
Density.xlsx
ElectricalConductivity.xlsx
HeatCapacity.xlsx
SurfaceTension.xlsx
ThermalConductivity.xlsx
Viscosity.xlsx
```

Each file is parsed and merged by ionic-liquid name, temperature, and pressure.

## Step 1: Merge Six Raw Property Datasets

Run from the repository root:

```powershell
cd D:\GGNN\IL-model\Sparse-Label-Prediction
```

Merge the six raw Excel files:

```powershell
python data\scripts\merge_property_datasets.py
```

Output:

```text
data/processed/ionic_liquid_6_properties_values_errors.xlsx
```

The merged workbook contains one sheet:

```text
Merged
```

Main columns:

```text
IL_Name
Cation_FullName
Cation_ShortName
Anion_FullName
Anion_ShortName
Temperature_K
Pressure_kPa
Density_ActualValue
Density_ErrorValue
ElectricalConductivity_ActualValue
ElectricalConductivity_ErrorValue
HeatCapacity_ActualValue
HeatCapacity_ErrorValue
SurfaceTension_ActualValue
SurfaceTension_ErrorValue
ThermalConductivity_ActualValue
ThermalConductivity_ErrorValue
Viscosity_ActualValue
Viscosity_ErrorValue
```

## Step 2: Fill IL_SMILES from ILThermoPy Compound Table

The local ILThermoPy project is located at:

```text
data/ILThermoPy-main/
```

The SMILES mapping table used by the script is:

```text
data/ILThermoPy-main/src/ilthermopy/data/compounds.csv
```

Run:

```powershell
python data\scripts\add_il_smiles_and_export_csv.py `
  --workbook data\processed\ionic_liquid_6_properties_values_errors.xlsx `
  --compounds data\ILThermoPy-main\src\ilthermopy\data\compounds.csv `
  --sheet Merged `
  --missing-report data\processed\ionic_liquid_missing_smiles_names.csv
```

This inserts or updates the `IL_SMILES` column immediately after `IL_Name`.

Unresolved ionic-liquid names are written to:

```text
data/processed/ionic_liquid_missing_smiles_names.csv
```


## Step 3: Strictly Fill Missing Property Values from ILThermoPy

Missing property values can be filled from ILThermoPy using strict matching by:

- IL name
- property name
- temperature
- pressure

The script does not fill a value unless a strict match is found.

Run all six properties:

```powershell
python data\scripts\fill_missing_properties_from_ilthermopy_strict.py `
  --input data\processed\ionic_liquid_6_properties_values_errors.xlsx `
  --output data\processed\ionic_liquid_6_properties_values_errors_ilthermo_strict.xlsx `
  --properties Density,ElectricalConductivity,HeatCapacity,SurfaceTension,ThermalConductivity,Viscosity
```

Output workbook:

```text
data/processed/ionic_liquid_6_properties_values_errors_ilthermo_strict.xlsx
```

Query cache:

```text
data/cache/ilthermopy_strict_property_cache.json
```

Fill report:

```text
data/processed/ilthermopy_strict_property_fill_report.csv
```

The report records whether each missing cell was:

- matched
- not matched
- ambiguous
- skipped because of query error

ILThermoPy queries use the NIST ILThermo service and may occasionally fail due to network or server-side TLS interruptions.

To retry failed queries:

```powershell
python data\scripts\fill_missing_properties_from_ilthermopy_strict.py `
  --input data\processed\ionic_liquid_6_properties_values_errors.xlsx `
  --output data\processed\ionic_liquid_6_properties_values_errors_ilthermo_strict.xlsx `
  --properties Density,ElectricalConductivity,HeatCapacity,SurfaceTension,ThermalConductivity,Viscosity `
  --retry-failed `
  --sleep-seconds 0.5 `
  --request-timeout 30
```

For a small test run:

```powershell
python data\scripts\fill_missing_properties_from_ilthermopy_strict.py `
  --input data\processed\ionic_liquid_6_properties_values_errors.xlsx `
  --output data\processed\ionic_liquid_6_properties_values_errors_ilthermo_strict.xlsx `
  --properties Density `
  --limit 20
```
