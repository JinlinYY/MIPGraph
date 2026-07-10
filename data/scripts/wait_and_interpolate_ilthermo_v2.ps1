param(
    [Parameter(Mandatory = $true)]
    [int]$IlthermoProcessId
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$statusLog = Join-Path $repo "data\logs\ilthermo_interpolation_status.log"
$stdout = Join-Path $repo "data\logs\ilthermo_interpolation_stdout.log"
$stderr = Join-Path $repo "data\logs\ilthermo_interpolation_stderr.log"
$inputFile = Join-Path $repo "data\processed\ionic_liquid_6_properties_values_errors_ilthermo_strict_v2.xlsx"

"waiting for ILThermo PID $IlthermoProcessId" | Set-Content $statusLog
Wait-Process -Id $IlthermoProcessId
if (-not (Test-Path $inputFile)) {
    "ILThermo process ended without producing $inputFile" | Add-Content $statusLog
    exit 1
}

& "E:\anaconda\envs\ggnn39\python.exe" `
    (Join-Path $repo "data\scripts\fill_missing_properties_by_interpolation.py") `
    --input $inputFile `
    --output (Join-Path $repo "data\processed\ionic_liquid_6_properties_values_errors_ilthermo_strict_v2_interpolated.xlsx") `
    --report (Join-Path $repo "data\processed\ilthermo_strict_v2_interpolation_report.csv") `
    --summary (Join-Path $repo "data\processed\ilthermo_strict_v2_interpolation_summary.json") `
    --max-temperature-gap 40 `
    --pressure-round-decimals 1 `
    --missing-pressure-kpa 101.325 `
    --max-replicate-relative-spread 0.15 `
    1>> $stdout 2>> $stderr

if ($LASTEXITCODE -ne 0) {
    "interpolation failed with exit code $LASTEXITCODE" | Add-Content $statusLog
    exit $LASTEXITCODE
}
& "E:\anaconda\envs\ggnn39\python.exe" `
    (Join-Path $repo "il_property_prediction\scripts\preprocess_augmented_data.py") `
    --input (Join-Path $repo "data\processed\ionic_liquid_6_properties_values_errors_ilthermo_strict_v2_interpolated.xlsx") `
    --evaluation-reference (Join-Path $repo "data\processed\ionic_liquid_6_properties_values_errors.xlsx") `
    --output-dir (Join-Path $repo "il_property_prediction\data\processed_ilthermo_interpolated") `
    --source-processed-dir (Join-Path $repo "il_property_prediction\data\processed") `
    1>> $stdout 2>> $stderr
if ($LASTEXITCODE -ne 0) {
    "augmented preprocessing failed with exit code $LASTEXITCODE" | Add-Content $statusLog
    exit $LASTEXITCODE
}
"interpolation complete" | Add-Content $statusLog
