$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "E:\anaconda\envs\ggnn39\python.exe"
$script = Join-Path $repo "data\scripts\fill_missing_properties_from_ilthermopy_strict.py"
$stdout = Join-Path $repo "data\logs\ilthermo_strict_v2_stdout.log"
$stderr = Join-Path $repo "data\logs\ilthermo_strict_v2_stderr.log"

New-Item -ItemType Directory -Force (Split-Path $stdout) | Out-Null

$arguments = @(
    $script,
    "--input", (Join-Path $repo "data\processed\ionic_liquid_6_properties_values_errors_ilthermo_strict.xlsx"),
    "--output", (Join-Path $repo "data\processed\ionic_liquid_6_properties_values_errors_ilthermo_strict_v2.xlsx"),
    "--cache", (Join-Path $repo "data\cache\ilthermopy_strict_property_cache.json"),
    "--report", (Join-Path $repo "data\processed\ilthermopy_strict_v2_property_fill_report.csv"),
    "--properties", "Density,ElectricalConductivity,HeatCapacity,SurfaceTension,ThermalConductivity,Viscosity",
    "--retry-failed",
    "--sleep-seconds", "0.5",
    "--checkpoint-every", "10",
    "--progress-every", "10",
    "--request-timeout", "30",
    "--request-retries", "2",
    "--retry-backoff", "1",
    "--no-tqdm"
)

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

$process.Id
