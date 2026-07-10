$ErrorActionPreference = "Stop"

$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = "E:\anaconda\envs\ggnn39\python.exe"
$outputRoot = Join-Path $project "outputs\groupkfold_ilthermo_interpolated_seed42"
$stdout = Join-Path $outputRoot "run_stdout.log"
$stderr = Join-Path $outputRoot "run_stderr.log"
New-Item -ItemType Directory -Force $outputRoot | Out-Null

$arguments = @(
    (Join-Path $project "scripts\run_groupkfold_cv.py"),
    "--config", (Join-Path $project "configs\default.yaml"),
    "--base-split", (Join-Path $project "data\processed_ilthermo_interpolated\splits\il_level_seed42.json"),
    "--clean-csv", (Join-Path $project "data\processed_ilthermo_interpolated\il_multiprop_clean.csv"),
    "--arrays-path", (Join-Path $project "data\processed_ilthermo_interpolated\il_multiprop_arrays.npz"),
    "--graph-cache", (Join-Path $project "data\processed_ilthermo_interpolated\graph_cache.pt"),
    "--output-root", $outputRoot,
    "--pool", "train",
    "--folds", "5",
    "--seed", "42",
    "--epochs", "80",
    "--patience", "20",
    "--batch-size", "512",
    "--validate-every", "4",
    "--num-workers", "0",
    "--disable-property-coupling",
    "--skip-existing"
)

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $project `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

$process.Id
