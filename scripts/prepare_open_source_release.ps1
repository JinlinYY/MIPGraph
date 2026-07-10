$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ReleaseRoot = Join-Path $RepoRoot "trained_weights\mipgraph_best"
$WeightsRoot = Join-Path $ReleaseRoot "weights"
$ArtifactsRoot = Join-Path $RepoRoot "training_artifacts"

function Assert-UnderRepo {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $full = [System.IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label is outside repository root: $full"
    }
    return $full
}

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$Base,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $baseUri = [System.Uri](([System.IO.Path]::GetFullPath($Base).TrimEnd('\') + '\'))
    $pathUri = [System.Uri]([System.IO.Path]::GetFullPath($Path))
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($pathUri).ToString()).Replace('/', '\')
}

function Convert-ToSlashPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return $Path.Replace('\', '/')
}

New-Item -ItemType Directory -Force -Path $ReleaseRoot, $WeightsRoot, $ArtifactsRoot | Out-Null

$SelectionRoot = Join-Path $RepoRoot "il_property_prediction\outputs\mps_weak_merged_validation"
$OutputsRoot = Join-Path $RepoRoot "il_property_prediction\outputs"
$SelectionFiles = @()
if (Test-Path -LiteralPath $SelectionRoot) {
    $SelectionFiles = Get-ChildItem -LiteralPath $SelectionRoot -Filter "selected_checkpoints.json" -Recurse
}

$manifest = [ordered]@{
    name = "MIPGraph release best checkpoints"
    created_by = "scripts/prepare_open_source_release.ps1"
    release_root = "trained_weights/mipgraph_best"
    selection_source = Convert-ToSlashPath (Get-RelativePath $RepoRoot $SelectionRoot)
    selection_rule = "Validation-only property-wise checkpoint selection from mps_weak_merged_validation."
    splits = [ordered]@{}
}

$movedBySource = @{}
$movedCount = 0

foreach ($selectionFile in $SelectionFiles) {
    $selection = Get-Content -Raw -LiteralPath $selectionFile.FullName | ConvertFrom-Json
    $splitName = [string]$selection.case
    $splitEntry = [ordered]@{
        label = [string]$selection.label
        selection_rule = [string]$selection.selection_rule
        selection_objective = [string]$selection.selection_objective
        test_used_for_selection = [bool]$selection.test_used_for_selection
        properties = [ordered]@{}
    }

    foreach ($property in $selection.properties.PSObject.Properties) {
        $item = $property.Value
        $src = Assert-UnderRepo ([string]$item.checkpoint) "checkpoint"
        if ($movedBySource.ContainsKey($src)) {
            $releaseWeightRel = $movedBySource[$src]
        } else {
            if ($src.StartsWith($OutputsRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                $weightSubPath = Get-RelativePath $OutputsRoot $src
            } else {
                $weightSubPath = Split-Path -Leaf $src
            }
            $dest = Assert-UnderRepo (Join-Path $WeightsRoot $weightSubPath) "release checkpoint"
            $destDir = Split-Path -Parent $dest
            New-Item -ItemType Directory -Force -Path $destDir | Out-Null

            if (Test-Path -LiteralPath $src) {
                if (Test-Path -LiteralPath $dest) {
                    throw "Destination already exists while source still exists: $dest"
                }
                Move-Item -LiteralPath $src -Destination $dest
                $movedCount += 1
            } elseif (-not (Test-Path -LiteralPath $dest)) {
                throw "Checkpoint source is missing and release destination does not exist: $src"
            }

            $releaseWeightRel = Convert-ToSlashPath (Get-RelativePath $ReleaseRoot $dest)
            $movedBySource[$src] = $releaseWeightRel
        }

        $splitEntry.properties[$property.Name] = [ordered]@{
            source = [string]$item.source
            weight = $releaseWeightRel
            val_score = [double]$item.val_score
            val_log_MAE = [double]$item.val_log_MAE
            original_checkpoint = Convert-ToSlashPath (Get-RelativePath $RepoRoot $src)
        }
    }

    $manifest.splits[$splitName] = $splitEntry
}

$manifestPath = Join-Path $ReleaseRoot "manifest.json"
$manifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$artifactMoves = @(
    @{ Source = "il_property_prediction\outputs"; Target = "training_artifacts\il_property_prediction_outputs" },
    @{ Source = "il_property_prediction\logs"; Target = "training_artifacts\il_property_prediction_logs" },
    @{ Source = "il_property_prediction\tmp"; Target = "training_artifacts\il_property_prediction_tmp" },
    @{ Source = "logs"; Target = "training_artifacts\root_logs" },
    @{ Source = "tmp"; Target = "training_artifacts\root_tmp" },
    @{ Source = "data\logs"; Target = "training_artifacts\data_logs" },
    @{ Source = "mipgraph_server_upload.tgz"; Target = "training_artifacts\generated_archives\mipgraph_server_upload.tgz" },
    @{ Source = "IL性质预测_改进版.zip"; Target = "training_artifacts\generated_archives\IL性质预测_改进版.zip" }
)

$movedArtifacts = @()
foreach ($entry in $artifactMoves) {
    $src = Assert-UnderRepo (Join-Path $RepoRoot $entry.Source) "artifact source"
    $dst = Assert-UnderRepo (Join-Path $RepoRoot $entry.Target) "artifact destination"
    if (-not (Test-Path -LiteralPath $src)) {
        continue
    }
    if (Test-Path -LiteralPath $dst) {
        throw "Artifact destination already exists: $dst"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Move-Item -LiteralPath $src -Destination $dst
    $movedArtifacts += [ordered]@{
        source = Convert-ToSlashPath $entry.Source
        destination = Convert-ToSlashPath $entry.Target
    }
}

$artifactManifest = @"
# Local Training Artifacts

This directory is intentionally ignored by git. It stores training outputs, logs,
temporary files, and generated archives moved out of the open-source tree.

Best release checkpoints are kept separately under `trained_weights/mipgraph_best/`.

Moved entries:
$($movedArtifacts | ForEach-Object { "- $($_.source) -> $($_.destination)" } | Out-String)
"@
$artifactManifest | Set-Content -LiteralPath (Join-Path $ArtifactsRoot "README.md") -Encoding UTF8

Write-Host "Release manifest: $manifestPath"
Write-Host "Moved release checkpoints: $movedCount"
Write-Host "Moved artifact roots: $($movedArtifacts.Count)"
