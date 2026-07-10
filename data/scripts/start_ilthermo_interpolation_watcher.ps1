param(
    [Parameter(Mandatory = $true)]
    [int]$IlthermoProcessId
)

$ErrorActionPreference = "Stop"
$watcher = Join-Path $PSScriptRoot "wait_and_interpolate_ilthermo_v2.ps1"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$watcherStdout = Join-Path $repo "data\logs\ilthermo_watcher_stdout.log"
$watcherStderr = Join-Path $repo "data\logs\ilthermo_watcher_stderr.log"
$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $watcher,
    "-IlthermoProcessId", $IlthermoProcessId
)
$process = Start-Process `
    -FilePath "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -ArgumentList $arguments `
    -WindowStyle Hidden `
    -RedirectStandardOutput $watcherStdout `
    -RedirectStandardError $watcherStderr `
    -PassThru
$process.Id
