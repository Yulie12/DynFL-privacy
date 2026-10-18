param(
    [switch]$PrecheckOnly,
    [switch]$DryRun,
    [int[]]$Seeds = @(2026, 2027, 2028)
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = 'D:\soft\Python310\python.exe'
if (-not (Test-Path $python)) { throw "Python not found: $python" }
$argsList = @('experiments/run_100round_batch.py', '--output-root', 'out/stage16_100round_batch', '--seeds') + @($Seeds | ForEach-Object { [string]$_ })
if ($PrecheckOnly) { $argsList += '--precheck-only' }
if ($DryRun) { $argsList += '--dry-run' }
& $python @argsList
if ($LASTEXITCODE -ne 0) { throw '100-round batch failed: see out/stage16_100round_batch/batch_state.json' }
