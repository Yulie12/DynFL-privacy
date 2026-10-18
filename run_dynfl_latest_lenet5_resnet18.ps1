[CmdletBinding()]
param(
    [int[]]$Seeds = @(2026, 2027, 2028),
    [string]$PythonExe = 'D:\soft\Python310\python.exe'
)

$ErrorActionPreference = 'Stop'
$repo = (Get-Location).Path
$runner = Join-Path $repo 'experiments\run_paper_config.py'
if (-not (Test-Path $PythonExe)) { throw "Python not found: $PythonExe" }
if (-not (Test-Path $runner)) { throw "Run this script from the DynFL-privacy repository root" }
if ($Seeds.Count -lt 1) { throw 'Provide at least one seed' }

# Frozen five-round STAGING configs, both at the latest Stage 15 code state.
$models = @(
    [PSCustomObject]@{
        Name = 'fmnist_lenet5'
        Config = 'configs\stage11_fmnist_lenet5_legacy_5r_STAGING.json'
    },
    [PSCustomObject]@{
        Name = 'cifar10_resnet18_layer4_head'
        Config = 'configs\stage11_cifar10_resnet18_layer4_legacy_5r_STAGING.json'
    }
)

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$session = "out/stage15_lenet5_resnet18_batch_$stamp"
$sessionAbs = Join-Path $repo $session
New-Item -ItemType Directory -Force -Path $sessionAbs | Out-Null

# Snapshot config files and local Git state for later reproducibility.
foreach ($m in $models) {
    $configAbs = Join-Path $repo $m.Config
    if (-not (Test-Path $configAbs)) { throw "Missing config: $configAbs" }
    Copy-Item $configAbs (Join-Path $sessionAbs ("config_{0}.json" -f $m.Name))
}
& git status --short 2>&1 | Out-File -FilePath (Join-Path $sessionAbs 'git_status.txt') -Encoding UTF8

Write-Host "Results directory: $sessionAbs" -ForegroundColor Cyan
Write-Host 'Preflight (dry-run) for both models...' -ForegroundColor Cyan
foreach ($m in $models) {
    & $PythonExe $runner --config $m.Config --seeds $Seeds[0] `
        --final-round-diagnostic-extra-epsilon 2 `
        --output-root "$session/$($m.Name)/preflight" --dry-run
    if ($LASTEXITCODE -ne 0) { throw "Preflight failed for $($m.Name)" }
}

# One GPU job at a time; four frozen policies are read from each JSON config.
foreach ($m in $models) {
    foreach ($seed in $Seeds) {
        $out = "$session/$($m.Name)/seed_$seed"
        Write-Host "`nRUN model=$($m.Name) seed=$seed" -ForegroundColor Cyan
        & $PythonExe $runner --config $m.Config --seeds $seed `
            --final-round-diagnostic-extra-epsilon 2 `
            --output-root $out
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $($m.Name) seed=$seed. Inspect $out"
        }
    }
}

$rows = @(
    foreach ($m in $models) {
        foreach ($seed in $Seeds) {
            $jobDir = Join-Path $sessionAbs "$($m.Name)\seed_$seed"
            $files = @(Get-ChildItem -Path $jobDir -Filter 'summary.json' -Recurse -File)
            if ($files.Count -ne 4) {
                throw "Expected 4 policy summaries in $jobDir, found $($files.Count)"
            }
            foreach ($f in $files) {
                $s = Get-Content -LiteralPath $f.FullName -Raw | ConvertFrom-Json
                if ($s.status -ne 'completed' -or [int]$s.rounds -ne 5) {
                    throw "Incomplete result: $($f.FullName)"
                }
                [PSCustomObject]@{
                    Model = $m.Name
                    Seed = [int]$seed
                    Policy = [string]$s.policy
                    AccuracyPct = [math]::Round(100.0 * [double]$s.final_test_accuracy, 3)
                    BestAccuracyPct = [math]::Round(100.0 * [double]$s.best_test_accuracy, 3)
                    LogicalTimeSec = [math]::Round([double]$s.total_logical_time, 3)
                    WallTimeSec = [math]::Round([double]$s.end_to_end_wall_time_sec, 3)
                    SkipRate = [double]$s.skip_rate
                    ModeDistribution = [string]$s.mode_distribution
                    ResultPath = $f.DirectoryName
                }
            }
        }
    }
)

$rows | Sort-Object Model, Seed, Policy |
    Export-Csv (Join-Path $sessionAbs 'batch_results.csv') -NoTypeInformation -Encoding UTF8
$means = @(
    $rows | Group-Object Model, Policy | ForEach-Object {
        $g = @($_.Group)
        [PSCustomObject]@{
            Model = $g[0].Model
            Policy = $g[0].Policy
            Runs = $g.Count
            MeanAccuracyPct = [math]::Round(($g | Measure-Object AccuracyPct -Average).Average, 3)
            MeanLogicalTimeSec = [math]::Round(($g | Measure-Object LogicalTimeSec -Average).Average, 3)
            MeanSkipRate = [math]::Round(($g | Measure-Object SkipRate -Average).Average, 4)
        }
    }
)
$means | Sort-Object Model, Policy |
    Export-Csv (Join-Path $sessionAbs 'mean_results.csv') -NoTypeInformation -Encoding UTF8
$means | Sort-Object Model, Policy | Format-Table -AutoSize
Write-Host "Completed: $($rows.Count) / $(8 * $Seeds.Count) policy runs" -ForegroundColor Green
Write-Host "CSV: $sessionAbs\batch_results.csv" -ForegroundColor Green
Write-Host "Means: $sessionAbs\mean_results.csv" -ForegroundColor Green
