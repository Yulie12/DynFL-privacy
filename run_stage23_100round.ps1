$ErrorActionPreference = "Continue"
$root = "E:\YTT\GROUP\DynFL-privacy"
Set-Location -LiteralPath $root

$python = "D:\soft\Python310\python.exe"
$config = "configs/stage21_resnet18_20c_2e_shadow_5r.json"
$outBase = "out\stage23_switch_100round"

New-Item -ItemType Directory -Force -Path $outBase | Out-Null

Start-Transcript -Path (Join-Path $outBase "master.log") -Force | Out-Null

$overall = 0
$env:DYNFL_EXPERIMENTAL_SAMPLE_MULTISTAGE = "1"

try {
    $policies = @(
        "fixed_mode_fixed_privacy",
        "dynamic_mode_fixed_privacy",
        "fixed_mode_dynamic_privacy",
        "full_dynfl"
    )

    foreach ($policy in $policies) {
        Write-Host ""
        Write-Host "=========================================="
        Write-Host "Starting: $policy"
        Write-Host "Rounds: 100 | Seed: 2026"
        Write-Host "=========================================="

        $policyOut = Join-Path $outBase $policy
        $log = Join-Path $outBase "$policy.log"

        New-Item -ItemType Directory -Force -Path $policyOut |
            Out-Null

        & $python -u experiments/run_paper_config.py `
            --config $config `
            --seeds 2026 `
            --policies $policy `
            --rounds 100 `
            --output-root $policyOut *> $log

        $code = $LASTEXITCODE
        Add-Content -LiteralPath $log -Value "`r`nEXIT_CODE=$code"

        if ($code -eq 0) {
            Write-Host "COMPLETED: $policy"
        }
        else {
            Write-Host "FAILED: $policy (exit code $code)"
            $overall = 1
        }
    }
}
finally {
    Remove-Item Env:DYNFL_EXPERIMENTAL_SAMPLE_MULTISTAGE `
        -ErrorAction SilentlyContinue
    Stop-Transcript | Out-Null
}

Add-Content -LiteralPath (Join-Path $outBase "master.log") `
    -Value "OVERALL_EXIT_CODE=$overall"

Write-Host "ALL POLICIES FINISHED. Overall exit code: $overall"
exit $overall
