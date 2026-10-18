# Analyze actual per-client mode changes from DynFL client_decisions.csv.
# PowerShell 5.1+; read-only for training output. No DynFL source edits.
param(
    [Parameter(Mandatory=$true)][string]$Root,
    [string]$Policy = 'full_dynfl',
    [string]$OutputDir = ''
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
    throw "Run directory not found: $Root"
}
$files = @(Get-ChildItem -LiteralPath $Root -Recurse -File -Filter 'client_decisions.csv' |
    Where-Object { $_.Directory.Name -eq $Policy } |
    Sort-Object LastWriteTime, FullName)
if ($files.Count -eq 0) {
    throw "No $Policy/client_decisions.csv found under: $Root. Complete at least one checkpoint chunk first."
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path (Resolve-Path -LiteralPath $Root).Path ('mode_analysis_' + $Policy)
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Later checkpoint logs replace older duplicate client/round keys. Never mix policies.
$byKey = @{}
$duplicateCount = 0
$conflictCount = 0
foreach ($file in $files) {
    foreach ($r in (Import-Csv -LiteralPath $file.FullName)) {
        if ($null -eq $r.round -or $null -eq $r.client_id -or $null -eq $r.mode) {
            throw "Missing round/client_id/mode column: $($file.FullName)"
        }
        $round = [int]$r.round
        $id = [string]$r.client_id
        $mode = [string]$r.mode
        $key = "$round`:$id"
        if ($byKey.ContainsKey($key)) {
            $duplicateCount++
            if ($byKey[$key].Mode -ne $mode) { $conflictCount++ }
        }
        $byKey[$key] = [pscustomobject]@{
            Round = $round; ClientId = $id; Mode = $mode
        }
    }
}
$rows = @($byKey.Values)
$roundStats = @(
    foreach ($g in ($rows | Group-Object Round | Sort-Object { [int]$_.Name })) {
        $round = [int]$g.Name
        $total = $g.Count
        foreach ($mg in ($g.Group | Group-Object Mode | Sort-Object Name)) {
            [pscustomobject]@{
                Round = $round + 1 # human-readable; internal CSV is 0-indexed
                Mode = $mg.Name
                Clients = $mg.Count
                SharePct = [math]::Round(100.0 * $mg.Count / $total, 2)
            }
        }
    }
)

# Only adjacent recorded rounds can establish a mode switch.
$pairs = @(
    foreach ($cg in ($rows | Group-Object ClientId)) {
        $history = @($cg.Group | Sort-Object Round)
        for ($i = 1; $i -lt $history.Count; $i++) {
            $a = $history[$i-1]
            $b = $history[$i]
            if ($b.Round -ne ($a.Round + 1)) { continue }
            $kind = if ($a.Mode -eq $b.Mode) { 'same' }
                    elseif ($a.Mode -eq 'SKIP') { 'resume_from_SKIP' }
                    elseif ($b.Mode -eq 'SKIP') { 'enter_SKIP' }
                    else { 'mode_switch' }
            [pscustomobject]@{
                FromRound = $a.Round + 1
                ToRound = $b.Round + 1
                ClientId = $cg.Name
                FromMode = $a.Mode
                ToMode = $b.Mode
                Kind = $kind
            }
        }
    }
)
$roundSwitch = @(
    foreach ($g in ($pairs | Group-Object ToRound | Sort-Object { [int]$_.Name })) {
        $pg = @($g.Group)
        [pscustomobject]@{
            FromRound = [int]$g.Name - 1
            ToRound = [int]$g.Name
            ComparedClients = $pg.Count
            ModeSwitch = @($pg | Where-Object Kind -eq 'mode_switch').Count
            EnterSKIP = @($pg | Where-Object Kind -eq 'enter_SKIP').Count
            ResumeFromSKIP = @($pg | Where-Object Kind -eq 'resume_from_SKIP').Count
            Unchanged = @($pg | Where-Object Kind -eq 'same').Count
        }
    }
)
$transitions = @(
    foreach ($g in ($pairs | Group-Object FromMode, ToMode | Sort-Object Name)) {
        $first = $g.Group[0]
        [pscustomobject]@{
            FromMode = $first.FromMode
            ToMode = $first.ToMode
            Count = $g.Count
        }
    }
)
$perClient = @(
    foreach ($g in ($pairs | Group-Object ClientId)) {
        [pscustomobject]@{
            ClientId = $g.Name
            ModeSwitch = @($g.Group | Where-Object Kind -eq 'mode_switch').Count
            EnterSKIP = @($g.Group | Where-Object Kind -eq 'enter_SKIP').Count
            ResumeFromSKIP = @($g.Group | Where-Object Kind -eq 'resume_from_SKIP').Count
        }
    }
)
$roundStats | Export-Csv -Path (Join-Path $OutputDir 'mode_by_round.csv') -NoTypeInformation -Encoding UTF8
$roundSwitch | Export-Csv -Path (Join-Path $OutputDir 'switch_by_round.csv') -NoTypeInformation -Encoding UTF8
$transitions | Export-Csv -Path (Join-Path $OutputDir 'mode_transitions.csv') -NoTypeInformation -Encoding UTF8
$perClient | Export-Csv -Path (Join-Path $OutputDir 'switch_by_client.csv') -NoTypeInformation -Encoding UTF8

Write-Host "Policy: $Policy | CSV inputs: $($files.Count) | distinct client-round rows: $($rows.Count)"
if ($duplicateCount -gt 0) {
    Write-Host "Overlapping checkpoint rows resolved: $duplicateCount (different modes: $conflictCount; newest file wins)"
}
Write-Host 'Mode distribution by round:'
$roundStats | Format-Table -AutoSize | Out-Host
Write-Host 'Adjacent-round changes (SKIP is NOT a mode switch):'
$roundSwitch | Format-Table -AutoSize | Out-Host
Write-Host 'Transition totals:'
$transitions | Format-Table -AutoSize | Out-Host
Write-Host "CSV reports: $OutputDir"
