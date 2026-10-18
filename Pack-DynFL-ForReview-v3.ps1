param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $preferred = "E:\YTT\GROUP\DynFL-privacy"
    if (Test-Path -LiteralPath $preferred) {
        $ProjectRoot = $preferred
    } else {
        $ProjectRoot = (Get-Location).Path
    }
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$desktop = [Environment]::GetFolderPath("Desktop")
if ([string]::IsNullOrWhiteSpace($desktop) -or !(Test-Path -LiteralPath $desktop)) {
    $desktop = $env:USERPROFILE
}
$stage = Join-Path $env:TEMP "DVR_$stamp"
$zipPath = Join-Path $desktop "DynFL_review_$stamp.zip"

$null = New-Item -ItemType Directory -Path $stage -Force
$null = New-Item -ItemType Directory -Path (Join-Path $stage "source") -Force
$null = New-Item -ItemType Directory -Path (Join-Path $stage "runtime_evidence") -Force

$sourceExtensions = @(
    ".py", ".pyi", ".ps1", ".bat", ".cmd", ".sh",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".md", ".rst", ".txt", ".patch", ".diff", ".html", ".sql"
)
$excludedDirRegex = @(
    '^(?i)\.git$', '^(?i)\.idea$', '^(?i)\.vscode$',
    '^(?i)\.venv$', '^(?i)venv$', '^(?i)env$',
    '^(?i)__pycache__$', '^(?i)node_modules$', '^(?i)site-packages$',
    '^(?i)\.pytest_cache$', '^(?i)\.mypy_cache$', '^(?i)\.ruff_cache$',
    '^(?i)\.tox$', '^(?i)\.nox$',
    '^(?i)\.console_backup_.*$', '^(?i)\.dashboard_backup_.*$',
    '^(?i)_cleanup_reports$', '^(?i)_cleanup_upload_tools$',
    '^(?i)out$', '^(?i)runs$', '^(?i)logs$', '^(?i)results$', '^(?i)artifacts$',
    '^(?i)datasets?$', '^(?i)data$', '^(?i)checkpoints?$',
    '^(?i)tensorboard$', '^(?i)wandb$', '^(?i)mlruns$',
    '^(?i)\.cache$', '^(?i)cache$', '^(?i)tmp$', '^(?i)temp$'
)
$secretNameRegex = '(?i)(^\.env($|\.)|(^|[._-])(secret|credential|password|token)([._-]|$)|id_rsa|id_ed25519|private.?key|\.pem$|\.pfx$|\.p12$|\.key$|\.keystore$)'
$skipExtensions = @(
    ".pt", ".pth", ".ckpt", ".onnx", ".pb", ".bin", ".safetensors",
    ".npy", ".npz", ".pkl", ".pickle", ".joblib",
    ".csv", ".parquet", ".feather", ".sqlite", ".db",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".mp4", ".avi",
    ".xlsx", ".xls", ".pptx", ".docx"
)
$maxSourceBytes = 2MB
$maxEvidenceBytes = 5MB
$maxEvidenceFiles = 80
$maxEvidenceTotalBytes = 50MB
# Keep staged file paths short enough for Windows PowerShell / Compress-Archive.
$maxDestinationPathLength = 220

$omitted = New-Object System.Collections.Generic.List[string]
$manifest = New-Object System.Collections.Generic.List[object]
$sourceCount = 0
$evidenceCount = 0
$evidenceBytes = 0L
$copyFailureCount = 0

function Get-RelativePathSafe([string]$Base, [string]$Full) {
    $baseUri = New-Object System.Uri(($Base.TrimEnd('\') + '\'))
    $fullUri = New-Object System.Uri($Full)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($fullUri).ToString()).Replace('/', '\')
}
function Test-ExcludedPath([string]$RelativePath) {
    $parts = $RelativePath -split '[\\/]'
    foreach ($part in $parts) {
        foreach ($rx in $excludedDirRegex) {
            if ($part -match $rx) { return $true }
        }
    }
    return $false
}
$secretContentRegexes = @(
    '(?i)\bAKIA[0-9A-Z]{16}\b',
    '(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{32,}',
    '(?i)-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    '(?i)://[^/\s:@]+:[^/\s@]+@',
    '(?i)(api[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token)\s*[:=]\s*["''][A-Za-z0-9/_+=.-]{20,}["'']'
)

Write-Host "Scanning project: $ProjectRoot"
Write-Host "Enumerating files. Original files will not be modified."
$allFiles = @(Get-ChildItem -LiteralPath $ProjectRoot -Recurse -File -Force -ErrorAction SilentlyContinue)
Write-Host ("Found {0} files. Collecting source files..." -f $allFiles.Count)

foreach ($file in $allFiles) {
    $rel = Get-RelativePathSafe $ProjectRoot $file.FullName
    if (Test-ExcludedPath $rel) { continue }
    if ($file.Extension.ToLowerInvariant() -notin $sourceExtensions) { continue }
    if ($file.Extension.ToLowerInvariant() -in $skipExtensions) { continue }
    if ($file.Length -gt $maxSourceBytes) {
        $omitted.Add("SOURCE_TOO_LARGE`t$rel`t$($file.Length) bytes")
        continue
    }
    if ($file.Name -match $secretNameRegex) {
        $omitted.Add("SENSITIVE_FILENAME_SKIPPED`t$rel")
        continue
    }

    $target = Join-Path (Join-Path $stage "source") $rel
    if ($target.Length -gt $maxDestinationPathLength) {
        $omitted.Add("PATH_TOO_LONG_SKIPPED`t$rel`tTargetLength=$($target.Length)")
        continue
    }

    $secretFound = $false
    try {
        $content = Get-Content -LiteralPath $file.FullName -Raw -ErrorAction Stop
        foreach ($rx in $secretContentRegexes) {
            if ($content -match $rx) { $secretFound = $true; break }
        }
    } catch {
        $omitted.Add("UNREADABLE_SOURCE_SKIPPED`t$rel")
        continue
    }
    if ($secretFound) {
        $omitted.Add("POSSIBLE_SECRET_CONTENT_SKIPPED_REVIEW_MANUALLY`t$rel")
        continue
    }

    try {
        $targetParent = Split-Path -Parent $target
        $null = New-Item -ItemType Directory -Path $targetParent -Force
        Copy-Item -LiteralPath $file.FullName -Destination $target -Force -ErrorAction Stop
        $sourceCount++
        $manifest.Add([pscustomobject]@{
            Kind = "source"
            RelativePath = $rel
            SizeBytes = $file.Length
            LastWriteTime = $file.LastWriteTime.ToString("s")
        })
    } catch {
        $copyFailureCount++
        $omitted.Add("SOURCE_COPY_FAILED`t$rel`t$($_.Exception.Message)")
        continue
    }
}
Write-Host ("Collected source/config/docs: {0}" -f $sourceCount)

Write-Host "Selecting recent runtime evidence..."
$evidenceNameRegex = '(?i)^(live_status\.json|online_j_ORACLE_NONCERTIFIED_round_.*\.json|.*(audit|summary|result|manifest|resolved_config|run_config|stage\d+_tests|test_results?).*\.(json|txt|log|md)|.*\.(log|out|err))$'
$evidenceDirs = @('out', 'runs', 'logs', 'results', 'artifacts')
$evidenceCandidates = @(
    foreach ($file in $allFiles) {
        $rel = Get-RelativePathSafe $ProjectRoot $file.FullName
        $parts = $rel -split '[\\/]'
        $insideEvidenceDir = $false
        foreach ($part in $parts) {
            if ($evidenceDirs -contains $part.ToLowerInvariant()) {
                $insideEvidenceDir = $true
                break
            }
        }
        if (!$insideEvidenceDir) { continue }
        if ($file.Name -notmatch $evidenceNameRegex) { continue }
        if ($file.Length -gt $maxEvidenceBytes) { continue }
        if ($file.Name -match $secretNameRegex) { continue }
        [pscustomobject]@{ File = $file; RelativePath = $rel }
    }
) | Sort-Object { $_.File.LastWriteTime } -Descending

foreach ($item in ($evidenceCandidates | Select-Object -First $maxEvidenceFiles)) {
    if (($evidenceBytes + $item.File.Length) -gt $maxEvidenceTotalBytes) {
        $omitted.Add("EVIDENCE_TOTAL_SIZE_LIMIT`t$($item.RelativePath)")
        continue
    }
    $target = Join-Path (Join-Path $stage "runtime_evidence") $item.RelativePath
    if ($target.Length -gt $maxDestinationPathLength) {
        $omitted.Add("EVIDENCE_PATH_TOO_LONG_SKIPPED`t$($item.RelativePath)")
        continue
    }

    $secretFound = $false
    try {
        $content = Get-Content -LiteralPath $item.File.FullName -Raw -ErrorAction Stop
        foreach ($rx in $secretContentRegexes) {
            if ($content -match $rx) { $secretFound = $true; break }
        }
    } catch {
        $omitted.Add("UNREADABLE_EVIDENCE_SKIPPED`t$($item.RelativePath)")
        continue
    }
    if ($secretFound) {
        $omitted.Add("POSSIBLE_SECRET_EVIDENCE_SKIPPED_REVIEW_MANUALLY`t$($item.RelativePath)")
        continue
    }

    try {
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force
        Copy-Item -LiteralPath $item.File.FullName -Destination $target -Force -ErrorAction Stop
        $evidenceCount++
        $evidenceBytes += $item.File.Length
        $manifest.Add([pscustomobject]@{
            Kind = "runtime_evidence"
            RelativePath = $item.RelativePath
            SizeBytes = $item.File.Length
            LastWriteTime = $item.File.LastWriteTime.ToString("s")
        })
    } catch {
        $copyFailureCount++
        $omitted.Add("EVIDENCE_COPY_FAILED`t$($item.RelativePath)`t$($_.Exception.Message)")
        continue
    }
}

$gitInfo = "Git status not collected or repository is not initialized. Git is not required for this package."
if ((Test-Path -LiteralPath (Join-Path $ProjectRoot ".git")) -and (Get-Command git -ErrorAction SilentlyContinue)) {
    try {
        $gitStatus = (& git -C $ProjectRoot status --short --branch 2>&1 | Out-String).Trim()
        $gitInfo = "Git status (informational only):`r`n$gitStatus"
    } catch {
        $gitInfo = "Found .git metadata, but could not read Git status."
    }
}

$readme = @"
DynFL review package
Project root: $ProjectRoot
Created at: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz")
Source files copied: $sourceCount
Runtime evidence files copied: $evidenceCount
Runtime evidence bytes copied: $evidenceBytes
Copy failures recorded: $copyFailureCount

Contents:
source\             Source/config/docs including untracked files.
runtime_evidence\   Selected small audit/status/result/log files.
MANIFEST.csv        Exact paths, sizes and modification times.
OMITTED_FILES.txt   Skipped large, sensitive, too-deep or failed files.

Safety:
- Original project files were not modified or deleted.
- Common model/data/archive/media files and dependency folders were excluded.
- Secret detection is best-effort, not a guarantee. Review the archive before uploading.
- This is a review bundle, not a runnable deployment package.

$gitInfo
"@
Set-Content -LiteralPath (Join-Path $stage "README_FOR_REVIEW.txt") -Value $readme -Encoding UTF8
if ($manifest.Count -gt 0) {
    $manifest | Export-Csv -LiteralPath (Join-Path $stage "MANIFEST.csv") -NoTypeInformation -Encoding UTF8
} else {
    "Kind,RelativePath,SizeBytes,LastWriteTime" | Set-Content -LiteralPath (Join-Path $stage "MANIFEST.csv") -Encoding UTF8
}
if ($omitted.Count -gt 0) {
    $omitted | Sort-Object -Unique | Set-Content -LiteralPath (Join-Path $stage "OMITTED_FILES.txt") -Encoding UTF8
} else {
    "No files were omitted by filters." | Set-Content -LiteralPath (Join-Path $stage "OMITTED_FILES.txt") -Encoding UTF8
}

Write-Host "Creating ZIP..."
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -CompressionLevel Optimal
if (!(Test-Path -LiteralPath $zipPath)) {
    throw "ZIP creation failed. Staging folder remains at: $stage"
}
Remove-Item -LiteralPath $stage -Recurse -Force

Write-Host ""
Write-Host "PACKAGING COMPLETED" -ForegroundColor Green
Write-Host ("ZIP: {0}" -f $zipPath) -ForegroundColor Yellow
Write-Host ("Source files: {0}" -f $sourceCount)
Write-Host ("Runtime evidence files: {0}" -f $evidenceCount)
Write-Host ("Skipped/failed entries: {0}" -f $omitted.Count)
Write-Host "Inspect README_FOR_REVIEW.txt and OMITTED_FILES.txt before uploading."
