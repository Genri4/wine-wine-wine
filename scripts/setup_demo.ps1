[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$ReleaseTag,

    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$GitHubRepository = 'Genri4/wine-wine-wine'
$AssetName = 'demo_runtime_assets_20260929.tar.gz'
$ExpectedSha256 = '0b29a840c27845feabe3ed5902346b2be3cee70ee649586dcdbcacfe6f41cb8d'
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$AssetUrl = "https://github.com/$GitHubRepository/releases/download/$ReleaseTag/$AssetName"
$ArchivePath = Join-Path $RepositoryRoot "artifacts\$AssetName"

function Invoke-DemoPreflight {
    $wsl = Get-Command 'wsl.exe' -ErrorAction SilentlyContinue
    if ($null -eq $wsl) {
        throw 'WSL is required for this project setup. Install/enable WSL, then rerun this script.'
    }

    $wslRootOutput = & $wsl.Source -e wslpath -a $RepositoryRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not convert the repository path for WSL. Check that a WSL Linux distribution is installed.'
    }
    $wslRoot = ($wslRootOutput -join "`n").Trim()
    if ([string]::IsNullOrWhiteSpace($wslRoot)) {
        throw 'WSL returned an empty repository path.'
    }

    $wslPreflight = "$($wslRoot.TrimEnd('/'))/scripts/check_demo_assets.py"
    & $wsl.Source -e python3 $wslPreflight
    if ($LASTEXITCODE -ne 0) {
        throw 'Demo asset preflight failed. See the messages above and web/README.md.'
    }
}

if (-not (Test-Path (Join-Path $RepositoryRoot 'scripts\check_demo_assets.py'))) {
    throw 'Run this script from a cloned My Wine repository.'
}

$referenceDirectory = Join-Path $RepositoryRoot 'data\processed\reference_images'
$checkpointPath = Join-Path $RepositoryRoot 'artifacts\experiments\so400m_hard_negative_lora_20260923T203613Z\checkpoints\epoch_005.pt'
if (-not $Force -and (Test-Path $referenceDirectory) -and (Test-Path $checkpointPath)) {
    Write-Host 'Runtime asset files already exist; checking them before downloading.'
    Invoke-DemoPreflight
    Write-Host 'Runtime assets are present and passed preflight.'
    Write-Host 'The preflight does not download or validate the Hugging Face/PaddleOCR model caches.'
    return
}

$tar = Get-Command 'tar.exe' -ErrorAction SilentlyContinue
if ($null -eq $tar) {
    throw 'Windows tar.exe was not found. Use a current Windows installation with tar support.'
}

New-Item -ItemType Directory -Force -Path (Split-Path $ArchivePath -Parent) | Out-Null
$temporaryArchive = Join-Path ([System.IO.Path]::GetTempPath()) ("my-wine-runtime-$([guid]::NewGuid().ToString('N')).tar.gz")

try {
    Write-Host "Downloading runtime assets from $AssetUrl"
    $curl = Get-Command 'curl.exe' -ErrorAction SilentlyContinue
    if ($null -ne $curl) {
        & $curl.Source --fail --location --retry 3 --output $temporaryArchive $AssetUrl
        if ($LASTEXITCODE -ne 0) {
            throw "Release asset download failed (exit code $LASTEXITCODE). Check that the release tag and asset have been published publicly."
        }
    }
    else {
        Invoke-WebRequest -Uri $AssetUrl -OutFile $temporaryArchive
    }

    $actualSha256 = (Get-FileHash -Path $temporaryArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $ExpectedSha256) {
        throw "Runtime archive SHA-256 mismatch. Expected $ExpectedSha256, received $actualSha256. Nothing was extracted."
    }
    Write-Host "PASS archive SHA-256: $actualSha256"

    $entries = & $tar.Source -tzf $temporaryArchive
    if ($LASTEXITCODE -ne 0) {
        throw 'The downloaded file is not a readable tar.gz archive.'
    }

    $fileEntries = @()
    foreach ($entry in $entries) {
        $normalizedEntry = ([string]$entry).Replace('\', '/').Trim()
        if ([string]::IsNullOrWhiteSpace($normalizedEntry)) {
            continue
        }
        if ($normalizedEntry.StartsWith('/') -or $normalizedEntry -match '(^|/)\.\.(/|$)' -or $normalizedEntry -match '^[A-Za-z]:') {
            throw "Unsafe path in runtime archive: $normalizedEntry"
        }
        if (-not $normalizedEntry.EndsWith('/')) {
            $fileEntries += $normalizedEntry
        }
    }

    if (-not $Force) {
        $collisions = @($fileEntries | Where-Object {
            Test-Path (Join-Path $RepositoryRoot $_.Replace('/', '\'))
        })
        if ($collisions.Count -gt 0) {
            $sample = ($collisions | Select-Object -First 5) -join ', '
            throw "Refusing to overwrite existing runtime files ($sample). Move them aside or rerun with -Force after reviewing the paths."
        }
    }

    Move-Item -Force -Path $temporaryArchive -Destination $ArchivePath
    & $tar.Source -xzf $ArchivePath -C $RepositoryRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Runtime archive extraction failed.'
    }

    Write-Host 'Running the standard-library asset preflight in WSL.'
    Invoke-DemoPreflight
    Write-Host 'Runtime assets installed and passed preflight.'
    Write-Host 'Next: install/check Python ML dependencies and the pinned SigLIP2 snapshot as described in web/README.md.'
}
finally {
    if (Test-Path $temporaryArchive) {
        Remove-Item -Force $temporaryArchive
    }
}
