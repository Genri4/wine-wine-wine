<#
.SYNOPSIS
    Run the generated_stress_dev AITUNNEL pilot without writing the API key to disk.

.DESCRIPTION
    The key is entered as a masked SecureString, exposed only to the child WSL
    process through AITUNNEL_API_KEY/WSLENV, and removed from this PowerShell
    process in the finally block. The default scope is one product × four
    existing scenarios. This wrapper cannot select the full 600-row plan.
#>

[CmdletBinding()]
param(
    [ValidateRange(1, 5)]
    [int]$PilotProducts = 1,
    [ValidatePattern('^[A-Za-z0-9._/-]+$')]
    [string]$Model = 'gpt-image-2.5-sunburst',
    [ValidatePattern('^[A-Za-z0-9._/-]+$')]
    [string]$GeneratedDir = 'data/benchmarks/generated_stress_dev/generated_raw',
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$drive = $projectRoot.Substring(0, 1).ToLowerInvariant()
$linuxProjectRoot = "/mnt/$drive" + $projectRoot.Substring(2).Replace('\', '/')

$secureKey = Read-Host "AITUNNEL_API_KEY" -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
$previousKey = $env:AITUNNEL_API_KEY
$previousWslEnv = $env:WSLENV

try {
    $env:AITUNNEL_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    $wslEntries = @()
    if ($previousWslEnv) {
        $wslEntries = @($previousWslEnv -split ':' | Where-Object { $_ -and $_ -ne 'AITUNNEL_API_KEY' })
    }
    $env:WSLENV = (@($wslEntries) + 'AITUNNEL_API_KEY') -join ':'

    $command = "cd '$linuxProjectRoot' && .venv/bin/python scripts/generate_stress_aitunnel.py --pilot-products $PilotProducts --model $Model --generated-dir '$GeneratedDir'"
    if ($Overwrite) {
        $command += ' --overwrite'
    }
    & wsl.exe bash -lc $command
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    if ($null -eq $previousKey) {
        Remove-Item Env:AITUNNEL_API_KEY -ErrorAction SilentlyContinue
    } else {
        $env:AITUNNEL_API_KEY = $previousKey
    }
    if ($null -eq $previousWslEnv) {
        Remove-Item Env:WSLENV -ErrorAction SilentlyContinue
    } else {
        $env:WSLENV = $previousWslEnv
    }
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
    Remove-Variable secureKey -ErrorAction SilentlyContinue
}
