$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "VoiceLab requires 64-bit Windows."
}

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Executable $($Arguments -join ' ')"
    }
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($launcher) {
        Invoke-Checked $launcher.Source @("-3.12", "-m", "venv", ".venv")
    } elseif ($python) {
        Invoke-Checked $python.Source @("-c", "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), 'Install Python 3.12 x64'")
        Invoke-Checked $python.Source @("-m", "venv", ".venv")
    } else {
        throw "Install Python 3.12 x64 from https://www.python.org/downloads/windows/ (include the py launcher), then run Start-VoiceLab.bat again."
    }
}

Invoke-Checked $venvPython @("-c", "import struct, sys; assert struct.calcsize('P') == 8, 'Use Python x64'; assert (3, 11) <= sys.version_info[:2] < (3, 14), 'Use Python 3.12'")
# Frozen transitive dependencies, with artifact hashes. The lock includes Windows wheels.
Invoke-Checked $venvPython @("-m", "pip", "install", "--require-hashes", "-r", "requirements.lock")
Invoke-Checked $venvPython @("-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", ".")
Write-Host "VoiceLab is installed. Start: Start-VoiceLab.bat"
