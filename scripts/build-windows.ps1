param([string]$CompilerPath = "")

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python -ErrorAction Stop).Source
}

if (-not $CompilerPath) {
    $compilerCommand = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($compilerCommand) {
        $CompilerPath = $compilerCommand.Source
    } else {
        foreach ($directory in @("Inno Setup 6", "Inno Setup 7")) {
            $candidate = Join-Path ${env:ProgramFiles(x86)} "$directory\ISCC.exe"
            if (Test-Path $candidate) { $CompilerPath = $candidate; break }
        }
    }
}
if (-not $CompilerPath -or -not (Test-Path $CompilerPath)) {
    throw "Install Inno Setup 6 or 7 (https://jrsoftware.org/isdl.php), then run Build-Windows.bat again."
}

& $python -m PyInstaller --noconfirm --clean --windowed --onedir --name VoiceLab --paths src --collect-all pedalboard --collect-all sounddevice --collect-all soundfile scripts/desktop_entry.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }
Copy-Item installer/Install-VirtualMicrophone.ps1 dist/VoiceLab/Install-VirtualMicrophone.ps1
$appVersion = (& $python -c "from voicelab import __version__; print(__version__)").Trim()
if ($LASTEXITCODE -ne 0) { throw "Could not read app version" }
& $CompilerPath "/DAppVersion=$appVersion" installer/VoiceLab.iss
if ($LASTEXITCODE -ne 0) { throw "Installer compiler failed ($LASTEXITCODE)" }

$installer = Join-Path (Get-Location) "dist\installer\VoiceLab-Setup.exe"
$digest = (Get-FileHash $installer -Algorithm SHA256).Hash.ToLowerInvariant()
"$digest  VoiceLab-Setup.exe" | Set-Content -Encoding ascii "$installer.sha256"
Write-Host "Ready: $installer"
