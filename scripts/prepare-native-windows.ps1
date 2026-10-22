# Restore the private compatibility pack from our already validated release.
# The original third-party installer asset is no longer available. Only our
# hash-pinned installer is run, silently, without the driver or GUI launch.

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Set-Location (Split-Path $PSScriptRoot -Parent)
gh release download v0.3.2 --pattern VoiceLab-Setup.exe --dir .analysis/input --clobber
if ($LASTEXITCODE -ne 0) { throw 'Preserved VoiceLab release download failed' }
$installer = (Resolve-Path .analysis/input/VoiceLab-Setup.exe).Path
if ((Get-FileHash $installer -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'f01c5f863c18878b07624e182946b44b777ce1155244062fcb26f119e599f2aa') { throw 'Unexpected preserved release hash' }
$previousDir = Join-Path (Get-Location) '.analysis/previous-app'
$install = Start-Process -FilePath $installer -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', '/TASKS=""', "/DIR=$previousDir" -Wait -PassThru
if ($install.ExitCode -ne 0) { throw 'Preserved release extraction failed' }
try {
    New-Item -ItemType Directory -Path dist -Force | Out-Null
    Copy-Item (Join-Path $previousDir 'native-sdk') dist/native-sdk -Recurse -Force
} finally {
    $uninstaller = Join-Path $previousDir 'unins000.exe'
    $uninstall = Start-Process -FilePath $uninstaller -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' -Wait -PassThru
    if ($uninstall.ExitCode -ne 0) { throw 'Preserved release cleanup failed' }
}
if (-not (Test-Path dist/native-sdk/manifest.json)) { throw 'Native SDK manifest is missing' }
Invoke-WebRequest 'https://raw.githubusercontent.com/dev6699/rvc-onnx/9d20b0ff54df72e83c2826e76c2e1eff5f1ab824/assets/input_1.wav' -OutFile .analysis/speech.wav
