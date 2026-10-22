# Prepare the private, user-supplied native compatibility pack on Windows.

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Set-Location (Split-Path $PSScriptRoot -Parent)
gh release download forai --pattern 'DubbingAI*.exe' --dir .analysis/input --clobber
if ($LASTEXITCODE -ne 0) { throw 'Installer download failed' }
gh release download 1.9 --repo dscharrer/innoextract --pattern innoextract-1.9-windows.zip --dir .analysis/tool --clobber
if ($LASTEXITCODE -ne 0) { throw 'Extractor download failed' }
$installer = (Get-ChildItem .analysis/input/*.exe | Select-Object -First 1).FullName
if ((Get-FileHash $installer -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'af7ea3aa9c743473d0cb927e9d388da192af64681870b3c2aedc11a7e987f231') { throw 'Unexpected installer hash' }
if ((Get-FileHash .analysis/tool/innoextract-1.9-windows.zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne '6989342c9b026a00a72a38f23b62a8e6a22cc5de69805cf47d68ac2fec993065') { throw 'Unexpected extractor hash' }
Expand-Archive .analysis/tool/innoextract-1.9-windows.zip .analysis/tool/unpacked -Force
& .analysis/tool/unpacked/innoextract.exe --extract --silent --output-dir .analysis/extracted $installer
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed' }
Invoke-WebRequest 'https://raw.githubusercontent.com/dev6699/rvc-onnx/9d20b0ff54df72e83c2826e76c2e1eff5f1ab824/assets/input_1.wav' -OutFile .analysis/speech.wav
python scripts/prepare-native-sdk.py --source .analysis/extracted/app
if ($LASTEXITCODE -ne 0) { throw 'Native preparation failed' }
