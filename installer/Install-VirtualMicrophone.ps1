# The driver is downloaded from its publisher, not redistributed in VoiceLab.
# Run the original, signed vendor installer so its prompts remain visible.
[CmdletBinding()]
param([switch]$VerifyOnly)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Test-CableInstalled {
    $devices = @(Get-CimInstance -ClassName Win32_SoundDevice -ErrorAction Stop |
        Where-Object { $_.Name -match 'VB-Audio Virtual Cable' -and $_.Status -eq 'OK' })
    return $devices.Count -gt 0
}

$temporaryDirectory = $null
try {
    if (-not $VerifyOnly -and (Test-CableInstalled)) {
        Write-Host "VB-CABLE is already installed."
        exit 0
    }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $VerifyOnly -and -not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        $arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $PSCommandPath
        $elevated = Start-Process -FilePath "$PSHOME\powershell.exe" -ArgumentList $arguments -Verb RunAs -Wait -PassThru
        exit $elevated.ExitCode
    }

    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $temporaryDirectory = Join-Path $env:TEMP ("VoiceLab-VBCABLE-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null
    $archive = Join-Path $temporaryDirectory "VBCABLE_Driver_Pack45.zip"
    $download = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"
    Write-Host "Downloading the official VB-CABLE driver..."
    Invoke-WebRequest -UseBasicParsing -Uri $download -OutFile $archive -TimeoutSec 180
    $extracted = Join-Path $temporaryDirectory "driver"
    Expand-Archive -LiteralPath $archive -DestinationPath $extracted
    $installers = @(Get-ChildItem -LiteralPath $extracted -Recurse -File -Filter "VBCABLE_Setup_x64.exe")
    if ($installers.Count -ne 1) { throw "The official package does not contain one x64 driver installer." }
    $vendorInstaller = $installers[0].FullName
    $signature = Get-AuthenticodeSignature -LiteralPath $vendorInstaller
    if ($signature.Status -ne 'Valid' -or
        $signature.SignerCertificate.Subject -notmatch '(?i)(VB[- ]?AUDIO|VINCENT\s+BUREL|BUREL\s+VINCENT|V\.\s*BUREL)') {
        throw "The VB-CABLE publisher signature could not be verified ($($signature.Status)): $($signature.StatusMessage). Nothing was executed."
    }
    if ($VerifyOnly) {
        Write-Host "Official VB-CABLE download and publisher signature verified. No driver was installed."
        exit 0
    }
    Write-Host "In the VB-CABLE window, choose Install Driver."
    $install = Start-Process -FilePath $vendorInstaller -Wait -PassThru
    if ($install.ExitCode -notin @(0, 3010, 1641)) {
        throw "VB-CABLE installer failed with code $($install.ExitCode)."
    }
    if (Test-CableInstalled) {
        Write-Host "VB-CABLE is installed. Choose CABLE Output in Discord or your game."
        exit 0
    }
    # The vendor installer may create the endpoint only after a restart.
    # This is a reboot request, not a claim that device readiness was verified.
    Write-Host "Restart Windows, then check the virtual microphone in VoiceLab."
    exit 3010
} catch {
    if ($env:GITHUB_ACTIONS -eq 'true') {
        $message = $_.Exception.Message.Replace('%', '%25').Replace("`r", '%0D').Replace("`n", '%0A')
        Write-Host "::error title=VB-CABLE verification::$message"
    }
    Write-Host "Virtual microphone setup could not complete: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "VoiceLab stays installed. Retry virtual microphone setup from the app."
    if (-not $VerifyOnly) { Read-Host "Press Enter to close" }
    exit 1
} finally {
    if ($temporaryDirectory -and (Test-Path -LiteralPath $temporaryDirectory)) {
        Remove-Item -LiteralPath $temporaryDirectory -Recurse -Force -ErrorAction SilentlyContinue
    }
}
