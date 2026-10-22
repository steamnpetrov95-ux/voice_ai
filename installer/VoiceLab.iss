; Compile after PyInstaller has created dist\VoiceLab.
; One downloadable installer; Python and Qt are included in the application.
#ifndef AppVersion
  #define AppVersion "0.3.3"
#endif

[Setup]
AppId={{2432E01A-A21E-48E1-A9DA-A30E0E296B10}
AppName=npetrov VoiceLab
AppVersion={#AppVersion}
AppPublisher=npetrov
AppPublisherURL=https://github.com/steamnpetrov95-ux/voice_ai
DefaultDirName={localappdata}\Programs\VoiceLab
DefaultGroupName=npetrov VoiceLab
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
#if VER >= EncodeVer(7, 0, 0)
SetupArchitecture=x64
#endif
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist\installer
OutputBaseFilename=VoiceLab-Setup
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\VoiceLab.exe
CloseApplications=yes
SetupLogging=yes

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
#ifdef IntegrateNeuralVoices
Name: "neuralvoices"; Description: "{cm:InstallNeuralVoices}"; GroupDescription: "{cm:AdditionalIcons}"
#endif
#ifdef IntegrateVBCable
Name: "virtualmic"; Description: "{cm:InstallVirtualMic}"; GroupDescription: "{cm:AdditionalIcons}"
#endif

[CustomMessages]
english.InstallNeuralVoices=Install six offline Russian neural voices
russian.InstallNeuralVoices=Установить шесть нейросетевых голосов русской озвучки (работают без интернета)
english.InstallVirtualMic=Install the VB-CABLE virtual microphone (Internet and administrator permission required)
english.InstallingVirtualMic=Installing the virtual microphone...
english.DriverFailed=npetrov VoiceLab is installed. Virtual microphone setup did not finish. You can retry it from npetrov VoiceLab.
russian.InstallVirtualMic=Установить виртуальный микрофон VB-CABLE (нужны интернет и права администратора)
russian.InstallingVirtualMic=Установка виртуального микрофона...
russian.DriverFailed=npetrov VoiceLab установлен. Установка виртуального микрофона не завершена. Её можно повторить из npetrov VoiceLab.

[Files]
Source: "..\dist\VoiceLab\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#ifdef IntegrateNativeSDK
Source: "..\dist\native-sdk\*"; DestDir: "{app}\native-sdk"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif
#ifdef IntegrateNeuralVoices
Source: "..\dist\neural-models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs; Tasks: neuralvoices
#endif

[InstallDelete]
; Remove only the shortcuts created by older versions of this application.
Type: files; Name: "{autoprograms}\VoiceLab.lnk"
Type: files; Name: "{autodesktop}\VoiceLab.lnk"

[Icons]
Name: "{autoprograms}\npetrov VoiceLab"; Filename: "{app}\VoiceLab.exe"
Name: "{autodesktop}\npetrov VoiceLab"; Filename: "{app}\VoiceLab.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VoiceLab.exe"; Description: "{cm:LaunchProgram,npetrov VoiceLab}"; Flags: nowait postinstall skipifsilent; Check: CanLaunchApp

[Code]
var
  DriverNeedsRestart: Boolean;

function CanLaunchApp: Boolean;
begin
  Result := not DriverNeedsRestart;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  Parameters: String;
begin
#ifdef IntegrateVBCable
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('virtualmic') then
  begin
    WizardForm.StatusLabel.Caption := CustomMessage('InstallingVirtualMic');
    Parameters := '-NoProfile -ExecutionPolicy Bypass -File "' +
      ExpandConstant('{app}\Install-VirtualMicrophone.ps1') + '"';
    if Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
        Parameters, '', SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode) then
    begin
      if (ResultCode = 3010) or (ResultCode = 1641) then
        DriverNeedsRestart := True
      else if ResultCode <> 0 then
        MsgBox(CustomMessage('DriverFailed'), mbInformation, MB_OK);
    end
    else
      MsgBox(CustomMessage('DriverFailed'), mbInformation, MB_OK);
  end;
#endif
end;

function NeedRestart: Boolean;
begin
  Result := DriverNeedsRestart;
end;
