; Compile after PyInstaller has created dist\VoiceLab.
; One downloadable installer; Python and Qt are included in the application.
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{2432E01A-A21E-48E1-A9DA-A30E0E296B10}
AppName=VoiceLab
AppVersion={#AppVersion}
AppPublisher=VoiceLab
AppPublisherURL=https://github.com/steamnpetrov95-ux/voice_ai
DefaultDirName={localappdata}\Programs\VoiceLab
DefaultGroupName=VoiceLab
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
Compression=lzma2
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
Name: "virtualmic"; Description: "{cm:InstallVirtualMic}"; GroupDescription: "{cm:AdditionalIcons}"

[CustomMessages]
english.InstallVirtualMic=Install the VB-CABLE virtual microphone (Internet and administrator permission required)
english.InstallingVirtualMic=Installing the virtual microphone...
english.DriverFailed=VoiceLab is installed. Virtual microphone setup did not finish. You can retry it from VoiceLab.
russian.InstallVirtualMic=Установить виртуальный микрофон VB-CABLE (нужны интернет и права администратора)
russian.InstallingVirtualMic=Установка виртуального микрофона...
russian.DriverFailed=VoiceLab установлен. Установка виртуального микрофона не завершена. Её можно повторить из VoiceLab.

[Files]
Source: "..\dist\VoiceLab\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\VoiceLab"; Filename: "{app}\VoiceLab.exe"
Name: "{autodesktop}\VoiceLab"; Filename: "{app}\VoiceLab.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VoiceLab.exe"; Description: "{cm:LaunchProgram,VoiceLab}"; Flags: nowait postinstall skipifsilent; Check: CanLaunchApp

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
end;

function NeedRestart: Boolean;
begin
  Result := DriverNeedsRestart;
end;
