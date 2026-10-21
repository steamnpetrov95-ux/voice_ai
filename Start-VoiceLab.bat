@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup-windows.ps1"
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m voicelab
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo VoiceLab could not start. Read the error above.
echo Setup / repair: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
pause
exit /b 1
