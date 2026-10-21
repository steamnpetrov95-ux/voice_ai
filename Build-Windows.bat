@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup-windows.ps1"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto failed
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\build-windows.ps1"
if errorlevel 1 goto failed
echo.
echo Ready: dist\installer\VoiceLab-Setup.exe
echo Share this ONE installer. Python is not required on the target computer.
pause
exit /b 0
:failed
echo.
echo Build failed. Read the error above.
pause
exit /b 1
