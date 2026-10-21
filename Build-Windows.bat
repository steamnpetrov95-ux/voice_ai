@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup-windows.ps1"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed --onedir --name VoiceLab --paths src --collect-all pedalboard --collect-all sounddevice --collect-all soundfile scripts\desktop_entry.py
if errorlevel 1 goto failed
echo.
echo Ready: dist\VoiceLab\VoiceLab.exe
echo Copy the ENTIRE dist\VoiceLab folder. It runs without Python.
pause
exit /b 0
:failed
echo.
echo Build failed. Read the error above.
pause
exit /b 1
