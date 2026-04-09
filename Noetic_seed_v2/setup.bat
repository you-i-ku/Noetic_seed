@echo off
chcp 65001 >nul 2>&1
echo [Noetic_seed_v2] Setting up virtual environment...

set VENV=%~dp0.venv
if not exist "%VENV%\Scripts\python.exe" (
    python -m venv "%VENV%"
)
"%VENV%\Scripts\pip.exe" install -q -r "%~dp0requirements.txt"
echo [Setup complete]
pause
