@echo off
chcp 65001 >nul 2>&1

set VENV=%~dp0.venv

if not exist "%VENV%\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

"%VENV%\Scripts\python.exe" "%~dp0_select_profile.py"
if errorlevel 1 (
    pause
    exit /b 1
)

for /f "delims=" %%P in (%~dp0_last_profile.tmp) do set PROFILE=%%P
del "%~dp0_last_profile.tmp" 2>nul

echo.
echo Starting profile: %PROFILE%
echo.
"%VENV%\Scripts\python.exe" "%~dp0profiles\%PROFILE%\main.py"
echo.
echo [Process exited. Check above for errors.]
pause
