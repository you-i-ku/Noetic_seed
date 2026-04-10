@echo off
chcp 65001 >nul 2>&1

set VENV=%~dp0.venv

if not exist "%VENV%\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo.
"%VENV%\Scripts\python.exe" "%~dp0server.py"
echo.
echo [Process exited.]
pause
