@echo off
chcp 65001 >nul 2>&1

set VENV=%~dp0.venv
rem Enable WM_DEBUG for smoke-level observation (propagated via subprocess.run)
set WM_DEBUG=1

if not exist "%VENV%\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo.
echo   WM_DEBUG: 1 (sandbox/wm_debug.jsonl)
"%VENV%\Scripts\python.exe" "%~dp0server.py"
echo.
echo [Process exited.]
pause
