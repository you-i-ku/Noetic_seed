@echo off
chcp 932 >nul 2>&1

set VENV=%~dp0.venv

if not exist "%VENV%\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

"%VENV%\Scripts\python.exe" "%~dp0main.py"
echo.
echo [Process exited. Check above for errors.]
pause
\r