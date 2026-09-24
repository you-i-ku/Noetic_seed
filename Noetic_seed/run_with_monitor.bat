@echo off
chcp 65001 >nul 2>&1

rem === Noetic_seed launcher: server.py + PC (browser) monitor ===
rem - server.py starts in its own window (profile-select mode, WS on :8765)
rem - monitor.html opens in the default browser with ?auto=1 (auto connect)
rem - Pick the profile from the browser screen.

set VENV=%~dp0.venv
set MONITOR=%~dp0..\Noetic_seed_monitor\web\monitor.html
set WM_DEBUG=1

if not exist "%VENV%\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at %VENV%
    echo Please run setup.bat first.
    pause
    exit /b 1
)

if not exist "%MONITOR%" (
    echo [ERROR] Monitor page not found:
    echo   %MONITOR%
    pause
    exit /b 1
)

rem --- normalize the monitor path into a file:/// URL (backslash -^> slash) ---
call :toURL "%MONITOR%"

echo.
echo === Noetic_seed + PC Monitor ===
echo   Server:   server.py  (ws://localhost:8765)
echo   Monitor:  %MONITOR%
echo   WM_DEBUG: 1
echo.

rem --- 1. server in its own window (keep it open to read the log / Ctrl+C there) ---
start "Noetic Seed Server" cmd /k ""%VENV%\Scripts\python.exe" "%~dp0server.py""

rem --- 2. give the WS port a moment, then open the browser ---
rem     (the page itself also retries for ~18s, so a short wait is enough)
timeout /t 3 /nobreak >nul
start "" "%MONURL%?auto=1"

echo   Launched. Pick a profile in the browser.
echo   To stop: close the "Noetic Seed Server" window (or press Ctrl+C there).
echo.
timeout /t 4 /nobreak >nul
exit /b 0

:toURL
set "MONURL=%~f1"
set "MONURL=%MONURL:\=/%"
set "MONURL=file:///%MONURL%"
goto :eof
