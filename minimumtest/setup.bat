@echo off
for /f "tokens=*" %%i in ('python --version 2^>"%TEMP%\pycheck.txt"') do set PYVER=%%i
if "%PYVER%"=="" goto :nopy
if "%PYVER%"=="Python" goto :nopy
python "%~dp0_setup.py"
goto :end

:nopy
echo Installing Python via winget...
winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
echo Please close this window and run setup.bat again.
pause

:end
