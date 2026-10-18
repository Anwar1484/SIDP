@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing_python
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1
if errorlevel 1 goto missing_python
".venv\Scripts\python.exe" "scripts\run_monitor.py" %*
exit /b %errorlevel%

:missing_python
echo The project Python environment needs setup or repair.
echo Run: powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
exit /b 1
