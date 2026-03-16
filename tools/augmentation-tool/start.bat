@echo off
REM Launch Augmentation GUI

REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ========================================
echo Soil Fertility Augmentation Tool
echo ========================================
echo.
echo Current directory: %CD%
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check for GUI file
if not exist augmentation_gui.py (
    echo ERROR: augmentation_gui.py not found in current directory
    echo Expected location: %CD%\augmentation_gui.py
    echo.
    dir augmentation_gui.py
    pause
    exit /b 1
)

echo Found: augmentation_gui.py
echo.
echo Launching GUI...
echo If the GUI doesn't appear, check for errors below:
echo.

python augmentation_gui.py

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo ERROR: Application exited with error code %errorlevel%
    echo ========================================
    pause
) else (
    echo.
    echo GUI closed normally
)
