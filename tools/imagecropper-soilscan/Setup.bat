@echo off
title SoilScan Setup
color 0A

:: Change to the directory where this batch file is located
cd /d "%~dp0"

echo.
echo  ============================================================
echo                     SOILSCAN SETUP
echo  ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo  [ERROR] Python is not installed!
    echo.
    echo  Please install Python 3.8+ from:
    echo  https://python.org/downloads
    echo.
    echo  IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

echo  [OK] Python found
python --version
echo.

:: Create venv if needed
if not exist "venv" (
    echo  Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        color 0C
        echo  [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created
    echo.
)

:: Install dependencies
echo  Installing dependencies (this may take a few minutes)...
echo.
call venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

if errorlevel 1 (
    color 0C
    echo.
    echo  [ERROR] Failed to install dependencies
    echo  Try running as Administrator
    pause
    exit /b 1
)

:: Verify installation
echo.
echo  Verifying installation...
python -c "import tkinter, PIL, rembg, onnxruntime, zipfile, json" 2>nul
if errorlevel 1 (
    color 0E
    echo  [WARNING] Some modules may not have installed correctly
    echo  Try running Setup.bat again
    echo.
) else (
    echo  [OK] All core modules verified
    echo.
)

echo  ============================================================
echo                    SETUP COMPLETE!
echo  ============================================================
echo.
echo  You can now launch SoilScan by double-clicking:
echo.
echo      SoilScan.pyw
echo.
echo  ============================================================
echo.
echo  NEW FEATURES IN THIS VERSION:
echo.
echo  [+] Multi-select image export (Ctrl+Click, Shift+Click)
echo  [+] Export Selected - Create ZIP from selected images
echo  [+] Export All Edited - Create ZIP from all processed images
echo  [+] Export tracking - Visual indicators (○ ✓ ⬆)
echo  [+] Export history log - All exports logged automatically
echo  [+] Auto Save - Automatically saves when you finish drawing
echo  [+] Auto Next - Automatically advances to next image
echo  [+] Keyboard shortcuts - Process images faster!
echo.
echo  Status Indicators:
echo      ○ Pending    ✓ Edited    ⬆ Exported
echo.
echo  Keyboard Shortcuts:
echo      Enter: Apply Lasso    Ctrl+Enter: AI+Lasso
echo      Space/→: Next         ←: Previous
echo      Del/Esc: Clear Lasso
echo.
echo  ============================================================
echo.
pause
