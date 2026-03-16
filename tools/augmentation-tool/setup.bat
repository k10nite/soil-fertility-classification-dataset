@echo off
REM ============================================================
REM Soil Fertility Augmentation Pipeline - Setup Script
REM ============================================================

echo.
echo ============================================================
echo  Soil Fertility Augmentation Pipeline - Setup
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Python detected:
python --version
echo.

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not installed!
    echo Please install pip or reinstall Python with pip included.
    pause
    exit /b 1
)

echo [2/3] Installing dependencies from requirements.txt...
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo [3/3] Verifying installation...
python -c "import cv2; import PIL; import numpy; import albumentations; import tqdm; print('All dependencies installed successfully!')"

if errorlevel 1 (
    echo.
    echo [WARNING] Some dependencies failed to import!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup Complete!
echo ============================================================
echo.
echo You can now run start.bat to launch the augmentation tool.
echo.
pause
