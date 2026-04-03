@echo off
TITLE NPK Rule Engine Launcher
SETLOCAL

:: Automatically detect the directory where the .bat file is located
SET PROJ_DIR=%~dp0
cd /d "%PROJ_DIR%"

echo =======================================================
echo          NPK FERTILIZER RULE ENGINE
echo =======================================================
echo.
echo [1/2] Checking dependencies (Streamlit, Pandas)...
:: Installs requirements if missing, stays quiet if already installed
python -m pip install -r requirements.txt --quiet --user

echo [2/2] Launching Web Interface...
echo Local Directory: %PROJ_DIR%
echo.
echo -------------------------------------------------------
:: python -m streamlit ensures the correct environment is used
python -m streamlit run src/app_with_fert_rec.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Streamlit failed to start. 
    echo Please ensure Python is added to your PATH.
    pause
)

ENDLOCAL
pause