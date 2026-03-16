@echo off
REM Launch Augmentation GUI without console window

REM Change to the directory where this batch file is located
cd /d "%~dp0"

REM Launch GUI using pythonw (no console window)
start "" pythonw augmentation_gui.py

REM Exit immediately
exit
