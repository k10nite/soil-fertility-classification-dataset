# SoilScan - Portable Setup Guide

This tool is **completely self-contained and portable**. Copy this folder anywhere and it will work!

## Quick Start

### First Time Setup (One-time only)

1. **Copy this folder anywhere you want**
   - Rename it to anything you like (the name doesn't matter!)
   - `imagecropper-soilscan`, `soilscan`, `my-cropper` - all work fine!

2. **Run Setup.bat** (only needed once)
   ```
   Double-click: Setup.bat
   ```
   This creates a virtual environment and installs all dependencies (PIL, rembg, onnxruntime, etc.)

3. **Launch the tool**
   ```
   Double-click: SoilScan.pyw
   ```
   Or use: SoilScan.bat (shows console for debugging)

## Features

### Self-Contained Design
The tool includes everything it needs:
- `soilscan_lite.py` - Main application
- `SoilScan.pyw` - GUI launcher (no console)
- `SoilScan.bat` - Console launcher (with status messages)
- `Setup.bat` - One-time setup installer
- `requirements.txt` - Python dependencies list
- `venv/` - Virtual environment (created on first setup)

### Works From Any Location
You can:
- Copy the folder anywhere on your computer
- Rename the folder to anything you want
- Move it to a USB drive
- Copy it to multiple locations
- **It just works!**

### No Hardcoded Paths
The launchers use smart path detection:
- `cd /d "%~dp0"` in batch files - Always runs from own directory
- `Path(__file__).parent` in Python - Finds files relative to script location
- No configuration needed

## File Overview

```
imagecropper-soilscan/           (← you can rename this!)
├── soilscan_lite.py             - Main application (70KB)
├── SoilScan.pyw                 - Silent launcher (no console)
├── SoilScan.bat                 - Console launcher (with messages)
├── Setup.bat                    - First-time setup
├── requirements.txt             - Dependencies list
├── README.md                    - User guide
├── LICENSE                      - MIT license
├── .gitignore                   - Git ignore rules
└── venv/                        - Virtual environment (auto-created)
```

## Launch Options

### Option 1: SoilScan.pyw (Recommended)
- Double-click to launch
- No console window
- Clean, professional launch
- **Use this for normal operation**

### Option 2: SoilScan.bat (Debugging)
- Shows console window
- Displays status messages
- Good for troubleshooting
- **Use if something goes wrong**

## Tool Features

### Image Processing
- **AI Background Removal** - Automatic soil extraction using rembg
- **Freeform Lasso** - Draw around soil samples manually
- **Hybrid Mode** - Combine AI + manual selection
- **Zoom Mode** - Detailed manual editing
- **Multi-select Export** - Export multiple images at once

### Workflow Automation
- **Auto Save** - Automatically saves when you finish drawing
- **Auto Next** - Automatically advances to next image
- **Keyboard Shortcuts** - Process images faster
  - Enter: Apply Lasso
  - Ctrl+Enter: AI + Lasso
  - Space/→: Next image
  - ←: Previous image
  - Del/Esc: Clear selection

### Export Options
- **Export Selected** - Create ZIP from selected images (Ctrl+Click, Shift+Click)
- **Export All Edited** - Create ZIP from all processed images
- **Export Tracking** - Visual indicators (○ Pending, ✓ Edited, ⬆ Exported)
- **Export History** - All exports logged automatically

## Troubleshooting

### "Python not found"
- Install Python from https://python.org/downloads/
- **Important:** Check "Add Python to PATH" during installation
- Restart your computer after installing

### "Failed to create venv"
- Make sure Python is installed correctly
- Try running Setup.bat as Administrator
- Check that you have write permissions in the folder

### "Module not found" errors
- Run `Setup.bat` first to install dependencies
- Make sure you have an internet connection (pip needs to download packages)
- Try deleting the `venv` folder and running Setup.bat again

### AI processing is slow
- First run downloads the AI model (~100MB)
- Subsequent runs are much faster
- DirectML acceleration used if available (GPU)

### Application won't start
1. Check the console window for errors (use SoilScan.bat)
2. Verify all dependencies installed successfully
3. Try deleting `venv` folder and re-running Setup.bat
4. Make sure all files are in the same folder

## Advanced Usage

### Running from Command Line

If you prefer command line over batch files:

```bash
# Navigate to the tool directory
cd path/to/your-renamed-folder

# First time setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Launch GUI
python soilscan_lite.py
```

### Copying to Multiple Locations

Need the tool in multiple places? Just copy the entire folder:
```
C:\Projects\soil-study-1\soilscan\
C:\Projects\soil-study-2\my-cropper\
D:\backup\imagecropper\
```

Each copy works independently with its own virtual environment!

### Sharing with Colleagues

To share with someone else:
1. Copy the entire folder (exclude `venv` to save space)
2. Send them the folder
3. They run Setup.bat on their computer
4. Done!

The `venv` folder is auto-created per installation, so you don't need to share it.

## Technical Details

### How Portability Works

1. **Batch files find themselves:**
   ```batch
   cd /d "%~dp0"
   ```
   This changes to the directory where the batch file lives.

2. **Python launcher finds itself:**
   ```python
   script_dir = Path(__file__).parent
   ```
   The Python script locates its own directory.

3. **Virtual environment is local:**
   - Each copy has its own `venv/` folder
   - No global Python pollution
   - Dependencies isolated per installation

4. **Result:** True portability!

### Directory-Agnostic Design

The tool doesn't care about:
- What you name the folder
- Where you put the folder
- What the parent directories are called
- Whether you're on C:\ or D:\ drive

It only cares that all files stay together in the same folder.

### Virtual Environment Benefits

Using a virtual environment means:
- **Isolation** - Dependencies don't conflict with system Python
- **Portability** - Each installation is independent
- **Clean** - Easy to delete (just remove the folder)
- **Safe** - No risk of breaking system Python packages

## System Requirements

- **OS:** Windows 7 or later
- **Python:** 3.8 or higher
- **RAM:** 4GB minimum, 8GB recommended (for AI processing)
- **Storage:** ~500MB (including virtual environment)
- **Internet:** Required for first-time setup (downloads dependencies)

## Support

For detailed usage instructions, see `README.md` in this folder.

For setup/installation issues, verify:
1. Python 3.8+ is installed and in PATH
2. Internet connection is working (for pip)
3. All files are in the same folder
4. You ran `Setup.bat` at least once

## What's New in This Version

- ✅ **Portable batch files** - Added `cd /d "%~dp0"` for true portability
- ✅ **Multi-select export** - Ctrl+Click, Shift+Click to select multiple images
- ✅ **Export tracking** - Visual indicators (○ ✓ ⬆) show what's been exported
- ✅ **Export history** - All exports logged automatically
- ✅ **Auto Save** - Automatically saves when you finish drawing
- ✅ **Auto Next** - Automatically advances to next image
- ✅ **Keyboard shortcuts** - Process images faster!

---

**Last Updated:** March 16, 2026
**Version:** Portable Edition
**License:** MIT
