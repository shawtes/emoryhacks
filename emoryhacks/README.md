Speech-based dementia/Alzheimer's screening (research)

This directory contains a Python pipeline for research-only screening using voice recordings. See PLAN.md for the complete plan and milestones.

## Virtual Environment Setup

**IMPORTANT:** This project uses a virtual environment to avoid conflicts with your system Python.

### First Time Setup
1. Create virtual environment: `py -3.13 -m venv venv`
2. Activate it:
   - **PowerShell:** `.\activate_venv.ps1` or `.\venv\Scripts\Activate.ps1`
   - **CMD:** `activate_venv.bat` or `venv\Scripts\activate.bat`
3. Install requirements: `python -m pip install -r emoryhacks\requirements.txt`

### When Returning to Project
**Always activate the virtual environment first!**
- **PowerShell:** `.\activate_venv.ps1`
- **CMD:** `activate_venv.bat`

You'll know it's activated when you see `(venv)` in your terminal prompt.

### Deactivate
When done, type: `deactivate`

## Quick Start (after activating virtual environment)
- Place audio in data/raw/
- Run preprocessing to produce cleaned audio in data/interim/ and features in data/processed/
- Train ML baseline to get cross-validated results in reports/metrics/

Disclaimer: Not a medical device. Research and educational use only.


