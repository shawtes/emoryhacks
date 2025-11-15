@echo off
REM Startup script for the API server (Windows)
REM This script navigates to the emoryhacks directory and starts the API

echo ========================================
echo Starting Backend API Server
echo ========================================
echo.

REM Change to emoryhacks directory
cd /d "%~dp0\emoryhacks"

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found!
    echo Make sure you're running this from the project root.
    echo Expected location: shawtestclone\emoryhacks\requirements.txt
    pause
    exit /b 1
)

echo Current directory: %CD%
echo.

REM Activate virtual environment if it exists
if exist "..\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ..\venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found!
    echo Creating one now...
    python -m venv ..\venv
    call ..\venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting FastAPI server...
echo API will be available at http://localhost:8000
echo API docs available at http://localhost:8000/docs
echo Press Ctrl+C to stop the server
echo.

REM Start the API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

