@echo off
REM Startup script for the frontend (Windows)
REM This script navigates to the webapp directory and starts the dev server

echo ========================================
echo Starting Frontend Development Server
echo ========================================
echo.

REM Change to webapp directory
cd /d "%~dp0\webapp"

REM Check if we're in the right directory
if not exist "package.json" (
    echo ERROR: package.json not found!
    echo Make sure you're running this from the project root.
    echo Expected location: shawtestclone\webapp\package.json
    pause
    exit /b 1
)

echo Current directory: %CD%
echo.

REM Install dependencies if node_modules doesn't exist
if not exist "node_modules" (
    echo Installing dependencies...
    echo This may take a few minutes...
    call npm install
    if errorlevel 1 (
        echo ERROR: npm install failed!
        pause
        exit /b 1
    )
    echo.
)

echo Starting Vite development server...
echo Frontend will be available at http://localhost:3000
echo Press Ctrl+C to stop the server
echo.

REM Start the development server
call npm run dev

