@echo off
REM Start Voiceprint Analysis API Server
REM This script activates the venv and starts the service

echo ========================================
echo Voiceprint Analysis API Server
echo ========================================
echo.

REM Set UTF-8 encoding
chcp 65001 >nul 2>&1

REM Set Python encoding
set PYTHONIOENCODING=utf-8

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated
echo.

REM Start the API server
echo Starting Voiceprint Analysis API...
echo.
python main.py api

REM If the server stops, pause to see any error messages
if errorlevel 1 (
    echo.
    echo ERROR: Server stopped with error code %errorlevel%
    pause
)

