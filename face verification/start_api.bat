@echo off
REM Start Face Verification API Server
REM Zero Trust Telehealth Platform

echo ========================================
echo Face Verification API - Starting
echo ========================================
echo.

cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Check if model file exists
if not exist "models\best_model.pt" (
    echo.
    echo WARNING: Model file not found!
    echo Expected location: models\best_model.pt
    echo.
    echo Please place your trained model file in the models folder.
    echo.
    pause
    exit /b 1
)

echo Starting Face Verification API on port 8004...
echo.

python main.py api

pause
