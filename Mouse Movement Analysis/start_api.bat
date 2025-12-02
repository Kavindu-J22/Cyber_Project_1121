@echo off
REM Startup script for Mouse Movement Analysis API (Windows)

echo ========================================
echo Mouse Movement Analysis API
echo Zero Trust Telehealth Platform
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo.

REM Check if model checkpoint exists
if not exist "checkpoints\best_model.pth" (
    echo WARNING: No trained model found at checkpoints\best_model.pth
    echo Please train the model first using: python train.py
    echo.
    echo The API will start but may not function properly without a trained model.
    echo.
    pause
)

REM Start the API server
echo Starting Mouse Movement Analysis API...
echo API will be available at: http://localhost:8003
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn src.api:app --host 0.0.0.0 --port 8003 --reload

REM Deactivate virtual environment on exit
deactivate

