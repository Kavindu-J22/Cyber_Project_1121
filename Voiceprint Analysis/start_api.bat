@echo off
echo ========================================
echo Voiceprint Analysis API Server
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
    echo Please run: python -m venv venv
    echo.
)

REM Start the API server
echo Starting API server...
echo API will be available at: http://localhost:8001
echo Interactive docs at: http://localhost:8001/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python main.py api

pause

