@echo off
echo ========================================
echo Zero Trust Telehealth Platform
echo ML Services Startup Script
echo ========================================
echo.

echo This script will start all 4 ML API services in separate terminals.
echo Each service will activate its own virtual environment.
echo.
echo Services:
echo   1. Voice Recognition API (Port 8001)
echo   2. Keystroke Dynamics API (Port 8002)
echo   3. Mouse Movement Analysis API (Port 8003)
echo   4. Face Recognition API (Port 8004)
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

echo.
echo Starting Voice Recognition API (Port 8001)...
start "Voice API - Port 8001" cmd /k "cd /d "%~dp0Voiceprint Analysis" && .\venv\Scripts\Activate.ps1 && python main.py api"
timeout /t 3 /nobreak >nul

echo Starting Keystroke Dynamics API (Port 8002)...
start "Keystroke API - Port 8002" cmd /k "cd /d "%~dp0Keystroke Dynamics" && .\venv\Scripts\Activate.ps1 && python main.py api"
timeout /t 3 /nobreak >nul

echo Starting Mouse Movement API (Port 8003)...
start "Mouse API - Port 8003" cmd /k "cd /d "%~dp0Mouse Movement Analysis" && .\venv\Scripts\Activate.ps1 && python main.py api"
timeout /t 3 /nobreak >nul

echo Starting Face Recognition API (Port 8004)...
start "Face API - Port 8004" cmd /k "cd /d "%~dp0face verification" && .\venv\Scripts\Activate.ps1 && python main.py api"
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo All ML services are starting...
echo ========================================
echo.
echo Please wait for all services to initialize.
echo Check each terminal window for status.
echo.
echo Services should be available at:
echo   - Voice API: http://localhost:8001/health
echo   - Keystroke API: http://localhost:8002/health
echo   - Mouse API: http://localhost:8003/health
echo   - Face API: http://localhost:8004/health
echo.
echo Press any key to exit this window...
pause >nul

