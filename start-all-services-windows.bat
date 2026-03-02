@echo off
REM Start all services for Zero Trust Telehealth Platform
REM This script starts all ML models, backend, and frontend in separate windows

echo ========================================
echo Zero Trust Telehealth Platform
echo Starting All Services
echo ========================================
echo.

REM Get the current directory
set ROOT_DIR=%cd%

echo Starting services in separate windows...
echo.

REM ========================================
REM Start Face Verification API (Port 8004)
REM ========================================
echo [1/6] Starting Face Verification API on port 8004...
start "Face Verification API - Port 8004" cmd /k "cd /d "%ROOT_DIR%\face verification" && call venv\Scripts\activate.bat && python main.py api"
timeout /t 2 /nobreak >nul

REM ========================================
REM Start Voiceprint Analysis API (Port 8001)
REM ========================================
echo [2/6] Starting Voiceprint Analysis API on port 8001...
start "Voiceprint Analysis API - Port 8001" cmd /k "cd /d "%ROOT_DIR%\Voiceprint Analysis" && call venv\Scripts\activate.bat && python main.py api"
timeout /t 2 /nobreak >nul

REM ========================================
REM Start Keystroke Dynamics API (Port 8002)
REM ========================================
echo [3/6] Starting Keystroke Dynamics API on port 8002...
start "Keystroke Dynamics API - Port 8002" cmd /k "cd /d "%ROOT_DIR%\Keystroke Dynamics" && call venv\Scripts\activate.bat && python main.py api"
timeout /t 2 /nobreak >nul

REM ========================================
REM Start Mouse Movement Analysis API (Port 8003)
REM ========================================
echo [4/6] Starting Mouse Movement Analysis API on port 8003...
start "Mouse Movement Analysis API - Port 8003" cmd /k "cd /d "%ROOT_DIR%\Mouse Movement Analysis" && call venv\Scripts\activate.bat && python main.py api"
timeout /t 2 /nobreak >nul

REM ========================================
REM Start Backend Server (Port 5000)
REM ========================================
echo [5/6] Starting Backend Server on port 5000...
start "Backend Server - Port 5000" cmd /k "cd /d "%ROOT_DIR%\Backend" && npm start"
timeout /t 2 /nobreak >nul

REM ========================================
REM Start Frontend Client (Port 5173)
REM ========================================
echo [6/6] Starting Frontend Client on port 5173...
start "Frontend Client - Port 5173" cmd /k "cd /d "%ROOT_DIR%\Client" && npm run dev"
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo All Services Started!
echo ========================================
echo.
echo Service URLs:
echo   Frontend:           http://localhost:5173
echo   Backend:            http://localhost:5000
echo   Face Verification:  http://localhost:8004/docs
echo   Voiceprint:         http://localhost:8001/docs
echo   Keystroke:          http://localhost:8002/docs
echo   Mouse Movement:     http://localhost:8003/docs
echo.
echo Each service is running in a separate window.
echo Close the window to stop that service.
echo.
echo Press any key to exit this launcher...
pause >nul

