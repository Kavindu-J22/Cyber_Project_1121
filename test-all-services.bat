@echo off
REM Test all services for Zero Trust Telehealth Platform
REM This script checks if all services are running and responding

echo ========================================
echo Zero Trust Telehealth Platform
echo Service Health Check
echo ========================================
echo.

REM Function to test URL
setlocal enabledelayedexpansion

echo Testing services...
echo.

REM Test Face Verification (Port 8004)
echo [1/6] Testing Face Verification API (Port 8004)...
curl -s http://localhost:8004/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ Face Verification is running
) else (
    echo   ✗ Face Verification is NOT running
)
echo.

REM Test Voiceprint Analysis (Port 8001)
echo [2/6] Testing Voiceprint Analysis API (Port 8001)...
curl -s http://localhost:8001/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ Voiceprint Analysis is running
) else (
    echo   ✗ Voiceprint Analysis is NOT running
)
echo.

REM Test Keystroke Dynamics (Port 8002)
echo [3/6] Testing Keystroke Dynamics API (Port 8002)...
curl -s http://localhost:8002/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ Keystroke Dynamics is running
) else (
    echo   ✗ Keystroke Dynamics is NOT running
)
echo.

REM Test Mouse Movement (Port 8003)
echo [4/6] Testing Mouse Movement API (Port 8003)...
curl -s http://localhost:8003/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ Mouse Movement is running
) else (
    echo   ✗ Mouse Movement is NOT running
)
echo.

REM Test Backend (Port 5000)
echo [5/6] Testing Backend Server (Port 5000)...
curl -s http://localhost:5000/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ Backend is running
) else (
    echo   ✗ Backend is NOT running
)
echo.

REM Test Frontend (Port 5173)
echo [6/6] Testing Frontend Client (Port 5173)...
curl -s http://localhost:5173 >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ Frontend is running
) else (
    echo   ✗ Frontend is NOT running
)
echo.

echo ========================================
echo Health Check Complete
echo ========================================
echo.
echo Service URLs:
echo   Frontend:           http://localhost:5173
echo   Backend:            http://localhost:5000/api/health
echo   Face Verification:  http://localhost:8004/docs
echo   Voiceprint:         http://localhost:8001/docs
echo   Keystroke:          http://localhost:8002/docs
echo   Mouse Movement:     http://localhost:8003/docs
echo.
echo If any service is not running, start it manually or use:
echo   start-all-services-windows.bat
echo.
pause

