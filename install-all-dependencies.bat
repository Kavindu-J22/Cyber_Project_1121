@echo off
REM Install all dependencies for Zero Trust Telehealth Platform
REM This script installs Python and Node.js dependencies for all services

echo ========================================
echo Zero Trust Telehealth Platform
echo Installing All Dependencies
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check Node.js installation
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo Node.js found:
node --version
echo.

REM ========================================
REM Install Face Verification Dependencies
REM ========================================
echo [1/6] Installing Face Verification dependencies...
cd "face verification"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt
call venv\Scripts\deactivate.bat
cd ..
echo Face Verification dependencies installed!
echo.

REM ========================================
REM Install Voiceprint Analysis Dependencies
REM ========================================
echo [2/6] Installing Voiceprint Analysis dependencies...
cd "Voiceprint Analysis"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt
call venv\Scripts\deactivate.bat
cd ..
echo Voiceprint Analysis dependencies installed!
echo.

REM ========================================
REM Install Keystroke Dynamics Dependencies
REM ========================================
echo [3/6] Installing Keystroke Dynamics dependencies...
cd "Keystroke Dynamics"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt
call venv\Scripts\deactivate.bat
cd ..
echo Keystroke Dynamics dependencies installed!
echo.

REM ========================================
REM Install Mouse Movement Dependencies
REM ========================================
echo [4/6] Installing Mouse Movement Analysis dependencies...
cd "Mouse Movement Analysis"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt
call venv\Scripts\deactivate.bat
cd ..
echo Mouse Movement Analysis dependencies installed!
echo.

REM ========================================
REM Install Backend Dependencies
REM ========================================
echo [5/6] Installing Backend dependencies...
cd Backend
if not exist node_modules (
    npm install
) else (
    echo Backend dependencies already installed
)
cd ..
echo Backend dependencies installed!
echo.

REM ========================================
REM Install Frontend Dependencies
REM ========================================
echo [6/6] Installing Frontend dependencies...
cd Client
if not exist node_modules (
    npm install
) else (
    echo Frontend dependencies already installed
)
cd ..
echo Frontend dependencies installed!
echo.

echo ========================================
echo All Dependencies Installed Successfully!
echo ========================================
echo.
echo You can now run the services using:
echo   - start-all-services.bat (to start all services)
echo   - Or start individual services from their folders
echo.
pause

