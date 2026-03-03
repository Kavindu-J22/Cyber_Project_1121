# Setup and Run All Services for Zero Trust Telehealth Platform
# This script creates virtual environments, installs dependencies, and starts all services

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Zero Trust Telehealth Platform Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if a port is in use
function Test-Port {
    param([int]$Port)
    $connection = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue -InformationLevel Quiet
    return $connection
}

# Function to setup Python ML service
function Setup-MLService {
    param(
        [string]$ServiceName,
        [string]$Path,
        [int]$Port
    )
    
    Write-Host "[$ServiceName] Setting up..." -ForegroundColor Yellow
    
    # Check if venv exists
    if (-not (Test-Path "$Path\venv")) {
        Write-Host "[$ServiceName] Creating virtual environment..." -ForegroundColor Gray
        Push-Location $Path
        python -m venv venv
        Pop-Location
    } else {
        Write-Host "[$ServiceName] Virtual environment already exists" -ForegroundColor Green
    }
    
    # Install requirements
    Write-Host "[$ServiceName] Installing requirements..." -ForegroundColor Gray
    Push-Location $Path
    & ".\venv\Scripts\Activate.ps1"
    pip install -r requirements.txt --quiet
    Pop-Location
    
    Write-Host "[$ServiceName] Setup complete!" -ForegroundColor Green
}

# Function to start ML service
function Start-MLService {
    param(
        [string]$ServiceName,
        [string]$Path,
        [int]$Port,
        [string]$Command = "python main.py api"
    )
    
    Write-Host "[$ServiceName] Starting on port $Port..." -ForegroundColor Yellow
    
    # Check if port is already in use
    if (Test-Port -Port $Port) {
        Write-Host "[$ServiceName] Already running on port $Port" -ForegroundColor Green
        return
    }
    
    # Start service in new window
    $scriptBlock = "cd '$Path' ; .\venv\Scripts\Activate.ps1 ; $Command ; pause"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $scriptBlock
    
    Write-Host "[$ServiceName] Started in new window" -ForegroundColor Green
}

# Setup all ML services
Write-Host "`n=== Setting up ML Services ===" -ForegroundColor Cyan

Setup-MLService -ServiceName "Face Verification" -Path "face verification" -Port 8004
Setup-MLService -ServiceName "Voiceprint Analysis" -Path "Voiceprint Analysis" -Port 8001
Setup-MLService -ServiceName "Keystroke Dynamics" -Path "Keystroke Dynamics" -Port 8002
Setup-MLService -ServiceName "Mouse Movement" -Path "Mouse Movement Analysis" -Port 8003

# Setup Backend
Write-Host "`n=== Setting up Backend ===" -ForegroundColor Cyan
if (-not (Test-Path "Backend\node_modules")) {
    Write-Host "[Backend] Installing npm dependencies..." -ForegroundColor Gray
    Push-Location Backend
    npm install
    Pop-Location
} else {
    Write-Host "[Backend] Dependencies already installed" -ForegroundColor Green
}

# Setup Frontend
Write-Host "`n=== Setting up Frontend ===" -ForegroundColor Cyan
if (-not (Test-Path "Client\node_modules")) {
    Write-Host "[Frontend] Installing npm dependencies..." -ForegroundColor Gray
    Push-Location Client
    npm install
    Pop-Location
} else {
    Write-Host "[Frontend] Dependencies already installed" -ForegroundColor Green
}

Write-Host "`n=== Starting All Services ===" -ForegroundColor Cyan

# Start ML Services
Start-MLService -ServiceName "Face Verification" -Path "face verification" -Port 8004
Start-MLService -ServiceName "Voiceprint Analysis" -Path "Voiceprint Analysis" -Port 8001
Start-MLService -ServiceName "Keystroke Dynamics" -Path "Keystroke Dynamics" -Port 8002
Start-MLService -ServiceName "Mouse Movement" -Path "Mouse Movement Analysis" -Port 8003

# Start Backend
Write-Host "[Backend] Starting on port 5000..." -ForegroundColor Yellow
if (-not (Test-Port -Port 5000)) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd Backend ; npm start ; pause"
    Write-Host "[Backend] Started in new window" -ForegroundColor Green
} else {
    Write-Host "[Backend] Already running on port 5000" -ForegroundColor Green
}

# Start Frontend
Write-Host "[Frontend] Starting on port 5173..." -ForegroundColor Yellow
if (-not (Test-Port -Port 5173)) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd Client ; npm run dev ; pause"
    Write-Host "[Frontend] Started in new window" -ForegroundColor Green
} else {
    Write-Host "[Frontend] Already running on port 5173" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All Services Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Yellow
Write-Host "  Frontend:           http://localhost:5173" -ForegroundColor White
Write-Host "  Backend:            http://localhost:5000" -ForegroundColor White
Write-Host "  Face Verification:  http://localhost:8004/docs" -ForegroundColor White
Write-Host "  Voiceprint:         http://localhost:8001/docs" -ForegroundColor White
Write-Host "  Keystroke:          http://localhost:8002/docs" -ForegroundColor White
Write-Host "  Mouse Movement:     http://localhost:8003/docs" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

