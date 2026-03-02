# Running All Services - Zero Trust Telehealth Platform

This guide explains how to set up and run all services for the Zero Trust Telehealth Platform.

## Prerequisites

- **Python 3.8+** installed and in PATH
- **Node.js 16+** installed and in PATH
- **MongoDB** running on localhost:27017 (optional for full functionality)
- **Windows OS** (scripts are Windows-specific)

## Quick Start (Automated)

### Option 1: Install and Run Everything

```batch
# Step 1: Install all dependencies (run once)
install-all-dependencies.bat

# Step 2: Start all services
start-all-services-windows.bat
```

### Option 2: Use PowerShell Script

```powershell
# Run the comprehensive setup and start script
.\setup-and-run-all.ps1
```

## Manual Setup (Step by Step)

### 1. Face Verification ML Model (Port 8004)

```batch
cd "face verification"
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python main.py api
```

**API Docs:** http://localhost:8004/docs

### 2. Voiceprint Analysis ML Model (Port 8001)

```batch
cd "Voiceprint Analysis"
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python main.py api
```

**API Docs:** http://localhost:8001/docs

### 3. Keystroke Dynamics ML Model (Port 8002)

```batch
cd "Keystroke Dynamics"
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python main.py api
```

**API Docs:** http://localhost:8002/docs

### 4. Mouse Movement Analysis ML Model (Port 8003)

```batch
cd "Mouse Movement Analysis"
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python main.py api
```

**API Docs:** http://localhost:8003/docs

### 5. Backend Server (Port 5000)

```batch
cd Backend
npm install
npm start
```

**API:** http://localhost:5000
**Health Check:** http://localhost:5000/api/health

### 6. Frontend Client (Port 5173)

```batch
cd Client
npm install
npm run dev
```

**Frontend:** http://localhost:5173

## Service Ports Summary

| Service | Port | URL | Docs |
|---------|------|-----|------|
| Frontend | 5173 | http://localhost:5173 | - |
| Backend | 5000 | http://localhost:5000 | - |
| Face Verification | 8004 | http://localhost:8004 | /docs |
| Voiceprint Analysis | 8001 | http://localhost:8001 | /docs |
| Keystroke Dynamics | 8002 | http://localhost:8002 | /docs |
| Mouse Movement | 8003 | http://localhost:8003 | /docs |

## Verification Steps

### 1. Check ML Services Health

```powershell
# Face Verification
curl http://localhost:8004/health

# Voiceprint Analysis
curl http://localhost:8001/health

# Keystroke Dynamics
curl http://localhost:8002/health

# Mouse Movement
curl http://localhost:8003/health
```

### 2. Check Backend Health

```powershell
curl http://localhost:5000/api/health
```

### 3. Open Frontend

Navigate to http://localhost:5173 in your browser.

## Troubleshooting

### Port Already in Use

If a port is already in use, you can:
1. Find and kill the process using that port
2. Change the port in the respective `config.yaml` file

```powershell
# Find process using port (e.g., 8004)
netstat -ano | findstr :8004

# Kill process by PID
taskkill /PID <PID> /F
```

### Virtual Environment Issues

If you encounter virtual environment issues:

```batch
# Delete and recreate venv
rmdir /s /q venv
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Missing Dependencies

If dependencies are missing:

```batch
# For Python services
pip install -r requirements.txt

# For Node.js services
npm install
```

### Model Files Missing

Some ML models may require trained model files. Check the respective service's README for:
- Model file locations
- Training instructions
- Pre-trained model downloads

## Stopping Services

- **Individual Windows:** Close the command window for that service
- **All Services:** Close all command windows opened by the start script

## Environment Variables

Create a `.env` file in the Backend folder:

```env
PORT=5000
CLIENT_URL=http://localhost:5173
MONGODB_URI=mongodb://localhost:27017/telehealth
JWT_SECRET=your_jwt_secret_here
NODE_ENV=development
```

## Next Steps

1. ✅ All services running
2. 📝 Register a doctor account
3. 🔐 Enroll biometric data
4. 🎥 Start a telehealth session
5. 📊 Monitor continuous authentication

## Support

For issues or questions, refer to:
- `Documentations/QUICK_START.md`
- `Documentations/SETUP_GUIDE.md`
- Individual service README files

