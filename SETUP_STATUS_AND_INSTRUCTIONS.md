# Setup Status and Instructions
## Zero Trust Telehealth Platform

**Date:** March 2, 2026  
**Status:** In Progress - Dependencies Installing

---

## Current Status

### ✅ Completed
1. **Virtual Environments Created** for all ML services:
   - Face Verification (`face verification/venv`)
   - Voiceprint Analysis (`Voiceprint Analysis/venv`)
   - Keystroke Dynamics (`Keystroke Dynamics/venv`)
   - Mouse Movement Analysis (`Mouse Movement Analysis/venv`)

2. **Node.js Dependencies Installed**:
   - ✅ Backend (`Backend/node_modules`)
   - ✅ Frontend (`Client/node_modules`)

### 🔄 In Progress
**Python ML Service Dependencies** are currently being installed in the background:
- Face Verification (Terminal 76798)
- Keystroke Dynamics (Terminal 87996)
- Mouse Movement Analysis (Terminal 94652)
- Voiceprint Analysis (Terminal 104097)

These installations may take 10-30 minutes depending on your internet speed and system performance, as they include large packages like PyTorch.

---

## Next Steps - Manual Approach

Since the automated installation is running in the background, you can proceed with the following manual steps:

### Step 1: Wait for Python Dependencies to Install

Check if the installations are complete by trying to import the main packages:

```powershell
# Test Face Verification
cd "face verification"
.\venv\Scripts\Activate.ps1
python -c "import torch, fastapi; print('✓ Dependencies installed')"
deactivate
cd ..

# Test Keystroke Dynamics
cd "Keystroke Dynamics"
.\venv\Scripts\Activate.ps1
python -c "import torch, fastapi; print('✓ Dependencies installed')"
deactivate
cd ..

# Test Mouse Movement
cd "Mouse Movement Analysis"
.\venv\Scripts\Activate.ps1
python -c "import torch, fastapi; print('✓ Dependencies installed')"
deactivate
cd ..

# Test Voiceprint
cd "Voiceprint Analysis"
.\venv\Scripts\Activate.ps1
python -c "import torch, fastapi; print('✓ Dependencies installed')"
deactivate
cd ..
```

### Step 2: Start MongoDB (Required for Backend)

The Backend requires MongoDB to be running. Install and start MongoDB:

```powershell
# If MongoDB is installed, start it:
net start MongoDB

# Or run mongod manually:
mongod --dbpath C:\data\db
```

**Alternative:** If you don't have MongoDB, you can modify the Backend to work without it temporarily.

### Step 3: Start All Services

Once dependencies are installed, use the provided batch script:

```batch
start-all-services-windows.bat
```

This will open 6 separate windows for:
1. Face Verification API (Port 8004)
2. Voiceprint Analysis API (Port 8001)
3. Keystroke Dynamics API (Port 8002)
4. Mouse Movement API (Port 8003)
5. Backend Server (Port 5000)
6. Frontend Client (Port 5173)

### Step 4: Verify Services are Running

Open your browser and check:

- **Frontend:** http://localhost:5173
- **Backend Health:** http://localhost:5000/api/health
- **Face Verification Docs:** http://localhost:8004/docs
- **Voiceprint Docs:** http://localhost:8001/docs
- **Keystroke Docs:** http://localhost:8002/docs
- **Mouse Movement Docs:** http://localhost:8003/docs

---

## Alternative: Start Services Individually

If the batch script doesn't work, start each service manually:

### Face Verification
```batch
cd "face verification"
venv\Scripts\activate.bat
python main.py api
```

### Voiceprint Analysis
```batch
cd "Voiceprint Analysis"
venv\Scripts\activate.bat
python main.py api
```

### Keystroke Dynamics
```batch
cd "Keystroke Dynamics"
venv\Scripts\activate.bat
python main.py api
```

### Mouse Movement Analysis
```batch
cd "Mouse Movement Analysis"
venv\Scripts\activate.bat
python main.py api
```

### Backend
```batch
cd Backend
npm start
```

### Frontend
```batch
cd Client
npm run dev
```

---

## Troubleshooting

### Issue: Python dependencies still installing
**Solution:** Wait for the background installations to complete. You can check Task Manager for Python processes.

### Issue: MongoDB connection error
**Solution:** 
1. Install MongoDB from https://www.mongodb.com/try/download/community
2. Start MongoDB service
3. Or modify `Backend/config/db.js` to skip MongoDB temporarily

### Issue: Port already in use
**Solution:**
```powershell
# Find process using port (e.g., 8004)
netstat -ano | findstr :8004

# Kill process
taskkill /PID <PID> /F
```

### Issue: Module not found errors
**Solution:** Reinstall dependencies:
```batch
cd "service-folder"
venv\Scripts\activate.bat
pip install -r requirements.txt
```

---

## Service Configuration

All services are configured via their respective `config.yaml` files:
- `face verification/config.yaml` - Port 8004
- `Voiceprint Analysis/config.yaml` - Port 8001
- `Keystroke Dynamics/config.yaml` - Port 8002
- `Mouse Movement Analysis/config.yaml` - Port 8003

Backend configuration is in `Backend/.env`

---

## Testing the System

Once all services are running:

1. Open http://localhost:5173
2. Register a new doctor account
3. Enroll biometric data (face, voice, keystroke, mouse)
4. Start a session
5. Monitor continuous authentication

---

## Support Files Created

- `install-all-dependencies.bat` - Install all dependencies
- `start-all-services-windows.bat` - Start all services
- `setup-and-run-all.ps1` - PowerShell setup script
- `RUNNING_SERVICES_GUIDE.md` - Detailed running guide

---

## Estimated Time to Complete

- Python dependencies installation: 10-30 minutes (in progress)
- Starting all services: 2-5 minutes
- Total: 15-35 minutes

---

**Last Updated:** March 2, 2026

