# 🚀 START HERE - Zero Trust Telehealth Platform

## Quick Start Guide

Welcome! This guide will help you get all services up and running.

---

## ⚡ Current Status

### ✅ Ready to Run
- **Backend** (Node.js) - Dependencies installed
- **Frontend** (React + Vite) - Dependencies installed

### 🔄 Installing (Background)
- **Face Verification ML** - Python dependencies installing
- **Voiceprint Analysis ML** - Python dependencies installing  
- **Keystroke Dynamics ML** - Python dependencies installing
- **Mouse Movement ML** - Python dependencies installing

**Note:** Python ML services are installing large packages (PyTorch, etc.) in the background. This may take 10-30 minutes.

---

## 🎯 Option 1: Automated Start (Recommended)

### Step 1: Wait for Python Dependencies (if not complete)

Check if installations are done:
```batch
cd "face verification"
venv\Scripts\activate.bat
python -c "import torch; print('Ready!')"
```

If you get an error, wait a few more minutes for the installation to complete.

### Step 2: Start All Services

```batch
start-all-services-windows.bat
```

This opens 6 windows for all services.

### Step 3: Verify Services

```batch
test-all-services.bat
```

Or visit:
- Frontend: http://localhost:5173
- Backend Health: http://localhost:5000/api/health

---

## 🔧 Option 2: Manual Start (If Automated Fails)

### Prerequisites
1. **MongoDB** must be running (for Backend)
   ```batch
   net start MongoDB
   ```
   Or install from: https://www.mongodb.com/try/download/community

### Start Each Service

Open 6 separate Command Prompt windows:

**Window 1 - Face Verification (Port 8004)**
```batch
cd "face verification"
venv\Scripts\activate.bat
python main.py api
```

**Window 2 - Voiceprint Analysis (Port 8001)**
```batch
cd "Voiceprint Analysis"
venv\Scripts\activate.bat
python main.py api
```

**Window 3 - Keystroke Dynamics (Port 8002)**
```batch
cd "Keystroke Dynamics"
venv\Scripts\activate.bat
python main.py api
```

**Window 4 - Mouse Movement (Port 8003)**
```batch
cd "Mouse Movement Analysis"
venv\Scripts\activate.bat
python main.py api
```

**Window 5 - Backend (Port 5000)**
```batch
cd Backend
npm start
```

**Window 6 - Frontend (Port 5173)**
```batch
cd Client
npm run dev
```

---

## 📊 Service Ports

| Service | Port | URL | API Docs |
|---------|------|-----|----------|
| **Frontend** | 5173 | http://localhost:5173 | - |
| **Backend** | 5000 | http://localhost:5000 | - |
| **Face Verification** | 8004 | http://localhost:8004 | /docs |
| **Voiceprint** | 8001 | http://localhost:8001 | /docs |
| **Keystroke** | 8002 | http://localhost:8002 | /docs |
| **Mouse Movement** | 8003 | http://localhost:8003 | /docs |

---

## ✅ Verification Checklist

After starting services, verify each one:

- [ ] Face Verification: http://localhost:8004/docs
- [ ] Voiceprint: http://localhost:8001/docs
- [ ] Keystroke: http://localhost:8002/docs
- [ ] Mouse Movement: http://localhost:8003/docs
- [ ] Backend: http://localhost:5000/api/health
- [ ] Frontend: http://localhost:5173

---

## 🐛 Troubleshooting

### Python dependencies still installing?
**Check:** Open Task Manager and look for Python processes  
**Wait:** Large packages like PyTorch take time to download

### MongoDB connection error?
**Solution:** Install and start MongoDB
```batch
# Install from: https://www.mongodb.com/try/download/community
# Then start:
net start MongoDB
```

### Port already in use?
**Find and kill the process:**
```powershell
netstat -ano | findstr :8004
taskkill /PID <PID> /F
```

### Module not found error?
**Reinstall dependencies:**
```batch
cd "service-folder"
venv\Scripts\activate.bat
pip install -r requirements.txt
```

---

## 📚 Additional Resources

- **Detailed Setup:** `SETUP_STATUS_AND_INSTRUCTIONS.md`
- **Running Guide:** `RUNNING_SERVICES_GUIDE.md`
- **Quick Start:** `Documentations/QUICK_START.md`
- **API Docs:** `Documentations/API_DOCUMENTATION.md`

---

## 🎓 Using the Platform

Once all services are running:

1. **Open Frontend:** http://localhost:5173
2. **Register:** Create a doctor account
3. **Enroll Biometrics:**
   - Upload face images
   - Record voice samples
   - Type password for keystroke pattern
   - Move mouse for behavioral pattern
4. **Start Session:** Begin a telehealth session
5. **Monitor:** Watch continuous authentication in real-time

---

## 🆘 Need Help?

1. Check `SETUP_STATUS_AND_INSTRUCTIONS.md` for detailed status
2. Run `test-all-services.bat` to check which services are running
3. Review individual service README files in their folders
4. Check logs in each service's `logs/` folder

---

## 📝 Scripts Available

- `install-all-dependencies.bat` - Install all dependencies
- `start-all-services-windows.bat` - Start all services
- `test-all-services.bat` - Test if services are running
- `setup-and-run-all.ps1` - PowerShell comprehensive script

---

**Last Updated:** March 2, 2026  
**Status:** Setup in progress - Python dependencies installing

🎉 **You're almost there! Just wait for Python dependencies to finish installing, then run the start script!**

