# Complete Setup Guide - Zero Trust Telehealth Platform

## 🎯 Quick Start (5 Services to Run)

This platform requires **5 services** to run simultaneously:

1. **Voice Recognition API** (Python) - Port 8001
2. **Keystroke Dynamics API** (Python) - Port 8002
3. **Mouse Movement API** (Python) - Port 8003
4. **Backend Server** (Node.js) - Port 5000
5. **Frontend App** (React) - Port 5173

---

## 📋 Step-by-Step Installation

### Step 1: Install Python ML Services

Open **3 separate terminals** for the Python services:

#### Terminal 1 - Voice Recognition API
```bash
cd "Voiceprint Analysis"
pip install -r requirements.txt
python main.py api
```
✅ Should see: `Uvicorn running on http://0.0.0.0:8001`

#### Terminal 2 - Keystroke Dynamics API
```bash
cd "Keystroke Dynamics"
pip install -r requirements.txt
python main.py api
```
✅ Should see: `Uvicorn running on http://0.0.0.0:8002`

#### Terminal 3 - Mouse Movement API
```bash
cd "Mouse Movement Analysis"
pip install -r requirements.txt
python main.py api
```
✅ Should see: `Uvicorn running on http://0.0.0.0:8003`

---

### Step 2: Install Backend Server

Open a **4th terminal**:

```bash
cd Backend
npm install
npm run dev
```

✅ Should see:
```
✓ MongoDB Connected: cluster0.xgvsfnu.mongodb.net
╔════════════════════════════════════════════════════════════╗
║   Zero Trust Telehealth Platform - Backend Server         ║
║   Server running on port 5000                              ║
╚════════════════════════════════════════════════════════════╝
```

---

### Step 3: Install Frontend App

Open a **5th terminal**:

```bash
cd Client
npm install
npm run dev
```

✅ Should see:
```
  VITE v5.0.8  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

---

## 🌐 Access the Application

Open your browser and navigate to:
```
http://localhost:5173
```

---

## 🔍 Verify All Services Are Running

### Check Python ML APIs

1. **Voice API**: http://localhost:8001/health
2. **Keystroke API**: http://localhost:8002/health
3. **Mouse API**: http://localhost:8003/health

### Check Backend API

http://localhost:5000/api/health

Should return:
```json
{
  "success": true,
  "message": "Server is running",
  "mlServices": {
    "voice": true,
    "keystroke": true,
    "mouse": true
  }
}
```

---

## 🎓 First Time Usage

### 1. Register a Doctor Account

1. Go to http://localhost:5173/register
2. Fill in the form:
   - **Personal Info**: First name, Last name, Email, Password
   - **Professional Info**: License number, Specialization, Years of experience
   - **Biometric Enrollment**:
     - **Voice**: Click "Start Recording", speak for 5-10 seconds, click "Stop Recording"
     - **Keystroke**: Click "Start Capture", type the phrase shown, click "Stop Capture" (do this 3 times)
     - **Mouse**: Click "Start Capture", move mouse naturally in the box for 10 seconds, click "Stop Capture"
3. Click "Complete Registration"

### 2. Login

1. Go to http://localhost:5173/login
2. Enter your email and password
3. Click "Sign In"

### 3. Start a Consultation

1. From the dashboard, click "Start Consultation"
2. Allow camera and microphone access when prompted
3. You'll see:
   - Your video feed
   - Real-time trust score (top right)
   - Verification status (right sidebar)
   - Biometric verification logs

### 4. During Consultation

The system will continuously verify:
- **Voice**: When you speak
- **Keystroke**: When you type (try typing in the browser console or any text field)
- **Mouse**: As you move your mouse

Watch the trust score and verification logs update in real-time!

---

## 🐛 Troubleshooting

### Python APIs Not Starting

**Problem**: `ModuleNotFoundError` or import errors

**Solution**:
```bash
# Make sure you're in the correct directory
cd "Voiceprint Analysis"  # or Keystroke Dynamics, or Mouse Movement Analysis

# Install dependencies
pip install -r requirements.txt

# If still failing, try:
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Backend Not Connecting to MongoDB

**Problem**: `MongoDB Connection Error`

**Solution**:
- Check your internet connection
- Verify the MongoDB URI in `Backend/.env`
- The provided URI should work: `mongodb+srv://kavindu:kavindu@cluster0.xgvsfnu.mongodb.net/zero-trust-telehealth`

### Frontend Not Loading

**Problem**: Blank page or errors

**Solution**:
```bash
cd Client
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Camera/Microphone Not Working

**Problem**: "Permission denied" or no video

**Solution**:
- Allow camera/microphone permissions in your browser
- Use Chrome or Firefox (recommended)
- Make sure no other app is using the camera

### ML Services Health Check Failing

**Problem**: Red X marks on dashboard

**Solution**:
1. Verify all 3 Python APIs are running
2. Check the terminal outputs for errors
3. Restart the Python services one by one
4. Refresh the dashboard

---

## 📊 Expected Behavior

### Registration
- Voice sample should be captured as a WAV file
- Keystroke patterns should show 3/3 samples
- Mouse pattern should show "✓ Captured"
- Registration should complete successfully

### Login
- Should redirect to dashboard immediately
- Should show your name and specialization

### Dashboard
- Should show ML services status (all green checkmarks)
- Should show your biometric enrollment status (all green)
- Should show list of registered doctors

### Meeting
- Video should start automatically
- Trust score should start at 100%
- Verification logs should appear every 10 seconds
- Biometric status bars should show green

---

## 🎯 Testing the System

### Test Scenario 1: Normal Operation
1. Start a consultation
2. Speak naturally
3. Type occasionally
4. Move mouse normally
5. **Expected**: Trust score stays above 80%, all verifications pass

### Test Scenario 2: Suspicious Activity
1. Start a consultation
2. Have someone else speak (different voice)
3. **Expected**: Voice verification fails, trust score drops, alerts appear

---

## 💡 Tips

1. **Keep all 5 terminals open** while using the application
2. **Check terminal outputs** for any errors
3. **Use Chrome or Firefox** for best compatibility
4. **Allow camera/microphone** when prompted
5. **Wait for all services** to fully start before using the app

---

## 📞 Support

If you encounter issues:
1. Check all 5 services are running
2. Review terminal outputs for errors
3. Verify MongoDB connection
4. Check browser console for frontend errors
5. Ensure Python dependencies are installed

---

**Happy Testing! 🚀**

