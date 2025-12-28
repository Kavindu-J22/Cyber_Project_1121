# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies (One Time Only)

**Windows:**
```bash
install-all.bat
```

**Mac/Linux:**
```bash
# Install Python dependencies
cd "Voiceprint Analysis" && pip install -r requirements.txt && cd ..
cd "Keystroke Dynamics" && pip install -r requirements.txt && cd ..
cd "Mouse Movement Analysis" && pip install -r requirements.txt && cd ..

# Install Node.js dependencies
cd Backend && npm install && cd ..
cd Client && npm install && cd ..
```

---

### Step 2: Start All Services

**Windows:**
```bash
start-all-services.bat
```

**Mac/Linux - Open 5 Terminals:**

Terminal 1:
```bash
cd "Voiceprint Analysis"
python main.py api
```

Terminal 2:
```bash
cd "Keystroke Dynamics"
python main.py api
```

Terminal 3:
```bash
cd "Mouse Movement Analysis"
python main.py api
```

Terminal 4:
```bash
cd Backend
npm run dev
```

Terminal 5:
```bash
cd Client
npm run dev
```

---

### Step 3: Use the Application

1. **Open Browser**: http://localhost:5173

2. **Register**:
   - Fill personal & professional info
   - Record voice (5-10 seconds)
   - Capture keystroke pattern (3 times)
   - Capture mouse movement (10 seconds)

3. **Login**:
   - Use your email and password

4. **Start Consultation**:
   - Click "Start Consultation"
   - Allow camera/microphone
   - See real-time verification!

---

## ✅ Verify Everything Works

### Check Services Health

Open these URLs in your browser:

- Voice API: http://localhost:8001/health
- Keystroke API: http://localhost:8002/health
- Mouse API: http://localhost:8003/health
- Backend API: http://localhost:5000/api/health
- Frontend: http://localhost:5173

All should return success responses!

---

## 🎯 What to Expect

### Registration Page
- 3-step wizard
- Biometric enrollment for all 3 modalities
- Success message on completion

### Dashboard
- ML services status (all green)
- Your biometric profile (all enrolled)
- List of registered doctors
- "Start Consultation" button

### Meeting Page
- Live video feed
- Trust score (starts at 100%)
- Real-time verification logs
- Biometric status indicators
- Alerts for suspicious activity

---

## 🔧 Common Issues

### "Cannot connect to MongoDB"
- Check internet connection
- MongoDB URI is pre-configured in Backend/.env

### "ML Service not responding"
- Ensure all 3 Python APIs are running
- Check terminal outputs for errors

### "Camera not working"
- Allow camera/microphone permissions
- Use Chrome or Firefox

### "Blank page"
- Wait for all services to fully start
- Check browser console for errors
- Refresh the page

---

## 📊 Test Data

You can register multiple doctors to test the system:

**Doctor 1:**
- Name: John Smith
- Email: john@example.com
- License: MD12345
- Specialization: Cardiology

**Doctor 2:**
- Name: Sarah Johnson
- Email: sarah@example.com
- License: MD67890
- Specialization: Pediatrics

---

## 💡 Pro Tips

1. **Keep terminals open** - Don't close any of the 5 terminal windows
2. **Check logs** - Watch terminal outputs for verification results
3. **Natural behavior** - Type and move mouse naturally during meetings
4. **Trust score** - Should stay above 80% for legitimate users
5. **Alerts** - Will appear if verification fails

---

## 🎓 Demo Flow

1. **Register** a doctor account with all biometrics
2. **Login** to see the dashboard
3. **Check** ML services are all green
4. **Start** a consultation
5. **Observe** real-time verification:
   - Speak to trigger voice verification
   - Type to trigger keystroke verification
   - Move mouse to trigger movement verification
6. **Watch** trust score and logs update
7. **End** consultation

---

## 📞 Need Help?

1. Read SETUP_GUIDE.md for detailed instructions
2. Check README.md for architecture details
3. Review terminal outputs for error messages
4. Ensure all 5 services are running

---

**Enjoy the Zero Trust Telehealth Platform! 🏥🔒**

