# 🚀 Getting Started - Voiceprint Analysis System

## Welcome!

This guide will help you get the Voiceprint Analysis System up and running in **under 10 minutes**.

---

## 📋 Prerequisites Checklist

Before you begin, ensure you have:

- [ ] Python 3.8 or higher installed
- [ ] pip (Python package manager)
- [ ] 4GB+ RAM available
- [ ] 2GB+ free disk space
- [ ] Internet connection (for downloading models)
- [ ] VoxCeleb dataset in the correct location

---

## ⚡ Quick Start (3 Steps)

### Step 1: Install Dependencies (5 minutes)

Open terminal/command prompt in the `Voiceprint Analysis` folder:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run Setup (1 minute)

```bash
python setup.py
```

This will:
- ✅ Create necessary directories
- ✅ Generate encryption keys
- ✅ Create `.env` file
- ✅ Verify dataset

### Step 3: Start the API Server (30 seconds)

**Windows:**
```bash
start_api.bat
```

**Linux/Mac:**
```bash
chmod +x start_api.sh
./start_api.sh
```

**Or manually:**
```bash
python main.py api
```

🎉 **Done!** The API is now running at: http://localhost:8001

---

## 🧪 Verify Installation

### Test 1: Health Check

Open your browser and visit:
```
http://localhost:8001/health
```

You should see:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {
    "verification_engine": true,
    "anti_spoofing": true
  }
}
```

### Test 2: Interactive API Docs

Visit:
```
http://localhost:8001/docs
```

You'll see the interactive Swagger UI with all API endpoints.

### Test 3: Run Test Suite

```bash
python main.py test
```

Expected output:
```
🧪 VOICEPRINT ANALYSIS SYSTEM - TEST SUITE
==================================================================
✓ Speaker enrollment successful
✓ Verification tests passed
✓ Continuous verification working
✓ Anti-spoofing detection active
✅ All tests completed!
```

---

## 📊 Evaluate Performance

Run the evaluation script to verify the system meets requirements:

```bash
python main.py train
```

Expected results:
```
📈 EVALUATION RESULTS
==================================================================
🎯 Equal Error Rate (EER): 2.45%
   Target: < 3.00%
   Status: ✓ PASS

⚡ Latency Performance:
   Average: 654.32 ms
   Target: < 800 ms
   Status: ✓ PASS
```

---

## 🎮 Try the Interactive Demo

```bash
python main.py demo
```

This will:
1. Enroll a sample speaker
2. Verify with genuine audio (should pass)
3. Test with impostor audio (should fail)

---

## 🔌 Your First API Call

### Using cURL

```bash
curl http://localhost:8001/api/v1/speakers
```

### Using Python

```python
import requests

# Health check
response = requests.get('http://localhost:8001/health')
print(response.json())
```

### Using JavaScript

```javascript
fetch('http://localhost:8001/health')
  .then(response => response.json())
  .then(data => console.log(data));
```

---

## 📚 Next Steps

### 1. Read the Documentation

- **README.md** - System overview and features
- **API_EXAMPLES.md** - Code examples in Python, JavaScript, cURL
- **INSTALLATION.md** - Detailed installation guide

### 2. Explore the API

Visit the interactive docs:
```
http://localhost:8001/docs
```

Try the endpoints:
- `/health` - Health check
- `/api/v1/speakers` - List enrolled speakers
- `/api/v1/enroll` - Enroll a new speaker
- `/api/v1/verify` - Verify speaker identity

### 3. Integrate with Your MERN App

See **API_EXAMPLES.md** for integration examples:
- React components
- Node.js backend proxy
- WebSocket real-time streaming

---

## 🔧 Common Commands

| Command | Description |
|---------|-------------|
| `python main.py api` | Start API server |
| `python main.py test` | Run test suite |
| `python main.py train` | Evaluate performance |
| `python main.py demo` | Interactive demo |
| `python setup.py` | Run setup |

---

## 🐛 Troubleshooting

### Issue: "Module not found"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Port 8001 already in use"

**Solution:** Edit `.env` and change:
```env
API_PORT=8002
```

### Issue: "Dataset not found"

**Solution:** Ensure the dataset is at:
```
Voiceprint Analysis/
└── Voice dataset - senath/
    ├── vox1_dev_wav/
    ├── vox1_test_wav/
    └── voiceprint_tuplets_dataset_5000.csv
```

### Issue: "GPU not available"

**Solution:** Edit `.env`:
```env
USE_GPU=false
```

The system will use CPU (slower but works).

---

## 📞 API Endpoints Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/enroll` | POST | Enroll speaker |
| `/api/v1/verify` | POST | Verify speaker |
| `/api/v1/verify/upload` | POST | Verify with file upload |
| `/api/v1/continuous-verify` | POST | Continuous verification |
| `/api/v1/speakers` | GET | List all speakers |
| `/api/v1/speakers/{id}` | GET | Get speaker info |
| `/api/v1/threshold` | PUT | Update threshold |

Full documentation: http://localhost:8001/docs

---

## 🎯 What You've Accomplished

✅ Installed the Voiceprint Analysis System  
✅ Started the API server  
✅ Verified the installation  
✅ Tested the system  
✅ Ready to integrate with your MERN app  

---

## 🌟 Key Features Available

- ✅ Real-time speaker verification
- ✅ < 3% Equal Error Rate
- ✅ < 800ms latency
- ✅ Anti-spoofing detection
- ✅ Privacy-first design (no raw audio storage)
- ✅ TLS 1.3 encryption
- ✅ REST API + WebSocket
- ✅ Continuous authentication

---

## 📖 Learn More

- **README.md** - Complete documentation
- **API_EXAMPLES.md** - Integration examples
- **PROJECT_SUMMARY.md** - Technical overview
- **Interactive Docs** - http://localhost:8001/docs

---

## 🎉 You're Ready!

The Voiceprint Analysis System is now running and ready to integrate with your Zero Trust Telehealth Platform.

**Next:** Check out **API_EXAMPLES.md** for integration code examples.

---

**Need Help?**

1. Check the troubleshooting section above
2. Review the logs in `logs/voiceprint_analysis.log`
3. Consult the documentation files

**Happy Coding! 🚀**

