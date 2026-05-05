# QUICK START - Face Verification Integration

## ⚡ Get Started in 3 Steps

### Step 1: Place Your Trained Model (REQUIRED)

```bash
# Navigate to face verification folder
cd "face verification"

.\.venv\Scripts\Activate.ps1

# Place your model file here:
# face verification/models/best_model.pt
```

**Important**: The API will NOT start without this file!

---

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- FastAPI & Uvicorn (API server)
- PyTorch & TorchVision (Deep learning)
- Pillow (Image processing)

---

### Step 3: Start the Service

#### Option A: Test First (Recommended)
```bash
python main.py test
```

#### Option B: Start API Server
```bash
# Windows
start_api.bat

# Mac/Linux
./start_api.sh

# Manual
python main.py api
```

#### Option C: Start All Services Together
```bash
# From project root
cd ..
start-all-services.bat
```

---

## ✅ Verify It's Working

Open in browser or use curl:

```bash
# Health check
curl http://localhost:8004/health

# Should return:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "device": "cpu",
#   "threshold": 0.78
# }
```

---

## 📡 Service Ports

After starting all services, you'll have:

- **Voice API**: http://localhost:8001
- **Keystroke API**: http://localhost:8002
- **Mouse API**: http://localhost:8003
- **Face API**: http://localhost:8004 ← **NEW**
- **Backend**: http://localhost:5000
- **Frontend**: http://localhost:5173

---

## 🎯 Quick Test

### Test with Two Images

```bash
curl -X POST http://localhost:8004/api/verify \
  -F "reference=@face1.jpg" \
  -F "probe=@face2.jpg"
```

### Expected Response

```json
{
  "similarity": 0.8542,
  "threshold": 0.78,
  "decision": "MATCH",
  "device": "cpu"
}
```

---

## 🔧 Troubleshooting

### Problem: Model Not Found
```
Error: Model checkpoint not found
```
**Solution**: Place `best_model.pt` in `models/` folder

### Problem: Port Already in Use
```
Error: Address already in use
```
**Solution**: 
1. Kill process using port 8004
2. Or change port in `config.yaml`

### Problem: Import Errors
```
Error: No module named 'fastapi'
```
**Solution**: Run `pip install -r requirements.txt`

### Problem: CUDA Errors (GPU)
```
Error: CUDA out of memory
```
**Solution**: Edit `config.yaml`:
```yaml
performance:
  device: "cpu"
```

---

## 📚 API Documentation

Once the service is running, visit:
- **Swagger UI**: http://localhost:8004/docs
- **ReDoc**: http://localhost:8004/redoc

---

## 🎓 How It Works

1. **Registration**:
   - User captures 3 face samples via webcam
   - Images sent to `/api/v1/enroll`
   - ResNet50 extracts 128-D embeddings
   - Embeddings stored for user

2. **Verification** (During Consultation):
   - Capture frame from video
   - Send to `/api/v1/verify`
   - Compare with enrolled embeddings
   - Return similarity score & decision

3. **Trust Score**:
   - High similarity → Trust maintained
   - Low similarity → Alert triggered
   - Continuous monitoring throughout consultation

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Model settings
model:
  embedding_dim: 128
  checkpoint_path: "models/best_model.pt"

# Verification threshold
verification:
  threshold: 0.78  # Adjust based on security needs
  
# API settings
api:
  port: 8004
  
# Performance
performance:
  device: "auto"  # auto, cpu, cuda
```

---

## 🚀 Next Steps

1. ✅ Service running? → Test with Registration: http://localhost:5173
2. ✅ Register a doctor account
3. ✅ Capture 3 face samples during registration
4. ✅ Start a consultation and see real-time face verification!

---

## 📞 Need Help?

- Read full documentation: `face verification/README.md`
- Check integration details: `FACE_VERIFICATION_INTEGRATION.md`
- API docs: http://localhost:8004/docs

---

**You're all set! 🎉**

Your Zero Trust Telehealth Platform now has **4-modal biometric authentication**:
- ✅ Voice
- ✅ Keystroke
- ✅ Mouse Movement
- ✅ Face Verification ← **NEW**
