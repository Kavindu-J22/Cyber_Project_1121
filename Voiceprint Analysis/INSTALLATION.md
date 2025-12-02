# 📦 Installation Guide - Voiceprint Analysis System

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM
- GPU (optional, but recommended for faster processing)

## Step-by-Step Installation

### 1. Clone or Navigate to Project Directory

```bash
cd "Voiceprint Analysis"
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and TorchAudio (Deep learning framework)
- SpeechBrain (Speaker recognition models)
- FastAPI and Uvicorn (API framework)
- Librosa and SoundFile (Audio processing)
- Cryptography (Security and encryption)
- And other required packages

**Note:** Installation may take 5-10 minutes depending on your internet connection.

### 4. Run Setup Script

```bash
python setup.py
```

This will:
- Create necessary directories
- Check dataset availability
- Generate encryption keys
- Create `.env` file
- Verify dependencies

### 5. Configure Environment Variables

Edit the `.env` file created by setup:

```env
# Security Keys (KEEP THESE SECRET!)
EMBEDDING_ENCRYPTION_KEY=<generated-key>
JWT_SECRET=<generated-key>

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/voiceprint_db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
ENVIRONMENT=development

# Performance
USE_GPU=true  # Set to false if no GPU available
NUM_WORKERS=4
```

### 6. Verify Installation

Run the test suite to verify everything is working:

```bash
python main.py test
```

Expected output:
```
🧪 VOICEPRINT ANALYSIS SYSTEM - TEST SUITE
==================================================================
🧪 TEST: Speaker Enrollment and Verification
==================================================================
✓ Enrollment successful!
✓ Verification tests passed
...
✅ All tests completed!
```

## 🚀 Running the System

### Option 1: API Server

Start the FastAPI server:

```bash
python main.py api
```

Access the API:
- API: http://localhost:8001
- Interactive Docs: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

### Option 2: Training/Evaluation

Evaluate the system on the VoxCeleb dataset:

```bash
python main.py train
```

This will:
- Load verification pairs
- Compute EER (Equal Error Rate)
- Measure latency
- Generate evaluation report

### Option 3: Interactive Demo

Run an interactive demonstration:

```bash
python main.py demo
```

### Option 4: Run Tests

Execute the test suite:

```bash
python main.py test
```

## 🐳 Docker Installation (Alternative)

If you prefer Docker:

### 1. Build Docker Image

```bash
docker-compose build
```

### 2. Start Services

```bash
docker-compose up -d
```

This will start:
- Voiceprint Analysis API (port 8001)
- MongoDB (port 27017)

### 3. Check Status

```bash
docker-compose ps
```

### 4. View Logs

```bash
docker-compose logs -f voiceprint-api
```

### 5. Stop Services

```bash
docker-compose down
```

## 🔧 Troubleshooting

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For GPU support:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "SpeechBrain model download fails"

**Solution:**
The first run will download pre-trained models (~200MB). Ensure you have:
- Stable internet connection
- Sufficient disk space (2GB+)

### Issue: "Audio file not found"

**Solution:**
Ensure the VoxCeleb dataset is in the correct location:
```
Voiceprint Analysis/
└── Voice dataset - senath/
    ├── vox1_dev_wav/
    ├── vox1_test_wav/
    └── voiceprint_tuplets_dataset_5000.csv
```

### Issue: "Port 8001 already in use"

**Solution:**
Change the port in `.env`:
```env
API_PORT=8002
```

Or in `config.yaml`:
```yaml
api:
  port: 8002
```

## 📊 Verifying Performance

After installation, verify the system meets requirements:

```bash
python main.py train
```

Expected results:
- ✅ EER < 3%
- ✅ Average latency < 800ms
- ✅ All tests passing

## 🔐 Security Checklist

Before deploying to production:

- [ ] Change default encryption keys in `.env`
- [ ] Set `ENVIRONMENT=production` in `.env`
- [ ] Enable TLS/SSL for API endpoints
- [ ] Configure MongoDB authentication
- [ ] Set up firewall rules
- [ ] Enable API rate limiting
- [ ] Configure alert email settings

## 📚 Next Steps

1. **Read the README.md** for API usage examples
2. **Review config.yaml** to customize settings
3. **Test the API** using the interactive docs at `/docs`
4. **Integrate with your MERN application**

## 💡 Tips

- Use GPU for faster processing (set `USE_GPU=true`)
- Adjust `verification.threshold` in `config.yaml` for different security levels
- Monitor logs in the `logs/` directory
- Keep encryption keys secure and backed up

## 🆘 Getting Help

If you encounter issues:
1. Check the logs in `logs/voiceprint_analysis.log`
2. Review the troubleshooting section above
3. Ensure all dependencies are correctly installed
4. Verify dataset paths are correct

---

**Installation complete! 🎉**

You're now ready to use the Voiceprint Analysis System for Zero Trust continuous speaker verification.

