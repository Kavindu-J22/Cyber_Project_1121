# Face Verification System
## Zero Trust Telehealth Platform - Biometric Authentication Module

### 🎯 Overview

Face verification module using **ResNet50 with Triplet Loss** for continuous authentication during telehealth consultations. This module provides real-time face recognition with high accuracy and low latency.

---

## 📋 Features

- **State-of-the-art Model**: ResNet50 backbone with triplet loss training
- **128-D Embeddings**: Compact L2-normalized face representations
- **Fast Inference**: < 500ms on CPU, < 50ms on GPU
- **RESTful API**: FastAPI-based endpoints for enrollment and verification
- **Flexible Deployment**: CPU and GPU support with automatic device detection
- **Security-First**: No raw image storage, only encrypted embeddings

---

## 🏗️ Architecture

```
Input Image (RGB)
    ↓
[Preprocessing Pipeline]
  - Resize to 224×224
  - ImageNet Normalization
    ↓
[ResNet50 Backbone]
  - 50 convolutional layers
  - Pre-trained on ImageNet
    ↓
[Embedding Layer]
  - 2048 → 512 → 128
  - BatchNorm + ReLU + Dropout
    ↓
[L2 Normalization]
    ↓
128-D Face Embedding
    ↓
[Cosine Similarity]
    ↓
Match/Mismatch Decision
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Trained model file: `best_resnet50_triplet.pth`

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Place your trained model
# models/best_resnet50_triplet.pth
```

### Start API Server

**Windows:**
```bash
start_api.bat
```

**Mac/Linux:**
```bash
chmod +x start_api.sh
./start_api.sh
```

**Manual:**
```bash
python main.py api
```

---

## 📡 API Endpoints

### Base URL
```
http://localhost:8004
```

### 1. Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "threshold": 0.8096
}
```

### 2. Enroll User
```http
POST /api/v1/enroll
Content-Type: multipart/form-data

Form Data:
- user_id: string (unique identifier)
- face_samples: file[] (3+ face images recommended)

Response:
{
  "success": true,
  "message": "User enrolled successfully",
  "data": {
    "user_id": "doctor_001",
    "num_samples": 3,
    "enrollment_quality": 0.92,
    "latency_ms": 450
  }
}
```

### 3. Verify User
```http
POST /api/v1/verify
Content-Type: multipart/form-data

Form Data:
- user_id: string
- face_sample: file (face image)
- threshold: float (optional, default: 0.8096)

Response:
{
  "success": true,
  "data": {
    "verified": true,
    "confidence_score": 0.87,
    "threshold": 0.8096,
    "decision": "MATCH",
    "latency_ms": 280
  }
}
```

### 4. Direct Comparison (No Enrollment)
```http
POST /api/verify
Content-Type: multipart/form-data

Form Data:
- reference: file (reference face)
- probe: file (probe face)
- threshold: float (optional)

Response:
{
  "similarity": 0.8542,
  "threshold": 0.8096,
  "decision": "MATCH",
  "device": "cpu"
}
```

### 5. Get Enrolled Users
```http
GET /api/v1/users

Response:
{
  "success": true,
  "count": 5,
  "users": ["doctor_001", "doctor_002", ...]
}
```

### 6. Remove User
```http
DELETE /api/v1/users/{user_id}

Response:
{
  "success": true,
  "message": "User doctor_001 removed successfully"
}
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize settings:

```yaml
# Model
model:
  embedding_dim: 128
  checkpoint_path: "models/best_resnet50_triplet.pth"

# Verification
verification:
  threshold: 0.8096  # Adjust based on security needs
  enrollment_samples: 3

# API
api:
  host: "0.0.0.0"
  port: 8004
  
# Performance
performance:
  device: "auto"  # auto, cpu, cuda
```

---

## 🎯 Model Specifications

| Parameter | Value |
|-----------|-------|
| **Architecture** | ResNet50 + Triplet Loss |
| **Embedding Dimension** | 128 |
| **Input Size** | 224×224×3 (RGB) |
| **Normalization** | ImageNet (mean/std) |
| **Similarity Metric** | Cosine Similarity |
| **Default Threshold** | 0.8096 |
| **Model Size** | ~90-100 MB |

---

## 📊 Performance Metrics

### Inference Speed
- **CPU**: ~200-500ms per verification
- **GPU**: ~20-50ms per verification

### Memory Usage
- Model: ~500 MB
- Per Request: ~50 MB

### Recommended Hardware
- **Minimum**: 2-core CPU, 4 GB RAM
- **Recommended**: 4-core CPU, 8 GB RAM, NVIDIA GPU (2+ GB VRAM)

---

## 🔐 Security Features

1. **No Raw Image Storage**: Images processed in-memory and discarded
2. **Embedding Encryption**: Optional encryption for stored embeddings
3. **Rate Limiting**: Prevent API abuse
4. **File Validation**: Image type and size validation
5. **TLS Support**: HTTPS encryption for production

---

## 🧪 Testing

### Run Tests
```bash
python main.py test
```

### Run Demo
```bash
python main.py demo
```

### Test with cURL
```bash
# Health check
curl http://localhost:8004/health

# Direct comparison
curl -X POST http://localhost:8004/api/verify \
  -F "reference=@face1.jpg" \
  -F "probe=@face2.jpg"
```

---

## 🔧 Troubleshooting

### Model Not Found
```
Error: Model checkpoint not found: models/best_resnet50_triplet.pth
```
**Solution**: Place your trained model file in the `models/` folder

### CUDA Out of Memory
```
Error: CUDA out of memory
```
**Solution**: Edit `config.yaml` and set `performance.device: "cpu"`

### Import Errors
```
Error: No module named 'fastapi'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

---

## 📁 Folder Structure

```
face verification/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── main.py                     # Entry point
├── start_api.bat              # Windows launcher
├── start_api.sh               # Unix launcher
├── README.md                  # This file
├── models/                    # Model storage
│   ├── best_resnet50_triplet.pth  # Trained model (place here)
│   └── checkpoints/
├── logs/                      # Log files
├── src/                       # Source code
│   ├── api.py                 # FastAPI server
│   ├── config_loader.py       # Config management
│   ├── face_model.py          # Model architecture
│   ├── face_preprocessing.py  # Image preprocessing
│   └── face_verification.py   # Verification engine
└── temp/                      # Temporary files
```

---

## 🔗 Integration with Telehealth Platform

This module integrates with the Zero Trust Telehealth Platform:

1. **Registration**: Capture 3 face samples during doctor registration
2. **Enrollment**: Send samples to `/api/v1/enroll`
3. **Continuous Verification**: During video consultation, periodically:
   - Capture frame from video
   - Send to `/api/v1/verify`
   - Update trust score based on confidence
4. **Alert System**: Trigger alerts when confidence < threshold

---

## 📚 References

- **Model**: ResNet50 (He et al., 2015)
- **Training**: Triplet Loss (Schroff et al., 2015)
- **Face Recognition**: Deep Face Recognition (Parkhi et al., 2015)

---

## 📝 License

Part of Zero Trust Telehealth Platform

.\.venv\Scripts\Activate.ps1