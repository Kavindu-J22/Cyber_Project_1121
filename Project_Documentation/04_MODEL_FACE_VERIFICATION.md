# ML Model 1: Face Verification (ResNet50 + Triplet Loss)

## Overview

| Property | Value |
|----------|-------|
| **Service Port** | 8004 |
| **Framework** | PyTorch + FastAPI |
| **Model Architecture** | ResNet50 backbone + 128-D embedding layer |
| **Pre-trained** | ImageNet weights for backbone |
| **Training Method** | Triplet Loss (metric learning) |
| **Input** | 224×224×3 RGB face images |
| **Output** | 128-dimensional L2-normalized embedding |
| **Similarity Metric** | Cosine Similarity |
| **Default Threshold** | 0.8096 |
| **Model Size** | ~90-100 MB |
| **Inference Speed** | ~280ms per verification |

---

## Architecture

```
Input Image (224×224×3 RGB)
    │
    ▼
┌─────────────────────────────────────────┐
│   ResNet50 Backbone (Pretrained ImageNet) │
│   ├── conv1 → bn1 → relu → maxpool      │
│   ├── layer1 (3 residual blocks)         │
│   ├── layer2 (4 residual blocks)         │
│   ├── layer3 (6 residual blocks)         │
│   ├── layer4 (3 residual blocks)         │
│   └── avgpool → 2048-D feature vector   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│   Embedding Layer                     │
│   ├── Linear(2048 → 128)            │
│   └── L2 Normalization (unit vector) │
└────────────────┬─────────────────────┘
                 │
                 ▼
         128-D Face Embedding
    (normalized to unit length = 1.0)
```

---

## How It Works

### Training (Triplet Loss)
The model learns to place **same-person faces close together** and **different-person faces far apart** in 128-dimensional embedding space.

**Triplet Formation:**
- **Anchor**: Reference face image
- **Positive**: Different image of the same person
- **Negative**: Image of a different person

**Loss Function:**
$$L = \max(d(\text{anchor}, \text{positive}) - d(\text{anchor}, \text{negative}) + \text{margin}, 0)$$

Where $d$ is the distance function and margin = 0.2-0.5.

### Image Preprocessing
```
Raw Image → Resize to 224×224 → Convert to Tensor [0,1] 
→ ImageNet Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
→ Add batch dimension → (1, 3, 224, 224) tensor
```

**Validation:** Min 50×50px, max 4000×4000px. Supports JPG, PNG, BMP, TIFF, WebP. All converted to RGB.

---

## Enrollment Flow

```
Doctor provides 3 face images during registration
    │
    ▼
For each image:
├── Preprocess (resize, normalize)
├── Forward through ResNet50 → 2048-D features
├── Embedding layer → 128-D vector
└── L2 normalize → unit vector
    │
    ▼
Compute Enrollment Quality:
├── Calculate pairwise cosine similarities between all 3 embeddings
├── Quality Score = mean(similarities) → e.g., 0.92
└── High quality (>0.90) = consistent samples from same person
    │
    ▼
Store: {
  embeddings: numpy array (3, 128),    ← the 3 sample embeddings
  num_samples: 3,
  quality_score: 0.92,
  mean_embedding: average vector,
  intra_sim_mean: 0.915,               ← for calibrated scoring
  intra_sim_std: 0.018
}
    │
    ▼
Persist to data/enrollments.pkl (survives restarts)
```

---

## Verification Flow

```
Camera frame captured during consultation (every 5 seconds)
    │
    ▼
FACE DETECTION GATE (Anti-Spoofing):
├── OpenCV Haar Cascade classifier
├── Detect face in image
├── If NO FACE → confidence = 0.05, decision = "NO_FACE_DETECTED"
│   (catches covered camera, dark frames, non-face images)
└── If face found → proceed
    │
    ▼
Extract Probe Embedding:
├── Preprocess image (224×224, normalize)
├── Forward through ResNet50 → 2048-D
├── Embedding layer → 128-D
└── L2 normalize
    │
    ▼
Compare with Enrolled Embeddings:
├── For each enrolled embedding:
│   similarity_i = cosine_similarity(probe, enrolled_i)
├── max_similarity = max(all similarities)
└── mean_similarity = mean(all similarities)
    │
    ▼
Calibrated Confidence Scoring (Sigmoid):
├── center = threshold - 0.03 = 0.7796
├── scale = 0.05
├── z_score = (max_similarity - center) / scale
├── confidence = sigmoid(z_score) = 1 / (1 + exp(-z_score))
└── Clamped to [0.0, 1.0]
    │
    ▼
Decision:
├── confidence >= 0.5 → MATCH (verified = true)
└── confidence < 0.5 → MISMATCH (verified = false)
```

### Confidence Score Examples

| Raw Cosine Similarity | Z-Score | Confidence | Decision |
|-----------------------|---------|------------|----------|
| 0.95 | +3.4 | **97%** | Clear match |
| 0.90 | +2.4 | **92%** | Strong match |
| 0.85 | +1.4 | **80%** | Match |
| 0.80 (at threshold) | +0.4 | **60%** | Marginal match |
| 0.75 | -0.6 | **35%** | Likely mismatch |
| 0.70 | -1.6 | **17%** | Clear mismatch |
| 0.05 (no face) | — | **5%** | No face detected |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info (version, model type, embedding_dim) |
| GET | `/health` | Health check (model_loaded, device, threshold) |
| GET | `/api/config` | Configuration details |
| POST | `/api/v1/enroll` | Enroll user with face images (multipart: user_id + face_samples[]) |
| POST | `/api/v1/verify` | Verify user identity (multipart: user_id + face_sample) |
| POST | `/api/verify` | Direct two-image comparison (multipart: reference + probe) |
| GET | `/api/v1/users` | List all enrolled users |
| DELETE | `/api/v1/users/{user_id}` | Remove user enrollment |

---

## Anti-Spoofing Mechanisms

1. **Face Detection Gate**: OpenCV Haar Cascade rejects frames without detected faces
2. **No Raw Image Storage**: Only 128-D embeddings stored — cannot reconstruct faces
3. **Configurable Face Detection**: Can upgrade to MTCNN or DNN-based detectors
4. **Enrollment Quality Check**: Ensures enrollment samples are consistent (pairwise similarity)

---

## Configuration (`face verification/config.yaml`)

```yaml
model:
  type: "resnet50_triplet"
  backbone: "resnet50"
  embedding_dim: 128
  pretrained: true
  checkpoint_path: "models/best_resnet50_triplet.pth"

verification:
  similarity_metric: "cosine"
  threshold: 0.8096                # Default threshold
  high_security_threshold: 0.85     # Strict mode
  balanced_threshold: 0.78          # Balanced mode
  lenient_threshold: 0.70           # Permissive mode
  enrollment_samples: 3

api:
  port: 8004
  workers: 1                        # Windows compatibility

security:
  store_raw_images: false            # Never store raw images
  encrypt_embeddings: true
```

---

## When This Model Is Used in the Web App

1. **Doctor Registration** → `POST /api/v1/enroll` — 3 face images enrolled
2. **Before Starting Consultation** → `POST /api/v1/verify` — Face verified as part of 3-step verification
3. **During Consultation** → `POST /api/v1/verify` every 5 seconds — Continuous face monitoring
4. **Frontend captures** → `FaceCapture.captureFrame()` from webcam video stream
5. **Backend routes** → `POST /api/verification/face` → `mlService.verifyFace()` → Face API at port 8004
