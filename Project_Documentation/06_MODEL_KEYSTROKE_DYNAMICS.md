# ML Model 3: Keystroke Dynamics (Deep Neural Network)

## Overview

| Property | Value |
|----------|-------|
| **Service Port** | 8002 |
| **Framework** | PyTorch + FastAPI |
| **Model Architecture** | 4-layer DNN with BatchNorm + Dropout |
| **Training Method** | Triplet Loss with Hard Mining |
| **Input** | 38-dimensional keystroke timing feature vector |
| **Output** | 128-dimensional L2-normalized embedding |
| **Similarity Metric** | Cosine Similarity |
| **Default Threshold** | 0.85 |
| **Parameters** | ~338,000 |
| **Inference Speed** | <20ms per verification |
| **Accuracy** | 96.20% |
| **False Accept Rate** | 0.39% |
| **False Reject Rate** | 7.48% |

---

## Architecture

```
38-D Keystroke Feature Vector
    │
    ▼
┌─────────────────────────────────────┐
│   Layer 1: Linear(38 → 256)        │
│   ├── Batch Normalization           │
│   ├── ReLU activation               │
│   └── Dropout (30%)                 │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Layer 2: Linear(256 → 512)       │
│   ├── Batch Normalization           │
│   ├── ReLU activation               │
│   └── Dropout (30%)                 │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Layer 3: Linear(512 → 256)       │
│   ├── Batch Normalization           │
│   ├── ReLU activation               │
│   └── Dropout (30%)                 │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Layer 4: Linear(256 → 128)       │
│   └── L2 Normalization              │
└────────────────┬────────────────────┘
                 │
                 ▼
       128-D Keystroke Embedding
   (unique typing fingerprint)
```

---

## How It Works

### What Are Keystroke Dynamics?
Every person types differently — different rhythm, different speed, different timing between keys. This behavioral biometric captures **how you type**, not **what you type**.

### Feature Extraction (38 Features)

The system captures timing from the phrase "The quick brown fox jumps over the lazy dog" (or similar text):

**Phase 1: Raw Timing Features (31 features)**

| Feature Type | Count | Description | Example |
|-------------|-------|-------------|---------|
| **Hold Time (H)** | 10 | Time a key is held down (keydown → keyup) | H.t = 85ms, H.h = 110ms |
| **Down-Down (DD)** | 10 | Time between consecutive key presses | DD.t→h = 142ms |
| **Up-Down (UD)** | 10 | Flight time: key release to next key press | UD.t→h = 57ms |
| Padding | 1 | To reach 31 | — |

```
Example Keystroke Recording:
  Press 'T'  at  0ms   (keydown)
  Release 'T' at  85ms  (keyup)    → H.T = 85ms
  Press 'h'  at  142ms (keydown)   → DD.T→h = 142ms, UD.T→h = 57ms
  Release 'h' at  252ms (keyup)    → H.h = 110ms
  Press 'e'  at  315ms (keydown)   → DD.h→e = 173ms, UD.h→e = 63ms
  ...
```

**Phase 2: Statistical Features (7 features)**

| Feature | Description |
|---------|-------------|
| Mean | Average of all timing values |
| Std | Standard deviation (typing consistency) |
| Median | Middle value |
| Min | Fastest timing |
| Max | Slowest timing |
| Q25 | 25th percentile |
| Q75 | 75th percentile |

**Total: 31 timing + 7 statistical = 38 features per sample**

### Preprocessing Pipeline
```
Raw keystrokes → Extract 31 timing features
    → Compute 7 statistical features → 38 total
    → Handle NaN values (replace with 0)
    → Detect outliers (z-score > 3.0)
    → Normalize with RobustScaler
    → Convert to PyTorch tensor
```

---

## Training Pipeline

| Component | Details |
|-----------|---------|
| **Dataset** | DSL-StrongPasswordData (51 subjects, 400 samples each) |
| **Loss Function** | Triplet Loss (margin=0.2) with hard negative mining |
| **Optimizer** | Adam (lr=0.0005, weight_decay=1e-5) |
| **Scheduler** | Cosine Annealing LR |
| **Batch Size** | 64 |
| **Epochs** | 150 (early stopping at patience=40) |
| **Data Split** | 60% train / 20% val / 20% test (by subject) |
| **Augmentation** | 5× augmentation: Gaussian noise + time warping + mixup |
| **Gradient Clipping** | Max norm 1.0 |

### Triplet Loss Objective
For each triplet (anchor, positive, negative):
$$L = \max(d(\text{anchor}, \text{positive}) - d(\text{anchor}, \text{negative}) + \text{margin}, 0)$$

**Hard Mining**: Selects the hardest positive (farthest same-user sample) and hardest negative (closest different-user sample) for more challenging training.

---

## Enrollment Flow

```
Doctor types the phrase 3 times during registration
    │
    ▼
For each typing sample:
├── Extract 38-D feature vector from keystroke timings
├── Normalize with RobustScaler
├── Forward through DNN → 128-D embedding
└── L2 normalize
    │
    ▼
Compute Template:
├── Mean embedding = average(all 3 embeddings)
├── Compute intra-class statistics:
│   ├── intra_sim_mean: ~0.90 (typical same-user similarity)
│   └── intra_sim_std: ~0.03 (variation)
└── Optional: Compute adaptive threshold (mean - 3σ)
    │
    ▼
Store: {
  template: 128-D tensor,          ← mean embedding
  embeddings: tensor(3, 128),      ← all enrollment embeddings
  intra_sim_mean: float,
  intra_sim_std: float,
  threshold: float
}
    │
    ▼
Persist to data/templates.pth (survives restarts)
```

---

## Verification Flow

```
Keystroke sample captured during consultation (every 10 seconds when chat is open)
    │
    ▼
Extract 38-D feature vector from keystroke timings
    │
    ▼
Normalize using fitted RobustScaler
    │
    ▼
Forward through DNN → 128-D embedding → L2 normalize
    │
    ▼
Compute cosine similarity with enrolled template:
    similarity = dot(test_embedding, template) / (||test|| × ||template||)
    │
    ▼
Calibrated Confidence Scoring (Sigmoid):
├── center = threshold - 0.03 = 0.82
├── scale = 0.05
├── z_score = (similarity - center) / scale
├── confidence = sigmoid(z_score) = 1 / (1 + exp(-z_score))
    │
    ▼
Classify Confidence Level:
├── confidence >= 0.90 → "high"
├── confidence >= 0.75 → "medium"
├── confidence >= 0.60 → "low"
└── confidence < 0.60 → "very_low" (alert triggered)
    │
    ▼
Decision:
├── confidence >= 0.5 → VERIFIED
└── confidence < 0.5 → NOT VERIFIED
```

### Confidence Score Examples (threshold = 0.85)

| Cosine Similarity | Z-Score | Confidence | Interpretation |
|-------------------|---------|------------|----------------|
| 0.95 | +2.6 | **93%** | Clearly authentic |
| 0.90 | +1.6 | **83%** | Likely authentic |
| 0.85 (threshold) | +0.6 | **65%** | At threshold |
| 0.80 | -0.4 | **40%** | Suspicious |
| 0.75 | -1.4 | **20%** | Likely impostor |
| 0.65 | -3.4 | **3%** | Definitely impostor |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health and model status |
| POST | `/enroll` | Enroll user with keystroke samples |
| POST | `/verify` | Single keystroke verification |
| POST | `/continuous-verify` | Streaming verification (50-100 samples) |
| GET | `/statistics/{user_id}` | Verification statistics |
| POST | `/detect-anomaly` | Anomaly detection on keystroke pattern |
| GET | `/enrolled-users` | List all enrolled users |

---

## Anomaly Detection

- **Method**: Isolation Forest
- **Contamination**: 0.1 (10% anomaly rate)
- **Consecutive Anomalies Alert**: 3 consecutive anomalies trigger an alert
- **Detection**: Identifies unusual typing patterns that don't match any enrolled user

---

## Adaptive Templates

The system supports template evolution over time:
- **Template Update Rate**: 0.1 (EMA - Exponential Moving Average)
- On each successful verification, the template is slightly updated:
  $$\text{template}_\text{new} = 0.9 \times \text{template}_\text{old} + 0.1 \times \text{new\_embedding}$$
- This adapts to gradual changes in typing behavior

---

## Configuration (`Keystroke Dynamics/config.yaml`)

```yaml
model:
  embedding_dim: 128
  hidden_dims: [256, 512, 256, 128]
  dropout: 0.3
  batch_norm: true

training:
  batch_size: 64
  epochs: 150
  learning_rate: 0.0005
  triplet_margin: 0.2
  mining: hard
  early_stopping_patience: 40

verification:
  similarity_metric: cosine
  threshold: 0.85
  confidence_levels: {high: 0.90, medium: 0.75, low: 0.60}
  alert_threshold: 0.60

enrollment:
  min_samples: 3
  max_samples: 200
  adaptive_templates: true
  template_update_rate: 0.1

api:
  port: 8002
  workers: 1
```

---

## When This Model Is Used in the Web App

1. **Doctor Registration** → `POST /keystroke/enroll` — 3 typing samples enrolled
2. **Before Starting Consultation** → `POST /keystroke/verify` — Doctor types "MediConsult Secure Access"
3. **During Consultation** → `POST /keystroke/verify` every 10 seconds — **Only when chat is open** (if chat closed, defaults to 50%)
4. **Frontend captures** → `KeystrokeCapture.start()` records key events → `KeystrokeCapture.stop()` returns 38-D feature vector
5. **Backend routes** → `POST /api/verification/keystroke` → `mlService.verifyKeystroke()` → Keystroke API at port 8002

### Why Keystroke Verification Only During Chat?
During a video consultation, the doctor only types when using the chat feature. When the chat window is closed, there are no keystrokes to analyze, so the system defaults to a neutral 50% confidence score. This prevents false negatives when the doctor is simply speaking to the patient.
