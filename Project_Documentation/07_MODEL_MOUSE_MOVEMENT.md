# ML Model 4: Mouse Movement Analysis (Siamese Neural Network)

## Overview

| Property | Value |
|----------|-------|
| **Service Port** | 8003 |
| **Framework** | PyTorch + FastAPI |
| **Model Architecture** | Siamese Network (shared-weight DNN) |
| **Training Method** | Triplet Loss with Hard Mining (margin=2.0) |
| **Input** | ~40-50 dimensional mouse movement feature vector |
| **Output** | 128-dimensional L2-normalized embedding |
| **Similarity Metric** | Cosine Similarity |
| **Default Threshold** | 0.85 |
| **Parameters** | ~500K+ |
| **Inference Speed** | <500ms per verification |
| **Training Dataset** | Balabit Mouse Dynamics Dataset |

---

## Architecture

```
~40-50 D Mouse Feature Vector
    │
    ▼
┌──────────────────────────────────────────┐
│   Siamese Network (Shared Weights)        │
│                                           │
│   Layer 1: Linear(input → 256)           │
│   ├── Batch Normalization                │
│   ├── ReLU                               │
│   └── Dropout (30%)                      │
│                                           │
│   Layer 2: Linear(256 → 512)             │
│   ├── Batch Normalization                │
│   ├── ReLU                               │
│   └── Dropout (30%)                      │
│                                           │
│   Layer 3: Linear(512 → 512)             │
│   ├── Batch Normalization                │
│   ├── ReLU                               │
│   └── Dropout (30%)                      │
│                                           │
│   Layer 4: Linear(512 → 256)             │
│   ├── Batch Normalization                │
│   ├── ReLU                               │
│   └── Dropout (30%)                      │
│                                           │
│   Layer 5: Linear(256 → 128)             │
│   └── L2 Normalization                   │
└────────────────┬─────────────────────────┘
                 │
                 ▼
        128-D Mouse Behavior Embedding
     (unique movement fingerprint)
```

### Why Siamese Network?
- **Shared weights**: Both anchor and comparison inputs go through the same network
- **Learns similarity**: Instead of classifying users, it learns a distance metric
- **Few-shot friendly**: Can verify new users with just 1-3 enrollment sessions
- **User-invariant**: Works for new users without retraining

---

## How It Works

### What Are Mouse Movement Biometrics?
Every person moves their mouse differently — different speed, acceleration, curvature, click patterns, and hesitation behavior. These subtle patterns are unique like a fingerprint.

### Feature Extraction (~40-50 Features)

Raw mouse events (timestamp, x, y, button, state) are processed in sliding windows of 100 events with stride 50:

#### A. Velocity Features (8-13 features)
| Feature | Description |
|---------|-------------|
| `velocity_mean` | Average cursor speed (pixels/second) |
| `velocity_std` | Speed variation |
| `velocity_median` | Median speed |
| `velocity_min/max` | Speed range |
| `velocity_q25/q75` | Speed percentiles |
| `velocity_skew/kurtosis` | Speed distribution shape |
| `vx_mean/std` | Horizontal velocity |
| `vy_mean/std` | Vertical velocity |

#### B. Acceleration Features (4 features)
| Feature | Description |
|---------|-------------|
| `acceleration_mean` | Average acceleration |
| `acceleration_std` | Acceleration variation |
| `acceleration_median` | Median acceleration |
| `acceleration_max_abs` | Peak acceleration |

#### C. Curvature & Angular Features (3 features)
| Feature | Description |
|---------|-------------|
| `curvature_mean` | How curved the movement paths are |
| `curvature_std` | Curvature variation |
| `angle_change_mean` | Average direction change between movements |

#### D. Jerk Features (2 features)
| Feature | Description |
|---------|-------------|
| `jerk_mean_abs` | Smoothness of acceleration changes |
| `jerk_std` | Jerk variation |

#### E. Click Dynamics (3-4 features)
| Feature | Description |
|---------|-------------|
| `click_ratio` | Fraction of events with button pressed |
| `click_hold_mean` | Average click duration |
| `click_hold_std` | Click duration variation |

#### F. Trajectory Features (3 features)
| Feature | Description |
|---------|-------------|
| `path_length` | Total distance traveled |
| `direct_distance` | Straight-line distance start→end |
| `trajectory_efficiency` | direct/path ratio (1.0 = perfectly straight) |

#### G. Temporal Features (4 features)
| Feature | Description |
|---------|-------------|
| `time_diff_mean` | Average time between events |
| `time_diff_std` | Timing variation |
| `num_pauses` | Count of significant pauses |
| `pause_duration_mean` | Average pause length |

**Normalization**: All features are normalized using RobustScaler (resistant to outliers).

---

## Training Pipeline

| Component | Details |
|-----------|---------|
| **Dataset** | Balabit Mouse Dynamics (10 users, ~200 sessions each) |
| **Loss Function** | Triplet Loss (margin=2.0) with hard negative mining |
| **Optimizer** | Adam (lr=0.0001, weight_decay=0.0001) |
| **Scheduler** | Cosine Annealing LR |
| **Batch Size** | 32 |
| **Epochs** | 200 (early stopping at patience=30) |
| **Data Split** | 60% train / 20% val / 20% test |
| **Augmentation** | Gaussian noise (2%) + time warping + spatial jitter (5%) |
| **Gradient Clipping** | Max norm 1.0 |

### Hard Triplet Mining
For each anchor sample:
- **Hardest Positive**: The farthest same-user sample (most different but still genuine)
- **Hardest Negative**: The closest different-user sample (most similar impostor)
- Forces the model to learn subtle behavioral differences

---

## Enrollment Flow

```
Doctor moves mouse naturally for 15 seconds during registration
    │
    ▼
Mouse events recorded: [
  {timestamp, x, y, button: "NoButton", state: "Move"},
  {timestamp, x, y, button: "Left", state: "Pressed"},
  ...100+ events
]
    │
    ▼
Feature Extraction:
├── Sliding windows of 100 events (stride 50)
├── Extract ~40-50 features per window
├── Normalize with RobustScaler
└── Generate 128-D embedding per window
    │
    ▼
Template Creation:
├── Mean embedding = average(all window embeddings)
├── Compute intra-class statistics:
│   ├── intra_sim_mean: ~0.90
│   └── intra_sim_std: ~0.03
    │
    ▼
Store: {
  template: 128-D tensor,
  embeddings: all window embeddings,
  intra_sim_mean: float,
  intra_sim_std: float,
  enrollment_time: timestamp,
  num_samples: count
}
    │
    ▼
Persist to data/templates.pth
```

---

## Verification Flow

```
Mouse events accumulated during consultation (sent every 10 seconds)
    │
    ▼
Minimum 50 events required (if less → neutral 0.5 score returned)
    │
    ▼
Feature extraction → Normalize → Model forward → 128-D embedding
    │
    ▼
Cosine similarity with enrolled template:
    raw_similarity = dot(test, template) / (||test|| × ||template||)
    │
    ▼
Calibrated Confidence Scoring (Tight Sigmoid):
    ┌──────────────────────────────────────────────┐
    │ center = 0.93  (midpoint of model's output)  │
    │ scale = 0.02   (very tight = 2% sim = 1σ)    │
    │ z = (raw_similarity - center) / scale         │
    │ confidence = sigmoid(z)                       │
    └──────────────────────────────────────────────┘
    │
    ▼
Decision:
├── confidence >= 0.85 → "high" (verified)
├── confidence >= 0.70 → "medium"
├── confidence >= 0.55 → "low"
└── confidence < 0.55 → "very_low" (alert)
```

### Confidence Score Examples

| Raw Cosine Similarity | Z-Score | Confidence | Level |
|-----------------------|---------|------------|-------|
| 0.99 | +3.0 | **95%** | Very close match |
| 0.97 | +2.0 | **88%** | Good match |
| 0.95 | +1.0 | **73%** | Acceptable |
| 0.93 (center) | 0.0 | **50%** | Midpoint |
| 0.91 | -1.0 | **27%** | Suspicious |
| 0.89 | -2.0 | **12%** | Likely impostor |
| 0.87 | -3.0 | **5%** | Clear mismatch |

**Note:** Mouse model output clusters in the 0.89-0.99 range, so the tight sigmoid (scale=0.02) spreads these scores to the full [0,1] confidence range.

---

## Anomaly Detection

### Automated Behavior Detection
| Check | Indicator | Risk Score |
|-------|-----------|------------|
| Constant velocity | `velocity_std < 1.0` | +0.4 (bot) |
| Perfect linear movements | `trajectory_efficiency > 0.98` | +0.4 (unnatural) |
| Impossible speed | `velocity_max > 10000 px/s` | +0.4 (physically impossible) |

### Remote Desktop Detection
| Check | Indicator | Risk Score |
|-------|-----------|------------|
| High jitter | `jerk_std > 1000` | +0.3 (RDP latency artifacts) |
| Excessive pauses | `num_pauses > 50` | +0.3 (RDP buffering) |

### User Substitution Detection
| Check | Indicator | Risk Score |
|-------|-----------|------------|
| Sudden confidence drop | 0.8 → 0.4 | +0.3 (person changed) |
| Consecutive low scores | ≥5 below 0.65 | +0.3 (persistent impostor) |

### Risk Level Assessment
| Total Risk Score | Level |
|-----------------|-------|
| ≥ 0.7 | **CRITICAL** |
| ≥ 0.4 | **HIGH** |
| ≥ 0.2 | **MEDIUM** |
| < 0.2 | **LOW** |

---

## Continuous Monitoring API

```
POST /monitor
{
  "user_id": "doctor_001",
  "session_id": "consultation_2024_1",
  "events": [... 200 mouse events over 10 seconds ...]
}

Response:
{
  "overall_verified": true,
  "mean_confidence": 0.88,
  "verification_rate": 0.95,    // 95% of windows passed
  "anomaly_detected": false,
  "risk_level": "low"
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` or `/health` | Health check, model status |
| POST | `/enroll` | Enroll user with mouse event data |
| POST | `/verify` | Single verification |
| POST | `/monitor` | Continuous monitoring during session |
| GET | `/users/{user_id}/stats` | Verification statistics |
| DELETE | `/users/{user_id}` | Remove user enrollment |

---

## Configuration (`Mouse Movement Analysis/config.yaml`)

```yaml
model:
  embedding_dim: 128
  hidden_dims: [256, 512, 512, 256, 128]
  dropout: 0.3
  batch_norm: true

training:
  batch_size: 32
  epochs: 200
  learning_rate: 0.0001
  triplet_margin: 2.0
  triplet_mining: hard
  early_stopping_patience: 30

verification:
  similarity_metric: cosine
  threshold: 0.85
  confidence_levels: {high: 0.85, medium: 0.70, low: 0.55}
  alert_threshold: 0.65

features:
  window_size: 100
  window_stride: 50
  min_events: 50
  velocity_features: enabled
  acceleration_features: enabled
  curvature_features: enabled
  jerk_features: enabled
  click_features: enabled
  trajectory_features: enabled
  temporal_features: enabled

anomaly:
  method: isolation_forest
  contamination: 0.1
  detect_automated_behavior: true
  detect_remote_desktop: true
  detect_user_substitution: true

api:
  port: 8003
  workers: 1
  rate_limit: 100 req/min
```

---

## When This Model Is Used in the Web App

1. **Doctor Registration** → `POST /mouse/enroll` — 15 seconds of natural mouse movement enrolled
2. **Before Starting Consultation** → `POST /mouse/verify` — Mouse movement verified as part of biometric check
3. **During Consultation** → `POST /mouse/verify` every 10 seconds — Passive mouse tracking
4. **Frontend captures** → `MouseCapture.start()` attaches to mousemove/mouseclick → `MouseCapture.getEvents()` returns event array
5. **Backend routes** → `POST /api/verification/mouse` → `mlService.verifyMouse()` → Mouse API at port 8003

### Why Mouse Verification Is Always Active
Unlike keystroke dynamics (which only works when typing), mouse movements are **always being generated** during a consultation. The doctor moves the mouse to interact with the interface, use controls, and navigate. This makes mouse biometrics ideal for continuous, passive authentication that requires no active participation from the user.
