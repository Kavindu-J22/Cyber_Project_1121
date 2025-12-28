# ML Models Quick Reference Guide
## Zero Trust Telehealth Platform

---

## 📊 Model Comparison at a Glance

| Feature | Voice (ECAPA-TDNN) | Keystroke (DNN) | Mouse (Siamese) |
|---------|-------------------|-----------------|-----------------|
| **Model Type** | CNN + Attention | Deep Neural Net | Siamese Network |
| **Architecture** | 5 TDNN layers | 4 FC layers | 5 FC layers |
| **Parameters** | ~6.5M | ~500K | ~800K |
| **Embedding Dim** | 192 | 128 | 128 |
| **Input Features** | 80 mel-bins | 31 timing features | 50+ movement features |
| **Accuracy** | 97%+ | 96% | 92%+ |
| **EER** | < 3% | < 5% | < 8% |
| **Latency** | < 800ms | < 500ms | < 300ms |
| **Storage** | 2KB | 512B | 512B |
| **Pre-trained** | ✅ Yes (VoxCeleb) | ❌ No | ❌ No |
| **Loss Function** | AAM-Softmax | Triplet Loss | Triplet Loss |
| **Anti-Spoofing** | ✅ CNN (3 layers) | ✅ Isolation Forest | ✅ Isolation Forest |

---

## 🎯 Why Each Model Was Chosen

### Voice: ECAPA-TDNN
**Reason:** State-of-the-art speaker verification with channel attention
- ✅ Pre-trained on 7,000+ speakers (no training needed)
- ✅ Handles noise and channel variations
- ✅ Compact 192-dim embeddings
- ✅ < 3% EER (industry-leading)

### Keystroke: Deep Neural Network
**Reason:** Non-linear pattern recognition for complex typing rhythms
- ✅ Captures temporal dependencies in typing
- ✅ Siamese architecture learns similarity metrics
- ✅ Triplet loss with hard mining
- ✅ 96% accuracy (vs. 85% with Random Forest)

### Mouse: Siamese Neural Network
**Reason:** Similarity metric learning for behavioral comparison
- ✅ Robust to task variability (clicking, scrolling)
- ✅ Deep architecture (5 layers) for complex patterns
- ✅ L2-normalized embeddings
- ✅ Detects bots and RDP artifacts

---

## 🔍 Feature Breakdown

### Voice Features (80 mel-bins)
```
Mel-Filterbank Features:
├── 80 frequency bands (20Hz - 8kHz)
├── Frame size: 25ms
├── Frame shift: 10ms
└── Purpose: Frequency representation of voice
```

### Keystroke Features (31 features)
```
Timing Features:
├── Hold Times (H.key): 11 features
│   └── Time key is held down
├── DD Times (keydown-keydown): 10 features
│   └── Time between pressing consecutive keys
└── UD Times (keyup-keydown): 10 features
    └── Time from releasing to pressing next key
```

### Mouse Features (50+ features)
```
Movement Features:
├── Velocity: speed, direction, statistics (10 features)
├── Acceleration: magnitude, statistics (5 features)
├── Jerk: smoothness metric (3 features)
├── Curvature: path curvature (5 features)
├── Click Dynamics: count, intervals, hold (5 features)
├── Trajectory: efficiency, straightness (5 features)
└── Temporal: pauses, bursts (5 features)
```

---

## 🏗️ Architecture Details

### Voice: ECAPA-TDNN
```
Input (80 mel-bins)
  ↓
Conv1D Layers [1024, 1024, 1024, 1024, 3072]
  ├── Kernel sizes: [5, 3, 3, 3, 1]
  ├── Dilations: [1, 2, 3, 4, 1]
  └── SE-Res2Net attention blocks
  ↓
Statistics Pooling (mean + std)
  ↓
192-dim Embedding
```

### Keystroke: Deep Neural Network
```
Input (31 features)
  ↓
FC Layer 1: 31 → 256 (+ BatchNorm + ReLU + Dropout)
  ↓
FC Layer 2: 256 → 512 (+ BatchNorm + ReLU + Dropout)
  ↓
FC Layer 3: 512 → 256 (+ BatchNorm + ReLU + Dropout)
  ↓
FC Layer 4: 256 → 128 (+ BatchNorm + ReLU + Dropout)
  ↓
128-dim Embedding (L2 normalized)
```

### Mouse: Siamese Network
```
Input (50+ features)
  ↓
FC Layer 1: 50 → 256 (+ BatchNorm + ReLU + Dropout)
  ↓
FC Layer 2: 256 → 512 (+ BatchNorm + ReLU + Dropout)
  ↓
FC Layer 3: 512 → 512 (+ BatchNorm + ReLU + Dropout)
  ↓
FC Layer 4: 512 → 256 (+ BatchNorm + ReLU + Dropout)
  ↓
FC Layer 5: 256 → 128 (+ BatchNorm + ReLU + Dropout)
  ↓
128-dim Embedding (L2 normalized)
```

---

## 📈 Performance Metrics

### Voice Authentication
- **EER:** < 3%
- **Accuracy:** 97%+
- **FAR (False Accept Rate):** < 2%
- **FRR (False Reject Rate):** < 4%
- **Latency:** < 800ms per verification
- **Threshold:** 0.65 (cosine similarity)

### Keystroke Dynamics
- **EER:** < 5%
- **Accuracy:** 96%
- **FAR:** < 3%
- **FRR:** < 7%
- **Latency:** < 500ms per verification
- **Threshold:** 0.75 (cosine similarity)

### Mouse Movement
- **EER:** < 8%
- **Accuracy:** 92%+
- **FAR:** < 5%
- **FRR:** < 11%
- **Latency:** < 300ms per verification
- **Threshold:** 0.70 (cosine similarity)

---

## 🛡️ Anti-Spoofing & Anomaly Detection

### Voice Anti-Spoofing (CNN)
```
Detects:
├── Replay Attacks (channel mismatch)
├── Synthetic Speech (TTS artifacts)
└── Voice Cloning (deepfake artifacts)

Architecture:
├── Conv2D Layer 1: 32 filters
├── Conv2D Layer 2: 64 filters
├── Conv2D Layer 3: 128 filters
└── FC Layers: 128 → 64 → 1 (binary)
```

### Keystroke Anomaly Detection (Isolation Forest)
```
Detects:
├── Typing Too Fast (< 50% baseline)
├── Typing Too Slow (> 200% baseline)
├── Irregular Rhythm (std > 200% baseline)
└── Unusual Patterns (z-score > 3.0)
```

### Mouse Anomaly Detection (Isolation Forest)
```
Detects:
├── Automated Behavior (constant velocity, linear paths)
├── Remote Desktop (high jitter, unusual pauses)
├── Impossible Speed (> 5000 pixels/second)
└── User Substitution (sudden confidence drops)
```

---

## 🔧 Training Configuration

### Voice (ECAPA-TDNN)
```yaml
Pre-trained: Yes (VoxCeleb)
Loss: AAM-Softmax
Optimizer: Adam (lr=0.001)
Dataset: VoxCeleb1 & VoxCeleb2 (7K speakers)
Training: Not required (use pre-trained)
```

### Keystroke (DNN)
```yaml
Pre-trained: No
Loss: Triplet Loss (margin=0.2)
Optimizer: Adam (lr=0.001)
Dataset: DSL-StrongPasswordData (51 users)
Epochs: 100
Batch Size: 32
Triplet Mining: Hard negatives
```

### Mouse (Siamese)
```yaml
Pre-trained: No
Loss: Triplet Loss (margin=0.2)
Optimizer: Adam (lr=0.001)
Dataset: Balabit Mouse Challenge (10 users)
Epochs: 100
Batch Size: 32
Triplet Mining: Hard negatives
```

---

## 💡 Key Concepts

### Siamese Network
- Twin networks with **shared weights**
- Learns **similarity metrics** directly
- Compares two samples in **embedding space**

### Triplet Loss
```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

Where:
- anchor: User's genuine sample
- positive: Same user's different sample
- negative: Different user's sample
- margin: Minimum separation (0.2)
```

### Embedding Space
- High-dimensional vector representation
- Similar samples → close in space
- Different samples → far apart
- Enables **cosine similarity** comparison

### Hard Negative Mining
- Select **most difficult** negative samples
- Negatives with **highest similarity** to anchor
- Focuses learning on **challenging cases**
- Improves **discrimination** capability

---

## 🚀 Deployment

### Model Sizes
- **Voice ECAPA-TDNN:** ~200MB (pre-trained)
- **Voice Anti-Spoof CNN:** ~5MB
- **Keystroke DNN:** ~10MB
- **Mouse Siamese:** ~15MB

### Inference Speed (CPU)
- **Voice:** ~800ms per 2.5s audio
- **Keystroke:** ~500ms per sample
- **Mouse:** ~300ms per sample

### Storage per User
- **Voice Embedding:** 192 floats = 768 bytes (encrypted: ~2KB)
- **Keystroke Embedding:** 128 floats = 512 bytes
- **Mouse Embedding:** 128 floats = 512 bytes
- **Total:** ~3KB per user

---

## 📚 Further Reading

For detailed explanations, see:
- **ML_MODELS_TECHNICAL_EXPLANATION.md** - Complete technical documentation
- **Voiceprint Analysis/README.md** - Voice authentication details
- **Keystroke Dynamics/README.md** - Keystroke dynamics details
- **Mouse Movement Analysis/README.md** - Mouse movement details

---

**Quick Reference Version:** 1.0  
**Last Updated:** December 9, 2025
