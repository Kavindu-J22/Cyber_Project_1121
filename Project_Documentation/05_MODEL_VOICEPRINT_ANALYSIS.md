# ML Model 2: Voiceprint Analysis (ECAPA-TDNN)

## Overview

| Property | Value |
|----------|-------|
| **Service Port** | 8001 |
| **Framework** | PyTorch + SpeechBrain + FastAPI |
| **Model Architecture** | ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN) |
| **Pre-trained** | VoxCeleb1 + VoxCeleb2 (7,363 speakers) |
| **Training Method** | Pre-trained speaker recognition (no custom training needed) |
| **Input** | Audio at 16kHz sample rate |
| **Output** | 192-dimensional speaker embedding |
| **Similarity Metric** | Cosine Similarity |
| **Default Threshold** | 0.65 |
| **Parameters** | ~6.2 million |
| **Inference Speed** | ~180-250ms per verification |

---

## Architecture

```
Raw Audio (16kHz)
    │
    ▼
┌──────────────────────────────────────────────┐
│   Audio Preprocessing                         │
│   ├── Resample to 16kHz                      │
│   ├── Noise Reduction (NoiseReduce library)  │
│   ├── Voice Activity Detection (energy-based)│
│   ├── Normalize amplitude to [-1, 1]         │
│   └── Segment into 2.5s windows (50% overlap)│
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│   Mel-Filterbank Extraction                   │
│   └── 80 mel-frequency bins from spectrogram │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│   ECAPA-TDNN Encoder (5 Conv1D Blocks)       │
│   ├── Block 1: 1024 ch, kernel=5, dilation=1│
│   ├── Block 2: 1024 ch, kernel=3, dilation=2│
│   ├── Block 3: 1024 ch, kernel=3, dilation=3│
│   ├── Block 4: 1024 ch, kernel=3, dilation=4│
│   └── Block 5: 3072 ch, kernel=1, dilation=1│
│   (Each block: SE-Res2Net attention)         │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│   Attentive Statistics Pooling               │
│   ├── Weighted mean + std computation        │
│   └── Output: 6144-D vector                 │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│   Fully Connected Layer                       │
│   ├── Projects 6144 → 192 dimensions        │
│   └── L2 Normalization                       │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
        192-D Speaker Embedding
   (unique voice fingerprint)
```

### Why ECAPA-TDNN?
- **Not RNN/LSTM**: Processes audio in parallel (CNN-based), faster inference
- **Dilated Convolutions**: Pattern [1, 2, 3, 4, 1] captures long-range dependencies
- **SE-Res2Net**: Squeeze-and-excitation attention for channel importance
- **Attentive Pooling**: Learns which time frames matter most for speaker identity
- **Pre-trained on massive data**: VoxCeleb1 + VoxCeleb2 = 7,363 speakers

---

## How It Works

### Audio Preprocessing Pipeline

```
Raw Audio File
    │
    ▼
[1] Load & Resample to 16kHz (torchaudio / librosa)
    │
    ▼
[2] Noise Reduction (NoiseReduce library)
    ├── Removes stationary background noise
    └── Preserves voice characteristics
    │
    ▼
[3] Voice Activity Detection (Energy-based)
    ├── Frame duration: 30ms
    ├── Energy threshold: 0.001
    └── Removes silence/non-speech sections
    │
    ▼
[4] Normalize amplitude to [-1, 1]
    │
    ▼
[5] Segment into 2.5-second windows (50% overlap)
    ├── Window samples: 40,000 (2.5s × 16kHz)
    ├── Hop samples: 20,000 (1.25s hop)
    └── Returns list of audio segments
```

---

## Enrollment Flow

```
Doctor provides 3 voice recordings (5-10 seconds each) during registration
    │
    ▼
For each audio file:
├── Preprocess (resample, noise reduction, VAD, normalize)
├── Segment into 2.5-second windows
├── For each segment:
│   ├── Extract mel-filterbank features (80 bins)
│   ├── Forward through ECAPA-TDNN → 192-D embedding
│   └── L2 normalize
└── Collect all segment embeddings
    │
    ▼
Compute Enrollment Template:
├── Mean embedding = average(all_segment_embeddings)
├── Enrollment quality = mean(intra-speaker similarities) → e.g., 0.92
└── Typically: 3 samples × 3 segments each = 9 embeddings averaged
    │
    ▼
Security:
├── Encrypt voiceprint with AES-256 encryption
└── Raw audio is NEVER stored (privacy compliance)
    │
    ▼
Store: {
  speaker_id: "doctor_001",
  voiceprint_template: <encrypted 192-D embedding>,
  num_samples: 3,
  num_embeddings: 9,
  enrollment_quality: 0.92,
  embedding_dim: 192
}
    │
    ▼
Persist to data/speakers.pkl (survives restarts)
```

---

## Verification Flow

```
5-second audio captured during consultation (every 20 seconds)
    │
    ▼
Audio Preprocessing:
├── Resample to 16kHz
├── Noise reduction
├── VAD (remove silence)
├── Normalize
└── Segment into 2.5s windows
    │
    ▼
For each segment:
├── Extract 192-D embedding via ECAPA-TDNN
├── L2 normalize
├── Retrieve enrolled voiceprint (decrypt AES-256)
└── Compute cosine similarity
    │
    ▼
Aggregate Across Segments:
├── confidence_score = mean(all segment similarities)
│   e.g., Segment 1: 0.82, Segment 2: 0.79, Segment 3: 0.81
│   → confidence = 0.807
    │
    ▼
Anti-Spoofing Check:
├── Extract LFCC features (Linear Frequency Cepstral Coefficients)
├── CNN classifier detects:
│   ├── Replay attacks
│   ├── Synthetic speech
│   └── Voice cloning
└── Returns: { is_genuine: true/false, spoofing_type: null/"replay" }
    │
    ▼
Decision:
├── confidence >= 0.65 (threshold) → VERIFIED
└── confidence < 0.65 → NOT VERIFIED
├── If confidence < 0.50 → Security alert triggered
```

### Score Distribution

| Scenario | Typical Confidence | Range |
|----------|-------------------|-------|
| Same speaker (genuine) | 0.92 | 0.75 — 0.99 |
| Different speaker (impostor) | 0.35 | 0.05 — 0.65 |
| Threshold (EER point) | 0.65 | — |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root health check |
| GET | `/health` | Health status |
| POST | `/api/v1/enroll` | Enroll with file paths |
| POST | `/api/v1/enroll/upload` | Enroll with uploaded audio files |
| POST | `/api/v1/verify` | Verify from file path |
| POST | `/api/v1/verify/upload` | Verify with uploaded audio |
| POST | `/api/v1/continuous-verify` | Continuous verification (multiple windows) |
| GET | `/api/v1/speakers` | List all enrolled speakers |
| GET | `/api/v1/speakers/{speaker_id}` | Get speaker info |
| DELETE | `/api/v1/speakers/{speaker_id}` | Remove speaker |
| PUT | `/api/v1/threshold` | Update verification threshold dynamically |
| GET | `/api/v1/alerts` | Get recent security alerts |
| WebSocket | `/ws/verify/{speaker_id}` | Real-time streaming verification |

---

## Anti-Spoofing Features

1. **Replay Attack Detection**: Detects recorded audio playback
2. **Synthetic Speech Detection**: Identifies computer-generated speech
3. **Voice Cloning Detection**: Detects deepfake voice attempts
4. **LFCC-based CNN Classifier**: Analyzes frequency characteristics for spoofing signatures
5. **No Raw Audio Storage**: Only encrypted embeddings stored

---

## Configuration (`Voiceprint Analysis/config.yaml`)

```yaml
model:
  type: "ecapa_tdnn"
  embedding_dim: 192
  pretrained: true

audio:
  sample_rate: 16000
  window_duration: 2.5              # seconds
  window_overlap: 0.5               # 50% overlap
  min_duration: 1.0
  max_duration: 10.0
  vad_enabled: true
  noise_reduction: true

verification:
  similarity_metric: "cosine"
  threshold: 0.65
  enrollment_samples: 1             # Minimum (3 recommended)

anti_spoofing:
  enabled: true
  detect_replay: true
  detect_synthetic: true
  detect_voice_cloning: true

security:
  encrypt_embeddings: true
  store_raw_audio: false

api:
  port: 8001
  workers: 1
```

---

## Docker Deployment

The voice service includes Docker support:
- **Dockerfile**: Python 3.10-slim with ffmpeg, libsndfile
- **docker-compose.yml**: Voice API + MongoDB container
- **Health check**: Every 30 seconds, 40-second start period

---

## When This Model Is Used in the Web App

1. **Doctor Registration** → `POST /api/v1/enroll/upload` — 3 voice samples enrolled
2. **Before Starting Consultation** → `POST /api/v1/verify/upload` — Voice verified as part of biometric check
3. **During Consultation** → `POST /api/v1/verify/upload` every 20 seconds — 5-second audio recordings sent
4. **Frontend captures** → `VoiceCapture.start()` records 5 seconds → `VoiceCapture.stop()` returns WAV blob
5. **Backend routes** → `POST /api/verification/voice` → `mlService.verifyVoice()` → Voice API at port 8001
6. **Audio conversion** → Backend converts uploaded audio to 16kHz WAV via ffmpeg before sending to ML service
