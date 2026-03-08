# ML Models Integration — How Models Connect to the Web App

## Overview: Biometric Pipeline

```
Frontend (React)                   Backend (Express)              ML Service (Python/FastAPI)
┌────────────────┐                ┌────────────────┐              ┌────────────────┐
│ Capture Data   │── HTTP POST ──→│ /api/verify/*  │── HTTP POST →│ Model Inference│
│ (JS classes)   │                │ Verification   │              │ Return Score   │
│                │                │ Controller     │              │                │
│                │←── Response ───│ mlService.js   │←── Response──│                │
└────────────────┘                └────────────────┘              └────────────────┘
```

---

## When Each Model Is Called

### During Doctor Registration (Enrollment)

| Step | Frontend | Backend | ML Service |
|------|----------|---------|------------|
| 1 | `FaceCapture.captureFrame()` × 3 | `mlService.enrollFace(userId, paths[])` | Face API: `POST /api/v1/enroll` (port 8004) |
| 2 | `VoiceCapture.start()` → `stop()` × 3 | `mlService.enrollVoiceMultiple(userId, paths[])` | Voice API: `POST /api/v1/enroll` (port 8001) |
| 3 | `KeystrokeCapture.start()` → `stop()` × 3 | `mlService.enrollKeystroke(userId, samples[])` | Keystroke API: `POST /keystroke/enroll` (port 8002) |
| 4 | `MouseCapture.start()` → `stop()` | `mlService.enrollMouse(userId, events[])` | Mouse API: `POST /mouse/enroll` (port 8003) |

**Registration bundles all data** into one `POST /api/auth/register` multipart request. The backend then calls each ML service sequentially.

### Before Starting Consultation (Verification Gate)

| Step | What Happens | API Call | Threshold |
|------|-------------|----------|-----------|
| Face | Capture 1 frame from webcam | `POST /api/verification/face` → port 8004 | 0.45 |
| Voice | Record 5s of speech | `POST /api/verification/voice` → port 8001 | 0.45 |
| Keystroke | Type "MediConsult Secure Access" | `POST /api/verification/keystroke` → port 8002 | 0.45 |
| Mouse | Track natural movement | `POST /api/verification/mouse` → port 8003 | 0.45 |

**Rule: 2 out of 4 must pass** (confidence ≥ 0.45) to proceed.

### During Live Consultation (Continuous)

| Modality | Frequency | Frontend Capture | Backend Route | ML Service |
|----------|-----------|-----------------|---------------|------------|
| **Face** | Every 5 seconds | `FaceCapture.captureFrame()` from video stream | `POST /api/verification/face` | Face API verify (port 8004) |
| **Voice** | Every 20 seconds | `VoiceCapture` records 5 seconds | `POST /api/verification/voice` | Voice API verify (port 8001) |
| **Keystroke** | Every 10 seconds | `KeystrokeCapture` (only when chat is open) | `POST /api/verification/keystroke` | Keystroke API verify (port 8002) |
| **Mouse** | Every 10 seconds | `MouseCapture` passive tracking | `POST /api/verification/mouse` | Mouse API verify (port 8003) |

---

## Detailed API Flow for Each Model

### Face Verification Flow

```
[Frontend - Meeting.jsx]
setInterval(5000):                     // Every 5 seconds
  frame = FaceCapture.captureFrame(video, 224, 224)  // Capture from webcam
  formData = { userId, face_sample: frame }
  
  POST /api/verification/face (multipart)
      │
      ▼
[Backend - verificationRoutes.js → mlService.js]
  mlService.verifyFace(userId, imagePath)
      │
      ▼
  POST http://localhost:8004/api/v1/verify
  FormData: { user_id, face_sample: file }
      │
      ▼
[Face API - Python]
  1. Load image → detect face (Haar Cascade)
  2. If no face → return confidence=0.05
  3. Preprocess: resize 224×224, normalize
  4. ResNet50 → 2048-D → Embedding → 128-D
  5. Cosine similarity vs enrolled embeddings
  6. Calibrated sigmoid confidence
  7. Return { verified, confidence_score, decision }
      │
      ▼
[Backend]
  Return result + update Session.verificationLogs
      │
      ▼
[Frontend]
  Update face confidence bar + overall trust score
```

### Voice Verification Flow

```
[Frontend - Meeting.jsx]
setInterval(20000):                    // Every 20 seconds
  VoiceCapture.start()                 // Record 5 seconds
  setTimeout(5000):
    blob = VoiceCapture.stop()         // Get WAV blob
    formData = { userId, audio: blob }
    
    POST /api/verification/voice (multipart)
        │
        ▼
[Backend - verificationRoutes.js → audioConverter.js → mlService.js]
  convertToWav(audioPath)              // ffmpeg: any format → WAV 16kHz mono
  mlService.verifyVoice(userId, wavPath)
        │
        ▼
  POST http://localhost:8001/api/v1/verify/upload
  FormData: { speaker_id, audio_file: wavFile }
        │
        ▼
[Voice API - Python]
  1. Load audio → resample 16kHz
  2. Noise reduction (NoiseReduce)
  3. VAD → remove silence
  4. Segment → 2.5s windows
  5. ECAPA-TDNN → 192-D embedding per segment
  6. Cosine similarity vs enrolled voiceprint
  7. Average confidence across segments
  8. Anti-spoofing check (replay/synthetic/cloning)
  9. Return { verified, confidence_score, anti_spoofing }
        │
        ▼
[Backend]
  Return result + update Session.verificationLogs
        │
        ▼
[Frontend]
  Update voice confidence bar + overall trust score
```

### Keystroke Verification Flow

```
[Frontend - Meeting.jsx]
When chat is OPEN:
  KeystrokeCapture.start()             // Attach to keydown/keyup events
  
  setInterval(10000):                  // Every 10 seconds
    features = KeystrokeCapture.stop() // 38-D feature vector
    POST /api/verification/keystroke (JSON)
    { userId, keystrokeData: features }
        │
        ▼
[Backend - verificationRoutes.js → mlService.js]
  mlService.verifyKeystroke(userId, keystrokeSample)
        │
        ▼
  POST http://localhost:8002/keystroke/verify
  JSON: { user_id, keystroke_sample: features38D }
        │
        ▼
[Keystroke API - Python]
  1. Validate feature vector (38 dimensions)
  2. Normalize with RobustScaler
  3. DNN forward → 128-D embedding
  4. Cosine similarity vs enrolled template
  5. Calibrated sigmoid confidence
  6. Return { verified, confidence, confidence_level }
        │
        ▼
[Backend]
  Return result + update Session.verificationLogs
        │
        ▼
[Frontend]
  Update keystroke confidence bar + overall trust score

When chat is CLOSED:
  Keystroke confidence defaults to 0.50 (neutral)
```

### Mouse Verification Flow

```
[Frontend - Meeting.jsx]
Always active (passive tracking):
  MouseCapture.start()                 // Attach to mousemove + mouseclick
  
  setInterval(10000):                  // Every 10 seconds
    events = MouseCapture.getEvents()  // Array of {timestamp, x, y, button, state}
    POST /api/verification/mouse (JSON)
    { userId, mouseEvents: events }
        │
        ▼
[Backend - verificationRoutes.js → mlService.js]
  mlService.verifyMouse(userId, mouseEvents)
        │
        ▼
  POST http://localhost:8003/mouse/verify
  JSON: { user_id, events: [...] }
        │
        ▼
[Mouse API - Python]
  1. Check minimum events (≥50, else return 0.5)
  2. Feature extraction:
     - Velocity, acceleration, curvature, jerk
     - Click dynamics, trajectory, temporal
     - ~40-50 features per window
  3. Normalize with RobustScaler
  4. Siamese Network → 128-D embedding
  5. Cosine similarity vs enrolled template
  6. Calibrated tight sigmoid confidence
  7. Anomaly detection (bot, RDP, substitution)
  8. Return { verified, confidence, risk_level }
        │
        ▼
[Backend]
  Return result + update Session.verificationLogs
        │
        ▼
[Frontend]
  Update mouse confidence bar + overall trust score
```

---

## Trust Score Aggregation

```
All 4 modality scores collected:
  Face:      0.87 (checked 5s ago)
  Voice:     0.78 (checked 15s ago)
  Keystroke: 0.50 (chat closed, default)
  Mouse:     0.82 (checked 8s ago)

Backend Session Controller:
  1. Get last 10 verification logs (all modalities combined)
  2. Average = mean(0.87, 0.78, 0.50, 0.82, ...) = 0.74
  3. overallTrustScore = 74
  4. Store in Session model

Frontend Display:
  Overall Trust: 74% [████████████░░░░░░░░] 
  Face:      87% [█████████████████░░░]
  Voice:     78% [███████████████░░░░░]
  Keystroke: 50% [██████████░░░░░░░░░░]
  Mouse:     82% [████████████████░░░░]
```

---

## Alert & Lockout Flow

```
If average confidence of last 10 logs < 50%:
    │
    ▼
  Session.status → "suspicious"
  Alert added to Session.alerts[]
  Socket.IO: 'verification-alert' → both participants
    │
    ▼
If 15 minutes of continuous low trust:
    │
    ▼
  LOCKOUT: Doctor must re-authenticate
  ├── OTP sent to doctor's email
  ├── Slider verification required
  └── Socket: 'doctor-lockout-status' → patient sees "Doctor locked out"
    │
    ▼
If 5 total lockout failures:
    │
    ▼
  Socket: 'lockout-max-attempts'
  ├── Consultation TERMINATED
  ├── Email sent to patient (lockout termination)
  ├── Session ended for all participants
  └── All streams disconnected
```

---

## Mock ML Services (Development Mode)

When real Python ML services aren't running, `Backend/mock-ml-services.js` provides:

| Service | Port | Behavior |
|---------|------|----------|
| Voice | 8001 | Always returns `verified: true`, confidence: 0.85-1.0 (random) |
| Keystroke | 8002 | Always returns `verified: true`, confidence: 0.88-1.0 (random) |
| Mouse | 8003 | Always returns `verified: true`, confidence: 0.82-1.0 (random) |

Face verification (port 8004) is not mocked — requires the real Python service.

Run mock services: `node Backend/mock-ml-services.js`

---

## Model Comparison Table

| Property | Face | Voice | Keystroke | Mouse |
|----------|------|-------|-----------|-------|
| **Architecture** | ResNet50 | ECAPA-TDNN | 4-layer DNN | Siamese Network |
| **Embedding Dim** | 128 | 192 | 128 | 128 |
| **Training** | Triplet Loss | Pre-trained (VoxCeleb) | Triplet Loss | Triplet Loss |
| **Input** | 224×224 image | 16kHz audio | 38 timing features | ~40-50 movement features |
| **Threshold** | 0.8096 | 0.65 | 0.85 | 0.85 |
| **Check Interval** | 5 seconds | 20 seconds | 10 seconds | 10 seconds |
| **Passive?** | Semi (camera always on) | Semi (mic always on) | No (only when typing) | Yes (always) |
| **Anti-spoofing** | Face detection gate | Replay/synthetic/cloning | Anomaly detection | Bot/RDP/substitution |
| **Storage** | enrollments.pkl | speakers.pkl (encrypted) | templates.pth | templates.pth |
| **Port** | 8004 | 8001 | 8002 | 8003 |
| **Inference Time** | ~280ms | ~180-250ms | <20ms | <500ms |
