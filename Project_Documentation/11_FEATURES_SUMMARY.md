# Features Summary

## Complete Feature List

---

## User-Facing Features

### For Patients
| Feature | Description |
|---------|-------------|
| **Registration** | Simple signup with name, age, gender, email, password + OTP verification |
| **Doctor Search** | Browse and search active doctors by name or specialization |
| **Doctor Details** | View doctor profile: image, specialization, experience, description |
| **Book Appointment** | Request an appointment with reason, preferred time, and preferred dates |
| **Appointment Tracking** | View all appointments with status (Pending/Approved/Rejected) |
| **Cancel Appointment** | Cancel pending appointments |
| **Join Consultation** | Join live video consultation within the scheduled time window |
| **Send Waiting Alert** | Notify doctor via email when waiting for consultation to start |
| **Video/Audio Call** | WebRTC peer-to-peer live video and audio |
| **Real-time Chat** | Text messaging during consultation |
| **View Trust Score** | See doctor's live biometric trust score during consultation |
| **Lockout Notification** | Get notified if doctor is biometrically locked out |
| **Email Notifications** | Receive emails for appointment approval/rejection, consultation start, lockout |

### For Doctors
| Feature | Description |
|---------|-------------|
| **Registration with Biometrics** | Multi-step signup with face, voice, keystroke, and mouse enrollment |
| **Profile Management** | Edit name, specialization, experience, description, profile image |
| **View Appointment Requests** | See pending patient appointment requests |
| **Approve Appointments** | Set date, time, and notes for approved appointments |
| **Reject Appointments** | Decline with reason |
| **3-Step Consultation Start** | Biometric + OTP + Human verification before starting |
| **Video/Audio Consultation** | Live WebRTC calls with patients |
| **Real-time Chat** | In-consultation text messaging |
| **Biometric Monitoring** | See live confidence scores for all 4 modalities |
| **Trust Score Display** | Overall trust percentage shown during consultation |
| **End Consultation** | Close consultation session |

### For Admin
| Feature | Description |
|---------|-------------|
| **Dashboard Overview** | System-wide statistics (total doctors, patients) |
| **ML Service Health** | Monitor status of all 4 biometric services |
| **Doctor Management** | View, search, edit, activate/deactivate, delete doctors |
| **Patient Management** | View all patients |
| **Appointment Oversight** | View all system appointments with advanced filtering |

---

## Security Features

| Feature | Description |
|---------|-------------|
| **Multi-Modal Biometric Authentication** | 4 independent modalities (face, voice, keystroke, mouse) |
| **Continuous Verification** | Biometrics checked periodically throughout consultation |
| **Zero Trust Architecture** | Never trust, always verify — even after initial authentication |
| **Trust Score System** | Real-time 0-100% confidence aggregated from all modalities |
| **Automatic Lockout** | 5 biometric failures or 15 minutes of low trust → session terminated |
| **OTP Email Verification** | 6-digit codes for registration and consultation start |
| **Human Verification Puzzle** | Image selection + slider verification to prevent automation |
| **JWT Token Authentication** | Stateless, role-based API access control |
| **Password Hashing** | bcrypt with salt for all passwords |
| **No Raw Biometric Storage** | Only mathematical embeddings stored (images/audio discarded) |
| **Encrypted Voice Embeddings** | AES-256 encryption for voiceprint data |
| **Face Detection Gate** | Rejects verification attempts with no face detected (covered camera) |
| **Anti-Spoofing (Voice)** | Detects replay attacks, synthetic speech, voice cloning |
| **Anomaly Detection (Mouse)** | Detects automated bots, remote desktop usage, user substitution |
| **CORS Protection** | API restricted to allowed frontend origins |
| **Helmet Security Headers** | XSS, clickjacking, MIME-sniff protection |
| **TTL Auto-Expiry** | OTPs auto-delete from database after 10 minutes |

---

## Real-Time Features

| Feature | Description |
|---------|-------------|
| **WebRTC Video Call** | Peer-to-peer video/audio without server relay |
| **Socket.IO Signaling** | WebRTC connection establishment (offer/answer/ICE) |
| **Live Chat** | Real-time text messaging with timestamps and read status |
| **Live Biometric Scores** | Doctor's scores broadcast to patient in real-time |
| **Lockout Status Broadcast** | Patient instantly notified of doctor lockout |
| **Trust Score Updates** | Real-time trust score changes visible to both parties |
| **Verification Alerts** | Instant notification of suspicious biometric results |

---

## Communication Features

| Feature | Description |
|---------|-------------|
| **OTP Emails** | Branded HTML emails with 6-digit codes |
| **Appointment Approval Email** | Green-themed notification with schedule details |
| **Appointment Rejection Email** | Red-themed notification with reason |
| **Consultation Started Email** | Green-themed "Join Now" notification to patient |
| **Patient Waiting Alert Email** | Amber-themed notification to doctor |
| **Lockout Termination Email** | Red-themed security notification to patient |

---

## ML/AI Features

| Feature | Model | Accuracy/Metric |
|---------|-------|-----------------|
| **Face Recognition** | ResNet50 + Triplet Loss | 128-D embeddings, threshold 0.8096 |
| **Voice Recognition** | ECAPA-TDNN (pre-trained VoxCeleb) | 192-D embeddings, threshold 0.65 |
| **Keystroke Analysis** | DNN + Triplet Loss | 96.20% accuracy, 0.39% FAR |
| **Mouse Movement Analysis** | Siamese Network + Triplet Loss | 128-D embeddings, anomaly detection |
| **Calibrated Confidence** | Sigmoid scoring on all models | Threshold-anchored, 0-1 range |
| **Face Detection** | OpenCV Haar Cascade | Anti-spoofing gate for face verification |
| **Voice Anti-Spoofing** | LFCC + CNN classifier | Replay, synthetic, cloning detection |
| **Mouse Anomaly Detection** | Isolation Forest | Bot, RDP, substitution detection |
| **Adaptive Templates** | EMA template updates | Keystroke patterns evolve over time |

---

## Technical Architecture Features

| Feature | Description |
|---------|-------------|
| **Microservice Architecture** | 4 independent ML services + 1 backend + 1 frontend |
| **Mock ML Services** | Node.js mock for development without Python services |
| **Docker Support** | Dockerized voice service with MongoDB |
| **Persistent Enrollment** | Biometric templates survive service restarts (pickle/pth files) |
| **Health Check System** | ML services expose health endpoints, backend checks all 4 |
| **Configurable Thresholds** | All verification thresholds adjustable via config files |
| **Multiple Security Modes** | High security, balanced, and lenient threshold options |
| **Role-Based Access Control** | 3 roles (doctor, patient, admin) with route-level enforcement |
| **Modular Codebase** | Clean separation: Routes → Controllers → Services → Models |
