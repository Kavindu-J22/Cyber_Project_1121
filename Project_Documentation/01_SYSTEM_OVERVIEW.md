# MediConsult - Zero Trust Telemedicine Platform

## Complete System Overview

### What Is This Application?

MediConsult is a **secure telemedicine (online doctor-patient consultation) web application** that uses **continuous biometric authentication** to ensure the identity of doctors throughout live video consultations. It implements a **Zero Trust Architecture** — meaning no one is trusted by default, and the doctor's identity is verified continuously using 4 different biometric modalities during every consultation session.

---

## How The Application Works (End-to-End Flow)

### Step 1: Registration

**Doctors** register with:
- Personal info (name, email, password)
- Professional info (medical license number, specialization, years of experience)
- **Biometric enrollment** — the system captures:
  - **3 face images** via webcam
  - **3 voice recordings** (5-10 seconds each)
  - **3 keystroke samples** (typing "The quick brown fox jumps over the lazy dog")
  - **15 seconds of mouse movement** patterns
- OTP email verification
- Human verification puzzle (image selection + slider)

**Patients** register with:
- Basic info (name, age, gender, email, password)
- OTP email verification
- No biometric enrollment required

### Step 2: Patient Books an Appointment

1. Patient logs in → sees list of active doctors
2. Patient selects a doctor → fills in appointment request (reason, preferred time, preferred dates)
3. Appointment status = "Pending"

### Step 3: Doctor Approves/Rejects Appointment

1. Doctor logs in → sees pending appointment requests
2. Doctor **approves** (sets specific date + time) or **rejects** (with reason)
3. Patient receives an **email notification** with the decision

### Step 4: Starting a Live Consultation

**Doctor side (3-step verification before starting):**
1. **Biometric Verification** — Face, Voice, Keystroke, Mouse verified against enrollment (minimum 2 of 4 must pass)
2. **OTP Verification** — 6-digit OTP sent to doctor's email
3. **Human Verification** — Slider puzzle
4. Consultation room opens → patient receives email to join

**Patient side:**
- Sees the consultation is ready → clicks "Join Consultation"
- Can send a "Waiting Alert" email to the doctor if they haven't started yet

### Step 5: During the Live Consultation

- **WebRTC peer-to-peer video/audio call** between doctor and patient
- **Real-time chat** messaging
- **Continuous biometric verification of the doctor**:
  - **Face recognition**: Every 5 seconds (camera frame capture)
  - **Voice recognition**: Every 20 seconds (5-second audio recording)
  - **Keystroke dynamics**: Every 10 seconds (when chat is open)
  - **Mouse movement**: Every 10 seconds (passive tracking)
- **Trust Score** (0-100%) displayed to both parties
- If trust drops too low → alerts are raised
- After 5 consecutive failures → **doctor is locked out**, consultation terminated, patient is notified via email

### Step 6: Ending the Consultation

- Doctor clicks "End Consultation"
- Session is marked as "Completed"
- Both participants are notified

---

## System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                       │
│                         Port: 5173                                   │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌───────────┐ │
│  │ Home    │ │ Register │ │ Dashboard │ │ Patient  │ │ Meeting   │ │
│  │ Page    │ │ (Doctor/ │ │ (Doctor)  │ │Dashboard │ │ Room      │ │
│  │         │ │ Patient) │ │           │ │          │ │ (WebRTC)  │ │
│  └─────────┘ └──────────┘ └───────────┘ └──────────┘ └───────────┘ │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────┐    │
│  │ Admin Dashboard │  │ Doctor Profile │  │ Biometric Capture   │    │
│  │                 │  │                │  │ (Face/Voice/Key/    │    │
│  │                 │  │                │  │  Mouse classes)     │    │
│  └────────────────┘  └────────────────┘  └─────────────────────┘    │
└───────────┬──────────────────────────────────────┬──────────────────┘
            │ HTTP (Axios)                         │ WebSocket (Socket.IO)
            ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    BACKEND (Express.js + Node.js)                    │
│                         Port: 5000                                   │
│  ┌──────────┐ ┌───────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │ Auth     │ │ Doctor/   │ │ Appointment  │ │ Consultation     │  │
│  │ Routes   │ │ Patient   │ │ Routes       │ │ Routes           │  │
│  │          │ │ Routes    │ │              │ │                  │  │
│  └──────────┘ └───────────┘ └──────────────┘ └──────────────────┘  │
│  ┌──────────┐ ┌───────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │ OTP      │ │ Session   │ │ Verification │ │ Socket.IO        │  │
│  │ Routes   │ │ Routes    │ │ Routes       │ │ (WebRTC Signal)  │  │
│  └──────────┘ └───────────┘ └──────────────┘ └──────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ ML Service Gateway (mlService.js)                            │   │
│  │ Routes biometric verification requests to Python ML APIs     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────┬────────────┬────────────┬────────────┬───────────┬───────────┘
      │            │            │            │           │
      ▼            ▼            ▼            ▼           ▼
┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ MongoDB  │ │ Voice API │ │Keystroke │ │ Mouse    │ │ Face     │
│ Atlas    │ │ Port:8001 │ │ API      │ │ API      │ │ API      │
│ (Cloud)  │ │ ECAPA-TDNN│ │Port:8002 │ │Port:8003 │ │Port:8004 │
│          │ │ (Python)  │ │ DNN      │ │ Siamese  │ │ResNet50  │
│          │ │           │ │ (Python) │ │ (Python) │ │(Python)  │
└──────────┘ └───────────┘ └──────────┘ └──────────┘ └──────────┘
```

---

## User Roles

| Role | Description | Access |
|------|-------------|--------|
| **Doctor** | Medical professional who conducts consultations | Dashboard, profile editor, appointment management, consultation room (with biometric verification) |
| **Patient** | Person seeking medical consultation | Patient dashboard, doctor search, booking appointments, joining consultations |
| **Admin** | System administrator | Admin dashboard, manage all doctors/patients, view all appointments, ML service health monitoring |

---

## Key Technologies Used

| Technology | Purpose | Where Used |
|------------|---------|------------|
| **React 18** | Frontend UI framework | All client-side pages and components |
| **Vite** | Fast frontend build tool & dev server | Client build pipeline |
| **Tailwind CSS** | Utility-first CSS framework | All frontend styling |
| **Express.js** | Backend REST API framework | All server-side routes and controllers |
| **MongoDB Atlas** | Cloud database | User data, appointments, consultations, sessions, OTPs |
| **Mongoose** | MongoDB ODM (Object Document Mapper) | Schema definitions, queries, validation |
| **Socket.IO** | Real-time bidirectional communication | WebRTC signaling, chat, biometric score updates |
| **WebRTC** | Peer-to-peer video/audio | Live consultation video calls |
| **JWT** | Authentication tokens | Stateless auth for API requests |
| **bcrypt** | Password hashing | Secure password storage |
| **Nodemailer** | Email service | OTP emails, appointment notifications, alerts |
| **Cloudinary** | Cloud image storage | Doctor profile images |
| **Multer** | File upload handling | Voice samples, face images |
| **Helmet** | HTTP security headers | Backend security hardening |
| **Morgan** | HTTP request logging | Backend request monitoring |
| **PyTorch** | Deep learning framework | All 4 ML model training/inference |
| **FastAPI** | Python REST API framework | All 4 ML microservice APIs |
| **SpeechBrain** | Speaker recognition toolkit | ECAPA-TDNN voice model |
| **OpenCV** | Computer vision | Face detection in face verification |
| **ffmpeg** | Audio format conversion | Voice sample WAV conversion |

---

## Security Features

1. **Continuous Biometric Authentication** — 4 modalities verified throughout consultation
2. **Zero Trust Architecture** — Never trust, always verify
3. **JWT Token Authentication** — Stateless, role-based access control
4. **OTP Email Verification** — For registration and consultation start
5. **Human Verification Puzzles** — Anti-bot protection
6. **Password Hashing** — bcrypt with salt rounds
7. **CORS Protection** — Restricted to allowed origins
8. **Helmet Security Headers** — XSS, clickjacking, MIME protection
9. **Automatic Lockout** — 5 biometric failures → session terminated + email notification
10. **Trust Score Monitoring** — Real-time 0-100% score visible to both parties
11. **No Raw Biometric Storage** — Only embeddings stored (images/audio discarded)
12. **Encrypted Voice Embeddings** — AES-256 encryption for voiceprint data
