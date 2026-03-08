# Complete Application Flow & Logic

## High-Level Application Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  REGISTRATION │ →  │  APPOINTMENT │ →  │ VERIFICATION │ →  │ CONSULTATION │
│  & LOGIN      │    │  BOOKING     │    │  GATE        │    │  WITH LIVE   │
│               │    │              │    │  (3-step)    │    │  BIOMETRICS  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## Flow 1: Doctor Registration (Biometric Enrollment)

```
┌─────── FRONTEND ──────┐           ┌─────── BACKEND ──────┐         ┌── ML SERVICES ──┐
│                        │           │                       │         │                  │
│ Step 1: Personal Info  │           │                       │         │                  │
│ ├── Name, Email, Pass  │──OTP────→│ POST /api/otp/send   │         │                  │
│ └── Enter OTP code     │──Verify─→│ POST /api/otp/verify │         │                  │
│                        │           │                       │         │                  │
│ Step 2: Professional   │           │                       │         │                  │
│ ├── License Number     │           │                       │         │                  │
│ ├── Specialization     │           │                       │         │                  │
│ └── Experience         │           │                       │         │                  │
│                        │           │                       │         │                  │
│ Step 3: Biometrics     │           │                       │         │                  │
│ ├── 3 Face Images ─────┤           │                       │         │                  │
│ ├── 3 Voice Samples ───┤           │                       │         │                  │
│ ├── 3 Keystroke Tries ─┤           │                       │         │                  │
│ ├── Mouse Movement ────┤───POST──→│ /api/auth/register    │         │                  │
│ └── Human Puzzle       │           │ ├── Create Doctor     │         │                  │
│                        │           │ ├── Convert audio→WAV │         │                  │
│                        │           │ ├── enrollVoice() ────┤────────→│ Port 8001 enroll │
│                        │           │ ├── enrollKeystroke()─┤────────→│ Port 8002 enroll │
│                        │           │ ├── enrollMouse() ────┤────────→│ Port 8003 enroll │
│                        │           │ ├── enrollFace() ─────┤────────→│ Port 8004 enroll │
│                        │           │ ├── Update biometric  │         │                  │
│                        │           │ │   status in DB       │         │                  │
│  ←── JWT Token ────────┤───────────│ └── Return token      │         │                  │
│                        │           │                       │         │                  │
│ Redirect to /dashboard │           │                       │         │                  │
└────────────────────────┘           └───────────────────────┘         └──────────────────┘
```

---

## Flow 2: Patient Registration

```
Patient → Enter name, age, gender, email, password
       → OTP verification
       → POST /api/auth/register-patient
       → JWT token returned
       → Redirect to /patient-dashboard
```
(No biometric enrollment needed)

---

## Flow 3: Appointment Booking & Approval

```
┌─── PATIENT ────┐        ┌── BACKEND ──┐        ┌──── DOCTOR ────┐
│                 │        │              │        │                 │
│ Search doctors  │──GET──→│ /api/doctors │        │                 │
│ View details    │        │              │        │                 │
│ Click "Book"    │        │              │        │                 │
│ Fill form:      │        │              │        │                 │
│ ├── Reason      │        │              │        │                 │
│ ├── Pref. time  │        │              │        │                 │
│ └── Pref. dates │        │              │        │                 │
│                 │──POST─→│ /api/        │        │                 │
│                 │        │ appointments │        │                 │
│                 │        │ Status:      │        │                 │
│                 │        │ "Pending"    │        │                 │
│                 │        │              │        │                 │
│                 │        │              │──GET──→│ See pending     │
│                 │        │              │        │ requests        │
│                 │        │              │        │                 │
│                 │        │              │←─PUT──│ APPROVE:        │
│                 │        │              │        │ ├── Set date    │
│ ←── EMAIL ──────┤←──Send─│ Notification │        │ ├── Set time   │
│ "Approved!"     │        │              │        │ └── Add note   │
│                 │        │              │        │                 │
│        OR       │        │              │←─PUT──│ REJECT:         │
│ ←── EMAIL ──────┤←──Send─│ Notification │        │ └── Reason     │
│ "Rejected"      │        │              │        │                 │
└─────────────────┘        └──────────────┘        └─────────────────┘
```

---

## Flow 4: Starting a Consultation (3-Step Verification Gate)

This is the most security-critical flow — the doctor must pass multiple verification layers:

```
Doctor clicks "Start Consultation" on dashboard
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: BIOMETRIC VERIFICATION (4 modalities)                    │
│                                                                   │
│  ┌── Face ──────────────────────────┐                            │
│  │ Capture frame from webcam        │                            │
│  │ POST /api/verification/face      │→ confidence: 0.87 ✓       │
│  └──────────────────────────────────┘                            │
│                                                                   │
│  ┌── Voice ─────────────────────────┐                            │
│  │ Record 5 seconds of speech       │                            │
│  │ POST /api/verification/voice     │→ confidence: 0.78 ✓       │
│  └──────────────────────────────────┘                            │
│                                                                   │
│  ┌── Keystroke ─────────────────────┐                            │
│  │ Type "MediConsult Secure Access" │                            │
│  │ POST /api/verification/keystroke │→ confidence: 0.72 ✓       │
│  └──────────────────────────────────┘                            │
│                                                                   │
│  ┌── Mouse ─────────────────────────┐                            │
│  │ Track natural mouse movement     │                            │
│  │ POST /api/verification/mouse     │→ confidence: 0.65         │
│  └──────────────────────────────────┘                            │
│                                                                   │
│  Requirement: 2 out of 4 must pass (threshold: 0.45)            │
│  Result: 3 passed ✓ → PROCEED                                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: OTP VERIFICATION                                         │
│                                                                   │
│  POST /api/otp/consultation/send → OTP sent to doctor's email   │
│  Doctor enters 6-digit code                                      │
│  POST /api/otp/consultation/verify → Code validated ✓            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: HUMAN VERIFICATION                                       │
│                                                                   │
│  Drag slider to 100% → confirms human presence ✓                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
POST /api/consultations/doctor/:appointmentId/start
    → Consultation status: "Active"
    → Room ID: room-{appointmentId}
    → Email sent to patient: "Join now!"
```

---

## Flow 5: Live Consultation (WebRTC + Continuous Biometrics)

```
┌──────── DOCTOR ────────┐      ┌──── SERVER ────┐      ┌─────── PATIENT ──────┐
│                         │      │                 │      │                       │
│ Navigate to             │      │                 │      │ Navigate to           │
│ /meeting/:sessionId     │      │                 │      │ /meeting/:sessionId   │
│                         │      │                 │      │                       │
│ getUserMedia()          │      │                 │      │ getUserMedia()        │
│ (camera + mic)          │      │                 │      │ (camera + mic)        │
│                         │      │                 │      │                       │
│ Socket: join-session ──→│─────→│ Track room ────→│─────→│ Socket: join-session  │
│                         │      │                 │      │                       │
│ ←── user-joined ────────│←─────│ Notify peers ───│←─────│ user-joined ──→       │
│                         │      │                 │      │                       │
│ Create RTCPeerConnection│      │                 │      │                       │
│ Create Offer ──────────→│─────→│ Relay offer ───→│─────→│ Set Remote Desc.     │
│                         │      │                 │      │ Create Answer ──────→ │
│ ←── Answer ─────────────│←─────│ ←── Relay ──────│←─────│                       │
│ Set Remote Description  │      │                 │      │                       │
│                         │      │                 │      │                       │
│ ←── ICE Candidates ────→│←────→│ ←── Relay ─────→│←────→│ ICE Candidates       │
│                         │      │                 │      │                       │
│ ═══ WebRTC P2P VIDEO/AUDIO ESTABLISHED (direct peer-to-peer) ═══             │
│                         │      │                 │      │                       │
│ CONTINUOUS BIOMETRIC    │      │                 │      │ PATIENT SEES:         │
│ VERIFICATION:           │      │                 │      │ ├── Doctor's video    │
│                         │      │                 │      │ ├── Trust score       │
│ Every 5s: Face ─────────│──API→│ → Face ML ──→  │      │ ├── Biometric scores  │
│ Every 20s: Voice ───────│──API→│ → Voice ML ──→ │      │ └── Lockout status    │
│ Every 10s: Keystroke ───│──API→│ → Key ML ───→  │      │                       │
│ Every 10s: Mouse ───────│──API→│ → Mouse ML ──→ │      │                       │
│                         │      │                 │      │                       │
│ ←── Trust Score Update──│←─────│ Calculate avg  │      │                       │
│                         │      │ of all 4 scores│      │                       │
│ Socket: biometric-update│─────→│ → patient ─────│─────→│ See doctor's scores   │
│                         │      │                 │      │                       │
│ Chat: type message ─────│──WS─→│ → broadcast ───│─────→│ See message           │
│                         │      │                 │      │ Type reply ───────────│
│ ←── Chat message ───────│←─────│ ←── broadcast──│←─────│                       │
│                         │      │                 │      │                       │
│ Click "End" ────────────│──WS─→│ end-session ───│─────→│ session-ended         │
│                         │      │ PUT /end        │      │ Consultation over     │
│                         │      │ status:Complete │      │                       │
└─────────────────────────┘      └─────────────────┘      └───────────────────────┘
```

---

## Flow 6: Lockout Mechanism

```
During consultation, doctor's biometric scores are consistently low:

Minute 0-5:   Trust = 85% → Normal
Minute 5-8:   Trust = 62% → Warning alerts shown
Minute 8-12:  Trust = 45% → Alerts escalating
Minute 12-15: Trust = 38% → 15 minutes of low trust
    │
    ▼
LOCKOUT TRIGGERED
├── Doctor sees: OTP + Slider re-authentication required
├── Patient sees: "Doctor is locked out" notification
├── Both see: Consultation paused
    │
    ▼
Doctor Re-authenticates:
├── Enters OTP sent to email
├── Completes slider verification
└── Session resumes OR...
    │
    ▼
If 5 total failures:
├── Socket: lockout-max-attempts
├── Consultation TERMINATED for all
├── Patient receives email: "Session terminated due to security"
├── Session status → "terminated"
└── Security incident logged
```

---

## Flow 7: Trust Score Calculation

```
Every biometric check returns a confidence score (0-1):

Face:      0.87  (checked every 5 seconds)
Voice:     0.78  (checked every 20 seconds)
Keystroke: 0.50  (default when chat closed, real score when typing)
Mouse:     0.82  (checked every 10 seconds)

Trust Score Calculation:
├── Collect last 10 verification logs from all modalities
├── Average all confidence scores
├── Trust Score = mean × 100 = e.g., 74%
├── Stored in Session.overallTrustScore (0-100)
    │
    ▼
If average confidence < 50%:
├── Session status → "suspicious"
├── Alert emitted via Socket.IO
├── Alert stored in Session.alerts array
└── Both participants notified
```

---

## Flow 8: Email Notification System

| Event | Recipient | Template Color | Content |
|-------|-----------|---------------|---------|
| Registration OTP | Doctor/Patient | Purple gradient | 6-digit code, 10-min validity |
| Consultation OTP | Doctor | Purple gradient | 6-digit code for starting consultation |
| Appointment Approved | Patient | Green gradient | Date, time, doctor name, notes |
| Appointment Rejected | Patient | Red gradient | Rejection reason |
| Consultation Started | Patient | Green gradient | "Join now" CTA |
| Patient Waiting | Doctor | Amber/Orange | "Patient is waiting" alert |
| Lockout Terminated | Patient | Red gradient | "Session ended due to security" |

---

## Authentication Logic

### JWT Token Structure
```json
{
  "id": "mongodb_object_id",
  "role": "doctor|patient|admin",
  "iat": 1700662800,
  "exp": 1701267600   // 7 days
}
```

### Token Flow
```
Login/Register → Server generates JWT → Stored in localStorage
    │
    ▼
Every API request → Authorization: Bearer <token>
    │
    ▼
Server middleware (protect):
├── Extract token from header
├── Verify with JWT_SECRET
├── Find user in Doctor/Patient collection
├── Populate req.user, req.userRole
└── Allow or deny access

Admin special case:
├── Hardcoded: admin@gmail.com / admin
├── Token contains role: "admin"
└── No database record needed
```

### Role-Based Access Control
```
restrictTo('doctor') → Only doctors can access
restrictTo('patient') → Only patients can access
restrictTo('admin') → Only admin can access
restrictTo('doctor', 'patient') → Both can access (e.g., meeting room)
```

---

## Key Business Logic Arguments

### Why 4 Biometric Modalities?
- **Face**: Strongest visual identifier, works continuously via webcam
- **Voice**: Active during audio consultation, captures vocal characteristics
- **Keystroke**: Behavioral biometric, works only when typing (chat)
- **Mouse**: Passive behavioral biometric, always active, requires no user effort

### Why 2 out of 4 to Start?
- Some modalities may fail due to environmental conditions (noisy room for voice, dark room for face)
- Requiring all 4 would cause too many false rejections
- 2/4 provides strong security while allowing operational flexibility

### Why Continuous Verification?
- Initial login only proves identity at one point in time
- Someone could take over the computer after login
- Continuous checking ensures the same doctor is present throughout the entire consultation

### Why OTP + Human Verification Before Consultation?
- Adds a second authentication factor (something you know + something you are)
- Prevents automated/scripted consultation starts
- Ensures physical human presence at the keyboard

### Why ±1 Hour Time Window?
- Doctors may be slightly early or late
- Too narrow would prevent legitimate consultations
- Too wide would allow misuse outside scheduled times
