# Frontend Architecture & Components

## Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2.0 | UI framework with functional components + hooks |
| React Router DOM | 6.20.1 | Client-side routing with nested and protected routes |
| Axios | 1.6.2 | HTTP client with request/response interceptors |
| Socket.IO Client | 4.6.0 | Real-time WebSocket for consultations |
| Vite | 5.0.8 | Fast dev server with HMR + production bundler |
| Tailwind CSS | 3.3.6 | Utility-first CSS framework |
| Lucide React | 0.294.0 | Icon library |
| React Hot Toast | 2.4.1 | Toast notification system |
| Recharts | 2.10.3 | Charting library (for analytics) |

---

## Routing Structure

```
/                    → Home (public landing page)
/login               → Login page (doctor/patient/admin)
/register            → Registration (multi-step, doctor/patient selector)
/dashboard           → Doctor Dashboard (protected: doctor only)
/doctor-profile      → Doctor Profile Editor (protected: doctor only)
/patient-dashboard   → Patient Dashboard (protected: patient only)
/admin-dashboard     → Admin Dashboard (protected: admin only)
/meeting/:sessionId  → Video Consultation Room (protected: doctor/patient)
```

All authenticated routes use `<PrivateRoute>` with role-based access control. Unauthenticated users are redirected to `/login`. Wrong-role users are redirected to their appropriate dashboard.

---

## Page Components

### Home Page (`Home.jsx`)
- Public landing page
- Hero section with CTA buttons (Login/Register)
- Features showcase: security, consultations, monitoring, HIPAA compliance
- Multi-modal biometric authentication explainer (Face, Voice, Keystroke, Mouse)
- "How It Works" 3-step guide

### Login Page (`Login.jsx`)
- Email + password form
- Calls `useAuth().login()` → `POST /api/auth/login`
- Redirects by role: doctor → `/dashboard`, patient → `/patient-dashboard`, admin → `/admin-dashboard`
- Zero Trust branding

### Registration Page (`Register.jsx`)
**Patient flow (1 step):** Name, age, gender, email, password → OTP verification → done

**Doctor flow (3 steps):**
1. **Step 1 — Personal Info:** Name, email, password + OTP email verification
2. **Step 2 — Professional Info:** Medical license number, specialization, years of experience
3. **Step 3 — Biometric Enrollment + Human Verification:**
   - **Face capture**: 3 images via webcam (224×224 resolution)
   - **Voice recording**: 3 samples of 5-10 seconds each
   - **Keystroke dynamics**: Type "The quick brown fox jumps over the lazy dog" 3 times
   - **Mouse movement**: 15-second natural mouse movement recording
   - **Human Verification Puzzle**: Image selection (medical emojis) + slider verification

Submits as multipart form data to `POST /api/auth/register`.

### Doctor Dashboard (`Dashboard.jsx`)
- Shows biometric enrollment status (Face/Voice/Keystroke/Mouse)
- ML Services health check (green/red indicators for each service)
- **Scheduled Consultations tab**: Upcoming consultations with "Start" button → opens `ConsultationStartVerificationModal`
- **Appointments tab**: Pending/Approved/Rejected requests with approve/reject actions
- Header: Doctor name, specialization, profile link, logout

### Patient Dashboard (`PatientDashboard.jsx`)
- Patient info card (name, age, gender)
- **Find Doctors tab**: Search/filter active doctors → view details → book appointments
- **My Appointments tab**: All appointments with status badges and actions
- **Confirmed Consultations tab**: Join live consultations, send waiting alerts

### Doctor Profile (`DoctorProfile.jsx`)
- Edit name, specialization, experience, description
- Upload profile image (preview + Cloudinary upload)
- Email and license number are read-only
- `PUT /api/doctors/:id` with multipart form data

### Admin Dashboard (`AdminDashboard.jsx`)
- ML Services health status dashboard
- Stats cards: Total doctors, total patients
- **Doctors tab**: Table with search/filter, toggle active, delete, view details
- **Patients tab**: All patients list
- **Appointments tab**: All appointments with advanced search and filtering

### Meeting Room (`Meeting.jsx`) — Most Complex Page
**Video/Audio:**
- WebRTC peer-to-peer connection using `RTCPeerConnection`
- STUN servers: `stun.l.google.com:19302`
- Socket.IO for signaling (join → offer → answer → ICE candidates)
- Local video as picture-in-picture, remote video full-screen
- Toggle video/audio independently
- Graceful fallback to audio-only if camera unavailable

**Continuous Biometric Verification (Doctor only):**
- Face: Every 5 seconds, captures frame from video stream → `POST /api/verification/face`
- Voice: Every 20 seconds, records 5 seconds of audio → `POST /api/verification/voice`
- Keystroke: Every 10 seconds when chat is open → `POST /api/verification/keystroke`
- Mouse: Every 10 seconds, passive movement tracking → `POST /api/verification/mouse`

**Trust Score System:**
- Aggregated from all 4 modalities (0-100%)
- Displayed prominently to both doctor and patient
- Low scores trigger visual alerts
- 15 minutes of continuous low trust → lockout with OTP + slider re-authentication
- 5 total failures → session terminated, patient emailed

**Chat System:**
- Real-time text messages via Socket.IO
- Unread message counter
- When chat opens → keystroke verification activates
- When chat closed → keystroke defaults to 50%

---

## Reusable Components

### PrivateRoute (`PrivateRoute.jsx`)
- Checks `user` and `userRole` from AuthContext
- Shows loading spinner during auth check
- Redirects unauthenticated → `/login`
- Redirects wrong-role → appropriate dashboard

### OTPVerification (`OTPVerification.jsx`)
- 6-digit input with auto-focus and paste support
- 60-second resend cooldown timer
- Auto-verifies when all 6 digits entered
- Calls `POST /api/otp/send`, `POST /api/otp/verify`, `POST /api/otp/resend`

### HumanVerificationPuzzle (`HumanVerificationPuzzle.jsx`)
- **Step 1**: Select medical-related emojis from shuffled grid
- **Step 2**: Drag slider to 95%+ to complete
- Shake animation on wrong selection
- Attempt counter

### ConsultationStartVerificationModal (`ConsultationStartVerificationModal.jsx`)
3-step verification before doctor can start consultation:

**Step 1 — Biometric Verification:**
- Face capture → `POST /api/verification/face`
- Voice recording (5 seconds) → `POST /api/verification/voice`
- Keystroke typing "MediConsult Secure Access" → `POST /api/verification/keystroke`
- Mouse movement tracking → `POST /api/verification/mouse`
- Requires 2 out of 4 to pass (threshold: 0.45 confidence)

**Step 2 — OTP Verification:**
- Sends OTP to doctor's email → `POST /api/otp/consultation/send`
- 6-digit code verification → `POST /api/otp/consultation/verify`

**Step 3 — Human Verification:**
- Slider drag to 100%
- On success → `onVerified()` callback starts consultation

### DoctorDetailsModal (`DoctorDetailsModal.jsx`)
- Modal with doctor profile: image, name, specialization, experience, description
- Used in patient dashboard when clicking "View Details"

### BookAppointmentModal (`BookAppointmentModal.jsx`)
- Form: reason, preferred time (radio), preferred dates (radio), notes
- `POST /api/appointments`

### ApproveAppointmentModal (`ApproveAppointmentModal.jsx`)
- Form: date picker, from time, to time, optional note
- `PUT /api/appointments/:id/approve`

### RejectAppointmentModal (`RejectAppointmentModal.jsx`)
- Form: rejection reason (optional)
- `PUT /api/appointments/:id/reject`

### MyAppointments (`MyAppointments.jsx`)
- Patient's appointments with tabs (All/Pending/Approved/Rejected)
- Cancel pending appointments

### DoctorAppointments (`DoctorAppointments.jsx`)
- Doctor's appointment requests with approve/reject actions

### ScheduledConsultations (`ScheduledConsultations.jsx`)
- Doctor's upcoming consultations with time status and start buttons

### ConfirmedConsultations (`ConfirmedConsultations.jsx`)
- Patient's joinable consultations with waiting alert option

### AdminDoctorModal & AdminAppointments
- Admin-specific management modals

---

## State Management (React Context API)

### AuthContext (`AuthContext.jsx`)
Central authentication state for the entire app:

**State:**
- `user` — Current user object (doctor/patient/admin data)
- `userRole` — Role string (doctor/patient/admin)
- `token` — JWT token
- `loading` — Auth initialization in progress

**Methods:**
- `login(email, password)` → `POST /api/auth/login` → stores token + role in localStorage → sets axios default header
- `register(formData)` → `POST /api/auth/register` (multipart, doctor with biometrics)
- `registerPatient(patientData)` → `POST /api/auth/register-patient`
- `logout()` → clears token, user, role from state and localStorage
- `fetchUser()` → `GET /api/doctors/me` or `GET /api/patients/me` based on role

**Persistence:** Token and role survive page refresh via localStorage.

---

## Biometric Capture Utilities (`biometricCapture.js`)

### KeystrokeCapture Class
- Records keydown/keyup events with timestamps
- Extracts **38-dimensional feature vector**:
  - 10 Hold Times (H) — how long each key is pressed
  - 10 Down-Down Times (DD) — time between consecutive key presses
  - 10 Up-Down Times (UD) — flight time between key release and next press
  - 1 Feature padding to reach 31
  - 7 Statistical Features: mean, std, median, min, max, Q25, Q75
- Used during registration (3 samples) and during consultation (continuous when chat open)

### MouseCapture Class
- Records mouse move events (timestamp, x, y, "Move")
- Records mouse click events (timestamp, x, y, button, "Pressed")
- Methods: `start()`, `stop()`, `getEvents()`, `handleMouseMove()`, `handleMouseClick()`
- Used during registration (15-second capture) and consultation (continuous passive tracking)

### VoiceCapture Class
- Records audio via `getUserMedia()` with echo cancellation + noise suppression
- Sample rate: 16kHz, mono channel
- Supports WebM/Opus or WAV formats
- `convertToWav()` method for format conversion
- Used during registration (3 × 5-10 second samples) and consultation (5-second samples every 20 seconds)

### FaceCapture Class
- Live camera stream via `getUserMedia()`
- Captures frames at 224×224 resolution
- `captureFrame()` draws video to canvas → converts to JPEG File object
- Methods: `startCamera()`, `stopCamera()`, `captureFrame()`, `getImages()`
- Used during registration (3 captures) and consultation (every 5 seconds)

---

## Frontend-Backend Communication Patterns

### API Calls (Axios)
- Base URL: `/api` (proxied to `http://localhost:5000` by Vite)
- All authenticated requests include `Authorization: Bearer <token>` header
- Multipart form data for file uploads (voice, face, profile image)
- JSON for all other requests

### Socket.IO (Real-time)
- Connects to `http://localhost:5000`
- Used only in Meeting page
- Handles: WebRTC signaling, chat messages, biometric score updates, lockout status

### WebRTC
- Peer-to-peer media streams (no server relay)
- STUN servers for NAT traversal
- Signaling via Socket.IO
