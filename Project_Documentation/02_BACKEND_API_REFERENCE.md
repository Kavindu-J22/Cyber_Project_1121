# Backend Architecture & API Reference

## Server Configuration

- **Framework**: Express.js (Node.js with ES6 modules)
- **Port**: 5000
- **Database**: MongoDB Atlas (Cloud)
- **Real-time**: Socket.IO for WebRTC signaling and live chat
- **File**: `Backend/server.js`

### Security Middleware Stack
| Middleware | Purpose |
|-----------|---------|
| Helmet | Sets secure HTTP headers (XSS protection, clickjacking prevention) |
| CORS | Restricts API access to `http://localhost:5173` (frontend) |
| Morgan | Logs all HTTP requests for monitoring |
| express.json | Parses JSON request bodies (limit: 50mb) |
| express.urlencoded | Parses URL-encoded form data (limit: 50mb) |

---

## Complete API Endpoints

### Authentication (`/api/auth`)

| Method | Endpoint | Description | Auth Required | Role |
|--------|----------|-------------|---------------|------|
| POST | `/api/auth/register` | Register doctor with biometric enrollment | No | — |
| POST | `/api/auth/register-patient` | Register patient | No | — |
| POST | `/api/auth/login` | Login (doctor/patient/admin) | No | — |

**Register Doctor Flow:**
1. Creates doctor record in MongoDB
2. Converts uploaded voice samples to WAV (via ffmpeg)
3. Calls `mlService.enrollVoiceMultiple()` → Voice API (port 8001)
4. Calls `mlService.enrollKeystroke()` → Keystroke API (port 8002)
5. Calls `mlService.enrollMouse()` → Mouse API (port 8003)
6. Calls `mlService.enrollFace()` → Face API (port 8004)
7. Updates doctor's `biometricData` field with enrollment status
8. Returns JWT token

**Login Flow:**
1. Accepts email + password
2. Checks Doctor collection, then Patient collection, then admin credentials
3. Compares bcrypt-hashed password
4. Returns JWT token with role (doctor/patient/admin) and user data

---

### Doctor Management (`/api/doctors`)

| Method | Endpoint | Description | Auth Required | Role |
|--------|----------|-------------|---------------|------|
| GET | `/api/doctors/me` | Get current doctor's profile | Yes | doctor |
| GET | `/api/doctors` | Get all doctors | Yes | any |
| GET | `/api/doctors/:id` | Get doctor by ID | Yes | any |
| PUT | `/api/doctors/:id` | Update doctor profile (with image upload to Cloudinary) | Yes | doctor |
| DELETE | `/api/doctors/:id` | Delete doctor | Yes | admin |
| PATCH | `/api/doctors/:id/toggle-active` | Activate/deactivate doctor | Yes | admin |

---

### Patient Management (`/api/patients`)

| Method | Endpoint | Description | Auth Required | Role |
|--------|----------|-------------|---------------|------|
| GET | `/api/patients/me` | Get current patient's profile | Yes | patient |
| GET | `/api/patients` | Get all patients | Yes | admin |
| GET | `/api/patients/doctors` | Get all active doctors | Yes | patient |
| PUT | `/api/patients/:id` | Update patient profile | Yes | patient |

---

### OTP (`/api/otp`)

| Method | Endpoint | Description | Auth Required | Role |
|--------|----------|-------------|---------------|------|
| POST | `/api/otp/send` | Send 6-digit OTP to email | No | — |
| POST | `/api/otp/verify` | Verify OTP code | No | — |
| POST | `/api/otp/resend` | Resend OTP | No | — |
| POST | `/api/otp/consultation/send` | Send consultation OTP to doctor | Yes | doctor |
| POST | `/api/otp/consultation/verify` | Verify consultation OTP | Yes | doctor |
| POST | `/api/otp/consultation/resend` | Resend consultation OTP | Yes | doctor |

**OTP Logic:** 6-digit random code → stored in MongoDB with 10-minute TTL → auto-deleted after expiry.

---

### Appointments (`/api/appointments`)

| Method | Endpoint | Description | Auth Required | Role |
|--------|----------|-------------|---------------|------|
| POST | `/api/appointments` | Create appointment request | Yes | patient |
| GET | `/api/appointments/my-appointments` | Get patient's appointments | Yes | patient |
| GET | `/api/appointments/doctor-appointments` | Get doctor's pending appointments | Yes | doctor |
| GET | `/api/appointments/all` | Get all appointments (with search/filter) | Yes | admin |
| GET | `/api/appointments/:id` | Get appointment by ID | Yes | any |
| PUT | `/api/appointments/:id/approve` | Approve + schedule appointment | Yes | doctor |
| PUT | `/api/appointments/:id/reject` | Reject appointment | Yes | doctor |
| DELETE | `/api/appointments/:id` | Cancel pending appointment | Yes | patient |

**Appointment Lifecycle:**
```
Patient creates request → Status: "Pending"
  ↓
Doctor approves (sets date/time) → Status: "Approved" → Email sent to patient
  OR
Doctor rejects (with reason) → Status: "Rejected" → Email sent to patient
  ↓
Patient can cancel pending appointments only
```

---

### Consultations (`/api/consultations`)

| Method | Endpoint | Description | Auth Required | Role |
|--------|----------|-------------|---------------|------|
| GET | `/api/consultations/patient/my-consultations` | Get patient's confirmed consultations | Yes | patient |
| POST | `/api/consultations/patient/:appointmentId/send-alert` | Patient sends waiting alert to doctor | Yes | patient |
| GET | `/api/consultations/doctor/my-consultations` | Get doctor's scheduled consultations | Yes | doctor |
| POST | `/api/consultations/doctor/:appointmentId/start` | Doctor starts consultation | Yes | doctor |
| PUT | `/api/consultations/:sessionId/end` | End consultation | Yes | doctor |
| GET | `/api/consultations/:appointmentId/status` | Get consultation status | Yes | any |

**Consultation Lifecycle:**
```
Approved appointment → Consultation status: "Scheduled"
  ↓
Patient joins and sends alert → Status: "PatientWaiting" → Email to doctor
  ↓
Doctor passes verification (biometric + OTP + human) → Status: "Active"
  → Room ID: room-{appointmentId}
  → Email sent to patient: "Join now"
  ↓
WebRTC video/audio established → Continuous biometric verification begins
  ↓
Doctor ends consultation → Status: "Completed"
```

**Time Window:** Consultations can only be started within ±1 hour of the scheduled time.

---

### Biometric Verification (`/api/verification`)

| Method | Endpoint | Description | Auth Required | Role |
|--------|----------|-------------|---------------|------|
| POST | `/api/verification/voice` | Verify voice sample | Yes | doctor |
| POST | `/api/verification/keystroke` | Verify keystroke pattern | Yes | doctor |
| POST | `/api/verification/mouse` | Verify mouse movement | Yes | doctor |
| POST | `/api/verification/face` | Verify face image | Yes | doctor |
| GET | `/api/verification/health` | Check all ML services status | Yes | any |

---

### Sessions (`/api/sessions`) — Biometric Session Tracking

| Method | Endpoint | Description | Auth Required | Role |
|--------|----------|-------------|---------------|------|
| POST | `/api/sessions` | Create verification session | Yes | any |
| GET | `/api/sessions/:sessionId` | Get session details | Yes | any |
| GET | `/api/sessions/doctor/:doctorId` | Get doctor's sessions (last 50) | Yes | any |
| PUT | `/api/sessions/:sessionId` | Update session status | Yes | any |
| POST | `/api/sessions/:sessionId/verification` | Add verification log entry | Yes | any |

---

## Database Models (MongoDB/Mongoose)

### Doctor Schema
```
firstName, lastName, email (unique), password (bcrypt hashed)
medicalLicenseNumber (unique), specialization, yearsOfExperience
profileImage (Cloudinary URL), description (max 1000 chars)
biometricData: {
  voiceEnrolled (bool), voiceEmbedding (reference ID)
  keystrokeEnrolled (bool), keystrokeProfile (reference ID)
  mouseEnrolled (bool), mouseProfile (reference ID)
  faceEnrolled (bool), faceProfile (reference ID)
}
isActive (default: true), createdAt, lastLogin
```

### Patient Schema
```
fullName, age (0-150), gender (Male/Female/Other)
email (unique), password (bcrypt hashed)
isActive (default: true), createdAt, lastLogin
```

### Appointment Schema
```
appointmentNumber (unique: APT-YYYYMMDD-XXXXX)
patientId (ref), doctorId (ref)
reason (max 500 chars), preferredTime (Morning/Afternoon/Evening/Night)
preferredDates (Weekdays/Weekends/Any), additionalNotes (max 1000 chars)
status (Pending/Approved/Rejected)
appointmentDate, appointmentTimeFrom, appointmentTimeTo
doctorNote (max 1000 chars)
createdAt, updatedAt, respondedAt
```

### Consultation Schema
```
appointmentId (ref, unique), patientId (ref), doctorId (ref)
status (Scheduled/PatientWaiting/Active/Completed/Missed)
scheduledDate, scheduledTimeFrom, scheduledTimeTo
doctorStartedAt, patientJoinedAt, endedAt, waitingAlertSentAt
consultationRoomId (room-{appointmentId})
createdAt, updatedAt
```

### Session Schema (Biometric Verification Tracking)
```
sessionId (UUID), doctorId (ref), patientId
startTime, endTime
status (active/completed/terminated/suspicious)
verificationLogs: [{ timestamp, verificationType, verified, confidence, details }]
alerts: [{ timestamp, type, severity, message, details }]
overallTrustScore (0-100, default: 100)
```

### OTP Schema
```
email, otp (6-digit string)
userType (doctor/patient/consultation)
createdAt (TTL: auto-delete after 600 seconds)
```

---

## Middleware

### Authentication Middleware (`middleware/auth.js`)
- **`protect()`** — Extracts Bearer token from Authorization header, verifies JWT, populates `req.user`, `req.doctor`, `req.patient`, `req.userRole`
- **`restrictTo(...roles)`** — Role-based gating (returns 403 if role mismatch)
- **`generateToken(id, role)`** — Creates JWT with 7-day expiry
- **Admin handling**: Hardcoded admin credentials (admin@gmail.com)

### File Upload Middleware (`middleware/upload.js`)
- Uses **Multer** with disk storage
- Destination: `Backend/uploads/` (auto-created)
- File naming: `{fieldname}-{timestamp}-{random}.{ext}`
- Max file size: 10MB
- Voice filter: Only `audio/*` MIME types
- Face filter: All files accepted

---

## Services

### ML Service Gateway (`services/mlService.js`)
Centralized HTTP client that routes biometric requests to Python ML microservices:

| Service | URL | Enroll Endpoint | Verify Endpoint |
|---------|-----|----------------|-----------------|
| Voice | `http://localhost:8001` | `POST /api/v1/enroll` | `POST /api/v1/verify/upload` |
| Keystroke | `http://localhost:8002` | `POST /keystroke/enroll` | `POST /keystroke/verify` |
| Mouse | `http://localhost:8003` | `POST /mouse/enroll` | `POST /mouse/verify` |
| Face | `http://localhost:8004` | `POST /api/v1/enroll` | `POST /api/v1/verify` |

### Email Service (`utils/emailService.js`)
- **SMTP**: Gmail with app password
- **Functions**: `sendOTPEmail()`, `sendAppointmentApprovalEmail()`, `sendAppointmentRejectionEmail()`, `sendConsultationStartedEmail()`, `sendWaitingAlertEmail()`, `sendLockoutTerminatedEmail()`
- All emails use branded HTML templates with color-coded themes

### Audio Converter (`utils/audioConverter.js`)
- Converts uploaded audio to WAV format using **ffmpeg**
- Specs: 16kHz, mono, 16-bit PCM
- Auto-deletes original file after conversion

---

## Socket.IO Real-Time Events

### WebRTC Signaling
| Event | Direction | Purpose |
|-------|-----------|---------|
| `join-session` | Client → Server | User joins consultation room |
| `user-joined` | Server → Client | Notify other participant joined |
| `offer` | Client → Client (via server) | WebRTC SDP offer |
| `answer` | Client → Client (via server) | WebRTC SDP answer |
| `ice-candidate` | Client → Client (via server) | NAT traversal candidate |
| `end-session` | Client → All in room | Doctor ends consultation |
| `session-ended` | Server → All in room | Session termination broadcast |

### Biometric Verification
| Event | Direction | Purpose |
|-------|-----------|---------|
| `verify-biometric` | Client → Server | Submit biometric for verification |
| `verification-result` | Server → Client | Return confidence + trust score |
| `verification-alert` | Server → Client | Low-confidence warning |
| `doctor-biometric-update` | Server → Patient | Doctor's live scores for display |
| `doctor-lockout-status` | Server → Patient | Doctor lockout notification |
| `lockout-max-attempts` | Server → All | 5 failures → terminate + email patient |

### Chat
| Event | Direction | Purpose |
|-------|-----------|---------|
| `chat-message` | Client → All in room | Real-time text message with timestamp |
