# Database Schema & Models Reference

> Complete MongoDB/Mongoose schema documentation for all 6 models.

---

## Entity Relationship Diagram

```
┌──────────────────┐          ┌──────────────────┐
│     DOCTOR       │          │     PATIENT      │
├──────────────────┤          ├──────────────────┤
│ _id (ObjectId)   │◄─────┐  │ _id (ObjectId)   │◄─────┐
│ firstName        │      │  │ fullName          │      │
│ lastName         │      │  │ age               │      │
│ email (unique)   │      │  │ gender            │      │
│ password (hash)  │      │  │ email (unique)    │      │
│ medicalLicenseNo │      │  │ password (hash)   │      │
│ specialization   │      │  │ isActive          │      │
│ yearsOfExp       │      │  │ createdAt         │      │
│ profileImage     │      │  │ lastLogin         │      │
│ description      │      │  └──────────────────┘      │
│ isActive         │      │                             │
│ biometricData{}  │      │                             │
│ createdAt        │      │                             │
│ lastLogin        │      │                             │
└──────────────────┘      │                             │
         ▲                │                             │
         │           ┌────┴─────────────────────────────┴──┐
         │           │         APPOINTMENT                  │
         │           ├──────────────────────────────────────┤
         │           │ _id (ObjectId)                       │
         │           │ appointmentNumber (unique, auto)     │
         └───────────│ doctorId → Doctor                    │
                     │ patientId → Patient ─────────────────┘
                     │ reason                               │
                     │ preferredTime (Morning/Afternoon/…)  │
                     │ preferredDates (Weekdays/Weekends/…) │
                     │ status (Pending/Approved/Rejected)   │
                     │ appointmentDate                      │
                     │ appointmentTimeFrom / TimeTo         │
                     │ doctorNote                           │
                     │ createdAt, updatedAt, respondedAt    │
                     └───────────┬──────────────────────────┘
                                 │ 1:1 (unique)
                                 ▼
                     ┌──────────────────────────────────────┐
                     │         CONSULTATION                  │
                     ├──────────────────────────────────────┤
                     │ _id (ObjectId)                       │
                     │ appointmentId → Appointment (unique)  │
                     │ doctorId → Doctor                    │
                     │ patientId → Patient                  │
                     │ status (Scheduled/Active/Completed…) │
                     │ scheduledDate, TimeFrom, TimeTo      │
                     │ doctorStartedAt, patientJoinedAt     │
                     │ endedAt                              │
                     │ consultationRoomId                   │
                     │ createdAt, updatedAt                 │
                     └──────────────────────────────────────┘

┌──────────────────────────────────────┐
│           SESSION                     │
├──────────────────────────────────────┤
│ _id (ObjectId)                       │
│ sessionId (unique)                   │
│ doctorId → Doctor                    │
│ patientId (String, default demo)     │
│ startTime, endTime                   │
│ status (active/completed/terminated) │
│ overallTrustScore (0-100)            │
│ metadata {}                          │
│                                      │
│ verificationLogs[] ──────────────┐   │
│ alerts[] ────────────────────┐   │   │
└──────────────────────────────┤───┤───┘
                               │   │
            ┌──────────────────┘   └──────────────────┐
            ▼                                          ▼
┌────────────────────┐              ┌──────────────────────┐
│ alert              │              │ verificationLog       │
├────────────────────┤              ├──────────────────────┤
│ timestamp          │              │ timestamp             │
│ type               │              │ verificationType      │
│ severity           │              │   (voice/keystroke/   │
│ message            │              │    mouse/combined)    │
│ details {}         │              │ verified (Boolean)    │
└────────────────────┘              │ confidence (0-1)      │
                                    │ details {}            │
                                    └──────────────────────┘

┌──────────────────────────────────────┐
│           OTP                         │
├──────────────────────────────────────┤
│ _id (ObjectId)                       │
│ email                                │
│ otp                                  │
│ userType (doctor/patient/consultation)│
│ createdAt (TTL: 10 min auto-delete)  │
└──────────────────────────────────────┘
```

---

## Model Details

### 1. Doctor

**Collection:** `doctors`

| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `firstName` | String | Yes | — | `trim` |
| `lastName` | String | Yes | — | `trim` |
| `email` | String | Yes | — | `unique`, `lowercase`, `trim`, regex validated |
| `password` | String | Yes | — | `minlength: 6`, `select: false` (excluded from queries) |
| `medicalLicenseNumber` | String | Yes | — | `unique`, `trim` |
| `specialization` | String | Yes | — | `trim` |
| `yearsOfExperience` | Number | Yes | — | `min: 0` |
| `profileImage` | String | No | `null` | Cloudinary URL |
| `description` | String | No | `null` | `maxlength: 1000` |
| `isActive` | Boolean | No | `true` | — |
| `createdAt` | Date | No | `Date.now` | — |
| `lastLogin` | Date | No | `null` | — |

**Nested Object: `biometricData`**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `voiceEnrolled` | Boolean | `false` | Voice biometric registered? |
| `voiceEmbedding` | String | `null` | Reference/status string |
| `keystrokeEnrolled` | Boolean | `false` | Keystroke profile registered? |
| `keystrokeProfile` | String | `null` | Reference/status string |
| `mouseEnrolled` | Boolean | `false` | Mouse profile registered? |
| `mouseProfile` | String | `null` | Reference/status string |
| `faceEnrolled` | Boolean | `false` | Face biometric registered? |
| `faceProfile` | String | `null` | Reference/status string |

**Hooks:**
- `pre('save')` — If `password` is modified, hash with bcrypt (10 salt rounds)

**Methods:**
- `comparePassword(entered)` — Returns `bcrypt.compare(entered, this.password)`

---

### 2. Patient

**Collection:** `patients`

| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `fullName` | String | Yes | — | `trim` |
| `age` | Number | Yes | — | `min: 0`, `max: 150` |
| `gender` | String | Yes | — | `enum: ['Male', 'Female', 'Other']`, `trim` |
| `email` | String | Yes | — | `unique`, `lowercase`, `trim`, regex validated |
| `password` | String | Yes | — | `minlength: 6`, `select: false` |
| `isActive` | Boolean | No | `true` | — |
| `createdAt` | Date | No | `Date.now` | — |
| `lastLogin` | Date | No | `null` | — |

**Hooks:**
- `pre('save')` — Same bcrypt hashing as Doctor

**Methods:**
- `comparePassword(entered)` — Same as Doctor

**Note:** Patients do NOT have biometric data — only doctors are biometrically enrolled and verified.

---

### 3. Appointment

**Collection:** `appointments`

| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `appointmentNumber` | String | No | Auto-generated | `unique` |
| `patientId` | ObjectId | Yes | — | `ref: 'Patient'` |
| `doctorId` | ObjectId | Yes | — | `ref: 'Doctor'` |
| `reason` | String | Yes | — | `trim`, `maxlength: 500` |
| `preferredTime` | String | Yes | — | `enum: ['Morning', 'Afternoon', 'Evening', 'Night']` |
| `preferredDates` | String | Yes | — | `enum: ['Weekdays', 'Weekends', 'Any']` |
| `additionalNotes` | String | No | `''` | `trim`, `maxlength: 1000` |
| `status` | String | No | `'Pending'` | `enum: ['Pending', 'Approved', 'Rejected']` |
| `appointmentDate` | Date | No | `null` | Set when approved |
| `appointmentTimeFrom` | String | No | `null` | Set when approved |
| `appointmentTimeTo` | String | No | `null` | Set when approved |
| `doctorNote` | String | No | `''` | `trim`, `maxlength: 1000` |
| `createdAt` | Date | No | `Date.now` | — |
| `updatedAt` | Date | No | `Date.now` | — |
| `respondedAt` | Date | No | `null` | Set when doctor responds |

**Auto-Generated Number:** Format `APT-YYYYMMDD-XXXXX` (e.g., `APT-20260308-47293`)

**Indexes:**
- `{ patientId: 1, createdAt: -1 }` — Patient's appointment history
- `{ doctorId: 1, status: 1, createdAt: -1 }` — Doctor's pending/approved/rejected
- `{ appointmentNumber: 1 }` — Lookup by number
- `{ status: 1 }` — Filter by status

**Lifecycle:**
```
Patient creates → status: "Pending"
Doctor approves → status: "Approved", appointmentDate/Time filled
Doctor rejects → status: "Rejected", doctorNote filled
```

---

### 4. Consultation

**Collection:** `consultations`

| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `appointmentId` | ObjectId | Yes | — | `ref: 'Appointment'`, `unique` |
| `patientId` | ObjectId | Yes | — | `ref: 'Patient'` |
| `doctorId` | ObjectId | Yes | — | `ref: 'Doctor'` |
| `status` | String | No | `'Scheduled'` | `enum` (see below) |
| `scheduledDate` | Date | Yes | — | — |
| `scheduledTimeFrom` | String | Yes | — | — |
| `scheduledTimeTo` | String | Yes | — | — |
| `doctorStartedAt` | Date | No | `null` | When doctor opens room |
| `patientJoinedAt` | Date | No | `null` | When patient joins |
| `endedAt` | Date | No | `null` | When consultation ends |
| `waitingAlertSentAt` | Date | No | `null` | When "doctor waiting" email sent |
| `consultationRoomId` | String | No | `null` | WebRTC room identifier |
| `createdAt` | Date | No | `Date.now` | — |
| `updatedAt` | Date | No | `Date.now` | — |

**Status Enum:**
```
Scheduled → PatientWaiting → Active → Completed
                                   ↘ Missed
```

**Indexes:**
- `{ appointmentId: 1 }` — Link to originating appointment
- `{ patientId: 1, status: 1 }` — Patient's consultations by status
- `{ doctorId: 1, status: 1 }` — Doctor's consultations by status
- `{ scheduledDate: 1, status: 1 }` — Date-based queries

**Key Constraint:** One consultation per appointment (`appointmentId` is `unique`).

---

### 5. Session (Biometric Monitoring)

**Collection:** `sessions`

| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `sessionId` | String | Yes | — | `unique` |
| `doctorId` | ObjectId | Yes | — | `ref: 'Doctor'` |
| `patientId` | String | No | `'demo-patient'` | Not a ref — plain string |
| `startTime` | Date | No | `Date.now` | — |
| `endTime` | Date | No | `null` | — |
| `status` | String | No | `'active'` | `enum: ['active', 'completed', 'terminated', 'suspicious']` |
| `verificationLogs` | [Sub-doc] | No | `[]` | Array of embedded verification records |
| `alerts` | [Object] | No | `[]` | Array of alert objects |
| `overallTrustScore` | Number | No | `100` | `min: 0`, `max: 100` |
| `metadata` | Mixed | No | `{}` | Flexible data storage |

**Embedded: `verificationLogs[]`**

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `timestamp` | Date | No | Default: `Date.now` |
| `verificationType` | String | Yes | `enum: ['voice', 'keystroke', 'mouse', 'combined']` |
| `verified` | Boolean | Yes | — |
| `confidence` | Number | No | `min: 0`, `max: 1` |
| `details` | Mixed | No | Any JSON |

**Embedded: `alerts[]`**

| Field | Type | Constraints |
|-------|------|-------------|
| `timestamp` | Date | — |
| `type` | String | — |
| `severity` | String | `enum: ['low', 'medium', 'high', 'critical']` |
| `message` | String | — |
| `details` | Mixed | Any JSON |

**Indexes:**
- `{ doctorId: 1, startTime: -1 }` — Doctor's sessions ordered by time
- `{ sessionId: 1 }` — Lookup by session ID

**Note:** `patientId` is stored as a plain String, not an ObjectId reference. This appears to be a design decision for session simplicity.

---

### 6. OTP (One-Time Password)

**Collection:** `otps`

| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `email` | String | Yes | — | `lowercase`, `trim` |
| `otp` | String | Yes | — | — |
| `userType` | String | Yes | — | `enum: ['doctor', 'patient', 'consultation']` |
| `createdAt` | Date | No | `Date.now` | **`expires: 600`** (TTL) |

**Indexes:**
- `{ email: 1, userType: 1 }` — Lookup OTP by email and type
- TTL index on `createdAt` — Documents auto-delete after 10 minutes

**TTL Behavior:** MongoDB automatically removes OTP documents 600 seconds (10 minutes) after `createdAt`. No manual cleanup needed.

---

## Relationships Summary

| From | To | Cardinality | Join Field |
|------|----|-------------|------------|
| Appointment | Doctor | N:1 | `doctorId` |
| Appointment | Patient | N:1 | `patientId` |
| Consultation | Appointment | 1:1 | `appointmentId` (unique) |
| Consultation | Doctor | N:1 | `doctorId` |
| Consultation | Patient | N:1 | `patientId` |
| Session | Doctor | N:1 | `doctorId` |

**Not Referenced:**
- Patient has no biometric data (only Doctor does)
- Session.patientId is a plain String, not a Mongoose ref
- OTP is standalone (no refs to any model)

---

## Data Flow Through Models

```
1. Doctor registers    → Doctor document created (biometricData populated)
2. Patient registers   → Patient document created
3. Patient books       → Appointment created (status: Pending)
4. Doctor approves     → Appointment updated (status: Approved, date/time set)
                       → Consultation created (status: Scheduled)
5. Doctor opens room   → Session created (overallTrustScore: 100)
                       → Consultation updated (status: PatientWaiting)
6. Biometric checks    → Session.verificationLogs[] grows
                       → Session.overallTrustScore recalculated
7. Alert triggered     → Session.alerts[] grows, Session.status → suspicious
8. Consultation ends   → Session.endTime set, status → completed
                       → Consultation.endedAt set, status → Completed
9. Lockout OTP sent    → OTP document created (auto-expires in 10 min)
```
