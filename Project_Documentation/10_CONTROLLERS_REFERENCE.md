# Controllers & Route Handlers Reference

## All Available Controllers

---

## 1. Auth Controller (`Backend/controllers/authController.js`)

### `register(req, res)` — Doctor Registration
**Route**: `POST /api/auth/register`  
**Auth**: None  
**Input**: Multipart form data with:
- `firstName`, `lastName`, `email`, `password`
- `medicalLicenseNumber`, `specialization`, `yearsOfExperience`
- `voiceSamples[]` (audio files, up to 5)
- `faceImages[]` (image files, up to 5)
- `keystrokeData` (JSON string of keystroke feature arrays)
- `mouseData` (JSON string of mouse event arrays)

**Logic**:
1. Check if email already registered (Doctor or Patient)
2. Create doctor document in MongoDB
3. Convert voice files to WAV via ffmpeg
4. Call 4 ML services for biometric enrollment (in parallel where possible)
5. Update doctor.biometricData with enrollment results
6. Generate JWT token
7. Return user data + biometric enrollment results

**Error Handling**: Biometric enrollment failures don't block registration — they're reported but the doctor is still created.

---

### `registerPatient(req, res)` — Patient Registration
**Route**: `POST /api/auth/register-patient`  
**Auth**: None  
**Input**: JSON `{ fullName, age, gender, email, password }`  
**Logic**: Create patient → generate JWT → return user data

---

### `login(req, res)` — Universal Login
**Route**: `POST /api/auth/login`  
**Auth**: None  
**Input**: JSON `{ email, password }`  
**Logic**:
1. Check admin credentials (admin@gmail.com / admin)
2. Check Doctor collection
3. Check Patient collection
4. Compare bcrypt password
5. Update lastLogin timestamp
6. Return JWT + user data + role

---

## 2. Doctor Controller (`Backend/controllers/doctorController.js`)

### `getMe(req, res)` — Get Current Doctor
**Route**: `GET /api/doctors/me`  
**Auth**: JWT (doctor)  
**Logic**: Return `req.doctor` from auth middleware

### `getAllDoctors(req, res)` — List Doctors
**Route**: `GET /api/doctors`  
**Auth**: JWT (any)  
**Logic**: Admin sees all doctors; patients see only `isActive: true`

### `getDoctorById(req, res)` — Get Doctor
**Route**: `GET /api/doctors/:id`  
**Auth**: JWT (any)

### `updateDoctor(req, res)` — Update Profile
**Route**: `PUT /api/doctors/:id`  
**Auth**: JWT (doctor/admin)  
**Input**: Multipart (name, specialization, experience, description, profileImage file)  
**Logic**: Upload new image to Cloudinary → delete old image → update doctor document

### `deleteDoctor(req, res)` — Delete Doctor
**Route**: `DELETE /api/doctors/:id`  
**Auth**: JWT (admin)

### `toggleDoctorActive(req, res)` — Toggle Active Status
**Route**: `PATCH /api/doctors/:id/toggle-active`  
**Auth**: JWT (admin)  
**Logic**: Flips `isActive` boolean

---

## 3. Patient Controller (`Backend/controllers/patientController.js`)

### `getMe(req, res)` — Get Current Patient
**Route**: `GET /api/patients/me`  
**Auth**: JWT (patient)

### `getAllPatients(req, res)` — List All Patients
**Route**: `GET /api/patients`  
**Auth**: JWT (admin)

### `getAllDoctors(req, res)` — Get Active Doctors (Patient View)
**Route**: `GET /api/patients/doctors`  
**Auth**: JWT (patient)  
**Logic**: Returns only `isActive: true` doctors

### `updatePatient(req, res)` — Update Profile
**Route**: `PUT /api/patients/:id`  
**Auth**: JWT (patient)  
**Input**: `{ fullName, age, gender }`

---

## 4. Appointment Controller (`Backend/controllers/appointmentController.js`)

### `createAppointment(req, res)` — Book Appointment
**Route**: `POST /api/appointments`  
**Auth**: JWT (patient)  
**Input**: `{ doctorId, reason, preferredTime, preferredDates, additionalNotes }`  
**Logic**:
1. Validate doctor exists and is active
2. Generate unique appointment number (APT-YYYYMMDD-XXXXX)
3. Create appointment with status "Pending"

### `getMyAppointments(req, res)` — Patient's Appointments
**Route**: `GET /api/appointments/my-appointments`  
**Auth**: JWT (patient)  
**Logic**: Find appointments by patientId, populate doctor info, sort by newest

### `getDoctorAppointments(req, res)` — Doctor's Requests
**Route**: `GET /api/appointments/doctor-appointments`  
**Auth**: JWT (doctor)  
**Logic**: Find pending appointments for this doctor, populate patient info

### `getAllAppointments(req, res)` — Admin View
**Route**: `GET /api/appointments/all`  
**Auth**: JWT (admin)  
**Logic**: Search by appointment number, patient name, doctor name. Filter by status and doctor.

### `getAppointmentById(req, res)` — Single Appointment
**Route**: `GET /api/appointments/:id`  
**Auth**: JWT (any — with ownership check)

### `approveAppointment(req, res)` — Doctor Approves
**Route**: `PUT /api/appointments/:id/approve`  
**Auth**: JWT (doctor)  
**Input**: `{ appointmentDate, appointmentTimeFrom, appointmentTimeTo, doctorNote }`  
**Logic**:
1. Verify appointment belongs to this doctor
2. Verify status is "Pending"
3. Set date, time, note
4. Status → "Approved"
5. Send approval email to patient

### `rejectAppointment(req, res)` — Doctor Rejects
**Route**: `PUT /api/appointments/:id/reject`  
**Auth**: JWT (doctor)  
**Input**: `{ doctorNote }`  
**Logic**: Set note → Status "Rejected" → Email to patient

### `cancelAppointment(req, res)` — Patient Cancels
**Route**: `DELETE /api/appointments/:id`  
**Auth**: JWT (patient)  
**Logic**: Only if status is "Pending" → delete appointment

---

## 5. Consultation Controller (`Backend/controllers/consultationController.js`)

### `getPatientConsultations(req, res)` — Patient's Consultations
**Route**: `GET /api/consultations/patient/my-consultations`  
**Auth**: JWT (patient)  
**Logic**:
1. Find approved appointments for this patient
2. Auto-create Consultation records if not exist
3. Check time validity (±1 hour of scheduled time)
4. Return status flags: `canJoin`, `canSendAlert`, `isMissed`, `isFuture`, `isWaiting`

### `getDoctorConsultations(req, res)` — Doctor's Consultations
**Route**: `GET /api/consultations/doctor/my-consultations`  
**Auth**: JWT (doctor)  
**Logic**: Same as patient but with flags: `canStart`, `isMissed`, `isFuture`, `isPatientWaiting`

### `sendWaitingAlert(req, res)` — Patient Alerts Doctor
**Route**: `POST /api/consultations/patient/:appointmentId/send-alert`  
**Auth**: JWT (patient)  
**Logic**:
1. Validate appointment is approved and within time window
2. Update consultation status → "PatientWaiting"
3. Send email to doctor: "Your patient is waiting"

### `startConsultation(req, res)` — Doctor Starts
**Route**: `POST /api/consultations/doctor/:appointmentId/start`  
**Auth**: JWT (doctor)  
**Logic**:
1. Validate time window (±1 hour)
2. Create/update consultation status → "Active"
3. Set `consultationRoomId` = `room-{appointmentId}`
4. Send email to patient: "Join now"

### `getConsultationStatus(req, res)` — Check Status
**Route**: `GET /api/consultations/:appointmentId/status`  
**Auth**: JWT (any)

### `endConsultation(req, res)` — Doctor Ends
**Route**: `PUT /api/consultations/:sessionId/end`  
**Auth**: JWT (doctor)  
**Logic**: Find by roomId → verify doctor ownership → status "Completed" → set endedAt

---

## 6. OTP Controller (`Backend/controllers/otpController.js`)

### `sendOTP(req, res)` — Send OTP
**Route**: `POST /api/otp/send`  
**Input**: `{ email, userType }`  
**Logic**: Validate userType → check email not registered → delete old OTPs → generate 6-digit → store with 10-min TTL → send email

### `verifyOTP(req, res)` — Verify OTP
**Route**: `POST /api/otp/verify`  
**Input**: `{ email, otp, userType }`  
**Logic**: Find matching OTP → confirm match → delete → return success

### `resendOTP(req, res)` — Resend OTP
**Route**: `POST /api/otp/resend`  
**Logic**: Same as sendOTP (deletes old, creates new)

### `sendConsultationOTP(req, res)` — Consultation OTP
**Route**: `POST /api/otp/consultation/send`  
**Auth**: JWT (doctor)  
**Logic**: Generate OTP for the authenticated doctor's email

### `verifyConsultationOTP(req, res)` — Verify Consultation OTP
**Route**: `POST /api/otp/consultation/verify`  
**Auth**: JWT (doctor)

### `resendConsultationOTP(req, res)` — Resend Consultation OTP
**Route**: `POST /api/otp/consultation/resend`  
**Auth**: JWT (doctor)

---

## 7. Session Controller (`Backend/controllers/sessionController.js`)

### `createSession(req, res)` — Create Biometric Session
**Route**: `POST /api/sessions`  
**Logic**: Generate UUID → create session with status "active"

### `getSession(req, res)` — Get Session
**Route**: `GET /api/sessions/:sessionId`

### `getDoctorSessions(req, res)` — Doctor's Session History
**Route**: `GET /api/sessions/doctor/:doctorId`  
**Logic**: Last 50 sessions, sorted by newest

### `updateSession(req, res)` — Update Session Status
**Route**: `PUT /api/sessions/:sessionId`  
**Input**: `{ status, endTime }`

### `addVerificationLog(req, res)` — Log Biometric Result
**Route**: `POST /api/sessions/:sessionId/verification`  
**Input**: `{ verificationType, verified, confidence, details }`  
**Logic**:
1. Push to verificationLogs array
2. Get last 10 logs
3. Calculate average confidence
4. Update overallTrustScore (0-100)
5. If average < 50%: generate alert, set status "suspicious"

---

## Summary: Controller Count

| Controller | Functions | Routes |
|------------|-----------|--------|
| Auth | 3 | 3 |
| Doctor | 5 | 6 |
| Patient | 4 | 4 |
| Appointment | 7 | 8 |
| Consultation | 6 | 6 |
| OTP | 6 | 6 |
| Session | 5 | 5 |
| **Total** | **36** | **38** |
