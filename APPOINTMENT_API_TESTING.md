# Appointment System API Testing Guide

## Prerequisites
1. Backend server running on `http://localhost:5000`
2. MongoDB connected
3. Valid JWT tokens for Patient, Doctor, and Admin users

---

## 🔑 Getting Authentication Tokens

### 1. Login as Patient:
```bash
POST http://localhost:5000/api/auth/login
Content-Type: application/json

{
  "email": "patient@example.com",
  "password": "password123"
}
```
**Response:** Copy the `token` from response

### 2. Login as Doctor:
```bash
POST http://localhost:5000/api/auth/login
Content-Type: application/json

{
  "email": "doctor@example.com",
  "password": "password123"
}
```
**Response:** Copy the `token` from response

### 3. Login as Admin:
```bash
POST http://localhost:5000/api/auth/login
Content-Type: application/json

{
  "email": "admin@gmail.com",
  "password": "admin"
}
```
**Response:** Copy the `token` from response

---

## 📋 Patient Endpoints Testing

### 1. Create Appointment Request
```bash
POST http://localhost:5000/api/appointments
Authorization: Bearer <PATIENT_TOKEN>
Content-Type: application/json

{
  "doctorId": "65f1234567890abcdef12345",
  "reason": "Regular health checkup and consultation for persistent headaches",
  "preferredTime": "Morning",
  "preferredDates": "Weekdays",
  "additionalNotes": "I prefer early morning slots between 8-10 AM"
}
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Appointment request created successfully",
  "data": {
    "appointment": {
      "appointmentNumber": "APT-20240315-12345",
      "status": "Pending",
      "patientId": {...},
      "doctorId": {...},
      ...
    }
  }
}
```

### 2. Get My Appointments (All)
```bash
GET http://localhost:5000/api/appointments/my-appointments
Authorization: Bearer <PATIENT_TOKEN>
```

### 3. Get My Appointments (Filtered by Status)
```bash
GET http://localhost:5000/api/appointments/my-appointments?status=Pending
Authorization: Bearer <PATIENT_TOKEN>
```

### 4. Get Single Appointment
```bash
GET http://localhost:5000/api/appointments/<APPOINTMENT_ID>
Authorization: Bearer <PATIENT_TOKEN>
```

### 5. Cancel Pending Appointment
```bash
DELETE http://localhost:5000/api/appointments/<APPOINTMENT_ID>
Authorization: Bearer <PATIENT_TOKEN>
```

---

## 👨‍⚕️ Doctor Endpoints Testing

### 1. Get Doctor's Appointments (Default: Pending)
```bash
GET http://localhost:5000/api/appointments/doctor-appointments
Authorization: Bearer <DOCTOR_TOKEN>
```

### 2. Get Doctor's Appointments (Filtered)
```bash
GET http://localhost:5000/api/appointments/doctor-appointments?status=Approved
Authorization: Bearer <DOCTOR_TOKEN>
```

### 3. Approve Appointment
```bash
PUT http://localhost:5000/api/appointments/<APPOINTMENT_ID>/approve
Authorization: Bearer <DOCTOR_TOKEN>
Content-Type: application/json

{
  "appointmentDate": "2024-03-20",
  "appointmentTimeFrom": "09:00 AM",
  "appointmentTimeTo": "09:30 AM",
  "doctorNote": "Please bring your previous medical records and arrive 10 minutes early"
}
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Appointment approved successfully",
  "data": {
    "appointment": {
      "status": "Approved",
      "appointmentDate": "2024-03-20T00:00:00.000Z",
      "appointmentTimeFrom": "09:00 AM",
      "appointmentTimeTo": "09:30 AM",
      ...
    }
  }
}
```
**Note:** Patient will receive approval email automatically

### 4. Reject Appointment
```bash
PUT http://localhost:5000/api/appointments/<APPOINTMENT_ID>/reject
Authorization: Bearer <DOCTOR_TOKEN>
Content-Type: application/json

{
  "doctorNote": "Unfortunately, I'm fully booked for the next two weeks. Please try booking with another specialist or contact us after March 30th."
}
```

**Note:** Patient will receive rejection email automatically

---

## 👑 Admin Endpoints Testing

### 1. Get All Appointments
```bash
GET http://localhost:5000/api/appointments/all
Authorization: Bearer <ADMIN_TOKEN>
```

### 2. Filter by Status
```bash
GET http://localhost:5000/api/appointments/all?status=Approved
Authorization: Bearer <ADMIN_TOKEN>
```

### 3. Filter by Doctor
```bash
GET http://localhost:5000/api/appointments/all?doctorId=65f1234567890abcdef12345
Authorization: Bearer <ADMIN_TOKEN>
```

### 4. Search Appointments
```bash
GET http://localhost:5000/api/appointments/all?search=APT-20240315
Authorization: Bearer <ADMIN_TOKEN>
```

### 5. Combined Filters
```bash
GET http://localhost:5000/api/appointments/all?status=Pending&search=headache
Authorization: Bearer <ADMIN_TOKEN>
```

---

## ✅ Testing Checklist

- [ ] Patient can create appointment (verify auto-generated appointment number)
- [ ] Patient can view all their appointments
- [ ] Patient can filter appointments by status
- [ ] Patient can cancel pending appointments
- [ ] Patient cannot cancel approved/rejected appointments
- [ ] Doctor sees pending appointments by default
- [ ] Doctor can filter by status (Pending/Approved/Rejected)
- [ ] Doctor can approve with date/time (all fields required)
- [ ] Doctor can reject with optional note
- [ ] Email sent on approval (check patient's email)
- [ ] Email sent on rejection (check patient's email)
- [ ] Admin can view all appointments
- [ ] Admin can filter by status
- [ ] Admin can filter by doctor
- [ ] Admin can search by various fields
- [ ] Authorization works (patient can't approve, doctor can't see other doctor's appointments)

---

## 🐛 Common Issues & Solutions

### Issue: "Not authorized, no token"
**Solution:** Make sure to include `Authorization: Bearer <TOKEN>` header

### Issue: "Doctor not found"
**Solution:** Use a valid doctor ID from your database. Get it from `/api/doctors` endpoint

### Issue: "Appointment not found"
**Solution:** Use a valid appointment ID from a created appointment

### Issue: "Not authorized to approve this appointment"
**Solution:** Make sure the doctor token matches the doctor who received the appointment

### Issue: Email not sent
**Solution:** Check email configuration in `.env` file (EMAIL_USER and EMAIL_PASS)

---

## 📧 Email Testing

After approving/rejecting appointments, check the patient's email inbox for:

**Approval Email:**
- Green gradient design
- Appointment number
- Doctor name
- Date and time slot
- Doctor's note (if provided)

**Rejection Email:**
- Red gradient design
- Appointment number
- Doctor name
- Reason for request
- Doctor's note (if provided)

---

## 🔄 Complete Test Flow

1. **Patient creates appointment** → Verify appointment number generated
2. **Doctor views pending appointments** → Verify appointment appears
3. **Doctor approves appointment** → Verify status changes to "Approved"
4. **Check patient email** → Verify approval email received
5. **Patient views appointments** → Verify approved appointment shows date/time
6. **Admin views all appointments** → Verify appointment appears with all details
7. **Create another appointment** → Doctor rejects it
8. **Check patient email** → Verify rejection email received
9. **Patient tries to cancel approved appointment** → Verify error message
10. **Patient cancels pending appointment** → Verify deletion successful

---

**Happy Testing! 🚀**

