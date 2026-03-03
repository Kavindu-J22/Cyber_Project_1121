# Appointment System - Quick Start Guide

## 🚀 What's Been Implemented

A complete appointment management system with:
- ✅ **Auto-generated appointment numbers** (APT-YYYYMMDD-XXXXX)
- ✅ **Patient request system** with auto-filled information
- ✅ **Doctor approval/rejection** workflow
- ✅ **Email notifications** (professional templates)
- ✅ **Admin monitoring** with search and filters
- ✅ **Role-based access control**

---

## 📂 New Files Created

```
Backend/
├── models/
│   └── Appointment.js              ✅ NEW - Appointment schema
├── controllers/
│   └── appointmentController.js    ✅ NEW - All appointment logic
├── routes/
│   └── appointmentRoutes.js        ✅ NEW - API routes
└── utils/
    └── emailService.js             ✅ UPDATED - Added email templates

Root/
├── APPOINTMENT_SYSTEM_IMPLEMENTATION.md  ✅ NEW - Full documentation
├── APPOINTMENT_API_TESTING.md            ✅ NEW - Testing guide
└── QUICK_START_GUIDE.md                  ✅ NEW - This file
```

---

## 🎯 Key Features by Role

### Patient Features:
1. **Create Appointment**
   - Auto-filled patient info (name, age, gender, email)
   - Specify reason for visit
   - Choose preferred time (Morning/Afternoon/Evening/Night)
   - Choose preferred dates (Weekdays/Weekends/Any)
   - Add additional notes
   - Get auto-generated appointment number

2. **View Appointments**
   - See all appointments with status
   - Filter by: Pending, Approved, Rejected
   - View appointment details

3. **Manage Appointments**
   - Cancel pending appointments
   - Receive email notifications

### Doctor Features:
1. **View Appointments**
   - Default view: Pending appointments
   - Filter by: Pending, Approved, Rejected
   - See patient details

2. **Approve Appointments**
   - Select appointment date
   - Set time slot (from - to)
   - Add optional note
   - Patient receives email automatically

3. **Reject Appointments**
   - Add optional note explaining reason
   - Patient receives email automatically

### Admin Features:
1. **Monitor All Appointments**
   - View all appointments system-wide
   - Filter by status
   - Filter by doctor
   - Search by:
     - Appointment number
     - Patient name
     - Doctor name
     - Reason

---

## 🔌 API Endpoints Summary

### Patient:
```
POST   /api/appointments                    Create appointment
GET    /api/appointments/my-appointments    Get my appointments
DELETE /api/appointments/:id                Cancel appointment
```

### Doctor:
```
GET    /api/appointments/doctor-appointments  Get my appointments
PUT    /api/appointments/:id/approve          Approve appointment
PUT    /api/appointments/:id/reject           Reject appointment
```

### Admin:
```
GET    /api/appointments/all                  Get all appointments
```

### Common:
```
GET    /api/appointments/:id                  Get single appointment
```

---

## 📊 Database Schema Quick Reference

```javascript
Appointment {
  appointmentNumber: "APT-20240315-12345"  // Auto-generated
  patientId: ObjectId                       // Reference to Patient
  doctorId: ObjectId                        // Reference to Doctor
  reason: String                            // Required, max 500 chars
  preferredTime: "Morning" | "Afternoon" | "Evening" | "Night"
  preferredDates: "Weekdays" | "Weekends" | "Any"
  additionalNotes: String                   // Optional, max 1000 chars
  status: "Pending" | "Approved" | "Rejected"
  
  // Filled when approved:
  appointmentDate: Date
  appointmentTimeFrom: "09:00 AM"
  appointmentTimeTo: "09:30 AM"
  
  doctorNote: String                        // Optional, max 1000 chars
  
  createdAt: Date
  updatedAt: Date
  respondedAt: Date
}
```

---

## 🧪 Quick Test

### 1. Start Backend Server:
```bash
cd Backend
npm start
```

### 2. Test Patient Creates Appointment:
```bash
# Login as patient first to get token
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"patient@example.com","password":"password123"}'

# Create appointment (replace <TOKEN> and <DOCTOR_ID>)
curl -X POST http://localhost:5000/api/appointments \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "doctorId": "<DOCTOR_ID>",
    "reason": "Regular checkup",
    "preferredTime": "Morning",
    "preferredDates": "Weekdays"
  }'
```

### 3. Test Doctor Views Appointments:
```bash
# Login as doctor
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"doctor@example.com","password":"password123"}'

# Get pending appointments
curl -X GET http://localhost:5000/api/appointments/doctor-appointments \
  -H "Authorization: Bearer <DOCTOR_TOKEN>"
```

---

## 📧 Email Templates

### Approval Email Features:
- ✅ Green gradient design
- ✅ Appointment number badge
- ✅ Doctor name, date, time
- ✅ Optional doctor's note
- ✅ Professional branding

### Rejection Email Features:
- ✅ Red gradient design
- ✅ Appointment number badge
- ✅ Doctor name, reason
- ✅ Optional doctor's note
- ✅ Encouragement to reapply

---

## 🔐 Security Features

- ✅ JWT authentication required for all endpoints
- ✅ Role-based authorization (Patient/Doctor/Admin)
- ✅ Patients can only access their own appointments
- ✅ Doctors can only manage their own appointments
- ✅ Admins have full system access
- ✅ Status validation (can't re-process appointments)
- ✅ Input validation and sanitization

---

## 📝 Next Steps for Frontend

You'll need to create these components:

### Patient Dashboard:
1. **Book Appointment Form**
   - Doctor selection dropdown
   - Reason textarea
   - Preferred time radio buttons
   - Preferred dates radio buttons
   - Additional notes textarea

2. **My Appointments Page**
   - Status filter tabs (All/Pending/Approved/Rejected)
   - Appointment cards with details
   - Cancel button for pending appointments

### Doctor Dashboard:
1. **Appointments Tab**
   - Status filter (Pending/Approved/Rejected)
   - Appointment list with patient details
   - Approve/Reject buttons

2. **Approval Modal**
   - Date picker
   - Time from/to inputs
   - Doctor note textarea

3. **Rejection Modal**
   - Doctor note textarea

### Admin Dashboard:
1. **All Appointments View**
   - Search bar
   - Status filter dropdown
   - Doctor filter dropdown
   - Appointments table with all details

---

## 🎨 Suggested UI/UX

### Status Badges:
- **Pending**: Yellow/Orange badge
- **Approved**: Green badge
- **Rejected**: Red badge

### Appointment Cards:
- Show appointment number prominently
- Display doctor/patient info with avatar
- Show date/time if approved
- Color-coded by status

### Filters:
- Tabs for status filtering
- Dropdown for doctor selection (admin)
- Search bar with placeholder text

---

## ✅ Testing Checklist

Before deploying:
- [ ] Test patient appointment creation
- [ ] Verify appointment number generation
- [ ] Test doctor approval flow
- [ ] Test doctor rejection flow
- [ ] Verify email notifications work
- [ ] Test patient viewing appointments
- [ ] Test patient canceling appointments
- [ ] Test admin viewing all appointments
- [ ] Test search and filter functionality
- [ ] Verify authorization rules

---

## 🆘 Troubleshooting

**Server won't start:**
- Check MongoDB is running
- Verify .env file exists with correct values

**Emails not sending:**
- Check EMAIL_USER and EMAIL_PASS in .env
- Verify Gmail app password is correct

**Authorization errors:**
- Ensure JWT_SECRET is set in .env
- Check token is included in Authorization header

**Appointment not found:**
- Verify appointment ID is correct
- Check user has permission to access

---

## 📞 Support

For detailed information, refer to:
- `APPOINTMENT_SYSTEM_IMPLEMENTATION.md` - Full documentation
- `APPOINTMENT_API_TESTING.md` - API testing guide

---

**Status:** ✅ Backend Complete - Ready for Frontend Integration
**Version:** 1.0.0
**Date:** March 3, 2026

