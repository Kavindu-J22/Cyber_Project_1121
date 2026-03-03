# Appointment Management System - Implementation Guide

## Overview
A comprehensive appointment management system has been implemented for the MediConsult Zero Trust Telehealth Platform. This system allows patients to request appointments, doctors to approve/reject them, and admins to monitor all appointments.

---

## 🎯 Features Implemented

### For Patients:
- ✅ Create appointment requests with auto-filled patient information
- ✅ Specify reason for appointment
- ✅ Select preferred time (Morning, Afternoon, Evening, Night)
- ✅ Select preferred dates (Weekdays, Weekends, Any)
- ✅ Add additional notes
- ✅ Auto-generated appointment number (Format: APT-YYYYMMDD-XXXXX)
- ✅ View all appointments with status filtering
- ✅ Email notifications for approval/rejection
- ✅ Cancel pending appointments

### For Doctors:
- ✅ View appointments with filter options (Pending, Approved, Rejected)
- ✅ Default view shows Pending appointments
- ✅ Approve appointments with:
  - Date selection
  - Time slot (from - to)
  - Optional doctor note
- ✅ Reject appointments with optional note
- ✅ Automatic email notifications sent to patients

### For Admin:
- ✅ View all appointments across the system
- ✅ Filter by status (Pending, Approved, Rejected)
- ✅ Filter by doctor
- ✅ Search functionality (appointment number, patient name, doctor name, reason)
- ✅ View complete appointment details including date/time if approved

---

## 📁 Files Created/Modified

### New Files:
1. **Backend/models/Appointment.js** - Appointment schema with auto-generated appointment numbers
2. **Backend/controllers/appointmentController.js** - All appointment business logic
3. **Backend/routes/appointmentRoutes.js** - API route definitions

### Modified Files:
1. **Backend/utils/emailService.js** - Added email templates for approval/rejection
2. **Backend/server.js** - Registered appointment routes

---

## 🔌 API Endpoints

### Patient Endpoints:
```
POST   /api/appointments                    - Create new appointment
GET    /api/appointments/my-appointments    - Get patient's appointments (with status filter)
DELETE /api/appointments/:id                - Cancel pending appointment
```

### Doctor Endpoints:
```
GET    /api/appointments/doctor-appointments - Get doctor's appointments (default: Pending)
PUT    /api/appointments/:id/approve         - Approve appointment
PUT    /api/appointments/:id/reject          - Reject appointment
```

### Admin Endpoints:
```
GET    /api/appointments/all                 - Get all appointments (with filters)
```

### Common Endpoints:
```
GET    /api/appointments/:id                 - Get single appointment by ID
```

---

## 📊 Database Schema

### Appointment Model:
```javascript
{
  appointmentNumber: String (unique, auto-generated),
  patientId: ObjectId (ref: Patient),
  doctorId: ObjectId (ref: Doctor),
  reason: String (required, max 500 chars),
  preferredTime: String (Morning/Afternoon/Evening/Night),
  preferredDates: String (Weekdays/Weekends/Any),
  additionalNotes: String (max 1000 chars),
  status: String (Pending/Approved/Rejected),
  
  // Filled when approved:
  appointmentDate: Date,
  appointmentTimeFrom: String,
  appointmentTimeTo: String,
  
  doctorNote: String (max 1000 chars),
  
  createdAt: Date,
  updatedAt: Date,
  respondedAt: Date
}
```

---

## 📧 Email Notifications

### Approval Email:
- ✅ Professional gradient design (green theme)
- ✅ Appointment number prominently displayed
- ✅ Doctor name, date, and time slot
- ✅ Optional doctor's note
- ✅ Branded footer

### Rejection Email:
- ✅ Professional gradient design (red theme)
- ✅ Appointment number
- ✅ Doctor name and reason for request
- ✅ Optional doctor's note explaining rejection
- ✅ Encouragement to submit new request

---

## 🔐 Security & Authorization

- ✅ All endpoints protected with JWT authentication
- ✅ Role-based access control (Patient, Doctor, Admin)
- ✅ Patients can only view/cancel their own appointments
- ✅ Doctors can only approve/reject their own appointments
- ✅ Admins have full visibility
- ✅ Status validation (can't approve/reject already processed appointments)

---

## 📝 Usage Examples

### 1. Patient Creates Appointment:
```javascript
POST /api/appointments
Headers: { Authorization: "Bearer <patient_token>" }
Body: {
  "doctorId": "doctor_id_here",
  "reason": "Regular checkup and consultation",
  "preferredTime": "Morning",
  "preferredDates": "Weekdays",
  "additionalNotes": "I prefer early morning slots"
}
```

### 2. Doctor Approves Appointment:
```javascript
PUT /api/appointments/:id/approve
Headers: { Authorization: "Bearer <doctor_token>" }
Body: {
  "appointmentDate": "2024-04-15",
  "appointmentTimeFrom": "09:00 AM",
  "appointmentTimeTo": "09:30 AM",
  "doctorNote": "Please bring your previous medical records"
}
```

### 3. Doctor Views Pending Appointments:
```javascript
GET /api/appointments/doctor-appointments?status=Pending
Headers: { Authorization: "Bearer <doctor_token>" }
```

### 4. Admin Searches Appointments:
```javascript
GET /api/appointments/all?search=APT-20240315&status=Approved
Headers: { Authorization: "Bearer <admin_token>" }
```

---

## ✅ Testing Checklist

- [ ] Patient can create appointment request
- [ ] Appointment number is auto-generated correctly
- [ ] Patient information is auto-filled from their profile
- [ ] Doctor receives appointment in their dashboard
- [ ] Doctor can filter by Pending/Approved/Rejected
- [ ] Doctor can approve with date/time
- [ ] Doctor can reject with note
- [ ] Patient receives email on approval
- [ ] Patient receives email on rejection
- [ ] Patient can view their appointments
- [ ] Patient can cancel pending appointments
- [ ] Admin can view all appointments
- [ ] Admin can filter and search appointments
- [ ] Authorization checks work correctly

---

## 🚀 Next Steps

To complete the full implementation, you'll need to:

1. **Frontend Components** (Patient Side):
   - Appointment request form
   - My appointments page with status tabs
   - Appointment details modal

2. **Frontend Components** (Doctor Side):
   - Appointments dashboard with filters
   - Approval modal with date/time picker
   - Rejection modal with note field

3. **Frontend Components** (Admin Side):
   - All appointments view with search/filter
   - Appointment details modal

4. **Testing**:
   - Start the backend server
   - Test all API endpoints
   - Verify email notifications
   - Test authorization rules

---

## 📞 Support

For any issues or questions regarding the appointment system:
- Check the API endpoints are correctly registered in server.js
- Verify MongoDB connection is active
- Ensure email service is configured (check .env file)
- Review console logs for detailed error messages

---

**Implementation Date:** March 3, 2026
**Version:** 1.0.0
**Status:** ✅ Backend Complete - Ready for Frontend Integration

