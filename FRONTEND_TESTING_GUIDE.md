# Frontend Testing Guide - Appointment System

## 🎯 Overview
This guide will help you test the complete appointment management system from frontend to backend.

---

## 📋 Prerequisites

1. **Backend Server Running**
   ```bash
   cd Backend
   npm start
   # Should be running on http://localhost:5000
   ```

2. **Frontend Server Running**
   ```bash
   cd Client
   npm run dev
   # Should be running on http://localhost:5173 (or similar)
   ```

3. **MongoDB Running**
   - Ensure MongoDB is connected and running

4. **Email Service Configured**
   - Check `.env` file has `EMAIL_USER` and `EMAIL_PASS` configured

---

## 🧪 Testing Flow

### Step 1: Patient Creates Appointment

1. **Login as Patient**
   - Navigate to login page
   - Use patient credentials
   - Should redirect to Patient Dashboard

2. **Navigate to Doctors Tab**
   - Should see "Find Doctors" tab (default view)
   - View list of available doctors

3. **Book Appointment**
   - Click "Book Appointment" button on any doctor card
   - Modal should open with doctor information
   - Fill in the form:
     - **Reason**: Enter reason for appointment (required)
     - **Preferred Time**: Select Morning/Afternoon/Evening/Night
     - **Preferred Dates**: Select Weekdays/Weekends/Any
     - **Additional Notes**: Optional notes
   - Click "Book Appointment"
   - Should see success toast with appointment number (APT-YYYYMMDD-XXXXX)
   - Modal should close

4. **View My Appointments**
   - Click "My Appointments" tab
   - Should see the newly created appointment
   - Status should be "Pending" (yellow badge)
   - Should display:
     - Appointment number
     - Doctor name and specialization
     - Reason, preferred time, preferred dates
     - Additional notes (if provided)
     - Cancel button (for pending appointments)

### Step 2: Doctor Reviews and Approves Appointment

1. **Login as Doctor**
   - Logout from patient account
   - Login with doctor credentials
   - Should redirect to Doctor Dashboard

2. **Navigate to Appointments Tab**
   - Click "Appointments" tab
   - Should see "Pending" filter selected by default
   - Should see the appointment created by patient

3. **Review Appointment Details**
   - Should see:
     - Appointment number
     - Patient name, age, gender, email
     - Reason for appointment
     - Preferred time and dates
     - Additional notes (if any)

4. **Approve Appointment**
   - Click "Approve" button
   - Modal should open
   - Fill in required fields:
     - **Appointment Date**: Select future date
     - **From Time**: Select start time
     - **To Time**: Select end time
     - **Doctor Note**: Optional note to patient
   - Click "Approve Appointment"
   - Should see success toast
   - Modal should close
   - Appointment should move to "Approved" filter

5. **Verify Email Sent**
   - Check patient's email inbox
   - Should receive approval email with:
     - Green gradient design
     - Appointment number
     - Doctor name
     - Date and time slot
     - Doctor's note (if provided)

### Step 3: Patient Views Approved Appointment

1. **Login as Patient Again**
   - Logout from doctor account
   - Login as patient
   - Navigate to "My Appointments" tab

2. **View Approved Appointment**
   - Click "Approved" filter
   - Should see the appointment with:
     - Green badge showing "Approved"
     - Scheduled date and time
     - Doctor's note (if provided)
   - Should NOT see cancel button (approved appointments can't be cancelled)

### Step 4: Test Rejection Flow

1. **Create Another Appointment** (as Patient)
   - Book another appointment with same or different doctor

2. **Reject Appointment** (as Doctor)
   - Login as doctor
   - Go to Appointments tab
   - Click "Reject" button on the new appointment
   - Modal should open
   - Optionally add rejection note
   - Click "Reject Appointment"
   - Should see success toast
   - Appointment should move to "Rejected" filter

3. **Verify Rejection Email**
   - Check patient's email
   - Should receive rejection email with:
     - Red gradient design
     - Appointment number
     - Doctor name
     - Rejection note (if provided)

4. **Patient Views Rejected Appointment**
   - Login as patient
   - Go to "My Appointments" → "Rejected" filter
   - Should see appointment with red badge
   - Should see rejection note (if provided)

### Step 5: Test Cancel Flow

1. **Create Appointment** (as Patient)
   - Book a new appointment

2. **Cancel Pending Appointment**
   - In "My Appointments" → "Pending" filter
   - Click "Cancel Appointment" button
   - Confirm cancellation in dialog
   - Should see success toast
   - Appointment should be removed from list

### Step 6: Admin Views All Appointments

1. **Login as Admin**
   - Use admin credentials (admin@gmail.com / admin)
   - Should redirect to Admin Dashboard

2. **Navigate to Appointments Tab**
   - Click "Appointments" tab
   - Should see all appointments from all patients and doctors

3. **Test Filters**
   - **Search**: Try searching by:
     - Appointment number
     - Patient name
     - Doctor name
     - Reason keywords
   - **Status Filter**: Filter by Pending/Approved/Rejected
   - **Doctor Filter**: Filter by specific doctor

4. **View Appointment Details**
   - Each appointment should show:
     - Appointment number and status badge
     - Patient information (name, age, gender, email)
     - Doctor information (name, specialization, email)
     - Reason, preferred time, preferred dates
     - If approved: scheduled date and time
     - If rejected: rejection note

---

## ✅ Testing Checklist

### Patient Features:
- [ ] Can view list of doctors
- [ ] Can search doctors by name
- [ ] Can filter doctors by specialization
- [ ] Can open book appointment modal
- [ ] Can fill and submit appointment form
- [ ] Appointment number is auto-generated
- [ ] Can view all appointments in "My Appointments"
- [ ] Can filter appointments by status (All/Pending/Approved/Rejected)
- [ ] Can cancel pending appointments
- [ ] Cannot cancel approved/rejected appointments
- [ ] Receives email on approval
- [ ] Receives email on rejection

### Doctor Features:
- [ ] Can view appointments in Appointments tab
- [ ] Default filter shows "Pending" appointments
- [ ] Can filter by Pending/Approved/Rejected
- [ ] Can see patient details in appointment card
- [ ] Can approve appointment with date/time
- [ ] Can reject appointment with optional note
- [ ] Approved appointments show in "Approved" filter
- [ ] Rejected appointments show in "Rejected" filter
- [ ] Email is sent to patient on approval
- [ ] Email is sent to patient on rejection

### Admin Features:
- [ ] Can view all appointments system-wide
- [ ] Can search appointments by multiple criteria
- [ ] Can filter by status
- [ ] Can filter by doctor
- [ ] Can see complete appointment details
- [ ] Can see patient and doctor information
- [ ] Results count updates with filters

### UI/UX:
- [ ] Status badges are color-coded (Yellow/Green/Red)
- [ ] Modals open and close properly
- [ ] Forms validate required fields
- [ ] Loading states show during API calls
- [ ] Success/error toasts appear
- [ ] Tabs switch correctly
- [ ] Responsive design works on mobile

---

## 🐛 Common Issues & Solutions

### Issue: "Failed to load appointments"
**Solution**: Check backend server is running and MongoDB is connected

### Issue: Email not received
**Solution**: 
- Check `.env` file has correct EMAIL_USER and EMAIL_PASS
- Check spam folder
- Verify email service is configured in backend

### Issue: "Not authorized"
**Solution**: 
- Clear browser cache and localStorage
- Login again to get fresh JWT token

### Issue: Appointment not showing
**Solution**:
- Refresh the page
- Check correct filter is selected
- Verify appointment was created successfully

---

## 🎨 UI Components Created

### Patient Components:
- `BookAppointmentModal.jsx` - Modal for booking appointments
- `MyAppointments.jsx` - Component to view patient's appointments

### Doctor Components:
- `DoctorAppointments.jsx` - List of doctor's appointments
- `ApproveAppointmentModal.jsx` - Modal for approving appointments
- `RejectAppointmentModal.jsx` - Modal for rejecting appointments

### Admin Components:
- `AdminAppointments.jsx` - View all appointments with filters

### Updated Pages:
- `PatientDashboard.jsx` - Added appointments tab
- `Dashboard.jsx` (Doctor) - Added appointments tab
- `AdminDashboard.jsx` - Added appointments tab

---

## 🚀 Deployment Checklist

Before deploying to production:

- [ ] Test all flows end-to-end
- [ ] Verify email notifications work
- [ ] Check all API endpoints are accessible
- [ ] Test with multiple users simultaneously
- [ ] Verify data persistence in MongoDB
- [ ] Test on different browsers
- [ ] Test responsive design on mobile devices
- [ ] Check console for any errors
- [ ] Verify all environment variables are set
- [ ] Test error handling scenarios

---

**Status**: ✅ Frontend Complete - Ready for Testing
**Date**: March 3, 2026

