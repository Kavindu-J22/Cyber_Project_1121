# Live Consultation System - Complete Implementation Guide

## 🎯 Overview
This system manages the complete flow from appointment booking to live consultation, with time-based controls and email notifications.

---

## 📋 Features Implemented

### **Patient Features:**
✅ View confirmed (approved) appointments in "Confirmed Live Consultations" tab  
✅ See appointment details (date, time, doctor, reason)  
✅ Time-based button states:
  - **Future appointments**: "Starts Soon" (disabled)
  - **Current time window**: "Send Waiting Alert to Doctor" or "Join Consultation"
  - **Past 1 hour**: "Consultation Missed" message
✅ Send waiting alert to doctor via email  
✅ Join consultation when doctor starts it  
✅ Automatic email notification when doctor starts consultation  

### **Doctor Features:**
✅ View scheduled consultations (approved appointments) in "Scheduled Consultations" tab  
✅ See patient details and appointment information  
✅ Time-based button states:
  - **Future appointments**: "Scheduled" (disabled)
  - **Current time window**: "Start Consultation" (enabled)
  - **Past 1 hour**: "Consultation Missed" message
✅ Visual alert when patient is waiting  
✅ Start consultation and automatically notify patient via email  
✅ Removed old "Start Live Consultation" blue section  

### **Time Zone:**
✅ All time calculations use **Sri Lankan timezone (UTC+5:30)**  
✅ Consultations available from scheduled time to 1 hour after  
✅ Auto-refresh every 30 seconds to update button states  

---

## 🗂️ Files Created/Modified

### **Backend Files Created:**
1. **`Backend/models/Consultation.js`**
   - Tracks consultation sessions
   - Fields: appointmentId, patientId, doctorId, status, scheduledDate, scheduledTimeFrom, scheduledTimeTo, doctorStartedAt, patientJoinedAt, waitingAlertSentAt, consultationRoomId

2. **`Backend/controllers/consultationController.js`**
   - `getPatientConsultations` - Get patient's confirmed consultations with time status
   - `getDoctorConsultations` - Get doctor's scheduled consultations with time status
   - `sendWaitingAlert` - Patient sends waiting alert to doctor
   - `startConsultation` - Doctor starts consultation and notifies patient
   - `getConsultationStatus` - Get consultation status
   - Helper functions for Sri Lankan timezone conversion

3. **`Backend/routes/consultationRoutes.js`**
   - Patient routes: GET /patient/my-consultations, POST /patient/:appointmentId/send-alert
   - Doctor routes: GET /doctor/my-consultations, POST /doctor/:appointmentId/start
   - Common routes: GET /:appointmentId/status

### **Backend Files Modified:**
1. **`Backend/utils/emailService.js`**
   - `sendConsultationStartedEmail` - Green gradient email when doctor starts consultation
   - `sendWaitingAlertEmail` - Orange gradient email when patient sends waiting alert

2. **`Backend/server.js`**
   - Registered consultation routes: `app.use('/api/consultations', consultationRoutes)`

### **Frontend Files Created:**
1. **`Client/src/components/ConfirmedConsultations.jsx`**
   - Displays patient's approved appointments
   - Time-based button logic (Starts Soon / Send Alert / Join / Missed)
   - Auto-refresh every 30 seconds
   - Shows waiting status when alert sent

2. **`Client/src/components/ScheduledConsultations.jsx`**
   - Displays doctor's approved appointments
   - Time-based button logic (Scheduled / Start Consultation / Missed)
   - Shows "Patient is Waiting" alert when patient sends alert
   - Auto-refresh every 30 seconds

### **Frontend Files Modified:**
1. **`Client/src/pages/PatientDashboard.jsx`**
   - Added "Confirmed Live Consultations" tab
   - Integrated ConfirmedConsultations component

2. **`Client/src/pages/Dashboard.jsx`** (Doctor Dashboard)
   - Changed default tab to "Scheduled Consultations"
   - Replaced "Dashboard" tab with "Scheduled Consultations"
   - Removed ML Services Status section
   - Removed Biometric Enrollment Status section
   - Removed "Start Live Consultation" blue section
   - Integrated ScheduledConsultations component

---

## 🔄 Complete Flow

### **Step 1: Appointment Approval**
1. Patient books appointment
2. Doctor approves appointment with date and time
3. Appointment status changes to "Approved"
4. Consultation record is automatically created

### **Step 2: Before Consultation Time**
**Patient Side:**
- Sees appointment in "Confirmed Live Consultations" tab
- Button shows "Starts Soon" (disabled)

**Doctor Side:**
- Sees appointment in "Scheduled Consultations" tab
- Button shows "Scheduled" (disabled)

### **Step 3: At Consultation Time (Within 1 Hour Window)**

**Scenario A: Patient Waits for Doctor**
1. Patient's button changes to "Send Waiting Alert to Doctor"
2. Patient clicks button
3. Doctor receives email notification
4. Doctor's dashboard shows "Patient is Waiting for You!" alert
5. Doctor clicks "Start Consultation"
6. Patient receives email notification
7. Patient's button changes to "Join Consultation"
8. Both can join the consultation room

**Scenario B: Doctor Starts First**
1. Doctor's button changes to "Start Consultation"
2. Doctor clicks button
3. Patient receives email notification immediately
4. Patient's button changes to "Join Consultation"
5. Both can join the consultation room

### **Step 4: After 1 Hour Past Start Time**
**Both Patient and Doctor:**
- See "Consultation Missed" message
- Cannot start or join consultation
- Patient must book a new appointment

---

## 🕐 Time Logic (Sri Lankan Timezone)

### **Time Windows:**
```
Scheduled Time: 10:00 AM

Before 10:00 AM → Future (buttons disabled)
10:00 AM - 11:00 AM → Active window (buttons enabled)
After 11:00 AM → Missed (consultation expired)
```

### **Button States:**

**Patient:**
| Time | Consultation Status | Button State |
|------|-------------------|--------------|
| Before scheduled time | Scheduled | "Starts Soon" (disabled) |
| Within 1 hour, doctor not started | Scheduled | "Send Waiting Alert to Doctor" |
| Within 1 hour, alert sent | PatientWaiting | "Waiting for Doctor" (info box) |
| Within 1 hour, doctor started | Active | "Join Consultation" (enabled) |
| After 1 hour | Missed | "Consultation Missed" (error box) |

**Doctor:**
| Time | Consultation Status | Button State |
|------|-------------------|--------------|
| Before scheduled time | Scheduled | "Scheduled" (disabled) |
| Within 1 hour, patient not waiting | Scheduled | "Start Consultation" (enabled) |
| Within 1 hour, patient waiting | PatientWaiting | "Patient is Waiting!" + "Start Consultation" |
| Within 1 hour, started | Active | Redirects to consultation room |
| After 1 hour | Missed | "Consultation Missed" (error box) |

---

## 📧 Email Notifications

### **1. Consultation Started Email (to Patient)**
**Trigger:** Doctor clicks "Start Consultation"  
**Design:** Green gradient  
**Content:**
- Appointment number
- Doctor name
- Scheduled date and time
- "Join Consultation" call-to-action
- Urgent message to join immediately

### **2. Waiting Alert Email (to Doctor)**
**Trigger:** Patient clicks "Send Waiting Alert to Doctor"  
**Design:** Orange gradient  
**Content:**
- Appointment number
- Patient name
- Scheduled date and time
- Urgent alert that patient is waiting
- Call-to-action to start consultation

---

## 🎨 UI/UX Features

### **Color Coding:**
- **Green**: Approved appointments, Join button, Consultation started
- **Yellow**: Waiting alerts, Send alert button
- **Red**: Missed consultations, Error states
- **Gray**: Disabled buttons (future/scheduled)
- **Blue**: Information boxes

### **Auto-Refresh:**
- Both components refresh every 30 seconds
- Ensures button states are always up-to-date
- No manual refresh needed

### **Visual Alerts:**
- Patient waiting: Yellow alert box on doctor's side
- Consultation missed: Red error box with explanation
- Waiting status: Yellow info box on patient's side

---

## 🧪 Testing Guide

### **Test 1: Future Appointment**
1. Create and approve appointment for tomorrow
2. **Patient**: Should see "Starts Soon" (disabled)
3. **Doctor**: Should see "Scheduled" (disabled)

### **Test 2: Current Appointment - Patient Sends Alert**
1. Create and approve appointment for current time
2. **Patient**: Click "Send Waiting Alert to Doctor"
3. **Check**: Doctor receives email
4. **Doctor**: Should see "Patient is Waiting for You!" alert
5. **Doctor**: Click "Start Consultation"
6. **Check**: Patient receives email
7. **Patient**: Should see "Join Consultation" button
8. **Both**: Can click to join consultation room

### **Test 3: Current Appointment - Doctor Starts First**
1. Create and approve appointment for current time
2. **Doctor**: Click "Start Consultation"
3. **Check**: Patient receives email immediately
4. **Patient**: Should see "Join Consultation" button
5. **Both**: Can join consultation

### **Test 4: Missed Appointment**
1. Create and approve appointment for 2 hours ago
2. **Both**: Should see "Consultation Missed" message
3. **Patient**: Should see message to book new appointment

### **Test 5: Auto-Refresh**
1. Create appointment for 5 minutes from now
2. Wait and watch buttons change automatically
3. Verify no manual refresh needed

---

## 🔧 API Endpoints

### **Patient Endpoints:**
```
GET /api/consultations/patient/my-consultations
- Returns: List of approved appointments with time status

POST /api/consultations/patient/:appointmentId/send-alert
- Action: Sends waiting alert email to doctor
- Updates: Consultation status to 'PatientWaiting'
```

### **Doctor Endpoints:**
```
GET /api/consultations/doctor/my-consultations
- Returns: List of approved appointments with time status

POST /api/consultations/doctor/:appointmentId/start
- Action: Starts consultation, sends email to patient
- Updates: Consultation status to 'Active'
- Returns: Consultation room ID
```

### **Common Endpoints:**
```
GET /api/consultations/:appointmentId/status
- Returns: Current consultation status
```

---

## 📊 Database Schema

### **Consultation Model:**
```javascript
{
  appointmentId: ObjectId (ref: Appointment, unique),
  patientId: ObjectId (ref: Patient),
  doctorId: ObjectId (ref: Doctor),
  status: String (Scheduled/PatientWaiting/Active/Completed/Missed),
  scheduledDate: Date,
  scheduledTimeFrom: String,
  scheduledTimeTo: String,
  doctorStartedAt: Date,
  patientJoinedAt: Date,
  endedAt: Date,
  waitingAlertSentAt: Date,
  consultationRoomId: String,
  createdAt: Date,
  updatedAt: Date
}
```

---

## ✅ Checklist

### **Backend:**
- [x] Consultation model created
- [x] Consultation controller with all functions
- [x] Consultation routes registered
- [x] Email templates for consultation start and waiting alert
- [x] Sri Lankan timezone conversion
- [x] Time validation logic (1-hour window)

### **Frontend:**
- [x] ConfirmedConsultations component
- [x] ScheduledConsultations component
- [x] Patient dashboard updated with new tab
- [x] Doctor dashboard updated (removed old sections)
- [x] Time-based button logic
- [x] Auto-refresh functionality
- [x] Visual alerts and status indicators

### **Testing:**
- [ ] Test future appointments
- [ ] Test current appointments (patient sends alert)
- [ ] Test current appointments (doctor starts first)
- [ ] Test missed appointments
- [ ] Test email notifications
- [ ] Test auto-refresh
- [ ] Test Sri Lankan timezone conversion

---

## 🚀 Deployment Notes

1. **Environment Variables:**
   - Ensure EMAIL_USER and EMAIL_PASS are configured
   - Verify timezone settings on server

2. **Database:**
   - Consultation collection will be created automatically
   - Indexes are set for performance

3. **Frontend:**
   - Auto-refresh runs every 30 seconds
   - Consider adjusting interval based on server load

4. **Email Service:**
   - Test email delivery in production
   - Check spam folder if emails not received

---

**Status**: ✅ Complete - Ready for Testing  
**Date**: March 3, 2026  
**Timezone**: Sri Lankan Time (UTC+5:30)

