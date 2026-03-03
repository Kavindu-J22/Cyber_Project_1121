# Fix for 403 Forbidden Error on Consultations

## 🔴 Problem

**Error:**
```
Error fetching consultations: AxiosError: Request failed with status code 403
```

**Cause:**
The `restrictTo` middleware in the consultation routes was checking for capitalized role names (`'Patient'` and `'Doctor'`), but the auth middleware sets `req.userRole` to lowercase (`'patient'` and `'doctor'`).

---

## ✅ Solution Applied

**File Modified:** `Backend/routes/consultationRoutes.js`

**Changed:**
```javascript
// BEFORE (WRONG - Capitalized)
router.get('/patient/my-consultations', restrictTo('Patient'), getPatientConsultations);
router.post('/patient/:appointmentId/send-alert', restrictTo('Patient'), sendWaitingAlert);
router.get('/doctor/my-consultations', restrictTo('Doctor'), getDoctorConsultations);
router.post('/doctor/:appointmentId/start', restrictTo('Doctor'), startConsultation);

// AFTER (CORRECT - Lowercase)
router.get('/patient/my-consultations', restrictTo('patient'), getPatientConsultations);
router.post('/patient/:appointmentId/send-alert', restrictTo('patient'), sendWaitingAlert);
router.get('/doctor/my-consultations', restrictTo('doctor'), getDoctorConsultations);
router.post('/doctor/:appointmentId/start', restrictTo('doctor'), startConsultation);
```

---

## 🔄 How to Apply the Fix

### **Step 1: Restart Backend Server**

**Option A: Using Terminal**
1. Open a terminal in the Backend folder
2. Stop the current server (Ctrl+C if running)
3. Run: `npm start`

**Option B: Using PowerShell**
1. Find and kill the node process:
   ```powershell
   # Find the process
   netstat -ano | findstr :5000
   
   # Kill it (replace PID with the actual process ID)
   taskkill /F /PID <PID>
   
   # Start the server
   cd Backend
   npm start
   ```

### **Step 2: Verify the Fix**
1. Wait for the server to start (you should see "Server running on port 5000")
2. Refresh your browser
3. Login as a doctor
4. Go to "Scheduled Consultations" tab
5. ✅ Should now load consultations without 403 error

---

## 📝 Technical Details

### **Auth Middleware Sets Lowercase Roles:**
```javascript
// From Backend/middleware/auth.js
if (decoded.role === 'doctor') {
  req.userRole = 'doctor';  // ← lowercase
} else if (decoded.role === 'patient') {
  req.userRole = 'patient';  // ← lowercase
}
```

### **restrictTo Middleware Checks:**
```javascript
// From Backend/middleware/auth.js
export const restrictTo = (...roles) => {
  return (req, res, next) => {
    if (!roles.includes(req.userRole)) {  // ← Must match exactly
      return res.status(403).json({
        success: false,
        message: 'You do not have permission to perform this action'
      });
    }
    next();
  };
};
```

### **Why It Failed:**
- `req.userRole = 'doctor'` (lowercase)
- `restrictTo('Doctor')` was checking for `'Doctor'` (capitalized)
- `'doctor' !== 'Doctor'` → 403 Forbidden

### **Why It Works Now:**
- `req.userRole = 'doctor'` (lowercase)
- `restrictTo('doctor')` checks for `'doctor'` (lowercase)
- `'doctor' === 'doctor'` → ✅ Allowed

---

## ✅ Status

**Fix Applied:** ✅ Complete  
**File Modified:** `Backend/routes/consultationRoutes.js`  
**Action Required:** Restart backend server  

---

## 🧪 Testing After Restart

### **Test 1: Doctor Consultations**
1. Login as doctor
2. Go to "Scheduled Consultations" tab
3. ✅ Should load approved appointments
4. ✅ Should show time-based buttons

### **Test 2: Patient Consultations**
1. Login as patient
2. Go to "Confirmed Live Consultations" tab
3. ✅ Should load approved appointments
4. ✅ Should show time-based buttons

### **Test 3: Send Waiting Alert**
1. Login as patient with current appointment
2. Click "Send Waiting Alert to Doctor"
3. ✅ Should send alert without 403 error
4. ✅ Doctor should receive email

### **Test 4: Start Consultation**
1. Login as doctor with current appointment
2. Click "Start Consultation"
3. ✅ Should start without 403 error
4. ✅ Patient should receive email

---

**Date:** March 3, 2026  
**Status:** ✅ Fix Applied - Restart Required

