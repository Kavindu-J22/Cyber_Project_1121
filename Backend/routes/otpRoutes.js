import express from 'express';
import { sendOTP, verifyOTP, resendOTP, sendConsultationOTP, verifyConsultationOTP, resendConsultationOTP } from '../controllers/otpController.js';
import { protect } from '../middleware/auth.js';

const router = express.Router();

// Send OTP (registration)
router.post('/send', sendOTP);

// Verify OTP (registration)
router.post('/verify', verifyOTP);

// Resend OTP (registration)
router.post('/resend', resendOTP);

// ─── Consultation OTP (authenticated doctor only) ───────────────────────────
router.post('/consultation/send',   protect, sendConsultationOTP);
router.post('/consultation/verify', protect, verifyConsultationOTP);
router.post('/consultation/resend', protect, resendConsultationOTP);

export default router;

