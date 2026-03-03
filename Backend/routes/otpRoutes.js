import express from 'express';
import { sendOTP, verifyOTP, resendOTP } from '../controllers/otpController.js';

const router = express.Router();

// Send OTP
router.post('/send', sendOTP);

// Verify OTP
router.post('/verify', verifyOTP);

// Resend OTP
router.post('/resend', resendOTP);

export default router;

