import express from 'express';
import { register, registerPatient, login } from '../controllers/authController.js';
import upload from '../middleware/upload.js';

const router = express.Router();

// Doctor registration - Accept multiple voice samples (up to 5) and face images (up to 5)
router.post('/register', upload.fields([
  { name: 'voiceSamples', maxCount: 5 },
  { name: 'faceImages', maxCount: 5 }
]), register);

// Patient registration - No biometric data required
router.post('/register-patient', registerPatient);

// Login for all user types (doctor, patient, admin)
router.post('/login', login);

export default router;

