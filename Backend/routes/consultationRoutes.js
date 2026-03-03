import express from 'express';
import {
  getPatientConsultations,
  getDoctorConsultations,
  sendWaitingAlert,
  startConsultation,
  getConsultationStatus,
  endConsultation
} from '../controllers/consultationController.js';
import { protect, restrictTo } from '../middleware/auth.js';

const router = express.Router();

// All routes require authentication
router.use(protect);

// Patient routes
router.get('/patient/my-consultations', restrictTo('patient'), getPatientConsultations);
router.post('/patient/:appointmentId/send-alert', restrictTo('patient'), sendWaitingAlert);

// Doctor routes
router.get('/doctor/my-consultations', restrictTo('doctor'), getDoctorConsultations);
router.post('/doctor/:appointmentId/start', restrictTo('doctor'), startConsultation);
router.put('/:sessionId/end', restrictTo('doctor'), endConsultation);

// Common routes
router.get('/:appointmentId/status', getConsultationStatus);

export default router;

