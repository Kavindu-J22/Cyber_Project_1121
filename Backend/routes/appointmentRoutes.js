import express from 'express';
import {
  createAppointment,
  getMyAppointments,
  getDoctorAppointments,
  getAllAppointments,
  getAppointmentById,
  approveAppointment,
  rejectAppointment,
  cancelAppointment
} from '../controllers/appointmentController.js';
import { protect } from '../middleware/auth.js';

const router = express.Router();

// Patient routes
router.post('/', protect, createAppointment);
router.get('/my-appointments', protect, getMyAppointments);
router.delete('/:id', protect, cancelAppointment);

// Doctor routes
router.get('/doctor-appointments', protect, getDoctorAppointments);
router.put('/:id/approve', protect, approveAppointment);
router.put('/:id/reject', protect, rejectAppointment);

// Admin routes
router.get('/all', protect, getAllAppointments);

// Common routes
router.get('/:id', protect, getAppointmentById);

export default router;

