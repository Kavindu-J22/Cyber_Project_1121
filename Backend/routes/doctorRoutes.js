import express from 'express';
import {
  getMe,
  getAllDoctors,
  getDoctorById,
  updateDoctor,
  deleteDoctor,
  toggleDoctorActive
} from '../controllers/doctorController.js';
import { protect } from '../middleware/auth.js';
import upload from '../middleware/upload.js';

const router = express.Router();

router.get('/me', protect, getMe);
router.get('/', protect, getAllDoctors);
router.get('/:id', protect, getDoctorById);
router.put('/:id', protect, upload.single('profileImage'), updateDoctor);
router.delete('/:id', protect, deleteDoctor);
router.patch('/:id/toggle-active', protect, toggleDoctorActive);

export default router;

