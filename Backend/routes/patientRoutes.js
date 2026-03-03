import express from 'express';
import { 
  getMe, 
  getAllPatients, 
  getAllDoctors,
  updatePatient 
} from '../controllers/patientController.js';
import { protect } from '../middleware/auth.js';

const router = express.Router();

router.get('/me', protect, getMe);
router.get('/', protect, getAllPatients);
router.get('/doctors', protect, getAllDoctors);
router.put('/:id', protect, updatePatient);

export default router;

