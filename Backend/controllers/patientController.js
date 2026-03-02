import Patient from '../models/Patient.js';
import Doctor from '../models/Doctor.js';

// @desc    Get current patient profile
// @route   GET /api/patients/me
// @access  Private (Patient)
export const getMe = async (req, res) => {
  try {
    const patient = await Patient.findById(req.user.id);
    
    if (!patient) {
      return res.status(404).json({
        success: false,
        message: 'Patient not found'
      });
    }

    res.json({
      success: true,
      data: {
        patient: {
          id: patient._id,
          fullName: patient.fullName,
          age: patient.age,
          gender: patient.gender,
          email: patient.email,
          createdAt: patient.createdAt,
          lastLogin: patient.lastLogin
        }
      }
    });
  } catch (error) {
    console.error('Get patient error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Get all patients (Admin only)
// @route   GET /api/patients
// @access  Private (Admin)
export const getAllPatients = async (req, res) => {
  try {
    const patients = await Patient.find().select('-password');
    
    res.json({
      success: true,
      count: patients.length,
      data: { patients }
    });
  } catch (error) {
    console.error('Get all patients error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Get all doctors (for patient to view and book appointments)
// @route   GET /api/patients/doctors
// @access  Private (Patient)
export const getAllDoctors = async (req, res) => {
  try {
    const doctors = await Doctor.find({ isActive: true }).select('-password');
    
    res.json({
      success: true,
      count: doctors.length,
      data: { doctors }
    });
  } catch (error) {
    console.error('Get doctors error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Update patient profile
// @route   PUT /api/patients/:id
// @access  Private (Patient/Admin)
export const updatePatient = async (req, res) => {
  try {
    const patient = await Patient.findById(req.params.id);

    if (!patient) {
      return res.status(404).json({
        success: false,
        message: 'Patient not found'
      });
    }

    // Update fields
    const { fullName, age, gender } = req.body;
    
    if (fullName) patient.fullName = fullName;
    if (age) patient.age = age;
    if (gender) patient.gender = gender;

    await patient.save();

    res.json({
      success: true,
      message: 'Patient updated successfully',
      data: {
        patient: {
          id: patient._id,
          fullName: patient.fullName,
          age: patient.age,
          gender: patient.gender,
          email: patient.email
        }
      }
    });
  } catch (error) {
    console.error('Update patient error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

