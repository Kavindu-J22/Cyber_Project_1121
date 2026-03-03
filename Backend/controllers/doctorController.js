import Doctor from '../models/Doctor.js';
import cloudinary from '../config/cloudinary.js';
import fs from 'fs';

// @desc    Get current doctor profile
// @route   GET /api/doctors/me
// @access  Private
export const getMe = async (req, res) => {
  try {
    const doctor = await Doctor.findById(req.doctor._id);

    res.json({
      success: true,
      data: {
        doctor: doctor
      }
    });
  } catch (error) {
    console.error('Get profile error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Get all doctors
// @route   GET /api/doctors
// @access  Private
export const getAllDoctors = async (req, res) => {
  try {
    // Check if request is from admin (show all) or patient (show only active)
    const userRole = req.userRole;
    const query = userRole === 'admin' ? {} : { isActive: true };

    const doctors = await Doctor.find(query)
      .select('-password')
      .sort({ createdAt: -1 });

    res.json({
      success: true,
      count: doctors.length,
      data: {
        doctors: doctors
      }
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

// @desc    Get doctor by ID
// @route   GET /api/doctors/:id
// @access  Private
export const getDoctorById = async (req, res) => {
  try {
    const doctor = await Doctor.findById(req.params.id).select('-password');

    if (!doctor) {
      return res.status(404).json({
        success: false,
        message: 'Doctor not found'
      });
    }

    res.json({
      success: true,
      data: doctor
    });
  } catch (error) {
    console.error('Get doctor error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Update doctor profile
// @route   PUT /api/doctors/:id
// @access  Private
export const updateDoctor = async (req, res) => {
  try {
    const { firstName, lastName, specialization, yearsOfExperience, description } = req.body;

    const doctor = await Doctor.findById(req.params.id);

    if (!doctor) {
      return res.status(404).json({
        success: false,
        message: 'Doctor not found'
      });
    }

    // Update fields
    if (firstName) doctor.firstName = firstName;
    if (lastName) doctor.lastName = lastName;
    if (specialization) doctor.specialization = specialization;
    if (yearsOfExperience !== undefined) doctor.yearsOfExperience = yearsOfExperience;
    if (description !== undefined) doctor.description = description;

    // Handle profile image upload to Cloudinary
    if (req.file) {
      try {
        // Upload to Cloudinary
        const result = await cloudinary.uploader.upload(req.file.path, {
          folder: 'cybermed',
          resource_type: 'image',
          transformation: [
            { width: 500, height: 500, crop: 'fill', gravity: 'face' },
            { quality: 'auto' }
          ]
        });

        // Delete old image from Cloudinary if exists
        if (doctor.profileImage) {
          const publicId = doctor.profileImage.split('/').pop().split('.')[0];
          await cloudinary.uploader.destroy(`cybermed/${publicId}`);
        }

        doctor.profileImage = result.secure_url;

        // Delete local file
        fs.unlinkSync(req.file.path);
      } catch (uploadError) {
        console.error('Cloudinary upload error:', uploadError);
        // Clean up local file
        if (req.file && fs.existsSync(req.file.path)) {
          fs.unlinkSync(req.file.path);
        }
        return res.status(500).json({
          success: false,
          message: 'Failed to upload profile image',
          error: uploadError.message
        });
      }
    }

    await doctor.save();

    res.json({
      success: true,
      message: 'Doctor profile updated successfully',
      data: {
        doctor: doctor
      }
    });
  } catch (error) {
    console.error('Update doctor error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Delete doctor
// @route   DELETE /api/doctors/:id
// @access  Private (Admin)
export const deleteDoctor = async (req, res) => {
  try {
    const doctor = await Doctor.findById(req.params.id);

    if (!doctor) {
      return res.status(404).json({
        success: false,
        message: 'Doctor not found'
      });
    }

    await Doctor.findByIdAndDelete(req.params.id);

    res.json({
      success: true,
      message: 'Doctor deleted successfully'
    });
  } catch (error) {
    console.error('Delete doctor error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Toggle doctor active status
// @route   PATCH /api/doctors/:id/toggle-active
// @access  Private (Admin)
export const toggleDoctorActive = async (req, res) => {
  try {
    const { isActive } = req.body;
    const doctor = await Doctor.findById(req.params.id);

    if (!doctor) {
      return res.status(404).json({
        success: false,
        message: 'Doctor not found'
      });
    }

    doctor.isActive = isActive;
    await doctor.save();

    res.json({
      success: true,
      message: `Doctor ${isActive ? 'activated' : 'deactivated'} successfully`,
      data: {
        doctor: doctor
      }
    });
  } catch (error) {
    console.error('Toggle doctor active error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

