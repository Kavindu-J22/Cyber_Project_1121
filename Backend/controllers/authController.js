import Doctor from '../models/Doctor.js';
import { generateToken } from '../middleware/auth.js';
import mlService from '../services/mlService.js';
import fs from 'fs';
import { convertMultipleToWav } from '../utils/audioConverter.js';

// @desc    Register a new doctor
// @route   POST /api/auth/register
// @access  Public
export const register = async (req, res) => {
  try {
    const {
      firstName,
      lastName,
      email,
      password,
      medicalLicenseNumber,
      specialization,
      yearsOfExperience,
      keystrokePattern,
      mousePattern,
      facePattern
    } = req.body;

    // Check if doctor already exists
    const doctorExists = await Doctor.findOne({ 
      $or: [{ email }, { medicalLicenseNumber }] 
    });

    if (doctorExists) {
      return res.status(400).json({
        success: false,
        message: 'Doctor with this email or license number already exists'
      });
    }

    // Create doctor
    const doctor = await Doctor.create({
      firstName,
      lastName,
      email,
      password,
      medicalLicenseNumber,
      specialization,
      yearsOfExperience
    });

    // Enroll biometric data
    const biometricResults = {
      voice: { success: false, error: null },
      keystroke: { success: false, error: null },
      mouse: { success: false, error: null },
      face: { success: false, error: null }
    };

    // Enroll voice if audio files provided (expecting 3 samples)
    if (req.files && req.files.voiceSamples && req.files.voiceSamples.length > 0) {
      let convertedPaths = [];
      try {
        console.log(`Enrolling voice for doctor ${doctor._id}... (${req.files.voiceSamples.length} samples)`);

        // Get all file paths
        const audioFilePaths = req.files.voiceSamples.map(file => file.path);

        // Convert to WAV format if needed (fallback if frontend conversion fails)
        console.log('Converting audio files to WAV format...');
        convertedPaths = await convertMultipleToWav(audioFilePaths);
        console.log(`✓ Audio files converted: ${convertedPaths.length} files`);

        // Enroll with multiple samples
        const voiceResult = await mlService.enrollVoiceMultiple(doctor._id.toString(), convertedPaths);
        doctor.biometricData.voiceEnrolled = true;
        doctor.biometricData.voiceEmbedding = doctor._id.toString();
        biometricResults.voice.success = true;
        console.log('✓ Voice enrollment successful');

        // Clean up uploaded files
        convertedPaths.forEach(path => {
          if (fs.existsSync(path)) {
            fs.unlinkSync(path);
          }
        });
      } catch (error) {
        console.error('✗ Voice enrollment failed:', error.message);
        biometricResults.voice.error = error.message;
        // Clean up uploaded files even on error
        if (convertedPaths.length > 0) {
          convertedPaths.forEach(path => {
            if (fs.existsSync(path)) {
              fs.unlinkSync(path);
            }
          });
        }
        if (req.files && req.files.voiceSamples && req.files.voiceSamples.length > 0) {
          req.files.voiceSamples.forEach(file => {
            if (file.path && fs.existsSync(file.path)) {
              fs.unlinkSync(file.path);
            }
          });
        }
      }
    }

    // Enroll keystroke pattern
    if (keystrokePattern) {
      try {
        console.log(`Enrolling keystroke for doctor ${doctor._id}...`);
        const keystrokeData = JSON.parse(keystrokePattern);
        console.log(`Keystroke samples: ${keystrokeData.length}`);
        const keystrokeResult = await mlService.enrollKeystroke(
          doctor._id.toString(),
          keystrokeData
        );
        doctor.biometricData.keystrokeEnrolled = true;
        doctor.biometricData.keystrokeProfile = doctor._id.toString();
        biometricResults.keystroke.success = true;
        console.log('✓ Keystroke enrollment successful');
      } catch (error) {
        console.error('✗ Keystroke enrollment failed:', error.message);
        biometricResults.keystroke.error = error.message;
      }
    }

    // Enroll mouse pattern
    if (mousePattern) {
      try {
        console.log(`Enrolling mouse for doctor ${doctor._id}...`);
        const mouseData = JSON.parse(mousePattern);
        console.log(`Mouse events: ${mouseData.length}`);
        const mouseResult = await mlService.enrollMouse(
          doctor._id.toString(),
          mouseData
        );
        doctor.biometricData.mouseEnrolled = true;
        doctor.biometricData.mouseProfile = doctor._id.toString();
        biometricResults.mouse.success = true;
        console.log('✓ Mouse enrollment successful');
      } catch (error) {
        console.error('✗ Mouse enrollment failed:', error.message);
        biometricResults.mouse.error = error.message;
      }
    }

    // Enroll face pattern (if face images provided)
    if (req.files && req.files.faceImages && req.files.faceImages.length > 0) {
      try {
        console.log(`Enrolling face for doctor ${doctor._id}... (${req.files.faceImages.length} samples)`);

        // Get all face image file paths
        const faceImagePaths = req.files.faceImages.map(file => file.path);
        console.log(`Face image paths: ${faceImagePaths.join(', ')}`);

        // Enroll face with image files
        const faceResult = await mlService.enrollFace(
          doctor._id.toString(),
          faceImagePaths
        );
        doctor.biometricData.faceEnrolled = true;
        doctor.biometricData.faceProfile = doctor._id.toString();
        biometricResults.face.success = true;
        console.log('✓ Face enrollment successful');

        // Clean up uploaded face images
        faceImagePaths.forEach(path => {
          if (fs.existsSync(path)) {
            fs.unlinkSync(path);
          }
        });
      } catch (error) {
        console.error('✗ Face enrollment failed:', error.message);
        biometricResults.face.error = error.message;
        // Clean up uploaded files even on error
        if (req.files && req.files.faceImages && req.files.faceImages.length > 0) {
          req.files.faceImages.forEach(file => {
            if (file.path && fs.existsSync(file.path)) {
              fs.unlinkSync(file.path);
            }
          });
        }
      }
    }

    // Save doctor record regardless of biometric enrollment results
    await doctor.save();
    console.log(`Doctor ${doctor._id} saved to database`);

    // Determine overall success message
    const allBiometricsSuccess = biometricResults.voice.success &&
                                  biometricResults.keystroke.success &&
                                  biometricResults.mouse.success &&
                                  biometricResults.face.success;

    const message = allBiometricsSuccess
      ? 'Doctor registered successfully with all biometric enrollments'
      : 'Doctor registered successfully. Some biometric enrollments may have failed - you can re-enroll later.';

    res.status(201).json({
      success: true,
      message,
      data: {
        doctor: {
          id: doctor._id,
          firstName: doctor.firstName,
          lastName: doctor.lastName,
          email: doctor.email,
          medicalLicenseNumber: doctor.medicalLicenseNumber,
          specialization: doctor.specialization,
          yearsOfExperience: doctor.yearsOfExperience,
          biometricData: doctor.biometricData
        },
        biometricResults,
        token: generateToken(doctor._id)
      }
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error during registration',
      error: error.message
    });
  }
};

// @desc    Login doctor
// @route   POST /api/auth/login
// @access  Public
export const login = async (req, res) => {
  try {
    const { email, password } = req.body;

    // Validate input
    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: 'Please provide email and password'
      });
    }

    // Check for doctor (include password for comparison)
    const doctor = await Doctor.findOne({ email }).select('+password');

    if (!doctor) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }

    // Check password
    const isMatch = await doctor.comparePassword(password);

    if (!isMatch) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }

    // Update last login
    doctor.lastLogin = new Date();
    await doctor.save();

    res.json({
      success: true,
      message: 'Login successful',
      data: {
        doctor: {
          id: doctor._id,
          firstName: doctor.firstName,
          lastName: doctor.lastName,
          email: doctor.email,
          medicalLicenseNumber: doctor.medicalLicenseNumber,
          specialization: doctor.specialization,
          yearsOfExperience: doctor.yearsOfExperience,
          biometricData: doctor.biometricData
        },
        token: generateToken(doctor._id)
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error during login',
      error: error.message
    });
  }
};

