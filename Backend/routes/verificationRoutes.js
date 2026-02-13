import express from 'express';
import { protect } from '../middleware/auth.js';
import mlService from '../services/mlService.js';
import upload from '../middleware/upload.js';
import fs from 'fs';

const router = express.Router();

// @desc    Verify voice
// @route   POST /api/verification/voice
// @access  Private
router.post('/voice', protect, upload.single('voiceSample'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'Voice sample file is required'
      });
    }

    const result = await mlService.verifyVoice(req.doctor._id.toString(), req.file.path);
    
    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    res.json({
      success: true,
      data: result
    });
  } catch (error) {
    console.error('Voice verification error:', error);
    res.status(500).json({
      success: false,
      message: 'Voice verification failed',
      error: error.message
    });
  }
});

// @desc    Verify keystroke
// @route   POST /api/verification/keystroke
// @access  Private
router.post('/keystroke', protect, async (req, res) => {
  try {
    const { keystrokeSample } = req.body;

    if (!keystrokeSample) {
      return res.status(400).json({
        success: false,
        message: 'Keystroke sample is required'
      });
    }

    const result = await mlService.verifyKeystroke(
      req.doctor._id.toString(),
      keystrokeSample
    );

    res.json({
      success: true,
      data: result
    });
  } catch (error) {
    console.error('Keystroke verification error:', error);
    res.status(500).json({
      success: false,
      message: 'Keystroke verification failed',
      error: error.message
    });
  }
});

// @desc    Verify mouse movement
// @route   POST /api/verification/mouse
// @access  Private
router.post('/mouse', protect, async (req, res) => {
  try {
    const { mouseEvents } = req.body;

    if (!mouseEvents || !Array.isArray(mouseEvents)) {
      return res.status(400).json({
        success: false,
        message: 'Mouse events array is required'
      });
    }

    const result = await mlService.verifyMouse(
      req.doctor._id.toString(),
      mouseEvents
    );

    res.json({
      success: true,
      data: result
    });
  } catch (error) {
    console.error('Mouse verification error:', error);
    res.status(500).json({
      success: false,
      message: 'Mouse verification failed',
      error: error.message
    });
  }
});

// @desc    Verify face
// @route   POST /api/verification/face
// @access  Private
router.post('/face', protect, upload.single('faceSample'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'Face sample file is required'
      });
    }

    const result = await mlService.verifyFace(req.doctor._id.toString(), req.file.path);
    
    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    res.json({
      success: true,
      data: result
    });
  } catch (error) {
    console.error('Face verification error:', error);
    res.status(500).json({
      success: false,
      message: 'Face verification failed',
      error: error.message
    });
  }
});

// @desc    Check ML services health
// @route   GET /api/verification/health
// @access  Private
router.get('/health', protect, async (req, res) => {
  try {
    const health = await mlService.checkHealth();

    res.json({
      success: true,
      data: health
    });
  } catch (error) {
    console.error('Health check error:', error);
    res.status(500).json({
      success: false,
      message: 'Health check failed',
      error: error.message
    });
  }
});

export default router;

