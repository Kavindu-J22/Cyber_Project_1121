import jwt from 'jsonwebtoken';
import Doctor from '../models/Doctor.js';
import Patient from '../models/Patient.js';

export const protect = async (req, res, next) => {
  let token;

  if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
    try {
      // Get token from header
      token = req.headers.authorization.split(' ')[1];

      // Verify token
      const decoded = jwt.verify(token, process.env.JWT_SECRET);

      // Check if admin
      if (decoded.role === 'admin') {
        req.user = { id: 'admin', role: 'admin', email: 'admin@gmail.com' };
        req.userRole = 'admin';
        return next();
      }

      // Get user based on role
      if (decoded.role === 'doctor') {
        req.user = await Doctor.findById(decoded.id).select('-password');
        req.doctor = req.user; // For backward compatibility
        req.userRole = 'doctor';
      } else if (decoded.role === 'patient') {
        req.user = await Patient.findById(decoded.id).select('-password');
        req.patient = req.user;
        req.userRole = 'patient';
      }

      if (!req.user) {
        return res.status(401).json({
          success: false,
          message: 'Not authorized, user not found'
        });
      }

      next();
    } catch (error) {
      console.error('Auth middleware error:', error);
      return res.status(401).json({
        success: false,
        message: 'Not authorized, token failed'
      });
    }
  }

  if (!token) {
    return res.status(401).json({
      success: false,
      message: 'Not authorized, no token'
    });
  }
};

// Generate JWT Token with role
export const generateToken = (id, role = 'doctor') => {
  return jwt.sign({ id, role }, process.env.JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRE
  });
};

