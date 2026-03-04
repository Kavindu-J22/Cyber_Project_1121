import mongoose from 'mongoose';

const otpSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    lowercase: true,
    trim: true
  },
  otp: {
    type: String,
    required: true
  },
  userType: {
    type: String,
    enum: ['doctor', 'patient', 'consultation'],
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now,
    expires: 600 // OTP expires after 10 minutes (600 seconds)
  }
});

// Index for faster queries
otpSchema.index({ email: 1, userType: 1 });

const OTP = mongoose.model('OTP', otpSchema);

export default OTP;

