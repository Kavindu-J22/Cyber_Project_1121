import mongoose from 'mongoose';

const consultationSchema = new mongoose.Schema({
  appointmentId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Appointment',
    required: true,
    unique: true
  },
  patientId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Patient',
    required: true
  },
  doctorId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Doctor',
    required: true
  },
  status: {
    type: String,
    enum: ['Scheduled', 'PatientWaiting', 'Active', 'Completed', 'Missed'],
    default: 'Scheduled'
  },
  scheduledDate: {
    type: Date,
    required: true
  },
  scheduledTimeFrom: {
    type: String,
    required: true
  },
  scheduledTimeTo: {
    type: String,
    required: true
  },
  doctorStartedAt: {
    type: Date,
    default: null
  },
  patientJoinedAt: {
    type: Date,
    default: null
  },
  endedAt: {
    type: Date,
    default: null
  },
  waitingAlertSentAt: {
    type: Date,
    default: null
  },
  consultationRoomId: {
    type: String,
    default: null
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

// Update timestamp before saving
consultationSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// Index for faster queries
consultationSchema.index({ appointmentId: 1 });
consultationSchema.index({ patientId: 1, status: 1 });
consultationSchema.index({ doctorId: 1, status: 1 });
consultationSchema.index({ scheduledDate: 1, status: 1 });

const Consultation = mongoose.model('Consultation', consultationSchema);

export default Consultation;

