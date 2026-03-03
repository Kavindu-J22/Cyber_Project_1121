import mongoose from 'mongoose';

const appointmentSchema = new mongoose.Schema({
  appointmentNumber: {
    type: String,
    unique: true
  },
  patientId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Patient',
    required: [true, 'Patient ID is required']
  },
  doctorId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Doctor',
    required: [true, 'Doctor ID is required']
  },
  reason: {
    type: String,
    required: [true, 'Reason for appointment is required'],
    trim: true,
    maxlength: [500, 'Reason cannot exceed 500 characters']
  },
  preferredTime: {
    type: String,
    enum: ['Morning', 'Afternoon', 'Evening', 'Night'],
    required: [true, 'Preferred time is required']
  },
  preferredDates: {
    type: String,
    enum: ['Weekdays', 'Weekends', 'Any'],
    required: [true, 'Preferred dates are required']
  },
  additionalNotes: {
    type: String,
    trim: true,
    maxlength: [1000, 'Additional notes cannot exceed 1000 characters'],
    default: ''
  },
  status: {
    type: String,
    enum: ['Pending', 'Approved', 'Rejected'],
    default: 'Pending'
  },
  // Appointment details (filled when approved)
  appointmentDate: {
    type: Date,
    default: null
  },
  appointmentTimeFrom: {
    type: String,
    default: null
  },
  appointmentTimeTo: {
    type: String,
    default: null
  },
  // Doctor's response
  doctorNote: {
    type: String,
    trim: true,
    maxlength: [1000, 'Doctor note cannot exceed 1000 characters'],
    default: ''
  },
  // Timestamps
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  },
  respondedAt: {
    type: Date,
    default: null
  }
});

// Generate appointment number and update timestamp before saving
appointmentSchema.pre('save', async function(next) {
  // Generate appointment number for new appointments
  if (this.isNew && !this.appointmentNumber) {
    // Generate appointment number: APT-YYYYMMDD-XXXXX
    const date = new Date();
    const dateStr = date.toISOString().slice(0, 10).replace(/-/g, '');
    const randomNum = Math.floor(10000 + Math.random() * 90000);
    this.appointmentNumber = `APT-${dateStr}-${randomNum}`;
  }

  // Update the updatedAt timestamp
  this.updatedAt = Date.now();

  next();
});

// Index for faster queries
appointmentSchema.index({ patientId: 1, createdAt: -1 });
appointmentSchema.index({ doctorId: 1, status: 1, createdAt: -1 });
appointmentSchema.index({ appointmentNumber: 1 });
appointmentSchema.index({ status: 1 });

const Appointment = mongoose.model('Appointment', appointmentSchema);

export default Appointment;

