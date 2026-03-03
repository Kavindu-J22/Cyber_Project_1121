import Appointment from '../models/Appointment.js';
import Patient from '../models/Patient.js';
import Doctor from '../models/Doctor.js';
import { sendAppointmentApprovalEmail, sendAppointmentRejectionEmail } from '../utils/emailService.js';

// @desc    Create new appointment (Patient)
// @route   POST /api/appointments
// @access  Private (Patient)
export const createAppointment = async (req, res) => {
  try {
    const { doctorId, reason, preferredTime, preferredDates, additionalNotes } = req.body;

    // Validate required fields
    if (!doctorId || !reason || !preferredTime || !preferredDates) {
      return res.status(400).json({
        success: false,
        message: 'Please provide all required fields'
      });
    }

    // Verify doctor exists and is active
    const doctor = await Doctor.findById(doctorId);
    if (!doctor) {
      return res.status(404).json({
        success: false,
        message: 'Doctor not found'
      });
    }

    if (!doctor.isActive) {
      return res.status(400).json({
        success: false,
        message: 'Doctor is not currently accepting appointments'
      });
    }

    // Create appointment
    const appointment = await Appointment.create({
      patientId: req.user.id,
      doctorId,
      reason,
      preferredTime,
      preferredDates,
      additionalNotes: additionalNotes || ''
    });

    // Populate patient and doctor details
    await appointment.populate('patientId', 'fullName email age gender');
    await appointment.populate('doctorId', 'firstName lastName email specialization');

    res.status(201).json({
      success: true,
      message: 'Appointment request created successfully',
      data: { appointment }
    });
  } catch (error) {
    console.error('Create appointment error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Get patient's appointments
// @route   GET /api/appointments/my-appointments
// @access  Private (Patient)
export const getMyAppointments = async (req, res) => {
  try {
    const { status } = req.query;
    
    const query = { patientId: req.user.id };
    if (status && ['Pending', 'Approved', 'Rejected'].includes(status)) {
      query.status = status;
    }

    const appointments = await Appointment.find(query)
      .populate('doctorId', 'firstName lastName email specialization profileImage')
      .sort({ createdAt: -1 });

    res.json({
      success: true,
      count: appointments.length,
      data: { appointments }
    });
  } catch (error) {
    console.error('Get my appointments error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Get doctor's appointments
// @route   GET /api/appointments/doctor-appointments
// @access  Private (Doctor)
export const getDoctorAppointments = async (req, res) => {
  try {
    const { status } = req.query;
    
    const query = { doctorId: req.user.id };
    if (status && ['Pending', 'Approved', 'Rejected'].includes(status)) {
      query.status = status;
    } else {
      // Default to Pending if no status specified
      query.status = 'Pending';
    }

    const appointments = await Appointment.find(query)
      .populate('patientId', 'fullName email age gender')
      .sort({ createdAt: -1 });

    res.json({
      success: true,
      count: appointments.length,
      data: { appointments }
    });
  } catch (error) {
    console.error('Get doctor appointments error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Get all appointments (Admin)
// @route   GET /api/appointments/all
// @access  Private (Admin)
export const getAllAppointments = async (req, res) => {
  try {
    const { status, search, doctorId } = req.query;
    
    const query = {};
    
    // Filter by status
    if (status && ['Pending', 'Approved', 'Rejected'].includes(status)) {
      query.status = status;
    }
    
    // Filter by doctor
    if (doctorId) {
      query.doctorId = doctorId;
    }

    let appointments = await Appointment.find(query)
      .populate('patientId', 'fullName email age gender')
      .populate('doctorId', 'firstName lastName email specialization profileImage')
      .sort({ createdAt: -1 });

    // Search filter
    if (search) {
      const searchLower = search.toLowerCase();
      appointments = appointments.filter(apt => 
        apt.appointmentNumber.toLowerCase().includes(searchLower) ||
        apt.patientId?.fullName.toLowerCase().includes(searchLower) ||
        `${apt.doctorId?.firstName} ${apt.doctorId?.lastName}`.toLowerCase().includes(searchLower) ||
        apt.reason.toLowerCase().includes(searchLower)
      );
    }

    res.json({
      success: true,
      count: appointments.length,
      data: { appointments }
    });
  } catch (error) {
    console.error('Get all appointments error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Get single appointment by ID
// @route   GET /api/appointments/:id
// @access  Private
export const getAppointmentById = async (req, res) => {
  try {
    const appointment = await Appointment.findById(req.params.id)
      .populate('patientId', 'fullName email age gender')
      .populate('doctorId', 'firstName lastName email specialization profileImage');

    if (!appointment) {
      return res.status(404).json({
        success: false,
        message: 'Appointment not found'
      });
    }

    // Check authorization
    const userRole = req.userRole;
    if (userRole === 'patient' && appointment.patientId._id.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to view this appointment'
      });
    }

    if (userRole === 'doctor' && appointment.doctorId._id.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to view this appointment'
      });
    }

    res.json({
      success: true,
      data: { appointment }
    });
  } catch (error) {
    console.error('Get appointment error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Approve appointment (Doctor)
// @route   PUT /api/appointments/:id/approve
// @access  Private (Doctor)
export const approveAppointment = async (req, res) => {
  try {
    const { appointmentDate, appointmentTimeFrom, appointmentTimeTo, doctorNote } = req.body;

    // Validate required fields
    if (!appointmentDate || !appointmentTimeFrom || !appointmentTimeTo) {
      return res.status(400).json({
        success: false,
        message: 'Please provide appointment date and time'
      });
    }

    const appointment = await Appointment.findById(req.params.id)
      .populate('patientId', 'fullName email')
      .populate('doctorId', 'firstName lastName');

    if (!appointment) {
      return res.status(404).json({
        success: false,
        message: 'Appointment not found'
      });
    }

    // Check if doctor owns this appointment
    if (appointment.doctorId._id.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to approve this appointment'
      });
    }

    // Check if already processed
    if (appointment.status !== 'Pending') {
      return res.status(400).json({
        success: false,
        message: `Appointment is already ${appointment.status.toLowerCase()}`
      });
    }

    // Update appointment
    appointment.status = 'Approved';
    appointment.appointmentDate = appointmentDate;
    appointment.appointmentTimeFrom = appointmentTimeFrom;
    appointment.appointmentTimeTo = appointmentTimeTo;
    appointment.doctorNote = doctorNote || '';
    appointment.respondedAt = new Date();

    await appointment.save();

    // Send approval email to patient
    try {
      await sendAppointmentApprovalEmail(appointment.patientId.email, {
        appointmentNumber: appointment.appointmentNumber,
        doctorName: `${appointment.doctorId.firstName} ${appointment.doctorId.lastName}`,
        appointmentDate: appointment.appointmentDate,
        appointmentTimeFrom: appointment.appointmentTimeFrom,
        appointmentTimeTo: appointment.appointmentTimeTo,
        doctorNote: appointment.doctorNote
      });
    } catch (emailError) {
      console.error('Error sending approval email:', emailError);
      // Continue even if email fails
    }

    res.json({
      success: true,
      message: 'Appointment approved successfully',
      data: { appointment }
    });
  } catch (error) {
    console.error('Approve appointment error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Reject appointment (Doctor)
// @route   PUT /api/appointments/:id/reject
// @access  Private (Doctor)
export const rejectAppointment = async (req, res) => {
  try {
    const { doctorNote } = req.body;

    const appointment = await Appointment.findById(req.params.id)
      .populate('patientId', 'fullName email')
      .populate('doctorId', 'firstName lastName');

    if (!appointment) {
      return res.status(404).json({
        success: false,
        message: 'Appointment not found'
      });
    }

    // Check if doctor owns this appointment
    if (appointment.doctorId._id.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to reject this appointment'
      });
    }

    // Check if already processed
    if (appointment.status !== 'Pending') {
      return res.status(400).json({
        success: false,
        message: `Appointment is already ${appointment.status.toLowerCase()}`
      });
    }

    // Update appointment
    appointment.status = 'Rejected';
    appointment.doctorNote = doctorNote || '';
    appointment.respondedAt = new Date();

    await appointment.save();

    // Send rejection email to patient
    try {
      await sendAppointmentRejectionEmail(appointment.patientId.email, {
        appointmentNumber: appointment.appointmentNumber,
        doctorName: `${appointment.doctorId.firstName} ${appointment.doctorId.lastName}`,
        reason: appointment.reason,
        doctorNote: appointment.doctorNote
      });
    } catch (emailError) {
      console.error('Error sending rejection email:', emailError);
      // Continue even if email fails
    }

    res.json({
      success: true,
      message: 'Appointment rejected',
      data: { appointment }
    });
  } catch (error) {
    console.error('Reject appointment error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

// @desc    Cancel appointment (Patient)
// @route   DELETE /api/appointments/:id
// @access  Private (Patient)
export const cancelAppointment = async (req, res) => {
  try {
    const appointment = await Appointment.findById(req.params.id);

    if (!appointment) {
      return res.status(404).json({
        success: false,
        message: 'Appointment not found'
      });
    }

    // Check if patient owns this appointment
    if (appointment.patientId.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to cancel this appointment'
      });
    }

    // Only allow canceling pending appointments
    if (appointment.status !== 'Pending') {
      return res.status(400).json({
        success: false,
        message: 'Only pending appointments can be cancelled'
      });
    }

    await appointment.deleteOne();

    res.json({
      success: true,
      message: 'Appointment cancelled successfully'
    });
  } catch (error) {
    console.error('Cancel appointment error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  }
};

