import Consultation from '../models/Consultation.js';
import Appointment from '../models/Appointment.js';
import Patient from '../models/Patient.js';
import Doctor from '../models/Doctor.js';
import { sendConsultationStartedEmail, sendWaitingAlertEmail } from '../utils/emailService.js';

// Helper function to convert time to Sri Lankan timezone
const getSriLankanTime = () => {
  const now = new Date();
  // Sri Lanka is UTC+5:30
  const sriLankaOffset = 5.5 * 60; // in minutes
  const localOffset = now.getTimezoneOffset(); // in minutes
  const sriLankaTime = new Date(now.getTime() + (sriLankaOffset + localOffset) * 60000);
  return sriLankaTime;
};

// Helper function to check if consultation time is valid
const isConsultationTimeValid = (scheduledDate, scheduledTimeFrom) => {
  const sriLankaTime = getSriLankanTime();
  const [hours, minutes] = scheduledTimeFrom.split(':');
  const scheduledDateTime = new Date(scheduledDate);
  scheduledDateTime.setHours(parseInt(hours), parseInt(minutes), 0, 0);
  
  // Check if current time is within the scheduled time window
  const timeDiff = sriLankaTime - scheduledDateTime;
  const hoursDiff = timeDiff / (1000 * 60 * 60);
  
  return {
    isValid: hoursDiff >= 0 && hoursDiff <= 1, // Within 1 hour after start time
    isPast: hoursDiff > 1, // More than 1 hour past
    isFuture: hoursDiff < 0 // Not yet time
  };
};

// Get patient's confirmed consultations (approved appointments)
export const getPatientConsultations = async (req, res) => {
  try {
    const patientId = req.user.id;
    
    // Get all approved appointments for this patient
    const appointments = await Appointment.find({
      patientId,
      status: 'Approved'
    })
    .populate('doctorId', 'firstName lastName specialization email')
    .sort({ appointmentDate: 1, appointmentTimeFrom: 1 });
    
    // Get consultation status for each appointment
    const consultationsWithStatus = await Promise.all(
      appointments.map(async (apt) => {
        let consultation = await Consultation.findOne({ appointmentId: apt._id });
        
        // Create consultation record if it doesn't exist
        if (!consultation) {
          consultation = await Consultation.create({
            appointmentId: apt._id,
            patientId: apt.patientId,
            doctorId: apt.doctorId,
            scheduledDate: apt.appointmentDate,
            scheduledTimeFrom: apt.appointmentTimeFrom,
            scheduledTimeTo: apt.appointmentTimeTo,
            status: 'Scheduled'
          });
        }
        
        // Check time validity
        const timeCheck = isConsultationTimeValid(apt.appointmentDate, apt.appointmentTimeFrom);
        
        return {
          appointment: apt,
          consultation,
          timeStatus: {
            canJoin: timeCheck.isValid && consultation.status === 'Active',
            canSendAlert: timeCheck.isValid && consultation.status === 'Scheduled',
            isMissed: timeCheck.isPast,
            isFuture: timeCheck.isFuture,
            isWaiting: consultation.status === 'PatientWaiting'
          }
        };
      })
    );
    
    res.status(200).json({
      success: true,
      data: { consultations: consultationsWithStatus }
    });
  } catch (error) {
    console.error('Get patient consultations error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch consultations',
      error: error.message
    });
  }
};

// Get doctor's scheduled consultations (approved appointments)
export const getDoctorConsultations = async (req, res) => {
  try {
    const doctorId = req.user.id;
    
    // Get all approved appointments for this doctor
    const appointments = await Appointment.find({
      doctorId,
      status: 'Approved'
    })
    .populate('patientId', 'fullName email age gender')
    .sort({ appointmentDate: 1, appointmentTimeFrom: 1 });
    
    // Get consultation status for each appointment
    const consultationsWithStatus = await Promise.all(
      appointments.map(async (apt) => {
        let consultation = await Consultation.findOne({ appointmentId: apt._id });
        
        // Create consultation record if it doesn't exist
        if (!consultation) {
          consultation = await Consultation.create({
            appointmentId: apt._id,
            patientId: apt.patientId,
            doctorId: apt.doctorId,
            scheduledDate: apt.appointmentDate,
            scheduledTimeFrom: apt.appointmentTimeFrom,
            scheduledTimeTo: apt.appointmentTimeTo,
            status: 'Scheduled'
          });
        }
        
        // Check time validity
        const timeCheck = isConsultationTimeValid(apt.appointmentDate, apt.appointmentTimeFrom);
        
        return {
          appointment: apt,
          consultation,
          timeStatus: {
            canStart: timeCheck.isValid,
            isMissed: timeCheck.isPast,
            isFuture: timeCheck.isFuture,
            isPatientWaiting: consultation.status === 'PatientWaiting'
          }
        };
      })
    );
    
    res.status(200).json({
      success: true,
      data: { consultations: consultationsWithStatus }
    });
  } catch (error) {
    console.error('Get doctor consultations error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch consultations',
      error: error.message
    });
  }
};

// Patient sends waiting alert to doctor
export const sendWaitingAlert = async (req, res) => {
  try {
    const patientId = req.user.id;
    const { appointmentId } = req.params;

    // Get appointment
    const appointment = await Appointment.findById(appointmentId)
      .populate('patientId', 'fullName email')
      .populate('doctorId', 'firstName lastName email');

    if (!appointment) {
      return res.status(404).json({
        success: false,
        message: 'Appointment not found'
      });
    }

    // Verify patient owns this appointment
    if (appointment.patientId._id.toString() !== patientId) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized'
      });
    }

    // Check if appointment is approved
    if (appointment.status !== 'Approved') {
      return res.status(400).json({
        success: false,
        message: 'Appointment is not approved'
      });
    }

    // Check time validity
    const timeCheck = isConsultationTimeValid(appointment.appointmentDate, appointment.appointmentTimeFrom);
    if (!timeCheck.isValid) {
      return res.status(400).json({
        success: false,
        message: timeCheck.isPast ? 'Consultation time has passed' : 'Consultation time has not started yet'
      });
    }

    // Get or create consultation
    let consultation = await Consultation.findOne({ appointmentId });
    if (!consultation) {
      consultation = await Consultation.create({
        appointmentId: appointment._id,
        patientId: appointment.patientId._id,
        doctorId: appointment.doctorId._id,
        scheduledDate: appointment.appointmentDate,
        scheduledTimeFrom: appointment.appointmentTimeFrom,
        scheduledTimeTo: appointment.appointmentTimeTo,
        status: 'Scheduled'
      });
    }

    // Update consultation status to PatientWaiting
    consultation.status = 'PatientWaiting';
    consultation.waitingAlertSentAt = new Date();
    await consultation.save();

    // Send email to doctor
    const scheduledDate = new Date(appointment.appointmentDate).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
    const scheduledTime = `${appointment.appointmentTimeFrom} - ${appointment.appointmentTimeTo}`;

    await sendWaitingAlertEmail(
      appointment.doctorId.email,
      `${appointment.doctorId.firstName} ${appointment.doctorId.lastName}`,
      appointment.patientId.fullName,
      appointment.appointmentNumber,
      scheduledDate,
      scheduledTime
    );

    res.status(200).json({
      success: true,
      message: 'Waiting alert sent to doctor successfully',
      data: { consultation }
    });
  } catch (error) {
    console.error('Send waiting alert error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to send waiting alert',
      error: error.message
    });
  }
};

// Doctor starts consultation
export const startConsultation = async (req, res) => {
  try {
    const doctorId = req.user.id;
    const { appointmentId } = req.params;

    // Get appointment
    const appointment = await Appointment.findById(appointmentId)
      .populate('patientId', 'fullName email')
      .populate('doctorId', 'firstName lastName email');

    if (!appointment) {
      return res.status(404).json({
        success: false,
        message: 'Appointment not found'
      });
    }

    // Verify doctor owns this appointment
    if (appointment.doctorId._id.toString() !== doctorId) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized'
      });
    }

    // Check if appointment is approved
    if (appointment.status !== 'Approved') {
      return res.status(400).json({
        success: false,
        message: 'Appointment is not approved'
      });
    }

    // Check time validity
    const timeCheck = isConsultationTimeValid(appointment.appointmentDate, appointment.appointmentTimeFrom);
    if (!timeCheck.isValid) {
      return res.status(400).json({
        success: false,
        message: timeCheck.isPast ? 'Consultation time has passed' : 'Consultation time has not started yet'
      });
    }

    // Get or create consultation
    let consultation = await Consultation.findOne({ appointmentId });
    if (!consultation) {
      consultation = await Consultation.create({
        appointmentId: appointment._id,
        patientId: appointment.patientId._id,
        doctorId: appointment.doctorId._id,
        scheduledDate: appointment.appointmentDate,
        scheduledTimeFrom: appointment.appointmentTimeFrom,
        scheduledTimeTo: appointment.appointmentTimeTo,
        status: 'Scheduled'
      });
    }

    // Update consultation status to Active
    consultation.status = 'Active';
    consultation.doctorStartedAt = new Date();
    consultation.consultationRoomId = `room-${appointment._id}`;
    await consultation.save();

    // Send email to patient
    const scheduledDate = new Date(appointment.appointmentDate).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
    const scheduledTime = `${appointment.appointmentTimeFrom} - ${appointment.appointmentTimeTo}`;

    await sendConsultationStartedEmail(
      appointment.patientId.email,
      appointment.patientId.fullName,
      `${appointment.doctorId.firstName} ${appointment.doctorId.lastName}`,
      appointment.appointmentNumber,
      scheduledDate,
      scheduledTime
    );

    res.status(200).json({
      success: true,
      message: 'Consultation started successfully. Patient has been notified.',
      data: { consultation }
    });
  } catch (error) {
    console.error('Start consultation error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to start consultation',
      error: error.message
    });
  }
};

// Get consultation status
export const getConsultationStatus = async (req, res) => {
  try {
    const { appointmentId } = req.params;

    const consultation = await Consultation.findOne({ appointmentId })
      .populate('appointmentId')
      .populate('patientId', 'fullName email')
      .populate('doctorId', 'firstName lastName email');

    if (!consultation) {
      return res.status(404).json({
        success: false,
        message: 'Consultation not found'
      });
    }

    res.status(200).json({
      success: true,
      data: { consultation }
    });
  } catch (error) {
    console.error('Get consultation status error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch consultation status',
      error: error.message
    });
  }
};

