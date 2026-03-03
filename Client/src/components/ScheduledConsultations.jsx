import { useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import { Calendar, Clock, Video, User, AlertCircle, CheckCircle, XCircle } from 'lucide-react';

const ScheduledConsultations = () => {
  const [consultations, setConsultations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(null);

  useEffect(() => {
    fetchConsultations();
    // Refresh every 30 seconds to update button states
    const interval = setInterval(fetchConsultations, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchConsultations = async () => {
    try {
      const response = await axios.get('/api/consultations/doctor/my-consultations');
      setConsultations(response.data.data.consultations || []);
    } catch (error) {
      console.error('Error fetching consultations:', error);
      toast.error('Failed to load consultations');
      setConsultations([]);
    } finally {
      setLoading(false);
    }
  };

  const handleStartConsultation = async (appointmentId) => {
    setStarting(appointmentId);
    try {
      const response = await axios.post(`/api/consultations/doctor/${appointmentId}/start`);
      toast.success('Consultation started! Patient has been notified via email.');

      // Navigate to meeting room
      const consultationRoomId = response.data.data.consultation.consultationRoomId;
      window.location.href = `/meeting/${consultationRoomId}`;
    } catch (error) {
      console.error('Error starting consultation:', error);
      toast.error(error.response?.data?.message || 'Failed to start consultation');
      setStarting(null);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const getButtonContent = (consultation) => {
    const { timeStatus, consultation: consult, appointment } = consultation;

    // Check if consultation has ended
    if (consult?.status === 'Completed') {
      return (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-center">
          <CheckCircle className="h-8 w-8 text-gray-600 mx-auto mb-2" />
          <p className="text-gray-800 font-semibold">Consultation Ended</p>
          <p className="text-sm text-gray-600 mt-1">
            This consultation has been completed.
          </p>
        </div>
      );
    }

    if (timeStatus.isMissed) {
      return (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
          <XCircle className="h-8 w-8 text-red-600 mx-auto mb-2" />
          <p className="text-red-800 font-semibold">Consultation Missed</p>
          <p className="text-sm text-red-600 mt-1">
            The consultation time has passed (more than 1 hour).
          </p>
        </div>
      );
    }

    if (timeStatus.isFuture) {
      return (
        <button
          disabled
          className="w-full px-6 py-3 bg-gray-300 text-gray-600 rounded-lg font-semibold cursor-not-allowed flex items-center justify-center"
        >
          <Clock className="h-5 w-5 mr-2" />
          Scheduled
        </button>
      );
    }

    if (timeStatus.canStart) {
      return (
        <div>
          {timeStatus.isPatientWaiting && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-3 text-center">
              <AlertCircle className="h-6 w-6 text-yellow-600 mx-auto mb-1" />
              <p className="text-yellow-800 font-semibold text-sm">Patient is Waiting for You!</p>
            </div>
          )}
          <button
            onClick={() => handleStartConsultation(appointment._id)}
            disabled={starting === appointment._id}
            className="w-full px-6 py-3 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition-colors flex items-center justify-center disabled:opacity-50"
          >
            <Video className="h-5 w-5 mr-2" />
            {starting === appointment._id ? 'Starting...' : 'Start Consultation'}
          </button>
        </div>
      );
    }

    return (
      <button
        disabled
        className="w-full px-6 py-3 bg-gray-300 text-gray-600 rounded-lg font-semibold cursor-not-allowed flex items-center justify-center"
      >
        <Clock className="h-5 w-5 mr-2" />
        Scheduled
      </button>
    );
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
        <p className="mt-4 text-gray-600">Loading consultations...</p>
      </div>
    );
  }

  // Separate active/upcoming consultations from ended ones
  const activeConsultations = consultations.filter(c => c.consultation?.status !== 'Completed');
  const endedConsultations = consultations.filter(c => c.consultation?.status === 'Completed');

  if (consultations.length === 0) {
    return (
      <div className="text-center py-12">
        <Video className="h-16 w-16 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-600">No scheduled consultations</p>
        <p className="text-sm text-gray-500 mt-2">
          Approved appointments will appear here when it's time for consultation
        </p>
      </div>
    );
  }

  const renderConsultationCard = ({ appointment, consultation, timeStatus }) => (
    <div
      key={appointment._id}
      className="bg-white border-l-4 border-primary-400 rounded-lg shadow-sm p-6"
    >
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-bold text-gray-900">{appointment.appointmentNumber}</h3>
          <p className="text-sm text-gray-600">{appointment.patientId.fullName}</p>
          <p className="text-xs text-gray-500">
            Age: {appointment.patientId.age} | Gender: {appointment.patientId.gender}
          </p>
        </div>
        <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
          <CheckCircle className="h-4 w-4 mr-1" />
          Approved
        </span>
      </div>

      {/* Consultation Details */}
      <div className="grid grid-cols-2 gap-4 mb-4 pb-4 border-b border-gray-200">
        <div className="flex items-center text-gray-700">
          <Calendar className="h-5 w-5 mr-2 text-primary-600" />
          <div>
            <p className="text-xs text-gray-500">Date</p>
            <p className="text-sm font-medium">{formatDate(appointment.appointmentDate)}</p>
          </div>
        </div>
        <div className="flex items-center text-gray-700">
          <Clock className="h-5 w-5 mr-2 text-primary-600" />
          <div>
            <p className="text-xs text-gray-500">Time</p>
            <p className="text-sm font-medium">
              {appointment.appointmentTimeFrom} - {appointment.appointmentTimeTo}
            </p>
          </div>
        </div>
      </div>

      {/* Patient Info */}
      <div className="mb-4">
        <p className="text-sm font-medium text-gray-700 mb-1">Patient Email:</p>
        <p className="text-sm text-gray-600">{appointment.patientId.email}</p>
      </div>

      {/* Reason */}
      <div className="mb-4">
        <p className="text-sm font-medium text-gray-700 mb-1">Reason for Consultation:</p>
        <p className="text-sm text-gray-600">{appointment.reason}</p>
      </div>

      {/* Action Button */}
      {getButtonContent({ appointment, consultation, timeStatus })}
    </div>
  );

  return (
    <div className="space-y-8">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start">
          <AlertCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
          <div className="text-sm text-blue-800">
            <p className="font-semibold mb-1">Consultation Guidelines:</p>
            <ul className="list-disc list-inside space-y-1">
              <li>Start consultations at the scheduled time</li>
              <li>If patient sends a waiting alert, you'll be notified via email</li>
              <li>Click "Start Consultation" to begin and notify the patient</li>
              <li>Consultations must start within 1 hour of scheduled time</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Active/Upcoming Consultations */}
      {activeConsultations.length > 0 && (
        <div>
          <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            <Video className="h-6 w-6 text-primary-600" />
            Active & Upcoming Consultations
            <span className="text-sm font-normal text-gray-500">({activeConsultations.length})</span>
          </h2>
          <div className="space-y-6">
            {activeConsultations.map(renderConsultationCard)}
          </div>
        </div>
      )}

      {/* Ended Consultations */}
      {endedConsultations.length > 0 && (
        <div>
          <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            <CheckCircle className="h-6 w-6 text-gray-600" />
            Ended Consultations
            <span className="text-sm font-normal text-gray-500">({endedConsultations.length})</span>
          </h2>
          <div className="space-y-6">
            {endedConsultations.map(renderConsultationCard)}
          </div>
        </div>
      )}

      {activeConsultations.length === 0 && endedConsultations.length === 0 && (
        <div className="text-center py-12">
          <Video className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">No consultations found</p>
        </div>
      )}
    </div>
  );
};

export default ScheduledConsultations;

