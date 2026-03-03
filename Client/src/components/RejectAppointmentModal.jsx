import { useState } from 'react';
import { X, StickyNote, XCircle } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const RejectAppointmentModal = ({ appointment, onClose, onSuccess }) => {
  const [doctorNote, setDoctorNote] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await axios.put(`/api/appointments/${appointment._id}/reject`, { doctorNote });
      toast.success('Appointment rejected. Patient will receive an email notification.');
      onSuccess && onSuccess();
    } catch (error) {
      console.error('Error rejecting appointment:', error);
      toast.error(error.response?.data?.message || 'Failed to reject appointment');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center rounded-t-lg">
          <h2 className="text-2xl font-bold text-gray-900">Reject Appointment</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Appointment Info */}
        <div className="px-6 py-4 bg-red-50 border-b border-red-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            {appointment.appointmentNumber}
          </h3>
          <p className="text-sm text-gray-700">Patient: {appointment.patientId.fullName}</p>
          <p className="text-sm text-gray-600">Reason: {appointment.reason}</p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="px-6 py-6 space-y-6">
          {/* Doctor Note */}
          <div>
            <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
              <StickyNote className="h-4 w-4 mr-2" />
              Reason for Rejection (Optional)
            </label>
            <textarea
              value={doctorNote}
              onChange={(e) => setDoctorNote(e.target.value)}
              placeholder="Explain why you're rejecting this appointment (optional but recommended)..."
              rows="5"
              maxLength="1000"
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
            />
            <p className="text-xs text-gray-500 mt-1">{doctorNote.length}/1000 characters</p>
          </div>

          {/* Warning Box */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
            <p className="text-sm text-yellow-800">
              <strong>Warning:</strong> The patient will receive an email notification about this rejection. 
              {doctorNote.trim() ? ' Your note will be included in the email.' : ' Consider adding a note to explain the reason.'}
            </p>
          </div>

          {/* Buttons */}
          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 flex items-center justify-center px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              disabled={loading}
            >
              <XCircle className="h-5 w-5 mr-2" />
              {loading ? 'Rejecting...' : 'Reject Appointment'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RejectAppointmentModal;

