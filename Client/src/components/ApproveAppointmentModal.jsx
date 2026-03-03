import { useState } from 'react';
import { X, Calendar, Clock, StickyNote, CheckCircle } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const ApproveAppointmentModal = ({ appointment, onClose, onSuccess }) => {
  const [formData, setFormData] = useState({
    appointmentDate: '',
    appointmentTimeFrom: '',
    appointmentTimeTo: '',
    doctorNote: ''
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!formData.appointmentDate || !formData.appointmentTimeFrom || !formData.appointmentTimeTo) {
      toast.error('Please fill in all required fields');
      return;
    }

    setLoading(true);

    try {
      await axios.put(`/api/appointments/${appointment._id}/approve`, formData);
      toast.success('Appointment approved! Patient will receive an email notification.');
      onSuccess && onSuccess();
    } catch (error) {
      console.error('Error approving appointment:', error);
      toast.error(error.response?.data?.message || 'Failed to approve appointment');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center">
          <h2 className="text-2xl font-bold text-gray-900">Approve Appointment</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Appointment Info */}
        <div className="px-6 py-4 bg-green-50 border-b border-green-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            {appointment.appointmentNumber}
          </h3>
          <p className="text-sm text-gray-700">Patient: {appointment.patientId.fullName}</p>
          <p className="text-sm text-gray-600">Reason: {appointment.reason}</p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="px-6 py-6 space-y-6">
          {/* Appointment Date */}
          <div>
            <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
              <Calendar className="h-4 w-4 mr-2" />
              Appointment Date <span className="text-red-500 ml-1">*</span>
            </label>
            <input
              type="date"
              value={formData.appointmentDate}
              onChange={(e) => setFormData({ ...formData, appointmentDate: e.target.value })}
              min={new Date().toISOString().split('T')[0]}
              required
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          {/* Time Slot */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
                <Clock className="h-4 w-4 mr-2" />
                From <span className="text-red-500 ml-1">*</span>
              </label>
              <input
                type="time"
                value={formData.appointmentTimeFrom}
                onChange={(e) => setFormData({ ...formData, appointmentTimeFrom: e.target.value })}
                required
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
                <Clock className="h-4 w-4 mr-2" />
                To <span className="text-red-500 ml-1">*</span>
              </label>
              <input
                type="time"
                value={formData.appointmentTimeTo}
                onChange={(e) => setFormData({ ...formData, appointmentTimeTo: e.target.value })}
                required
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* Doctor Note */}
          <div>
            <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
              <StickyNote className="h-4 w-4 mr-2" />
              Note to Patient (Optional)
            </label>
            <textarea
              value={formData.doctorNote}
              onChange={(e) => setFormData({ ...formData, doctorNote: e.target.value })}
              placeholder="Any instructions or information for the patient..."
              rows="4"
              maxLength="1000"
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
            />
            <p className="text-xs text-gray-500 mt-1">{formData.doctorNote.length}/1000 characters</p>
          </div>

          {/* Info Box */}
          <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
            <p className="text-sm text-blue-800">
              <strong>Note:</strong> The patient will receive an email notification with the appointment details once you approve.
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
              className="flex-1 flex items-center justify-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              disabled={loading}
            >
              <CheckCircle className="h-5 w-5 mr-2" />
              {loading ? 'Approving...' : 'Approve Appointment'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ApproveAppointmentModal;

