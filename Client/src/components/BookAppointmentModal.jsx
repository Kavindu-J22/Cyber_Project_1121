import { useState } from 'react';
import { X, Calendar, Clock, FileText, StickyNote } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const BookAppointmentModal = ({ doctor, onClose, onSuccess }) => {
  const [formData, setFormData] = useState({
    reason: '',
    preferredTime: 'Morning',
    preferredDates: 'Weekdays',
    additionalNotes: ''
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.reason.trim()) {
      toast.error('Please provide a reason for the appointment');
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post('/api/appointments', {
        doctorId: doctor._id,
        ...formData
      });

      toast.success(`Appointment created! Number: ${response.data.data.appointment.appointmentNumber}`);
      onSuccess && onSuccess(response.data.data.appointment);
      onClose();
    } catch (error) {
      console.error('Error creating appointment:', error);
      toast.error(error.response?.data?.message || 'Failed to create appointment');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center">
          <h2 className="text-2xl font-bold text-gray-900">Book Appointment</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Doctor Info */}
        <div className="px-6 py-4 bg-primary-50 border-b border-primary-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Dr. {doctor.firstName} {doctor.lastName}
          </h3>
          <p className="text-sm text-gray-600">{doctor.specialization}</p>
          <p className="text-sm text-gray-600">{doctor.yearsOfExperience} years of experience</p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="px-6 py-6 space-y-6">
          {/* Reason */}
          <div>
            <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
              <FileText className="h-4 w-4 mr-2" />
              Reason for Appointment <span className="text-red-500 ml-1">*</span>
            </label>
            <textarea
              value={formData.reason}
              onChange={(e) => setFormData({ ...formData, reason: e.target.value })}
              placeholder="Please describe your symptoms or reason for consultation..."
              rows="4"
              maxLength="500"
              required
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
            />
            <p className="text-xs text-gray-500 mt-1">{formData.reason.length}/500 characters</p>
          </div>

          {/* Preferred Time */}
          <div>
            <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
              <Clock className="h-4 w-4 mr-2" />
              Preferred Time
            </label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {['Morning', 'Afternoon', 'Evening', 'Night'].map((time) => (
                <label
                  key={time}
                  className={`flex items-center justify-center px-4 py-3 border-2 rounded-md cursor-pointer transition-all ${
                    formData.preferredTime === time
                      ? 'border-primary-600 bg-primary-50 text-primary-700'
                      : 'border-gray-300 hover:border-gray-400'
                  }`}
                >
                  <input
                    type="radio"
                    name="preferredTime"
                    value={time}
                    checked={formData.preferredTime === time}
                    onChange={(e) => setFormData({ ...formData, preferredTime: e.target.value })}
                    className="sr-only"
                  />
                  <span className="text-sm font-medium">{time}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Preferred Dates */}
          <div>
            <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
              <Calendar className="h-4 w-4 mr-2" />
              Preferred Dates
            </label>
            <div className="grid grid-cols-3 gap-3">
              {['Weekdays', 'Weekends', 'Any'].map((dateType) => (
                <label
                  key={dateType}
                  className={`flex items-center justify-center px-4 py-3 border-2 rounded-md cursor-pointer transition-all ${
                    formData.preferredDates === dateType
                      ? 'border-primary-600 bg-primary-50 text-primary-700'
                      : 'border-gray-300 hover:border-gray-400'
                  }`}
                >
                  <input
                    type="radio"
                    name="preferredDates"
                    value={dateType}
                    checked={formData.preferredDates === dateType}
                    onChange={(e) => setFormData({ ...formData, preferredDates: e.target.value })}
                    className="sr-only"
                  />
                  <span className="text-sm font-medium">{dateType}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Additional Notes */}
          <div>
            <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
              <StickyNote className="h-4 w-4 mr-2" />
              Additional Notes (Optional)
            </label>
            <textarea
              value={formData.additionalNotes}
              onChange={(e) => setFormData({ ...formData, additionalNotes: e.target.value })}
              placeholder="Any additional information you'd like to share..."
              rows="3"
              maxLength="1000"
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
            />
            <p className="text-xs text-gray-500 mt-1">{formData.additionalNotes.length}/1000 characters</p>
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
              className="flex-1 px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              disabled={loading}
            >
              {loading ? 'Submitting...' : 'Book Appointment'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default BookAppointmentModal;

