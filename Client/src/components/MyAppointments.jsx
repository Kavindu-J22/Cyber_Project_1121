import { useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import { Calendar, Clock, User, FileText, X, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

const MyAppointments = () => {
  const [appointments, setAppointments] = useState([]);
  const [filter, setFilter] = useState('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAppointments();
  }, [filter]);

  const fetchAppointments = async () => {
    setLoading(true);
    try {
      const url = filter === 'all'
        ? '/api/appointments/my-appointments'
        : `/api/appointments/my-appointments?status=${filter}`;

      const response = await axios.get(url);
      setAppointments(response.data.data.appointments || []);
    } catch (error) {
      console.error('Error fetching appointments:', error);
      toast.error('Failed to load appointments');
      setAppointments([]);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async (appointmentId, appointmentNumber) => {
    if (!window.confirm(`Are you sure you want to cancel appointment ${appointmentNumber}?`)) {
      return;
    }

    try {
      await axios.delete(`/api/appointments/${appointmentId}`);
      toast.success('Appointment cancelled successfully');
      fetchAppointments();
    } catch (error) {
      console.error('Error cancelling appointment:', error);
      toast.error(error.response?.data?.message || 'Failed to cancel appointment');
    }
  };

  const getStatusBadge = (status) => {
    const badges = {
      Pending: { bg: 'bg-yellow-100', text: 'text-yellow-800', icon: AlertCircle },
      Approved: { bg: 'bg-green-100', text: 'text-green-800', icon: CheckCircle },
      Rejected: { bg: 'bg-red-100', text: 'text-red-800', icon: XCircle }
    };

    const badge = badges[status] || badges.Pending;
    const Icon = badge.icon;

    return (
      <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${badge.bg} ${badge.text}`}>
        <Icon className="h-4 w-4 mr-1" />
        {status}
      </span>
    );
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-6">My Appointments</h2>

      {/* Filter Tabs */}
      <div className="flex gap-2 mb-6 border-b border-gray-200">
        {['all', 'Pending', 'Approved', 'Rejected'].map((status) => (
          <button
            key={status}
            onClick={() => setFilter(status)}
            className={`px-4 py-2 font-medium text-sm transition-colors border-b-2 ${
              filter === status
                ? 'border-primary-600 text-primary-600'
                : 'border-transparent text-gray-600 hover:text-gray-900'
            }`}
          >
            {status === 'all' ? 'All' : status}
          </button>
        ))}
      </div>

      {/* Loading State */}
      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading appointments...</p>
        </div>
      ) : appointments.length === 0 ? (
        <div className="text-center py-12">
          <Calendar className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">
            {filter === 'all' ? 'No appointments yet' : `No ${filter.toLowerCase()} appointments`}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {appointments.map((apt) => (
            <div
              key={apt._id}
              className={`border-l-4 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow ${
                apt.status === 'Pending' ? 'border-yellow-400 bg-yellow-50' :
                apt.status === 'Approved' ? 'border-green-400 bg-green-50' :
                'border-red-400 bg-red-50'
              }`}
            >
              {/* Header */}
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-bold text-gray-900">{apt.appointmentNumber}</h3>
                  <p className="text-sm text-gray-600">
                    Created on {formatDate(apt.createdAt)}
                  </p>
                </div>
                {getStatusBadge(apt.status)}
              </div>

              {/* Doctor Info */}
              <div className="mb-4 pb-4 border-b border-gray-200">
                <div className="flex items-center text-gray-700 mb-2">
                  <User className="h-5 w-5 mr-2 text-primary-600" />
                  <span className="font-medium">
                    Dr. {apt.doctorId.firstName} {apt.doctorId.lastName}
                  </span>
                </div>
                <p className="text-sm text-gray-600 ml-7">{apt.doctorId.specialization}</p>
              </div>

              {/* Appointment Details */}
              <div className="space-y-3">
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-1">Reason:</p>
                  <p className="text-sm text-gray-600">{apt.reason}</p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-700">Preferred Time:</p>
                    <p className="text-sm text-gray-600">{apt.preferredTime}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-700">Preferred Dates:</p>
                    <p className="text-sm text-gray-600">{apt.preferredDates}</p>
                  </div>
                </div>

                {apt.additionalNotes && (
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-1">Additional Notes:</p>
                    <p className="text-sm text-gray-600">{apt.additionalNotes}</p>
                  </div>
                )}

                {/* Approved Details */}
                {apt.status === 'Approved' && (
                  <div className="mt-4 pt-4 border-t border-green-200 bg-green-100 -mx-6 -mb-6 px-6 py-4 rounded-b-lg">
                    <p className="text-sm font-semibold text-green-900 mb-2">Appointment Scheduled:</p>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center text-green-800">
                        <Calendar className="h-4 w-4 mr-2" />
                        <span className="text-sm font-medium">{formatDate(apt.appointmentDate)}</span>
                      </div>
                      <div className="flex items-center text-green-800">
                        <Clock className="h-4 w-4 mr-2" />
                        <span className="text-sm font-medium">{apt.appointmentTimeFrom} - {apt.appointmentTimeTo}</span>
                      </div>
                    </div>
                    {apt.doctorNote && (
                      <div className="mt-3">
                        <p className="text-sm font-medium text-green-900 mb-1">Doctor's Note:</p>
                        <p className="text-sm text-green-800">{apt.doctorNote}</p>
                      </div>
                    )}
                  </div>
                )}

                {/* Rejected Details */}
                {apt.status === 'Rejected' && apt.doctorNote && (
                  <div className="mt-4 pt-4 border-t border-red-200 bg-red-100 -mx-6 -mb-6 px-6 py-4 rounded-b-lg">
                    <p className="text-sm font-semibold text-red-900 mb-1">Reason for Rejection:</p>
                    <p className="text-sm text-red-800">{apt.doctorNote}</p>
                  </div>
                )}

                {/* Cancel Button for Pending */}
                {apt.status === 'Pending' && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <button
                      onClick={() => handleCancel(apt._id, apt.appointmentNumber)}
                      className="flex items-center px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors text-sm"
                    >
                      <X className="h-4 w-4 mr-2" />
                      Cancel Appointment
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MyAppointments;

