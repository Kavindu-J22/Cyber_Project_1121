import { useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import { Calendar, Clock, User, FileText, CheckCircle, XCircle, AlertCircle, Search, Filter } from 'lucide-react';

const AdminAppointments = () => {
  const [appointments, setAppointments] = useState([]);
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [doctorFilter, setDoctorFilter] = useState('all');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [appointmentsRes, doctorsRes] = await Promise.all([
        axios.get('/api/appointments/all'),
        axios.get('/api/doctors')
      ]);
      setAppointments(appointmentsRes.data.data.appointments || []);
      setDoctors(doctorsRes.data.data.doctors || []);
    } catch (error) {
      console.error('Error fetching data:', error);
      toast.error('Failed to load appointments');
      setAppointments([]);
      setDoctors([]);
    } finally {
      setLoading(false);
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

  // Filter appointments
  const filteredAppointments = appointments.filter(apt => {
    const matchesSearch = searchTerm === '' ||
      apt.appointmentNumber.toLowerCase().includes(searchTerm.toLowerCase()) ||
      apt.patientId.fullName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      `${apt.doctorId.firstName} ${apt.doctorId.lastName}`.toLowerCase().includes(searchTerm.toLowerCase()) ||
      apt.reason.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesStatus = statusFilter === 'all' || apt.status === statusFilter;
    const matchesDoctor = doctorFilter === 'all' || apt.doctorId._id === doctorFilter;

    return matchesSearch && matchesStatus && matchesDoctor;
  });

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-6">All Appointments</h2>

      {/* Filters */}
      <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search appointments..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Status Filter */}
        <div className="relative">
          <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent appearance-none"
          >
            <option value="all">All Status</option>
            <option value="Pending">Pending</option>
            <option value="Approved">Approved</option>
            <option value="Rejected">Rejected</option>
          </select>
        </div>

        {/* Doctor Filter */}
        <div className="relative">
          <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <select
            value={doctorFilter}
            onChange={(e) => setDoctorFilter(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent appearance-none"
          >
            <option value="all">All Doctors</option>
            {doctors.map(doctor => (
              <option key={doctor._id} value={doctor._id}>
                Dr. {doctor.firstName} {doctor.lastName}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Results Count */}
      <div className="mb-4 text-sm text-gray-600">
        Showing {filteredAppointments.length} of {appointments.length} appointments
      </div>

      {/* Loading State */}
      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading appointments...</p>
        </div>
      ) : filteredAppointments.length === 0 ? (
        <div className="text-center py-12">
          <Calendar className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">
            {appointments.length === 0 ? 'No appointments in the system' : 'No appointments match your filters'}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredAppointments.map((apt) => (
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

              {/* Patient and Doctor Info */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4 pb-4 border-b border-gray-200">
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-1">Patient:</p>
                  <p className="text-sm text-gray-900 font-medium">{apt.patientId.fullName}</p>
                  <p className="text-xs text-gray-600">
                    Age: {apt.patientId.age} | Gender: {apt.patientId.gender}
                  </p>
                  <p className="text-xs text-gray-600">{apt.patientId.email}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-1">Doctor:</p>
                  <p className="text-sm text-gray-900 font-medium">
                    Dr. {apt.doctorId.firstName} {apt.doctorId.lastName}
                  </p>
                  <p className="text-xs text-gray-600">{apt.doctorId.specialization}</p>
                  <p className="text-xs text-gray-600">{apt.doctorId.email}</p>
                </div>
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
                    <p className="text-sm font-semibold text-red-900 mb-1">Rejection Note:</p>
                    <p className="text-sm text-red-800">{apt.doctorNote}</p>
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

export default AdminAppointments;

