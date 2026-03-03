import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import toast from 'react-hot-toast';
import { User, Calendar, LogOut, Stethoscope, Mail, Award, Briefcase, Eye, Search, Filter, ClipboardList, Video } from 'lucide-react';
import DoctorDetailsModal from '../components/DoctorDetailsModal';
import BookAppointmentModal from '../components/BookAppointmentModal';
import MyAppointments from '../components/MyAppointments';
import ConfirmedConsultations from '../components/ConfirmedConsultations';

const PatientDashboard = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSpecialization, setSelectedSpecialization] = useState('all');
  const [activeTab, setActiveTab] = useState('doctors'); // 'doctors', 'appointments', or 'consultations'
  const [bookingDoctor, setBookingDoctor] = useState(null);

  useEffect(() => {
    fetchDoctors();
  }, []);

  const fetchDoctors = async () => {
    try {
      const response = await axios.get('/api/patients/doctors');
      setDoctors(response.data.data.doctors || []);
    } catch (error) {
      console.error('Error fetching doctors:', error);
      toast.error('Failed to load doctors');
      setDoctors([]);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleBookAppointment = (doctor) => {
    setBookingDoctor(doctor);
  };

  const handleAppointmentSuccess = () => {
    setActiveTab('appointments');
  };

  // Get unique specializations for filter
  const specializations = ['all', ...new Set(doctors.map(d => d.specialization))];

  // Filter doctors based on search and specialization
  const filteredDoctors = doctors.filter(doctor => {
    const matchesSearch = searchTerm === '' ||
      `${doctor.firstName} ${doctor.lastName}`.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSpecialization = selectedSpecialization === 'all' ||
      doctor.specialization === selectedSpecialization;
    return matchesSearch && matchesSpecialization;
  });

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <User className="h-8 w-8 text-primary-600 mr-3" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Patient Dashboard</h1>
                <p className="text-sm text-gray-600">Welcome, {user?.fullName}</p>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
            >
              <LogOut className="h-5 w-5 mr-2" />
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Patient Info Card */}
        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Your Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-600">Full Name</p>
              <p className="text-lg font-medium text-gray-900">{user?.fullName}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Age</p>
              <p className="text-lg font-medium text-gray-900">{user?.age} years</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Gender</p>
              <p className="text-lg font-medium text-gray-900">{user?.gender}</p>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="mb-6 border-b border-gray-200">
          <div className="flex gap-4">
            <button
              onClick={() => setActiveTab('doctors')}
              className={`flex items-center px-4 py-3 font-medium text-sm border-b-2 transition-colors ${
                activeTab === 'doctors'
                  ? 'border-primary-600 text-primary-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              <Stethoscope className="h-5 w-5 mr-2" />
              Find Doctors
            </button>
            <button
              onClick={() => setActiveTab('appointments')}
              className={`flex items-center px-4 py-3 font-medium text-sm border-b-2 transition-colors ${
                activeTab === 'appointments'
                  ? 'border-primary-600 text-primary-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              <ClipboardList className="h-5 w-5 mr-2" />
              My Appointments
            </button>
            <button
              onClick={() => setActiveTab('consultations')}
              className={`flex items-center px-4 py-3 font-medium text-sm border-b-2 transition-colors ${
                activeTab === 'consultations'
                  ? 'border-primary-600 text-primary-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              <Video className="h-5 w-5 mr-2" />
              Confirmed Live Consultations
            </button>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'doctors' ? (
          /* Doctors List */
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Available Doctors</h2>

          {/* Search and Filter */}
          <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search by doctor name..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>

            {/* Filter by Specialization */}
            <div className="relative">
              <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <select
                value={selectedSpecialization}
                onChange={(e) => setSelectedSpecialization(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent appearance-none"
              >
                {specializations.map(spec => (
                  <option key={spec} value={spec}>
                    {spec === 'all' ? 'All Specializations' : spec}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {loading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
              <p className="mt-4 text-gray-600">Loading doctors...</p>
            </div>
          ) : filteredDoctors.length === 0 ? (
            <div className="text-center py-8">
              <Stethoscope className="h-16 w-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">
                {doctors.length === 0 ? 'No doctors available at the moment' : 'No doctors found matching your search'}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredDoctors.map((doctor) => (
                <div key={doctor._id} className="border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
                  {/* Profile Image */}
                  <div className="flex flex-col items-center mb-4">
                    <div className="w-24 h-24 rounded-full overflow-hidden bg-gray-200 flex items-center justify-center mb-3">
                      {doctor.profileImage ? (
                        <img
                          src={doctor.profileImage}
                          alt={`Dr. ${doctor.firstName} ${doctor.lastName}`}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <User className="h-12 w-12 text-gray-400" />
                      )}
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 text-center">
                      Dr. {doctor.firstName} {doctor.lastName}
                    </h3>
                  </div>

                  <div className="space-y-2 mb-4">
                    <div className="flex items-center text-sm text-gray-600">
                      <Award className="h-4 w-4 mr-2" />
                      <span>{doctor.specialization}</span>
                    </div>
                    <div className="flex items-center text-sm text-gray-600">
                      <Briefcase className="h-4 w-4 mr-2" />
                      <span>{doctor.yearsOfExperience} years experience</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <button
                      onClick={() => setSelectedDoctor(doctor)}
                      className="w-full flex items-center justify-center px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
                    >
                      <Eye className="h-5 w-5 mr-2" />
                      View Details
                    </button>
                    <button
                      onClick={() => handleBookAppointment(doctor)}
                      className="w-full flex items-center justify-center px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
                    >
                      <Calendar className="h-5 w-5 mr-2" />
                      Book Appointment
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        ) : activeTab === 'appointments' ? (
          /* My Appointments */
          <MyAppointments />
        ) : (
          /* Confirmed Live Consultations */
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Confirmed Live Consultations</h2>
            <ConfirmedConsultations />
          </div>
        )}
      </main>

      {/* Doctor Details Modal */}
      {selectedDoctor && (
        <DoctorDetailsModal
          doctor={selectedDoctor}
          onClose={() => setSelectedDoctor(null)}
        />
      )}

      {/* Book Appointment Modal */}
      {bookingDoctor && (
        <BookAppointmentModal
          doctor={bookingDoctor}
          onClose={() => setBookingDoctor(null)}
          onSuccess={handleAppointmentSuccess}
        />
      )}
    </div>
  );
};

export default PatientDashboard;

