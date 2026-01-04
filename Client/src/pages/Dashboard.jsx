import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import toast from 'react-hot-toast';
import { Shield, Video, LogOut, Users, Activity, CheckCircle, XCircle } from 'lucide-react';

const Dashboard = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [mlHealth, setMlHealth] = useState({ face: false, voice: false, keystroke: false, mouse: false });

  useEffect(() => {
    fetchDoctors();
    checkMLHealth();
  }, []);

  const fetchDoctors = async () => {
    try {
      const response = await axios.get('/api/doctors');
      setDoctors(response.data.data);
    } catch (error) {
      console.error('Failed to fetch doctors:', error);
      toast.error('Failed to load doctors list');
    } finally {
      setLoading(false);
    }
  };

  const checkMLHealth = async () => {
    try {
      const response = await axios.get('/api/verification/health');
      setMlHealth(response.data.data);
    } catch (error) {
      console.error('Failed to check ML health:', error);
    }
  };

  const handleStartConsultation = async () => {
    try {
      const response = await axios.post('/api/sessions', {
        patientId: 'demo-patient'
      });
      
      const sessionId = response.data.data.sessionId;
      toast.success('Starting consultation...');
      navigate(`/meeting/${sessionId}`);
    } catch (error) {
      console.error('Failed to start consultation:', error);
      toast.error('Failed to start consultation');
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
    toast.success('Logged out successfully');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <Shield className="h-8 w-8 text-primary-600 mr-3" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Zero Trust Telehealth</h1>
                <p className="text-sm text-gray-600">Continuous Biometric Authentication</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">
                  Dr. {user?.firstName} {user?.lastName}
                </p>
                <p className="text-xs text-gray-600">{user?.specialization}</p>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
              >
                <LogOut className="h-4 w-4 mr-2" />
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* ML Services Status */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Activity className="h-5 w-5 mr-2 text-primary-600" />
            ML Services Status
          </h2>
          <div className="grid grid-cols-4 gap-4">
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-700">Face Recognition</span>
              {mlHealth.face ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : (
                <XCircle className="h-5 w-5 text-red-600" />
              )}
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-700">Voice Recognition</span>
              {mlHealth.voice ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : (
                <XCircle className="h-5 w-5 text-red-600" />
              )}
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-700">Keystroke Dynamics</span>
              {mlHealth.keystroke ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : (
                <XCircle className="h-5 w-5 text-red-600" />
              )}
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-700">Mouse Movement</span>
              {mlHealth.mouse ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : (
                <XCircle className="h-5 w-5 text-red-600" />
              )}
            </div>
          </div>
        </div>

        {/* Biometric Enrollment Status */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Your Biometric Profile</h2>
          <div className="grid grid-cols-4 gap-4">
            <div className={`p-4 rounded-lg ${user?.biometricData?.faceEnrolled ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
              <p className="text-sm font-medium text-gray-700">Face</p>
              <p className="text-xs text-gray-600 mt-1">
                {user?.biometricData?.faceEnrolled ? 'Enrolled ✓' : 'Not Enrolled ✗'}
              </p>
            </div>
            <div className={`p-4 rounded-lg ${user?.biometricData?.voiceEnrolled ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
              <p className="text-sm font-medium text-gray-700">Voice</p>
              <p className="text-xs text-gray-600 mt-1">
                {user?.biometricData?.voiceEnrolled ? 'Enrolled ✓' : 'Not Enrolled ✗'}
              </p>
            </div>
            <div className={`p-4 rounded-lg ${user?.biometricData?.keystrokeEnrolled ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
              <p className="text-sm font-medium text-gray-700">Keystroke</p>
              <p className="text-xs text-gray-600 mt-1">
                {user?.biometricData?.keystrokeEnrolled ? 'Enrolled ✓' : 'Not Enrolled ✗'}
              </p>
            </div>
            <div className={`p-4 rounded-lg ${user?.biometricData?.mouseEnrolled ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
              <p className="text-sm font-medium text-gray-700">Mouse</p>
              <p className="text-xs text-gray-600 mt-1">
                {user?.biometricData?.mouseEnrolled ? 'Enrolled ✓' : 'Not Enrolled ✗'}
              </p>
            </div>
          </div>
        </div>

        {/* Start Consultation */}
        <div className="bg-gradient-to-r from-primary-600 to-indigo-600 rounded-lg shadow-lg p-8 mb-6 text-white">
          <h2 className="text-2xl font-bold mb-2">Start Live Consultation</h2>
          <p className="mb-6 opacity-90">Begin a secure video consultation with continuous biometric verification</p>
          <button
            onClick={handleStartConsultation}
            className="flex items-center px-6 py-3 bg-white text-primary-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
          >
            <Video className="h-5 w-5 mr-2" />
            Start Consultation
          </button>
        </div>

        {/* Registered Doctors List */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Users className="h-5 w-5 mr-2 text-primary-600" />
            Registered Doctors ({doctors.length})
          </h2>
          
          {loading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Specialization</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">License</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Experience</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Biometric Status</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {doctors.map((doctor) => (
                    <tr key={doctor._id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">
                          Dr. {doctor.firstName} {doctor.lastName}
                        </div>
                        <div className="text-sm text-gray-500">{doctor.email}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {doctor.specialization}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {doctor.medicalLicenseNumber}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {doctor.yearsOfExperience} years
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex gap-2">
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            doctor.biometricData?.faceEnrolled ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                          }`}>
                            Face
                          </span>
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            doctor.biometricData?.voiceEnrolled ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                          }`}>
                            Voice
                          </span>
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            doctor.biometricData?.keystrokeEnrolled ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                          }`}>
                            Key
                          </span>
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            doctor.biometricData?.mouseEnrolled ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                          }`}>
                            Mouse
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default Dashboard;

