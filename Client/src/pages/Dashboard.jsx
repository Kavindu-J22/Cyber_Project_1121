import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import toast from 'react-hot-toast';
import { Shield, Video, LogOut, Activity, CheckCircle, XCircle, User, ClipboardList } from 'lucide-react';
import DoctorAppointments from '../components/DoctorAppointments';

const Dashboard = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [mlHealth, setMlHealth] = useState({ face: false, voice: false, keystroke: false, mouse: false });
  const [activeTab, setActiveTab] = useState('dashboard'); // 'dashboard' or 'appointments'

  useEffect(() => {
    checkMLHealth();
  }, []);

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
                <h1 className="text-2xl font-bold text-gray-900">MediConsult</h1>
                <p className="text-sm text-gray-600">Zero Trust Secure Telehealth Platform</p>
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
                onClick={() => navigate('/doctor-profile')}
                className="flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
              >
                <User className="h-4 w-4 mr-2" />
                Profile
              </button>
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
        {/* Tabs */}
        <div className="mb-6 border-b border-gray-200">
          <div className="flex gap-4">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`flex items-center px-4 py-3 font-medium text-sm border-b-2 transition-colors ${
                activeTab === 'dashboard'
                  ? 'border-primary-600 text-primary-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              <Shield className="h-5 w-5 mr-2" />
              Dashboard
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
              Appointments
            </button>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'dashboard' ? (
          <>
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
          </>
        ) : (
          /* Appointments Tab */
          <DoctorAppointments />
        )}
      </main>
    </div>
  );
};

export default Dashboard;

