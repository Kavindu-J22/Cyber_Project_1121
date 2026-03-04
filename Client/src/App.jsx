import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { useAuth } from './context/AuthContext';
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import DoctorProfile from './pages/DoctorProfile';
import PatientDashboard from './pages/PatientDashboard';
import AdminDashboard from './pages/AdminDashboard';
import Meeting from './pages/Meeting';
import PrivateRoute from './components/PrivateRoute';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Toaster position="top-right" />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/dashboard"
            element={
              <PrivateRoute allowedRoles={['doctor']}>
                <Dashboard />
              </PrivateRoute>
            }
          />
          <Route
            path="/doctor-profile"
            element={
              <PrivateRoute allowedRoles={['doctor']}>
                <DoctorProfile />
              </PrivateRoute>
            }
          />
          <Route
            path="/patient-dashboard"
            element={
              <PrivateRoute allowedRoles={['patient']}>
                <PatientDashboard />
              </PrivateRoute>
            }
          />
          <Route
            path="/admin-dashboard"
            element={
              <PrivateRoute allowedRoles={['admin']}>
                <AdminDashboard />
              </PrivateRoute>
            }
          />
          <Route
            path="/meeting/:sessionId"
            element={
              <PrivateRoute allowedRoles={['doctor', 'patient']}>
                <Meeting />
              </PrivateRoute>
            }
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

