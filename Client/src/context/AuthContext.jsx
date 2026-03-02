import { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [userRole, setUserRole] = useState(localStorage.getItem('userRole'));
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      fetchUser();
    } else {
      setLoading(false);
    }
  }, [token]);

  const fetchUser = async () => {
    try {
      const role = localStorage.getItem('userRole');

      if (role === 'admin') {
        setUser({ id: 'admin', email: 'admin', role: 'admin', fullName: 'Administrator' });
        setUserRole('admin');
      } else if (role === 'doctor') {
        const response = await axios.get('/api/doctors/me');
        setUser({ ...response.data.data.doctor, role: 'doctor' });
        setUserRole('doctor');
      } else if (role === 'patient') {
        const response = await axios.get('/api/patients/me');
        setUser({ ...response.data.data.patient, role: 'patient' });
        setUserRole('patient');
      }
    } catch (error) {
      console.error('Failed to fetch user:', error);
      logout();
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    const response = await axios.post('/api/auth/login', { email, password });
    const { token, user, role } = response.data.data;

    localStorage.setItem('token', token);
    localStorage.setItem('userRole', role);
    setToken(token);
    setUser({ ...user, role });
    setUserRole(role);
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

    return response.data;
  };

  const register = async (formData) => {
    const response = await axios.post('/api/auth/register', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    const { token, doctor } = response.data.data;

    localStorage.setItem('token', token);
    localStorage.setItem('userRole', 'doctor');
    setToken(token);
    setUser({ ...doctor, role: 'doctor' });
    setUserRole('doctor');
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

    return response.data;
  };

  const registerPatient = async (patientData) => {
    const response = await axios.post('/api/auth/register-patient', patientData);

    const { token, patient } = response.data.data;

    localStorage.setItem('token', token);
    localStorage.setItem('userRole', 'patient');
    setToken(token);
    setUser({ ...patient, role: 'patient' });
    setUserRole('patient');
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

    return response.data;
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('userRole');
    setToken(null);
    setUser(null);
    setUserRole(null);
    delete axios.defaults.headers.common['Authorization'];
  };

  const value = {
    user,
    userRole,
    token,
    loading,
    login,
    register,
    registerPatient,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

