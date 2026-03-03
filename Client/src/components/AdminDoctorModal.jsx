import { useState, useEffect } from 'react';
import { X, Save, User, Award, Briefcase, Mail, FileText, CreditCard } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const AdminDoctorModal = ({ doctor, onClose, onUpdate, isEditMode = false }) => {
  const [isEditing, setIsEditing] = useState(isEditMode);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    specialization: '',
    yearsOfExperience: '',
    description: ''
  });

  useEffect(() => {
    if (doctor) {
      setFormData({
        firstName: doctor.firstName || '',
        lastName: doctor.lastName || '',
        email: doctor.email || '',
        specialization: doctor.specialization || '',
        yearsOfExperience: doctor.yearsOfExperience || '',
        description: doctor.description || ''
      });
    }
  }, [doctor]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      await axios.put(`/api/doctors/${doctor._id}`, formData);
      toast.success('Doctor updated successfully!');
      setIsEditing(false);
      onUpdate();
    } catch (error) {
      console.error('Update error:', error);
      toast.error(error.response?.data?.message || 'Failed to update doctor');
    } finally {
      setLoading(false);
    }
  };

  if (!doctor) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex justify-between items-center p-6 border-b sticky top-0 bg-white z-10">
          <h2 className="text-2xl font-bold text-gray-900">
            {isEditing ? 'Edit Doctor Details' : 'Doctor Details'}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Profile Image */}
          <div className="flex flex-col items-center mb-6">
            <div className="w-32 h-32 rounded-full overflow-hidden bg-gray-200 flex items-center justify-center mb-4">
              {doctor.profileImage ? (
                <img 
                  src={doctor.profileImage} 
                  alt={`Dr. ${doctor.firstName} ${doctor.lastName}`}
                  className="w-full h-full object-cover"
                />
              ) : (
                <User className="h-16 w-16 text-gray-400" />
              )}
            </div>
          </div>

          {/* Form Fields */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* First Name */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <User className="inline h-4 w-4 mr-1" />
                First Name
              </label>
              {isEditing ? (
                <input
                  type="text"
                  name="firstName"
                  value={formData.firstName}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500"
                  required
                />
              ) : (
                <p className="text-gray-900 bg-gray-50 p-3 rounded-md">{doctor.firstName}</p>
              )}
            </div>

            {/* Last Name */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <User className="inline h-4 w-4 mr-1" />
                Last Name
              </label>
              {isEditing ? (
                <input
                  type="text"
                  name="lastName"
                  value={formData.lastName}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500"
                  required
                />
              ) : (
                <p className="text-gray-900 bg-gray-50 p-3 rounded-md">{doctor.lastName}</p>
              )}
            </div>

            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <Mail className="inline h-4 w-4 mr-1" />
                Email
              </label>
              {isEditing ? (
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500"
                  required
                />
              ) : (
                <p className="text-gray-900 bg-gray-50 p-3 rounded-md">{doctor.email}</p>
              )}
            </div>

            {/* Medical License Number - READ ONLY */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <CreditCard className="inline h-4 w-4 mr-1" />
                Medical License Number
              </label>
              <p className="text-gray-900 bg-gray-100 p-3 rounded-md border border-gray-300">
                {doctor.medicalLicenseNumber}
              </p>
              <p className="text-xs text-gray-500 mt-1">This field cannot be edited</p>
            </div>

            {/* Specialization */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <Award className="inline h-4 w-4 mr-1" />
                Specialization
              </label>
              {isEditing ? (
                <input
                  type="text"
                  name="specialization"
                  value={formData.specialization}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500"
                  required
                />
              ) : (
                <p className="text-gray-900 bg-gray-50 p-3 rounded-md">{doctor.specialization}</p>
              )}
            </div>

            {/* Years of Experience */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <Briefcase className="inline h-4 w-4 mr-1" />
                Years of Experience
              </label>
              {isEditing ? (
                <input
                  type="number"
                  name="yearsOfExperience"
                  value={formData.yearsOfExperience}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500"
                  required
                  min="0"
                />
              ) : (
                <p className="text-gray-900 bg-gray-50 p-3 rounded-md">{doctor.yearsOfExperience} years</p>
              )}
            </div>
          </div>

          {/* Description - Full Width */}
          <div className="mt-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <FileText className="inline h-4 w-4 mr-1" />
              Description
            </label>
            {isEditing ? (
              <textarea
                name="description"
                value={formData.description}
                onChange={handleChange}
                rows="4"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500"
                placeholder="Doctor's description..."
              />
            ) : (
              <p className="text-gray-900 bg-gray-50 p-3 rounded-md whitespace-pre-wrap">
                {doctor.description || 'No description provided'}
              </p>
            )}
          </div>

          {/* Biometric Status - Read Only */}
          <div className="mt-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Biometric Verification Status</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className={`p-3 rounded-md text-center ${
                doctor.biometricData?.faceEnrolled ? 'bg-green-100' : 'bg-gray-100'
              }`}>
                <p className="text-sm font-medium">Face</p>
                <p className={`text-lg font-bold ${
                  doctor.biometricData?.faceEnrolled ? 'text-green-800' : 'text-gray-800'
                }`}>
                  {doctor.biometricData?.faceEnrolled ? '✓ Enrolled' : '✗ Not Enrolled'}
                </p>
              </div>
              <div className={`p-3 rounded-md text-center ${
                doctor.biometricData?.voiceEnrolled ? 'bg-green-100' : 'bg-gray-100'
              }`}>
                <p className="text-sm font-medium">Voice</p>
                <p className={`text-lg font-bold ${
                  doctor.biometricData?.voiceEnrolled ? 'text-green-800' : 'text-gray-800'
                }`}>
                  {doctor.biometricData?.voiceEnrolled ? '✓ Enrolled' : '✗ Not Enrolled'}
                </p>
              </div>
              <div className={`p-3 rounded-md text-center ${
                doctor.biometricData?.keystrokeEnrolled ? 'bg-green-100' : 'bg-gray-100'
              }`}>
                <p className="text-sm font-medium">Keystroke</p>
                <p className={`text-lg font-bold ${
                  doctor.biometricData?.keystrokeEnrolled ? 'text-green-800' : 'text-gray-800'
                }`}>
                  {doctor.biometricData?.keystrokeEnrolled ? '✓ Enrolled' : '✗ Not Enrolled'}
                </p>
              </div>
              <div className={`p-3 rounded-md text-center ${
                doctor.biometricData?.mouseEnrolled ? 'bg-green-100' : 'bg-gray-100'
              }`}>
                <p className="text-sm font-medium">Mouse</p>
                <p className={`text-lg font-bold ${
                  doctor.biometricData?.mouseEnrolled ? 'text-green-800' : 'text-gray-800'
                }`}>
                  {doctor.biometricData?.mouseEnrolled ? '✓ Enrolled' : '✗ Not Enrolled'}
                </p>
              </div>
            </div>
          </div>

          {/* Account Information */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-50 p-3 rounded-md">
              <p className="text-sm text-gray-600">Created At</p>
              <p className="text-gray-900 font-medium">
                {new Date(doctor.createdAt).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric'
                })}
              </p>
            </div>
            <div className="bg-gray-50 p-3 rounded-md">
              <p className="text-sm text-gray-600">Last Login</p>
              <p className="text-gray-900 font-medium">
                {doctor.lastLogin ? new Date(doctor.lastLogin).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric'
                }) : 'Never'}
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-6 border-t bg-gray-50">
          {isEditing ? (
            <>
              <button
                onClick={() => {
                  setIsEditing(false);
                  setFormData({
                    firstName: doctor.firstName || '',
                    lastName: doctor.lastName || '',
                    email: doctor.email || '',
                    specialization: doctor.specialization || '',
                    yearsOfExperience: doctor.yearsOfExperience || '',
                    description: doctor.description || ''
                  });
                }}
                className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-100 transition-colors"
                disabled={loading}
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={loading}
                className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                <Save className="h-4 w-4" />
                {loading ? 'Saving...' : 'Save Changes'}
              </button>
            </>
          ) : (
            <>
              <button
                onClick={onClose}
                className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-100 transition-colors"
              >
                Close
              </button>
              <button
                onClick={() => setIsEditing(true)}
                className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
              >
                Edit
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdminDoctorModal;

