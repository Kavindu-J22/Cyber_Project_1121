import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import toast from 'react-hot-toast';
import { ArrowLeft, Upload, User, FileText, Save, Mail, CreditCard, Award, Briefcase } from 'lucide-react';

const DoctorProfile = () => {
  const navigate = useNavigate();
  const { user, fetchUser } = useAuth();
  const [loading, setLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    specialization: '',
    yearsOfExperience: '',
    description: '',
    profileImage: null
  });

  useEffect(() => {
    if (user) {
      setFormData({
        firstName: user.firstName || '',
        lastName: user.lastName || '',
        email: user.email || '',
        specialization: user.specialization || '',
        yearsOfExperience: user.yearsOfExperience || '',
        description: user.description || '',
        profileImage: null
      });
      if (user.profileImage) {
        setImagePreview(user.profileImage);
      }
    }
  }, [user]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 5 * 1024 * 1024) {
        toast.error('Image size should be less than 5MB');
        return;
      }
      setFormData(prev => ({
        ...prev,
        profileImage: file
      }));
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const submitData = new FormData();
      submitData.append('firstName', formData.firstName);
      submitData.append('lastName', formData.lastName);
      submitData.append('email', formData.email);
      submitData.append('specialization', formData.specialization);
      submitData.append('yearsOfExperience', formData.yearsOfExperience);
      submitData.append('description', formData.description);
      
      if (formData.profileImage) {
        submitData.append('profileImage', formData.profileImage);
      }

      await axios.put(`/api/doctors/${user._id}`, submitData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      toast.success('Profile updated successfully!');
      await fetchUser(); // Refresh user data
      navigate('/dashboard');
    } catch (error) {
      console.error('Profile update error:', error);
      toast.error(error.response?.data?.message || 'Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <button
          onClick={() => navigate('/dashboard')}
          className="flex items-center text-primary-600 hover:text-primary-700 mb-6"
        >
          <ArrowLeft className="h-5 w-5 mr-2" />
          Back to Dashboard
        </button>

        <div className="bg-white rounded-lg shadow p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Update Profile</h2>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Profile Image */}
            <div className="flex flex-col items-center mb-6">
              <div className="relative">
                <div className="w-32 h-32 rounded-full overflow-hidden bg-gray-200 flex items-center justify-center">
                  {imagePreview ? (
                    <img src={imagePreview} alt="Profile" className="w-full h-full object-cover" />
                  ) : (
                    <User className="h-16 w-16 text-gray-400" />
                  )}
                </div>
                <label className="absolute bottom-0 right-0 bg-primary-600 text-white p-2 rounded-full cursor-pointer hover:bg-primary-700">
                  <Upload className="h-5 w-5" />
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageChange}
                    className="hidden"
                  />
                </label>
              </div>
              <p className="text-sm text-gray-600 mt-2">Click to upload profile picture</p>
            </div>

            {/* Name Fields */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  <User className="inline h-4 w-4 mr-1" />
                  First Name
                </label>
                <input
                  type="text"
                  name="firstName"
                  value={formData.firstName}
                  onChange={handleChange}
                  className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  <User className="inline h-4 w-4 mr-1" />
                  Last Name
                </label>
                <input
                  type="text"
                  name="lastName"
                  value={formData.lastName}
                  onChange={handleChange}
                  className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500"
                  required
                />
              </div>
            </div>

            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                <Mail className="inline h-4 w-4 mr-1" />
                Email
              </label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500"
                required
              />
            </div>

            {/* Medical License Number - READ ONLY */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                <CreditCard className="inline h-4 w-4 mr-1" />
                Medical License Number
              </label>
              <input
                type="text"
                value={user?.medicalLicenseNumber || ''}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md bg-gray-100 cursor-not-allowed"
                disabled
                readOnly
              />
              <p className="text-xs text-gray-500 mt-1">This field cannot be edited</p>
            </div>

            {/* Specialization */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                <Award className="inline h-4 w-4 mr-1" />
                Specialization
              </label>
              <input
                type="text"
                name="specialization"
                value={formData.specialization}
                onChange={handleChange}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500"
                required
              />
            </div>

            {/* Years of Experience */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                <Briefcase className="inline h-4 w-4 mr-1" />
                Years of Experience
              </label>
              <input
                type="number"
                name="yearsOfExperience"
                value={formData.yearsOfExperience}
                onChange={handleChange}
                min="0"
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500"
                required
              />
            </div>

            {/* Description */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                <FileText className="inline h-4 w-4 mr-1" />
                Description
              </label>
              <textarea
                name="description"
                value={formData.description}
                onChange={handleChange}
                rows="5"
                maxLength="1000"
                placeholder="Tell patients about yourself, your expertise, and approach to healthcare..."
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
              />
              <p className="text-sm text-gray-500 mt-1">
                {formData.description.length}/1000 characters
              </p>
            </div>

            {/* Submit Button */}
            <div className="flex justify-end">
              <button
                type="submit"
                disabled={loading}
                className="flex items-center px-6 py-3 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50"
              >
                <Save className="h-5 w-5 mr-2" />
                {loading ? 'Saving...' : 'Save Profile'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default DoctorProfile;
