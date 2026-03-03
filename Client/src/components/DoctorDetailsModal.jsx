import { X, Award, Briefcase, User } from 'lucide-react';

const DoctorDetailsModal = ({ doctor, onClose }) => {
  if (!doctor) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex justify-between items-center p-6 border-b">
          <h2 className="text-2xl font-bold text-gray-900">Doctor Details</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Profile Image and Name */}
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
            <h3 className="text-2xl font-bold text-gray-900">
              Dr. {doctor.firstName} {doctor.lastName}
            </h3>
          </div>

          {/* Doctor Information */}
          <div className="space-y-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center mb-2">
                <Award className="h-5 w-5 text-primary-600 mr-2" />
                <span className="font-semibold text-gray-900">Specialization</span>
              </div>
              <p className="text-gray-700 ml-7">{doctor.specialization}</p>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center mb-2">
                <Briefcase className="h-5 w-5 text-primary-600 mr-2" />
                <span className="font-semibold text-gray-900">Experience</span>
              </div>
              <p className="text-gray-700 ml-7">{doctor.yearsOfExperience} years</p>
            </div>

            {doctor.description && (
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="mb-2">
                  <span className="font-semibold text-gray-900">About</span>
                </div>
                <p className="text-gray-700 whitespace-pre-wrap">{doctor.description}</p>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t bg-gray-50">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default DoctorDetailsModal;

