import { useState, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import toast from 'react-hot-toast';
import { Shield, User, Mail, Lock, FileText, Briefcase, Calendar, Mic, Keyboard, Mouse, CheckCircle, Camera } from 'lucide-react';
import { KeystrokeCapture, MouseCapture, VoiceCapture, FaceCapture } from '../utils/biometricCapture';

const Register = () => {
  const navigate = useNavigate();
  const { register, registerPatient } = useAuth();
  const [userType, setUserType] = useState(null); // 'doctor' or 'patient'
  const [step, setStep] = useState(0); // 0 = user type selection, 1+ = registration steps
  const [loading, setLoading] = useState(false);

  const [formData, setFormData] = useState({
    // Common fields
    email: '',
    password: '',
    confirmPassword: '',
    // Doctor fields
    firstName: '',
    lastName: '',
    medicalLicenseNumber: '',
    specialization: '',
    yearsOfExperience: '',
    // Patient fields
    fullName: '',
    age: '',
    gender: '',
  });

  // Biometric data
  const [voiceBlobs, setVoiceBlobs] = useState([]);
  const [keystrokeData, setKeystrokeData] = useState([]);
  const [mouseData, setMouseData] = useState([]);
  const [faceImages, setFaceImages] = useState([]);

  // Capture instances
  const keystrokeCapture = useRef(new KeystrokeCapture());
  const mouseCapture = useRef(new MouseCapture());
  const voiceCapture = useRef(new VoiceCapture());
  const faceCapture = useRef(new FaceCapture());

  // Recording states
  const [isRecordingVoice, setIsRecordingVoice] = useState(false);
  const [isCapturingKeystroke, setIsCapturingKeystroke] = useState(false);
  const [isCapturingMouse, setIsCapturingMouse] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isCapturingFace, setIsCapturingFace] = useState(false);
  const [isProcessingCapture, setIsProcessingCapture] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [voiceRecordingTime, setVoiceRecordingTime] = useState(0);
  const [mouseRecordingTime, setMouseRecordingTime] = useState(0);
  const [currentKeystrokeSample, setCurrentKeystrokeSample] = useState(0);
  const [typedText, setTypedText] = useState('');

  // Computed values
  const faceEnrolled = faceImages.length >= 3;

  const videoRef = useRef(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleUserTypeSelection = (type) => {
    setUserType(type);
    setStep(1);
  };

  const handleNext = () => {
    // Validate current step for Doctor
    if (userType === 'doctor') {
      if (step === 1) {
        if (!formData.firstName || !formData.lastName || !formData.email ||
            !formData.password || !formData.confirmPassword) {
          toast.error('Please fill in all fields');
          return;
        }
        if (formData.password !== formData.confirmPassword) {
          toast.error('Passwords do not match');
          return;
        }
        if (formData.password.length < 6) {
          toast.error('Password must be at least 6 characters');
          return;
        }
      }

      if (step === 2) {
        if (!formData.medicalLicenseNumber || !formData.specialization || !formData.yearsOfExperience) {
          toast.error('Please fill in all professional details');
          return;
        }
      }
    }

    // Validate for Patient
    if (userType === 'patient') {
      if (step === 1) {
        if (!formData.fullName || !formData.age || !formData.gender ||
            !formData.email || !formData.password || !formData.confirmPassword) {
          toast.error('Please fill in all fields');
          return;
        }
        if (formData.password !== formData.confirmPassword) {
          toast.error('Passwords do not match');
          return;
        }
        if (formData.password.length < 6) {
          toast.error('Password must be at least 6 characters');
          return;
        }
      }
    }

    setStep(step + 1);
  };

  const handleBack = () => {
    setStep(step - 1);
  };

  // Face Recognition Capture (3 images)
  const startFaceCapture = async () => {
    try {
      setCameraReady(false);
      setIsCameraActive(false);

      // Set isCapturingFace to true first to render the video element
      setIsCapturingFace(true);

      // Wait for next render cycle so video element is in DOM
      await new Promise(resolve => setTimeout(resolve, 100));

      // Now check if video element exists
      if (!videoRef.current) {
        toast.error('Video element not found. Please try again.');
        setIsCapturingFace(false);
        return;
      }

      console.log('Starting camera...');
      const started = await faceCapture.current.startCamera(videoRef.current);

      if (started) {
        setIsCameraActive(true);
        console.log('Camera stream started successfully');

        // Wait for video to be ready and playing
        await new Promise((resolve) => {
          const video = videoRef.current;

          if (!video) {
            console.error('Video element disappeared');
            resolve();
            return;
          }

          const markReady = () => {
            console.log('Camera ready! Video dimensions:', video.videoWidth, 'x', video.videoHeight);
            setCameraReady(true);
            resolve();
          };

          // Check if video is already ready
          if (video.readyState >= 3 && video.videoWidth > 0) {
            console.log('Video already ready');
            markReady();
            return;
          }

          // Set up event listeners
          const onLoadedMetadata = () => {
            console.log('Video metadata loaded');
            if (video.videoWidth > 0) {
              markReady();
            }
          };

          const onLoadedData = () => {
            console.log('Video data loaded, readyState:', video.readyState);
            if (video.readyState >= 2 && video.videoWidth > 0) {
              markReady();
            }
          };

          const onCanPlay = () => {
            console.log('Video can play');
            markReady();
          };

          video.addEventListener('loadedmetadata', onLoadedMetadata, { once: true });
          video.addEventListener('loadeddata', onLoadedData, { once: true });
          video.addEventListener('canplay', onCanPlay, { once: true });

          // Safety timeout - mark as ready after 3 seconds anyway
          setTimeout(() => {
            console.log('Camera ready timeout triggered');
            setCameraReady(true);
            resolve();
          }, 3000);
        });

        toast.success('📷 Camera activated! Wait for the green indicator, then capture.');
      } else {
        toast.error('Failed to access camera. Please check permissions.');
        setIsCapturingFace(false);
        setIsCameraActive(false);
      }
    } catch (error) {
      console.error('Camera start error:', error);
      toast.error('Failed to start camera. Please check permissions.');
      setIsCapturingFace(false);
      setIsCameraActive(false);
      setCameraReady(false);
    }
  };

  const captureFaceSample = async () => {
    if (isProcessingCapture) return; // Prevent double-clicks
    
    try {
      setIsProcessingCapture(true);
      
      // Validate video is ready
      if (!videoRef.current) {
        toast.error('Video element not found.');
        return;
      }

      if (!videoRef.current.srcObject) {
        toast.error('Camera stream not available. Please restart the camera.');
        return;
      }

      // Less strict check - just warn if readyState is low but still try
      if (videoRef.current.readyState < 1) {
        console.warn('Video readyState:', videoRef.current.readyState);
        toast.error('Video is still loading. Please wait a moment.');
        return;
      }

      // Try to capture
      await faceCapture.current.captureFrame(videoRef.current, 224, 224);
      const count = faceCapture.current.getImageCount();
      setFaceImages(faceCapture.current.getImages());
      
      if (count >= 3) {
        toast.success(`✅ All ${count} face samples captured!`);
        stopFaceCapture();
      } else {
        toast.success(`✅ Face sample ${count}/3 captured! Capture ${3 - count} more.`);
      }
    } catch (error) {
      console.error('Face capture error:', error);
      console.error('Video element state:', {
        exists: !!videoRef.current,
        hasStream: !!videoRef.current?.srcObject,
        readyState: videoRef.current?.readyState,
        videoWidth: videoRef.current?.videoWidth,
        videoHeight: videoRef.current?.videoHeight
      });
      toast.error(`Failed to capture: ${error.message}`);
    } finally {
      setIsProcessingCapture(false);
    }
  };

  const stopFaceCapture = () => {
    faceCapture.current.stopCamera();
    setIsCameraActive(false);
    setIsCapturingFace(false);
    setCameraReady(false);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  // Voice Recording with timer - Collect 3 samples
  const startVoiceRecording = async () => {
    const started = await voiceCapture.current.start();
    if (started) {
      setIsRecordingVoice(true);
      setVoiceRecordingTime(0);
      toast.success(`🎤 Recording sample ${voiceBlobs.length + 1}/3... Please speak clearly for 5-10 seconds`);

      // Auto-stop after 10 seconds
      const timer = setInterval(() => {
        setVoiceRecordingTime(prev => {
          if (prev >= 10) {
            clearInterval(timer);
            stopVoiceRecording();
            return 10;
          }
          return prev + 1;
        });
      }, 1000);
    } else {
      toast.error('Failed to access microphone. Please check permissions.');
    }
  };

  const stopVoiceRecording = async () => {
    const blob = await voiceCapture.current.stop();
    setVoiceBlobs(prev => [...prev, blob]);
    setIsRecordingVoice(false);
    const newCount = voiceBlobs.length + 1;
    if (newCount >= 3) {
      toast.success(`✅ All ${newCount} voice samples captured!`);
    } else {
      toast.success(`✅ Voice sample ${newCount}/3 captured! Please record ${3 - newCount} more.`);
    }
  };

  // Keystroke Capture with validation
  const REQUIRED_PHRASE = "The quick brown fox jumps over the lazy dog";

  const startKeystrokeCapture = () => {
    keystrokeCapture.current.start();
    setIsCapturingKeystroke(true);
    setTypedText('');
    setCurrentKeystrokeSample(keystrokeData.length + 1);
    toast.success(`⌨️ Sample ${keystrokeData.length + 1}/3: Type the exact phrase shown below`);
  };

  const handleKeystrokeTextChange = (e) => {
    // Prevent paste
    const newText = e.target.value;
    setTypedText(newText);
  };

  const handleKeystrokePaste = (e) => {
    e.preventDefault();
    toast.error('❌ Copy/paste is not allowed. Please type the phrase manually.');
  };

  const completeKeystrokeSample = () => {
    // Validate the typed text matches exactly
    if (typedText.trim() !== REQUIRED_PHRASE) {
      toast.error('❌ Text does not match! Please type the exact phrase.');
      return;
    }

    const features = keystrokeCapture.current.stop();

    // Validate we have enough keystroke events
    if (features.length < 38) {
      toast.error('❌ Not enough keystroke data captured. Please try again.');
      setIsCapturingKeystroke(false);
      setTypedText('');
      return;
    }

    setKeystrokeData(prev => [...prev, features]);
    setIsCapturingKeystroke(false);
    setTypedText('');
    toast.success(`✅ Keystroke sample ${keystrokeData.length + 1}/3 captured!`);
  };

  // Mouse Capture with timer
  const startMouseCapture = () => {
    mouseCapture.current.start();
    setIsCapturingMouse(true);
    setMouseRecordingTime(0);
    toast.success('🖱️ Move your mouse naturally in the area below');

    // Auto-stop after 15 seconds
    const timer = setInterval(() => {
      setMouseRecordingTime(prev => {
        if (prev >= 15) {
          clearInterval(timer);
          stopMouseCapture();
          return 15;
        }
        return prev + 1;
      });
    }, 1000);
  };

  const stopMouseCapture = () => {
    const events = mouseCapture.current.stop();
    setMouseData(events);
    setIsCapturingMouse(false);
    toast.success(`✅ Mouse pattern captured! (${events.length} events recorded)`);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Patient registration (simple, no biometrics)
    if (userType === 'patient') {
      setLoading(true);
      try {
        await registerPatient({
          fullName: formData.fullName,
          age: formData.age,
          gender: formData.gender,
          email: formData.email,
          password: formData.password
        });
        toast.success('Patient registration successful!');
        navigate('/patient-dashboard');
      } catch (error) {
        console.error('Patient registration error:', error);
        toast.error(error.response?.data?.message || 'Registration failed. Please try again.');
      } finally {
        setLoading(false);
      }
      return;
    }

    // Doctor registration (with biometrics)
    if (voiceBlobs.length < 3) {
      toast.error(`Please record 3 voice samples (${voiceBlobs.length}/3 completed)`);
      return;
    }

    if (keystrokeData.length < 3) {
      toast.error('Please capture at least 3 keystroke samples');
      return;
    }

    if (mouseData.length === 0) {
      toast.error('Please capture a mouse movement pattern');
      return;
    }

    if (faceImages.length < 3) {
      toast.error(`Please capture 3 face samples (${faceImages.length}/3 completed)`);
      return;
    }

    setLoading(true);

    try {
      const submitData = new FormData();
      submitData.append('firstName', formData.firstName);
      submitData.append('lastName', formData.lastName);
      submitData.append('email', formData.email);
      submitData.append('password', formData.password);
      submitData.append('medicalLicenseNumber', formData.medicalLicenseNumber);
      submitData.append('specialization', formData.specialization);
      submitData.append('yearsOfExperience', formData.yearsOfExperience);

      // Append all 3 voice samples
      voiceBlobs.forEach((blob, index) => {
        submitData.append('voiceSamples', blob, `voice-sample-${index + 1}.wav`);
      });

      submitData.append('keystrokePattern', JSON.stringify(keystrokeData));
      submitData.append('mousePattern', JSON.stringify(mouseData));

      // Append all 3 face images
      faceImages.forEach((image, index) => {
        submitData.append('faceImages', image, `face-sample-${index + 1}.jpg`);
      });

      await register(submitData);
      toast.success('Doctor registration successful!');
      navigate('/dashboard');
    } catch (error) {
      console.error('Registration error:', error);
      toast.error(error.response?.data?.message || 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex justify-center">
            <div className="bg-primary-600 p-3 rounded-full">
              <Shield className="h-12 w-12 text-white" />
            </div>
          </div>
          <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
            {step === 0 ? 'Registration' : userType === 'doctor' ? 'Doctor Registration' : 'Patient Registration'}
          </h2>
          {step > 0 && (
            <p className="mt-2 text-sm text-gray-600">
              {userType === 'doctor' ? `Step ${step} of 3` : 'Complete Your Profile'}
            </p>
          )}
        </div>

        <div className="bg-white rounded-lg shadow-xl p-8">
          {/* Step 0: User Type Selection */}
          {step === 0 && (
            <div className="space-y-6">
              <h3 className="text-lg font-medium text-gray-900 mb-6 text-center">Choose Registration Type</h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <button
                  type="button"
                  onClick={() => handleUserTypeSelection('doctor')}
                  className="p-8 border-2 border-gray-300 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all duration-200 group"
                >
                  <Briefcase className="h-16 w-16 mx-auto text-primary-600 mb-4" />
                  <h4 className="text-xl font-semibold text-gray-900 mb-2">Register as Doctor</h4>
                  <p className="text-sm text-gray-600">
                    Complete registration with biometric authentication for secure access
                  </p>
                </button>

                <button
                  type="button"
                  onClick={() => handleUserTypeSelection('patient')}
                  className="p-8 border-2 border-gray-300 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all duration-200 group"
                >
                  <User className="h-16 w-16 mx-auto text-primary-600 mb-4" />
                  <h4 className="text-xl font-semibold text-gray-900 mb-2">Register as Patient</h4>
                  <p className="text-sm text-gray-600">
                    Quick registration to book appointments and access healthcare services
                  </p>
                </button>
              </div>

              <div className="text-center mt-6">
                <p className="text-sm text-gray-600">
                  Already have an account?{' '}
                  <Link to="/login" className="font-medium text-primary-600 hover:text-primary-500">
                    Login here
                  </Link>
                </p>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit}>
            {/* Patient Registration Form */}
            {userType === 'patient' && step === 1 && (
              <div className="space-y-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Patient Information</h3>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Full Name</label>
                  <input
                    type="text"
                    name="fullName"
                    value={formData.fullName}
                    onChange={handleChange}
                    className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Age</label>
                    <input
                      type="number"
                      name="age"
                      value={formData.age}
                      onChange={handleChange}
                      min="0"
                      max="150"
                      className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">Gender</label>
                    <select
                      name="gender"
                      value={formData.gender}
                      onChange={handleChange}
                      className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                      required
                    >
                      <option value="">Select Gender</option>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Email Address</label>
                  <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Password</label>
                    <input
                      type="password"
                      name="password"
                      value={formData.password}
                      onChange={handleChange}
                      className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">Confirm Password</label>
                    <input
                      type="password"
                      name="confirmPassword"
                      value={formData.confirmPassword}
                      onChange={handleChange}
                      className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                      required
                    />
                  </div>
                </div>

                <div className="flex justify-between pt-4">
                  <button
                    type="button"
                    onClick={() => setStep(0)}
                    className="px-6 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
                  >
                    Back
                  </button>
                  <button
                    type="submit"
                    disabled={loading}
                    className="px-6 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50"
                  >
                    {loading ? 'Registering...' : 'Register'}
                  </button>
                </div>
              </div>
            )}

            {/* Doctor Registration - Step 1: Personal Information */}
            {userType === 'doctor' && step === 1 && (
              <div className="space-y-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Personal Information</h3>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">First Name</label>
                    <input
                      type="text"
                      name="firstName"
                      value={formData.firstName}
                      onChange={handleChange}
                      className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                      required
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Last Name</label>
                    <input
                      type="text"
                      name="lastName"
                      value={formData.lastName}
                      onChange={handleChange}
                      className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Email Address</label>
                  <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Password</label>
                    <input
                      type="password"
                      name="password"
                      value={formData.password}
                      onChange={handleChange}
                      className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">Confirm Password</label>
                    <input
                      type="password"
                      name="confirmPassword"
                      value={formData.confirmPassword}
                      onChange={handleChange}
                      className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                      required
                    />
                  </div>
                </div>

                <div className="flex gap-4">
                  <button
                    type="button"
                    onClick={() => setStep(0)}
                    className="flex-1 py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                  >
                    Back
                  </button>
                  <button
                    type="button"
                    onClick={handleNext}
                    className="flex-1 py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}

            {/* Doctor Registration - Step 2: Professional Information */}
            {userType === 'doctor' && step === 2 && (
              <div className="space-y-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Professional Information</h3>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Medical License Number</label>
                  <input
                    type="text"
                    name="medicalLicenseNumber"
                    value={formData.medicalLicenseNumber}
                    onChange={handleChange}
                    className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Specialization</label>
                  <input
                    type="text"
                    name="specialization"
                    value={formData.specialization}
                    onChange={handleChange}
                    placeholder="e.g., Cardiology, Pediatrics"
                    className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Years of Experience</label>
                  <input
                    type="number"
                    name="yearsOfExperience"
                    value={formData.yearsOfExperience}
                    onChange={handleChange}
                    min="0"
                    className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                    required
                  />
                </div>

                <div className="flex gap-4">
                  <button
                    type="button"
                    onClick={handleBack}
                    className="flex-1 py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                  >
                    Back
                  </button>
                  <button
                    type="button"
                    onClick={handleNext}
                    className="flex-1 py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}

            {/* Doctor Registration - Step 3: Biometric Enrollment */}
            {userType === 'doctor' && step === 3 && (
              <div className="space-y-6">
                <div className="text-center mb-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-2">Biometric Enrollment</h3>
                  <p className="text-sm text-gray-600">
                    Complete all biometric enrollments for secure continuous authentication
                  </p>
                </div>

                {/* Face Recognition */}
                <div className={`border-2 rounded-lg p-6 transition-all ${
                  faceEnrolled ? 'border-green-500 bg-green-50' : 'border-gray-200 bg-white'
                }`}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center">
                      <div className={`p-2 rounded-full ${faceEnrolled ? 'bg-green-500' : 'bg-primary-600'}`}>
                        <Camera className="h-5 w-5 text-white" />
                      </div>
                      <div className="ml-3">
                        <h4 className="font-semibold text-gray-900">Face Recognition</h4>
                        <p className="text-xs text-gray-500">Capture 3 face samples ({faceImages.length}/3 completed)</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      {faceEnrolled ? (
                        <div className="flex items-center text-green-600">
                          <CheckCircle className="h-5 w-5 mr-1" />
                          <span className="text-sm font-medium">Complete</span>
                        </div>
                      ) : (
                        <span className="text-sm font-medium text-gray-600">Not captured</span>
                      )}
                    </div>
                  </div>

                  {isCapturingFace && (
                    <div className="mb-4">
                      <div className={`mb-2 p-3 border rounded-md ${
                        cameraReady 
                          ? 'bg-green-50 border-green-200' 
                          : 'bg-yellow-50 border-yellow-200'
                      }`}>
                        <p className={`text-sm font-medium mb-1 ${
                          cameraReady ? 'text-green-900' : 'text-yellow-900'
                        }`}>
                          {cameraReady ? '✅ Camera Ready!' : '⏳ Camera Loading...'}
                        </p>
                        <p className={`text-xs ${
                          cameraReady ? 'text-green-700' : 'text-yellow-700'
                        }`}>
                          {cameraReady 
                            ? 'Position your face in the frame and click Capture' 
                            : 'Please wait for the camera to initialize'}
                        </p>
                      </div>
                      <div className="relative w-full bg-black rounded-lg overflow-hidden">
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          muted
                          className="w-full h-64 object-cover"
                          style={{ transform: 'scaleX(-1)' }}
                        />
                        <div className="absolute inset-0 border-4 border-dashed border-blue-400 m-8 rounded-lg pointer-events-none"></div>
                      </div>
                      
                      {faceImages.length > 0 && (
                        <div className="mt-3">
                          <p className="text-xs font-medium text-gray-700 mb-2">Captured Samples:</p>
                          <div className="flex gap-2">
                            {faceImages.map((img, index) => (
                              <div key={index} className="w-16 h-16 rounded-lg overflow-hidden border-2 border-green-500">
                                <img
                                  src={URL.createObjectURL(img)}
                                  alt={`Face sample ${index + 1}`}
                                  className="w-full h-full object-cover"
                                />
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {isCapturingFace ? (
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={captureFaceSample}
                        disabled={isProcessingCapture || !cameraReady}
                        className={`flex-1 py-3 px-4 rounded-md text-sm font-medium transition-all ${
                          isProcessingCapture || !cameraReady
                            ? 'bg-gray-400 cursor-not-allowed text-white'
                            : 'bg-green-600 hover:bg-green-700 text-white'
                        }`}
                      >
                        {isProcessingCapture ? '⏳ Capturing...' : !cameraReady ? '⏳ Camera Loading...' : `✅ Capture Sample (${faceImages.length}/3)`}
                      </button>
                      <button
                        type="button"
                        onClick={stopFaceCapture}
                        disabled={isProcessingCapture}
                        className="flex-1 py-3 px-4 rounded-md text-sm font-medium bg-gray-600 hover:bg-gray-700 text-white transition-all disabled:bg-gray-400 disabled:cursor-not-allowed"
                      >
                        ❌ Cancel
                      </button>
                    </div>
                  ) : (
                    <>
                      {faceImages.length > 0 && !faceEnrolled && (
                        <div className="mb-4">
                          <p className="text-xs font-medium text-gray-700 mb-2">Captured Samples ({faceImages.length}/3):</p>
                          <div className="flex gap-2">
                            {faceImages.map((img, index) => (
                              <div key={index} className="w-20 h-20 rounded-lg overflow-hidden border-2 border-green-500">
                                <img
                                  src={URL.createObjectURL(img)}
                                  alt={`Face sample ${index + 1}`}
                                  className="w-full h-full object-cover"
                                />
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      <button
                        type="button"
                        onClick={startFaceCapture}
                        disabled={faceEnrolled}
                        className={`w-full py-3 px-4 rounded-md text-sm font-medium transition-all ${
                          faceEnrolled
                            ? 'bg-green-600 text-white cursor-not-allowed'
                            : 'bg-primary-600 hover:bg-primary-700 text-white'
                        }`}
                      >
                        {faceEnrolled ? '✅ Face Samples Captured (3/3)' : '📷 Start Camera'}
                      </button>
                    </>
                  )}
                </div>

                {/* Voice Sample */}
                <div className={`border-2 rounded-lg p-6 transition-all ${
                  voiceBlobs.length >= 3 ? 'border-green-500 bg-green-50' : 'border-gray-200 bg-white'
                }`}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center">
                      <div className={`p-2 rounded-full ${voiceBlobs.length >= 3 ? 'bg-green-500' : 'bg-primary-600'}`}>
                        <Mic className="h-5 w-5 text-white" />
                      </div>
                      <div className="ml-3">
                        <h4 className="font-semibold text-gray-900">Voice Biometric</h4>
                        <p className="text-xs text-gray-500">Record 3 voice samples (5-10 seconds each)</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <span className={`text-sm font-medium ${voiceBlobs.length >= 3 ? 'text-green-600' : 'text-gray-600'}`}>
                        {voiceBlobs.length}/3 samples
                      </span>
                      {voiceBlobs.length >= 3 && (
                        <CheckCircle className="h-5 w-5 ml-2 text-green-600" />
                      )}
                    </div>
                  </div>

                  {isRecordingVoice && (
                    <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-md">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-red-900">🎤 Recording sample {voiceBlobs.length + 1}/3...</span>
                        <span className="text-lg font-bold text-red-600">{voiceRecordingTime}s / 10s</span>
                      </div>
                      <div className="w-full bg-red-200 rounded-full h-2">
                        <div
                          className="bg-red-600 h-2 rounded-full transition-all duration-1000"
                          style={{ width: `${(voiceRecordingTime / 10) * 100}%` }}
                        ></div>
                      </div>
                      <p className="text-xs text-red-700 mt-2">
                        💡 Speak clearly: "My name is [Your Name] and I am a medical professional"
                      </p>
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={isRecordingVoice ? stopVoiceRecording : startVoiceRecording}
                    disabled={voiceBlobs.length >= 3 && !isRecordingVoice}
                    className={`w-full py-3 px-4 rounded-md text-sm font-medium transition-all ${
                      isRecordingVoice
                        ? 'bg-red-600 hover:bg-red-700 text-white animate-pulse'
                        : voiceBlobs.length >= 3
                        ? 'bg-green-600 text-white cursor-not-allowed'
                        : 'bg-primary-600 hover:bg-primary-700 text-white'
                    }`}
                  >
                    {isRecordingVoice ? '⏹️ Stop Recording' : voiceBlobs.length >= 3 ? '✅ All Samples Captured' : `🎤 Record Sample ${voiceBlobs.length + 1}/3`}
                  </button>
                </div>

                {/* Keystroke Pattern */}
                <div className={`border-2 rounded-lg p-6 transition-all ${
                  keystrokeData.length >= 3 ? 'border-green-500 bg-green-50' : 'border-gray-200 bg-white'
                }`}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center">
                      <div className={`p-2 rounded-full ${keystrokeData.length >= 3 ? 'bg-green-500' : 'bg-primary-600'}`}>
                        <Keyboard className="h-5 w-5 text-white" />
                      </div>
                      <div className="ml-3">
                        <h4 className="font-semibold text-gray-900">Keystroke Dynamics</h4>
                        <p className="text-xs text-gray-500">Capture 3 typing samples</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      {keystrokeData.length >= 3 ? (
                        <div className="flex items-center text-green-600">
                          <CheckCircle className="h-5 w-5 mr-1" />
                          <span className="text-sm font-medium">Complete</span>
                        </div>
                      ) : (
                        <span className="text-sm font-medium text-gray-600">{keystrokeData.length}/3 samples</span>
                      )}
                    </div>
                  </div>

                  {isCapturingKeystroke && (
                    <div className="mb-4">
                      <div className="mb-2 p-3 bg-blue-50 border border-blue-200 rounded-md">
                        <p className="text-sm font-medium text-blue-900 mb-1">
                          Sample {currentKeystrokeSample}/3: Type this phrase EXACTLY (no copy/paste):
                        </p>
                        <p className="text-sm font-mono text-blue-700 font-semibold">
                          "The quick brown fox jumps over the lazy dog"
                        </p>
                      </div>
                      <textarea
                        className="w-full p-3 border-2 border-primary-500 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                        rows="3"
                        placeholder="Start typing here..."
                        value={typedText}
                        onChange={handleKeystrokeTextChange}
                        onPaste={handleKeystrokePaste}
                        onCut={(e) => e.preventDefault()}
                        onCopy={(e) => e.preventDefault()}
                        onKeyDown={(e) => keystrokeCapture.current.handleKeyDown(e)}
                        onKeyUp={(e) => keystrokeCapture.current.handleKeyUp(e)}
                        autoFocus
                      />
                      <div className="flex justify-between items-center mt-1">
                        <p className="text-xs text-gray-500">
                          Characters typed: {typedText.length} / {REQUIRED_PHRASE.length}
                        </p>
                        {typedText.length > 0 && (
                          <p className={`text-xs font-medium ${
                            typedText.trim() === REQUIRED_PHRASE ? 'text-green-600' : 'text-orange-600'
                          }`}>
                            {typedText.trim() === REQUIRED_PHRASE ? '✓ Match!' : '⚠ Keep typing...'}
                          </p>
                        )}
                      </div>
                    </div>
                  )}

                  {isCapturingKeystroke ? (
                    <button
                      type="button"
                      onClick={completeKeystrokeSample}
                      disabled={typedText.trim() !== REQUIRED_PHRASE}
                      className={`w-full py-3 px-4 rounded-md text-sm font-medium transition-all ${
                        typedText.trim() === REQUIRED_PHRASE
                          ? 'bg-green-600 hover:bg-green-700 text-white'
                          : 'bg-gray-400 text-white cursor-not-allowed'
                      }`}
                    >
                      {typedText.trim() === REQUIRED_PHRASE ? '✅ Complete Sample' : '⏳ Finish typing...'}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={startKeystrokeCapture}
                      disabled={keystrokeData.length >= 3}
                      className={`w-full py-3 px-4 rounded-md text-sm font-medium transition-all ${
                        keystrokeData.length >= 3
                          ? 'bg-green-600 text-white cursor-not-allowed'
                          : 'bg-primary-600 hover:bg-primary-700 text-white'
                      }`}
                    >
                      {keystrokeData.length >= 3
                        ? '✅ All Samples Captured'
                        : `⌨️ Capture Sample ${keystrokeData.length + 1}/3`
                      }
                    </button>
                  )}
                </div>

                {/* Mouse Pattern */}
                <div className={`border-2 rounded-lg p-6 transition-all ${
                  mouseData.length > 0 ? 'border-green-500 bg-green-50' : 'border-gray-200 bg-white'
                }`}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center">
                      <div className={`p-2 rounded-full ${mouseData.length > 0 ? 'bg-green-500' : 'bg-primary-600'}`}>
                        <Mouse className="h-5 w-5 text-white" />
                      </div>
                      <div className="ml-3">
                        <h4 className="font-semibold text-gray-900">Mouse Movement Pattern</h4>
                        <p className="text-xs text-gray-500">Record natural mouse movements</p>
                      </div>
                    </div>
                    {mouseData.length > 0 && (
                      <div className="flex items-center text-green-600">
                        <CheckCircle className="h-5 w-5 mr-1" />
                        <span className="text-sm font-medium">Complete ({mouseData.length} events)</span>
                      </div>
                    )}
                  </div>

                  {isCapturingMouse && (
                    <div className="mb-4">
                      <div className="mb-2 p-3 bg-purple-50 border border-purple-200 rounded-md">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-purple-900">🖱️ Recording mouse movements...</span>
                          <span className="text-lg font-bold text-purple-600">{mouseRecordingTime}s / 15s</span>
                        </div>
                        <div className="w-full bg-purple-200 rounded-full h-2 mt-2">
                          <div
                            className="bg-purple-600 h-2 rounded-full transition-all duration-1000"
                            style={{ width: `${(mouseRecordingTime / 15) * 100}%` }}
                          ></div>
                        </div>
                      </div>
                      <div
                        className="w-full h-48 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg border-2 border-dashed border-purple-300 cursor-crosshair relative overflow-hidden"
                        onMouseMove={(e) => mouseCapture.current.handleMouseMove(e)}
                        onClick={(e) => mouseCapture.current.handleMouseClick(e)}
                      >
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="text-center">
                            <Mouse className="h-12 w-12 text-purple-400 mx-auto mb-2 animate-bounce" />
                            <p className="text-sm text-purple-600 font-medium">Move your mouse naturally</p>
                            <p className="text-xs text-purple-500">Click, drag, and move around</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={isCapturingMouse ? stopMouseCapture : startMouseCapture}
                    disabled={mouseData.length > 0 && !isCapturingMouse}
                    className={`w-full py-3 px-4 rounded-md text-sm font-medium transition-all ${
                      mouseData.length > 0
                        ? 'bg-green-600 text-white cursor-not-allowed'
                        : 'bg-primary-600 hover:bg-primary-700 text-white'
                    }`}
                  >
                    {isCapturingMouse
                      ? '⏹️ Stop Recording'
                      : mouseData.length > 0
                      ? '✅ Mouse Pattern Captured'
                      : '🖱️ Start Mouse Recording'
                    }
                  </button>
                </div>

                <div className="flex gap-4">
                  <button
                    type="button"
                    onClick={handleBack}
                    className="flex-1 py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                  >
                    Back
                  </button>
                  <button
                    type="submit"
                    disabled={loading || voiceBlobs.length < 3 || keystrokeData.length < 3 || mouseData.length === 0}
                    className="flex-1 py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 disabled:opacity-50"
                  >
                    {loading ? 'Registering...' : 'Complete Registration'}
                  </button>
                </div>
              </div>
            )}
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600">
              Already have an account?{' '}
              <Link to="/login" className="font-medium text-primary-600 hover:text-primary-500">
                Sign in here
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Register;

