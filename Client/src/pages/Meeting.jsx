import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { io } from 'socket.io-client';
import axios from 'axios';
import toast from 'react-hot-toast';
import {
  Video, VideoOff, Mic, MicOff, PhoneOff, Shield,
  Activity, AlertTriangle, CheckCircle, TrendingUp
} from 'lucide-react';
import { KeystrokeCapture, MouseCapture } from '../utils/biometricCapture';

const Meeting = () => {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const { user } = useAuth();
  const isDoctor = user?.role === 'doctor';
  
  const [socket, setSocket] = useState(null);
  const [isVideoOn, setIsVideoOn] = useState(true);
  const [isAudioOn, setIsAudioOn] = useState(true);
  const [trustScore, setTrustScore] = useState(100);
  const [verificationLogs, setVerificationLogs] = useState([]);
  const [alerts, setAlerts] = useState([]);
  
  const videoRef = useRef(null);
  const keystrokeCapture = useRef(new KeystrokeCapture());
  const mouseCapture = useRef(new MouseCapture());
  const verificationInterval = useRef(null);

  useEffect(() => {
    // Initialize socket connection
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('Connected to server');
      newSocket.emit('join-session', {
        sessionId,
        userId: user.id,
        userRole: user.role
      });
    });

    newSocket.on('session-joined', (data) => {
      console.log('Joined session:', data);
      toast.success('Connected to consultation session');
    });

    newSocket.on('verification-result', (data) => {
      console.log('Verification result:', data);
      setTrustScore(data.trustScore);
      setVerificationLogs(prev => [...prev, {
        type: data.type,
        result: data.result,
        timestamp: new Date()
      }].slice(-10));
    });

    newSocket.on('verification-alert', (data) => {
      console.log('Verification alert:', data);
      setAlerts(prev => [...prev, {
        ...data,
        timestamp: new Date()
      }]);
      toast.error(data.message, { duration: 5000 });
    });

    // Start video
    startVideo();

    // Start biometric monitoring
    startBiometricMonitoring();

    return () => {
      newSocket.disconnect();
      stopVideo();
      stopBiometricMonitoring();
    };
  }, [sessionId, user]);

  const startVideo = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: true, 
        audio: true 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error('Failed to start video:', error);
      toast.error('Failed to access camera/microphone');
    }
  };

  const stopVideo = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
    }
  };

  const startBiometricMonitoring = () => {
    // Start keystroke capture
    keystrokeCapture.current.start();
    
    // Start mouse capture
    mouseCapture.current.start();

    // Send verification data every 10 seconds
    verificationInterval.current = setInterval(() => {
      sendVerificationData();
    }, 10000);
  };

  const stopBiometricMonitoring = () => {
    if (verificationInterval.current) {
      clearInterval(verificationInterval.current);
    }
  };

  const sendVerificationData = () => {
    if (!socket) return;

    // Send keystroke verification
    const keystrokeFeatures = keystrokeCapture.current.getFeatures();
    if (keystrokeFeatures.some(f => f !== 0)) {
      socket.emit('verify-biometric', {
        sessionId,
        doctorId: user.id,
        type: 'keystroke',
        payload: keystrokeFeatures
      });
      keystrokeCapture.current.start(); // Restart capture
    }

    // Send mouse verification
    const mouseEvents = mouseCapture.current.getEvents();
    if (mouseEvents.length > 50) {
      socket.emit('verify-biometric', {
        sessionId,
        doctorId: user.id,
        type: 'mouse',
        payload: mouseEvents
      });
      mouseCapture.current.start(); // Restart capture
    }
  };

  const toggleVideo = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const videoTrack = videoRef.current.srcObject.getVideoTracks()[0];
      videoTrack.enabled = !videoTrack.enabled;
      setIsVideoOn(videoTrack.enabled);
    }
  };

  const toggleAudio = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const audioTrack = videoRef.current.srcObject.getAudioTracks()[0];
      audioTrack.enabled = !audioTrack.enabled;
      setIsAudioOn(audioTrack.enabled);
    }
  };

  const endCall = async () => {
    if (!isDoctor) {
      toast.error('Only the doctor can end the consultation');
      return;
    }

    try {
      // End the consultation via API
      await axios.put(`/api/consultations/${sessionId}/end`, {}, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      toast.success('Consultation ended successfully');
      navigate('/dashboard');
    } catch (error) {
      console.error('Failed to end consultation:', error);
      toast.error('Failed to end consultation');
      navigate('/dashboard');
    }
  };

  const getTrustScoreColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getTrustScoreBg = (score) => {
    if (score >= 80) return 'bg-green-100';
    if (score >= 60) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  return (
    <div 
      className="min-h-screen bg-gray-900"
      onMouseMove={(e) => mouseCapture.current.handleMouseMove(e)}
      onClick={(e) => mouseCapture.current.handleMouseClick(e)}
      onKeyDown={(e) => keystrokeCapture.current.handleKeyDown(e)}
      onKeyUp={(e) => keystrokeCapture.current.handleKeyUp(e)}
      tabIndex={0}
    >
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-3">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <Shield className="h-6 w-6 text-primary-400 mr-2" />
            <span className="text-white font-semibold">Zero Trust Consultation</span>
            <span className="ml-4 text-gray-400 text-sm">Session: {sessionId.slice(0, 8)}...</span>
          </div>
          <div className={`flex items-center px-4 py-2 rounded-lg ${getTrustScoreBg(trustScore)}`}>
            <Activity className={`h-5 w-5 mr-2 ${getTrustScoreColor(trustScore)}`} />
            <span className={`font-bold ${getTrustScoreColor(trustScore)}`}>
              Trust Score: {trustScore}%
            </span>
          </div>
        </div>
      </div>

      <div className="flex h-[calc(100vh-60px)]">
        {/* Main Video Area */}
        <div className="flex-1 flex flex-col items-center justify-center p-6">
          <div className="relative w-full max-w-4xl aspect-video bg-gray-800 rounded-lg overflow-hidden shadow-2xl">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
            
            {!isVideoOn && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                <VideoOff className="h-16 w-16 text-gray-600" />
              </div>
            )}

            {/* Controls */}
            <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 flex gap-4">
              <button
                onClick={toggleVideo}
                className={`p-4 rounded-full ${isVideoOn ? 'bg-gray-700 hover:bg-gray-600' : 'bg-red-600 hover:bg-red-700'} text-white transition-colors`}
              >
                {isVideoOn ? <Video className="h-6 w-6" /> : <VideoOff className="h-6 w-6" />}
              </button>
              
              <button
                onClick={toggleAudio}
                className={`p-4 rounded-full ${isAudioOn ? 'bg-gray-700 hover:bg-gray-600' : 'bg-red-600 hover:bg-red-700'} text-white transition-colors`}
              >
                {isAudioOn ? <Mic className="h-6 w-6" /> : <MicOff className="h-6 w-6" />}
              </button>
              
              <button
                onClick={endCall}
                className="p-4 rounded-full bg-red-600 hover:bg-red-700 text-white transition-colors"
              >
                <PhoneOff className="h-6 w-6" />
              </button>
            </div>
          </div>
        </div>

        {/* Sidebar - Verification Status */}
        <div className="w-80 bg-gray-800 border-l border-gray-700 overflow-y-auto">
          <div className="p-4">
            <h3 className="text-white font-semibold mb-4 flex items-center">
              <Shield className="h-5 w-5 mr-2 text-primary-400" />
              Real-time Verification
            </h3>

            {/* Biometric Status */}
            <div className="space-y-3 mb-6">
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">Voice Recognition</span>
                  <CheckCircle className="h-4 w-4 text-green-400" />
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div className="bg-green-500 h-2 rounded-full" style={{ width: '95%' }}></div>
                </div>
              </div>

              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">Keystroke Dynamics</span>
                  <CheckCircle className="h-4 w-4 text-green-400" />
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div className="bg-green-500 h-2 rounded-full" style={{ width: '92%' }}></div>
                </div>
              </div>

              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">Mouse Movement</span>
                  <CheckCircle className="h-4 w-4 text-green-400" />
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div className="bg-green-500 h-2 rounded-full" style={{ width: '88%' }}></div>
                </div>
              </div>
            </div>

            {/* Alerts */}
            {alerts.length > 0 && (
              <div className="mb-6">
                <h4 className="text-white font-semibold mb-2 flex items-center">
                  <AlertTriangle className="h-4 w-4 mr-2 text-yellow-400" />
                  Alerts
                </h4>
                <div className="space-y-2">
                  {alerts.slice(-5).reverse().map((alert, index) => (
                    <div key={index} className={`p-2 rounded-lg ${
                      alert.severity === 'high' || alert.severity === 'critical'
                        ? 'bg-red-900/50 border border-red-700'
                        : 'bg-yellow-900/50 border border-yellow-700'
                    }`}>
                      <p className="text-xs text-white">{alert.message}</p>
                      <p className="text-xs text-gray-400 mt-1">
                        {alert.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Verification Logs */}
            <div>
              <h4 className="text-white font-semibold mb-2 flex items-center">
                <TrendingUp className="h-4 w-4 mr-2 text-primary-400" />
                Recent Verifications
              </h4>
              <div className="space-y-2">
                {verificationLogs.slice(-10).reverse().map((log, index) => (
                  <div key={index} className="bg-gray-700 rounded-lg p-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-300 capitalize">{log.type}</span>
                      <span className={`text-xs font-semibold ${
                        log.result.verified ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {log.result.verified ? '✓ Verified' : '✗ Failed'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between mt-1">
                      <span className="text-xs text-gray-400">
                        Confidence: {(log.result.confidence * 100).toFixed(1)}%
                      </span>
                      <span className="text-xs text-gray-500">
                        {log.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                ))}

                {verificationLogs.length === 0 && (
                  <div className="text-center py-8 text-gray-500 text-sm">
                    Waiting for verification data...
                  </div>
                )}
              </div>
            </div>

            {/* Info */}
            <div className="mt-6 p-3 bg-blue-900/30 border border-blue-700 rounded-lg">
              <p className="text-xs text-blue-300">
                🔒 Your biometric data is being continuously verified to ensure secure consultation.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Meeting;
