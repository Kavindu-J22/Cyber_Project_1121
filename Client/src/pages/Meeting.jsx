import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { io } from 'socket.io-client';
import axios from 'axios';
import toast from 'react-hot-toast';
import {
  Video, VideoOff, Mic, MicOff, PhoneOff, Shield,
  Activity, AlertTriangle, CheckCircle, TrendingUp,
  MessageSquare, Send, Users, X, Lock, ArrowRight, Loader2, Mail
} from 'lucide-react';
import { KeystrokeCapture, MouseCapture, FaceCapture, VoiceCapture } from '../utils/biometricCapture';

const ICE_SERVERS = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
  ],
};

const Meeting = () => {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const { user } = useAuth();
  const isDoctor = user?.role === 'doctor';

  // Connection
  const [socket, setSocket] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('waiting'); // waiting | connecting | connected | ended

  // Media controls
  const [isVideoOn, setIsVideoOn] = useState(true);
  const [isAudioOn, setIsAudioOn] = useState(true);
  const [hasRemoteStream, setHasRemoteStream] = useState(false);

  // Biometric verification
  const [trustScore, setTrustScore] = useState(100);
  const [verificationLogs, setVerificationLogs] = useState([]);
  const [alerts, setAlerts] = useState([]);

  // Per-modality confidence scores (0-1, null = no data yet)
  const [faceConf, setFaceConf] = useState(null);
  const [voiceConf, setVoiceConf] = useState(null);
  const [keystrokeConf, setKeystrokeConf] = useState(0.5); // 50 % default when chat is closed
  const [mouseConf, setMouseConf] = useState(null);

  // Doctor's biometric scores as seen by the patient
  const [doctorScores, setDoctorScores] = useState({ face: null, voice: null, keystroke: null, mouse: null });

  // Chat
  const [showChat, setShowChat] = useState(false);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [unreadCount, setUnreadCount] = useState(0);

  // Session state
  const [consultationEnded, setConsultationEnded] = useState(false);

  // Lockout state (15-min continuous low-trust)
  const [isLockedOut, setIsLockedOut] = useState(false);
  const [lockoutStep, setLockoutStep] = useState('otp'); // 'otp' | 'slide'
  const [lockoutOtp, setLockoutOtp] = useState(['', '', '', '', '', '']);
  const [lockoutOtpLoading, setLockoutOtpLoading] = useState(false);
  const [lockoutSliderPos, setLockoutSliderPos] = useState(0);
  const [lockoutIsSliding, setLockoutIsSliding] = useState(false);
  const [doctorIsLockedOut, setDoctorIsLockedOut] = useState(false); // patient view

  // Refs
  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const localStreamRef = useRef(null);
  const socketRef = useRef(null);
  const remoteSocketIdRef = useRef(null);
  const chatBottomRef = useRef(null);
  const showChatRef = useRef(false);
  const keystrokeCapture = useRef(new KeystrokeCapture());
  const mouseCapture = useRef(new MouseCapture());
  const faceCapture = useRef(new FaceCapture());
  const voiceCaptureRef = useRef(new VoiceCapture());
  const verificationInterval = useRef(null);
  const faceIntervalRef = useRef(null);
  const voiceIntervalRef = useRef(null);
  // Refs to track media state inside async interval callbacks (avoid stale closures)
  const isVideoOnRef = useRef(true);
  const isAudioOnRef = useRef(true);
  // Lockout refs
  const lowTrustStartRef = useRef(null);
  const isLockedOutRef = useRef(false);
  const lockoutOtpRefs = useRef([]);
  const lockoutSliderRef = useRef(null);

  useEffect(() => {
    let mounted = true;

    // Helper: build a RTCPeerConnection and wire up events
    const createPeerConnection = () => {
      const pc = new RTCPeerConnection(ICE_SERVERS);

      pc.onicecandidate = (event) => {
        if (event.candidate && remoteSocketIdRef.current) {
          socketRef.current?.emit('ice-candidate', {
            candidate: event.candidate,
            targetSocketId: remoteSocketIdRef.current,
            sessionId,
          });
        }
      };

      pc.ontrack = (event) => {
        if (remoteVideoRef.current && event.streams[0]) {
          remoteVideoRef.current.srcObject = event.streams[0];
          setHasRemoteStream(true);
          setConnectionStatus('connected');
        }
      };

      pc.onconnectionstatechange = () => {
        if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
          setConnectionStatus('waiting');
          setHasRemoteStream(false);
        }
      };

      // Attach existing local tracks so remote side receives them
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach(track =>
          pc.addTrack(track, localStreamRef.current)
        );
      }
      peerConnectionRef.current = pc;
      return pc;
    };

    const init = async () => {
      // 1. Get local camera + mic (with graceful fallback)
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        localStreamRef.current = stream;
        if (localVideoRef.current) localVideoRef.current.srcObject = stream;
      } catch {
        // Fallback 1: try audio-only
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: false, audio: true });
          localStreamRef.current = stream;
          if (localVideoRef.current) localVideoRef.current.srcObject = stream;
          toast('Camera unavailable — joined with audio only', { icon: '⚠️' });
        } catch {
          // Fallback 2: notify user — they can still see remote video
          toast.error('Could not access camera/microphone. Check browser permissions and try again.');
        }
      }

      // 2. Connect socket
      const sock = io('http://localhost:5000');
      socketRef.current = sock;
      setSocket(sock);

      const myName = isDoctor
        ? `Dr. ${user.firstName || user.name || 'Doctor'}`
        : (user.fullName || user.name || 'Patient');

      sock.on('connect', () => {
        sock.emit('join-session', { sessionId, userId: user.id, userRole: user.role, userName: myName });
      });

      sock.on('session-joined', () => toast.success('Connected to consultation room'));

      // Other participant already in room → WE create the offer
      sock.on('user-joined', async ({ socketId }) => {
        if (!mounted) return;
        remoteSocketIdRef.current = socketId;
        setConnectionStatus('connecting');
        const pc = createPeerConnection();
        try {
          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);
          sock.emit('offer', { offer, targetSocketId: socketId, sessionId });
        } catch (err) { console.error('Offer error:', err); }
      });

      // We received an offer → create answer
      sock.on('offer', async ({ offer, fromSocketId }) => {
        if (!mounted) return;
        remoteSocketIdRef.current = fromSocketId;
        setConnectionStatus('connecting');
        const pc = createPeerConnection();
        try {
          await pc.setRemoteDescription(new RTCSessionDescription(offer));
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);
          sock.emit('answer', { answer, targetSocketId: fromSocketId, sessionId });
        } catch (err) { console.error('Answer error:', err); }
      });

      sock.on('answer', async ({ answer }) => {
        if (!mounted || !peerConnectionRef.current) return;
        try {
          await peerConnectionRef.current.setRemoteDescription(new RTCSessionDescription(answer));
        } catch (err) { console.error('setRemoteDescription error:', err); }
      });

      sock.on('ice-candidate', async ({ candidate }) => {
        if (!mounted || !peerConnectionRef.current) return;
        try {
          await peerConnectionRef.current.addIceCandidate(new RTCIceCandidate(candidate));
        } catch (err) { console.error('ICE error:', err); }
      });

      // Other user left the room
      sock.on('user-left', ({ socketId }) => {
        if (socketId === remoteSocketIdRef.current) {
          remoteSocketIdRef.current = null;
          setHasRemoteStream(false);
          setConnectionStatus('waiting');
          if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null;
          toast('Other participant left the room', { icon: '👋' });
        }
      });

      // Doctor ended the session → both sides close
      sock.on('session-ended', () => {
        if (!mounted) return;
        setConsultationEnded(true);
        localStreamRef.current?.getTracks().forEach(t => t.stop());
        peerConnectionRef.current?.close();
        toast.error('Consultation has been ended');
        setTimeout(() => { if (mounted) navigate('/dashboard'); }, 5000);
      });

      // Chat messages
      sock.on('chat-message', (msg) => {
        if (!mounted) return;
        setMessages(prev => [...prev, msg]);
        if (!showChatRef.current) setUnreadCount(prev => prev + 1);
      });

      // Biometric verification results
      sock.on('verification-result', (data) => {
        if (!mounted) return;
        setTrustScore(data.trustScore);
        setVerificationLogs(prev => [...prev, {
          type: data.type, result: data.result, timestamp: new Date()
        }].slice(-10));
      });

      sock.on('verification-alert', (data) => {
        if (!mounted) return;
        setAlerts(prev => [...prev, { ...data, timestamp: new Date() }]);
        toast.error(data.message, { duration: 5000 });
      });

      // Doctor's biometric scores (received by the patient)
      sock.on('doctor-biometric-update', ({ scores }) => {
        if (!mounted) return;
        setDoctorScores(scores);
      });

      // Doctor lockout status (received by the patient)
      sock.on('doctor-lockout-status', ({ isLocked }) => {
        if (!mounted) return;
        setDoctorIsLockedOut(isLocked);
      });

      // ── Biometric Monitoring (doctor only) ─────────────────────
      if (isDoctor) {
        keystrokeCapture.current.start();
        mouseCapture.current.start();

        // Keystroke + Mouse: every 10 s via REST
        verificationInterval.current = setInterval(async () => {
          const token = localStorage.getItem('token');

          // Keystroke — ONLY verify when chat is open and the doctor has typed something.
          // When chat is closed the score stays at the 50 % neutral default.
          if (showChatRef.current) {
            const typedEvents = keystrokeCapture.current.events;
            if (typedEvents && typedEvents.length >= 5) {
              const kf = keystrokeCapture.current.getFeatures();
              try {
                const r = await axios.post('/api/verification/keystroke',
                  { keystrokeSample: kf },
                  { headers: { Authorization: `Bearer ${token}` } }
                );
                if (mounted) {
                  setKeystrokeConf(r.data.data?.confidence ?? 0.5);
                  keystrokeCapture.current.start(); // reset buffer ONLY after a successful API call
                }
              } catch (e) { console.error('Keystroke verification error:', e); }
              // On failure: keep accumulating — do NOT reset buffer
            }
            // < 5 events: keep accumulating across intervals; do NOT reset
          } else {
            // Chat closed — keep score at 50 % (already set when chat closed)
            keystrokeCapture.current.start(); // ensure buffer is clear
          }

          // Mouse — accumulate events until we have enough (≥ 20 events).
          const me = mouseCapture.current.getEvents();
          if (me.length >= 20) {
            try {
              const r = await axios.post('/api/verification/mouse',
                { mouseEvents: me },
                { headers: { Authorization: `Bearer ${token}` } }
              );
              if (mounted) setMouseConf(r.data.data?.confidence ?? null);
              mouseCapture.current.start(); // reset buffer only on success
            } catch (e) { console.error('Mouse verification error:', e); }
            // On failure: keep accumulating — do NOT reset
          } else {
            // Not enough mouse events this cycle → apply Zero Trust decay.
            // Each inactive 10-s cycle nudges 15 % toward the neutral 50 % score
            // so that a frozen score doesn't persist indefinitely when the user
            // stops moving the mouse.  null stays null (widget shows "Move mouse…")
            if (mounted) {
              setMouseConf(prev =>
                prev === null ? null : +(prev * 0.85 + 0.5 * 0.15).toFixed(4)
              );
            }
          }
        }, 10000);

        // Face: every 5 s via REST
        faceIntervalRef.current = setInterval(async () => {
          if (!isVideoOnRef.current || !localVideoRef.current?.srcObject) return;
          try {
            const frame = await faceCapture.current.captureFrame(localVideoRef.current);
            const fd = new FormData();
            fd.append('faceSample', frame, 'face.jpg');
            const token = localStorage.getItem('token');
            const r = await axios.post('/api/verification/face', fd, {
              headers: { Authorization: `Bearer ${token}` }
            });
            if (mounted) setFaceConf(r.data.data?.confidence_score ?? null);
          } catch (e) { console.error('Face verification error:', e); }
        }, 5000);

        // Voice: record 5s every 20s via REST — reuse existing WebRTC audio track
        let isRecordingVoice = false;
        voiceIntervalRef.current = setInterval(async () => {
          if (!isAudioOnRef.current || isRecordingVoice || !localStreamRef.current) return;
          const audioTracks = localStreamRef.current.getAudioTracks();
          if (!audioTracks.length) return;
          isRecordingVoice = true;
          try {
            const audioStream = new MediaStream(audioTracks);
            const chunks = [];
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
              ? 'audio/webm;codecs=opus'
              : MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
            const recorder = new MediaRecorder(audioStream, mimeType ? { mimeType } : {});
            recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
            recorder.start();
            await new Promise(res => setTimeout(res, 5000));
            recorder.stop();
            await new Promise(res => { recorder.onstop = res; });
            if (!mounted || !isAudioOnRef.current) return;
            const rawBlob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
            const wavBlob = await voiceCaptureRef.current.convertToWav(rawBlob);
            if (wavBlob) {
              const fd = new FormData();
              fd.append('voiceSample', wavBlob, 'voice.wav');
              const token = localStorage.getItem('token');
              const r = await axios.post('/api/verification/voice', fd, {
                headers: { Authorization: `Bearer ${token}` }
              });
              if (mounted) setVoiceConf(r.data.data?.confidence_score ?? null);
            }
          } catch (e) { console.error('Voice verification error:', e); }
          finally { isRecordingVoice = false; }
        }, 20000);
      }
    };

    init();

    return () => {
      mounted = false;
      localStreamRef.current?.getTracks().forEach(t => t.stop());
      peerConnectionRef.current?.close();
      socketRef.current?.disconnect();
      if (verificationInterval.current) clearInterval(verificationInterval.current);
      if (faceIntervalRef.current) clearInterval(faceIntervalRef.current);
      if (voiceIntervalRef.current) clearInterval(voiceIntervalRef.current);
      // Stop any in-progress voice recording
      voiceCaptureRef.current.stop().catch(() => {});
    };
  }, [sessionId, user, isDoctor, navigate]);

  // Keep showChatRef in sync so the socket callback can read it without stale closure
  useEffect(() => {
    showChatRef.current = showChat;
    if (showChat) {
      setUnreadCount(0);
      // Start a fresh keystroke buffer each time chat opens
      keystrokeCapture.current.start();
    } else {
      // Chat closed — reset keystroke score to neutral 50 % and clear the buffer
      setKeystrokeConf(0.5);
      keystrokeCapture.current.start();
    }
  }, [showChat]);

  // Auto-scroll chat to bottom on new message
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Compute overall trust score from live per-modality confidences (doctor only)
  // Also runs the 15-minute low-trust lockout timer.
  useEffect(() => {
    if (!isDoctor) return;
    const values = [faceConf, voiceConf, keystrokeConf, mouseConf].filter(v => v !== null);
    if (values.length > 0) {
      const avg = values.reduce((a, b) => a + b, 0) / values.length;
      const score = Math.round(avg * 100);
      setTrustScore(score);

      // 15-minute low-trust lockout logic
      if (!isLockedOutRef.current) {
        if (score < 50) {
          if (!lowTrustStartRef.current) {
            lowTrustStartRef.current = Date.now();
          } else if (Date.now() - lowTrustStartRef.current >= 15 * 60 * 1000) {
            // Trust has been below 50% for 15 continuous minutes → trigger lockout
            isLockedOutRef.current = true;
            setIsLockedOut(true);
            // Disable mic and camera
            const vt = localStreamRef.current?.getVideoTracks()[0];
            const at = localStreamRef.current?.getAudioTracks()[0];
            if (vt) { vt.enabled = false; setIsVideoOn(false); isVideoOnRef.current = false; setFaceConf(null); }
            if (at) { at.enabled = false; setIsAudioOn(false); isAudioOnRef.current = false; setVoiceConf(null); }
            // Notify patient
            socketRef.current?.emit('doctor-lockout-status', { sessionId, isLocked: true });
            toast.error('⚠️ Security alert: Trust score below 50% for 15 minutes. Verification required.', { duration: 8000 });
          }
        } else {
          // Trust recovered — reset timer
          lowTrustStartRef.current = null;
        }
      }
    }
  }, [faceConf, voiceConf, keystrokeConf, mouseConf, isDoctor, sessionId]);

  // Emit doctor's biometric scores to the patient via socket
  useEffect(() => {
    if (!isDoctor || !socketRef.current || !sessionId) return;
    socketRef.current.emit('doctor-biometric-update', {
      sessionId,
      scores: { face: faceConf, voice: voiceConf, keystroke: keystrokeConf, mouse: mouseConf }
    });
  }, [faceConf, voiceConf, keystrokeConf, mouseConf, isDoctor, sessionId]);

  const toggleVideo = () => {
    const t = localStreamRef.current?.getVideoTracks()[0];
    if (t) {
      t.enabled = !t.enabled;
      setIsVideoOn(t.enabled);
      isVideoOnRef.current = t.enabled;
      // Clear stale face confidence when camera turns off
      if (!t.enabled) setFaceConf(null);
    }
  };

  const toggleAudio = () => {
    const t = localStreamRef.current?.getAudioTracks()[0];
    if (t) {
      t.enabled = !t.enabled;
      setIsAudioOn(t.enabled);
      isAudioOnRef.current = t.enabled;
      // Clear stale voice confidence when mic turns off
      if (!t.enabled) setVoiceConf(null);
    }
  };

  // ── Lockout: auto-send OTP when lockout triggers (doctor only) ──────────────
  useEffect(() => {
    if (!isDoctor || !isLockedOut) return;
    const token = localStorage.getItem('token');
    setLockoutStep('otp');
    setLockoutOtp(['', '', '', '', '', '']);
    setLockoutSliderPos(0);
    axios.post('/api/otp/consultation/send', {}, {
      headers: { Authorization: `Bearer ${token}` }
    }).catch(err => console.error('Failed to send lockout OTP:', err));
  }, [isLockedOut, isDoctor]);

  // ── Lockout: OTP digit change ─────────────────────────────────────────────
  const handleLockoutOtpChange = (index, value) => {
    const v = value.replace(/[^0-9]/g, '').slice(-1);
    const next = [...lockoutOtp];
    next[index] = v;
    setLockoutOtp(next);
    if (v && index < 5) {
      lockoutOtpRefs.current[index + 1]?.focus();
    }
  };

  const handleLockoutOtpKeyDown = (index, e) => {
    if (e.key === 'Backspace' && !lockoutOtp[index] && index > 0) {
      lockoutOtpRefs.current[index - 1]?.focus();
    }
  };

  // ── Lockout: verify OTP ───────────────────────────────────────────────────
  const handleLockoutOtpSubmit = async () => {
    const code = lockoutOtp.join('');
    if (code.length < 6) return;
    setLockoutOtpLoading(true);
    try {
      const token = localStorage.getItem('token');
      await axios.post('/api/otp/consultation/verify', { otp: code }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setLockoutStep('slide');
    } catch (err) {
      toast.error('Invalid OTP. Please try again.');
      setLockoutOtp(['', '', '', '', '', '']);
      lockoutOtpRefs.current[0]?.focus();
    } finally {
      setLockoutOtpLoading(false);
    }
  };

  // ── Lockout: resend OTP ───────────────────────────────────────────────────
  const resendLockoutOtp = async () => {
    try {
      const token = localStorage.getItem('token');
      await axios.post('/api/otp/consultation/resend', {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      toast.success('OTP resent to your registered email.');
    } catch (err) {
      toast.error('Could not resend OTP. Try again.');
    }
  };

  // ── Lockout: resolve — restore mic + camera ───────────────────────────────
  const resolveLockout = () => {
    const vt = localStreamRef.current?.getVideoTracks()[0];
    const at = localStreamRef.current?.getAudioTracks()[0];
    if (vt) { vt.enabled = true; setIsVideoOn(true); isVideoOnRef.current = true; }
    if (at) { at.enabled = true; setIsAudioOn(true); isAudioOnRef.current = true; }
    isLockedOutRef.current = false;
    lowTrustStartRef.current = null;
    setIsLockedOut(false);
    setLockoutStep('otp');
    setLockoutOtp(['', '', '', '', '', '']);
    setLockoutSliderPos(0);
    socketRef.current?.emit('doctor-lockout-status', { sessionId, isLocked: false });
    toast.success('✅ Identity verified. Mic and camera restored.');
  };

  // ── Lockout: slider drag logic ────────────────────────────────────────────
  const handleSliderMouseDown = (e) => {
    e.preventDefault();
    setLockoutIsSliding(true);
  };
  const handleSliderTouchStart = () => setLockoutIsSliding(true);

  useEffect(() => {
    if (!lockoutIsSliding) return;
    const track = lockoutSliderRef.current;

    const onMove = (clientX) => {
      if (!track) return;
      const rect = track.getBoundingClientRect();
      const pct = Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
      setLockoutSliderPos(pct);
      if (pct >= 85) {
        setLockoutIsSliding(false);
        setLockoutSliderPos(100);
        setTimeout(resolveLockout, 300);
      }
    };

    const onMouseMove = (e) => onMove(e.clientX);
    const onTouchMove = (e) => onMove(e.touches[0].clientX);
    const onEnd = () => {
      setLockoutIsSliding(false);
      setLockoutSliderPos(p => (p < 85 ? 0 : p)); // snap back if not far enough
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onEnd);
    window.addEventListener('touchmove', onTouchMove);
    window.addEventListener('touchend', onEnd);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onEnd);
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('touchend', onEnd);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lockoutIsSliding]);

  const endCall = async () => {
    if (!isDoctor) { toast.error('Only the doctor can end the consultation'); return; }
    try {
      await axios.put(`/api/consultations/${sessionId}/end`, {}, {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      });
      // Tell all participants the session is over
      socketRef.current?.emit('end-session', { sessionId });
      setConsultationEnded(true);
      localStreamRef.current?.getTracks().forEach(t => t.stop());
      peerConnectionRef.current?.close();
      toast.success('Consultation ended successfully');
      setTimeout(() => navigate('/dashboard'), 3000);
    } catch (error) {
      console.error('Failed to end consultation:', error);
      toast.error('Failed to end consultation');
    }
  };

  const sendChatMessage = () => {
    if (!newMessage.trim() || !socketRef.current) return;
    const msg = {
      sessionId,
      message: newMessage.trim(),
      senderName: isDoctor
        ? `Dr. ${user.firstName || user.name || 'Doctor'}`
        : (user.fullName || user.name || 'Patient'),
      senderRole: user.role,
      senderId: user.id,
      timestamp: new Date().toISOString(),
    };
    socketRef.current.emit('chat-message', msg);
    setNewMessage('');
  };

  const trustColor = (s) => s >= 80 ? 'text-green-400' : s >= 60 ? 'text-yellow-400' : 'text-red-400';

  // ── Consultation Ended Overlay ───────────────────────────────
  if (consultationEnded) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="bg-gray-800 rounded-2xl p-12 shadow-2xl border border-gray-700 max-w-md mx-auto text-center">
          <CheckCircle className="h-20 w-20 text-green-400 mx-auto mb-6" />
          <h2 className="text-3xl font-bold text-white mb-4">Consultation Ended</h2>
          <p className="text-gray-400 mb-8">
            The consultation has been completed. You will be redirected to your dashboard shortly.
          </p>
          <button
            onClick={() => navigate('/dashboard')}
            className="w-full px-6 py-3 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition-colors"
          >
            Go to Dashboard Now
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
    <div
      className="min-h-screen bg-gray-900 flex flex-col"
      onMouseMove={(e) => mouseCapture.current.handleMouseMove(e)}
      onClick={(e) => mouseCapture.current.handleMouseClick(e)}
      tabIndex={0}
    >
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-3 flex justify-between items-center shrink-0">
        <div className="flex items-center gap-3">
          <Shield className="h-6 w-6 text-primary-400" />
          <span className="text-white font-semibold">Zero Trust Consultation</span>
          <span className="text-gray-400 text-sm">Session: {sessionId.slice(0, 8)}...</span>
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
            connectionStatus === 'connected'  ? 'bg-green-900/60 text-green-300' :
            connectionStatus === 'connecting' ? 'bg-yellow-900/60 text-yellow-300' :
            'bg-gray-700 text-gray-400'
          }`}>
            {connectionStatus === 'connected'  ? '● Connected' :
             connectionStatus === 'connecting' ? '● Connecting...' :
             '● Waiting for participant...'}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center px-3 py-2 rounded-lg bg-gray-700">
            <Activity className={`h-4 w-4 mr-2 ${trustColor(trustScore)}`} />
            <span className={`font-bold text-sm ${trustColor(trustScore)}`}>Trust: {trustScore}%</span>
          </div>
          {/* Chat toggle button */}
          <button
            onClick={() => setShowChat(p => !p)}
            className={`relative p-2 rounded-lg transition-colors ${showChat ? 'bg-primary-600' : 'bg-gray-700 hover:bg-gray-600'} text-white`}
            title="Toggle Chat"
          >
            <MessageSquare className="h-5 w-5" />
            {unreadCount > 0 && (
              <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                {unreadCount}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Video Area ─────────────────────────────────────────── */}
        <div className="flex-1 flex flex-col relative bg-gray-900 p-4">

          {/* Remote video (large – shows the other participant) */}
          <div className="flex-1 relative bg-gray-800 rounded-xl overflow-hidden shadow-2xl">
            <video ref={remoteVideoRef} autoPlay playsInline className="w-full h-full object-cover" />
            {!hasRemoteStream && (
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <Users className="h-24 w-24 text-gray-600 mb-4" />
                <p className="text-gray-400 text-lg font-medium">
                  {connectionStatus === 'waiting' ? 'Waiting for other participant to join…' : 'Connecting video…'}
                </p>
                <p className="text-gray-500 text-sm mt-2">
                  {isDoctor ? 'Patient will appear here when they join' : 'Doctor will appear here when connected'}
                </p>
              </div>
            )}
            {hasRemoteStream && (
              <div className="absolute top-4 left-4 bg-black/50 text-white text-sm px-3 py-1 rounded-lg">
                {isDoctor ? '🧑 Patient' : '👨‍⚕️ Doctor'}
              </div>
            )}
            {/* Patient: waiting overlay when doctor is locked out */}
            {!isDoctor && doctorIsLockedOut && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm z-20">
                <div className="bg-gray-800 border border-yellow-600 rounded-2xl p-8 max-w-sm text-center shadow-2xl">
                  <Lock className="h-12 w-12 text-yellow-400 mx-auto mb-4 animate-pulse" />
                  <h3 className="text-white text-xl font-bold mb-2">Security Verification</h3>
                  <p className="text-gray-300 text-sm mb-4">
                    Your doctor is currently undergoing identity verification. This is a routine Zero Trust security check.
                  </p>
                  <div className="flex items-center justify-center gap-2 text-yellow-400 text-sm">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Waiting for doctor to complete verification…</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Local video – picture-in-picture (bottom-right) */}
          <div className="absolute bottom-24 right-8 w-48 h-36 bg-gray-800 rounded-xl overflow-hidden border-2 border-gray-600 shadow-xl z-10">
            <video ref={localVideoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
            {!isVideoOn && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                <VideoOff className="h-8 w-8 text-gray-500" />
              </div>
            )}
            <div className="absolute bottom-1 left-2 text-white text-xs bg-black/50 px-2 py-0.5 rounded">You</div>
          </div>

          {/* Controls bar */}
          <div className="flex justify-center items-center gap-4 py-4 shrink-0">
            <button onClick={toggleAudio} title={isAudioOn ? 'Mute' : 'Unmute'}
              className={`p-4 rounded-full shadow-lg text-white transition-colors ${isAudioOn ? 'bg-gray-700 hover:bg-gray-600' : 'bg-red-600 hover:bg-red-700'}`}>
              {isAudioOn ? <Mic className="h-6 w-6" /> : <MicOff className="h-6 w-6" />}
            </button>
            <button onClick={toggleVideo} title={isVideoOn ? 'Turn off camera' : 'Turn on camera'}
              className={`p-4 rounded-full shadow-lg text-white transition-colors ${isVideoOn ? 'bg-gray-700 hover:bg-gray-600' : 'bg-red-600 hover:bg-red-700'}`}>
              {isVideoOn ? <Video className="h-6 w-6" /> : <VideoOff className="h-6 w-6" />}
            </button>
            {isDoctor && (
              <button onClick={endCall} title="End Consultation"
                className="p-4 rounded-full bg-red-600 hover:bg-red-700 text-white shadow-lg transition-colors">
                <PhoneOff className="h-6 w-6" />
              </button>
            )}
          </div>
        </div>

        {/* ── Right Sidebar ──────────────────────────────────────── */}
        <div className="w-80 bg-gray-800 border-l border-gray-700 flex flex-col overflow-hidden">
          {showChat ? (
            /* Chat Panel */
            <div className="flex flex-col h-full">
              <div className="p-4 border-b border-gray-700 flex items-center justify-between shrink-0">
                <h3 className="text-white font-semibold flex items-center gap-2">
                  <MessageSquare className="h-5 w-5 text-primary-400" />Chat
                </h3>
                <button onClick={() => setShowChat(false)} className="text-gray-400 hover:text-white">
                  <X className="h-4 w-4" />
                </button>
              </div>
              {/* Message list */}
              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {messages.length === 0 && (
                  <p className="text-center text-gray-500 text-sm mt-8">No messages yet. Start the conversation!</p>
                )}
                {messages.map((msg, idx) => {
                  const isOwn = msg.senderId === user.id;
                  return (
                    <div key={idx} className={`flex flex-col ${isOwn ? 'items-end' : 'items-start'}`}>
                      <span className="text-xs text-gray-400 mb-1">
                        {msg.senderName} · {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                      <div className={`max-w-[90%] px-4 py-2 rounded-2xl text-sm break-words ${
                        isOwn ? 'bg-primary-600 text-white rounded-br-sm' : 'bg-gray-700 text-gray-100 rounded-bl-sm'
                      }`}>
                        {msg.message}
                      </div>
                    </div>
                  );
                })}
                <div ref={chatBottomRef} />
              </div>
              {/* Message input */}
              <div className="p-4 border-t border-gray-700 shrink-0">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    onKeyDown={(e) => {
                      // Capture keystroke timing for biometric verification (chat-only scope)
                      if (isDoctor) keystrokeCapture.current.handleKeyDown(e);
                      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
                    }}
                    onKeyUp={(e) => {
                      if (isDoctor) keystrokeCapture.current.handleKeyUp(e);
                    }}
                    placeholder="Type a message…"
                    className="flex-1 bg-gray-700 text-white placeholder-gray-400 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                  />
                  <button onClick={sendChatMessage} disabled={!newMessage.trim()}
                    className="p-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                    <Send className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          ) : isDoctor ? (
            /* ── Doctor: Live Biometric Verification Panel ── */
            <div className="overflow-y-auto p-4 space-y-3">
              <h3 className="text-white font-semibold flex items-center gap-2 mb-1">
                <Shield className="h-5 w-5 text-primary-400" />Real-time Verification
              </h3>

              {/* Overall Trust Score */}
              <div className="bg-gray-700/60 rounded-lg p-3 mb-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-gray-400 uppercase tracking-wide">Overall Trust</span>
                  <span className={`font-bold text-base ${trustColor(trustScore)}`}>{trustScore}%</span>
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div className={`h-2 rounded-full transition-all duration-700 ${
                    trustScore >= 80 ? 'bg-green-500' : trustScore >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                  }`} style={{ width: `${trustScore}%` }} />
                </div>
              </div>

              {/* ── Face Recognition ── */}
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Video className="h-4 w-4 text-blue-400" />
                    <span className="text-sm text-gray-300">Face Recognition</span>
                  </div>
                  {isVideoOn && faceConf !== null
                    ? (faceConf >= 0.5 ? <CheckCircle className="h-4 w-4 text-green-400" /> : <AlertTriangle className="h-4 w-4 text-red-400" />)
                    : <span className="text-xs text-gray-500">{isVideoOn ? '⏳' : '📷'}</span>
                  }
                </div>
                {!isVideoOn ? (
                  <p className="text-xs text-gray-500 italic">Camera Off — face verification paused</p>
                ) : faceConf === null ? (
                  <p className="text-xs text-gray-500 italic">Waiting for first capture…</p>
                ) : (
                  <>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                      <div className={`h-2 rounded-full transition-all duration-700 ${faceConf >= 0.8 ? 'bg-green-500' : faceConf >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${Math.round(faceConf * 100)}%` }} />
                    </div>
                    <p className="text-xs text-gray-400 mt-1 text-right">{Math.round(faceConf * 100)}% confidence</p>
                  </>
                )}
              </div>

              {/* ── Voice Recognition ── */}
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Mic className="h-4 w-4 text-purple-400" />
                    <span className="text-sm text-gray-300">Voice Recognition</span>
                  </div>
                  {isAudioOn && voiceConf !== null
                    ? (voiceConf >= 0.5 ? <CheckCircle className="h-4 w-4 text-green-400" /> : <AlertTriangle className="h-4 w-4 text-red-400" />)
                    : <span className="text-xs text-gray-500">{isAudioOn ? '⏳' : '🎤'}</span>
                  }
                </div>
                {!isAudioOn ? (
                  <p className="text-xs text-gray-500 italic">Mic Off — voice verification paused</p>
                ) : voiceConf === null ? (
                  <p className="text-xs text-gray-500 italic">Recording first sample (5s)…</p>
                ) : (
                  <>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                      <div className={`h-2 rounded-full transition-all duration-700 ${voiceConf >= 0.8 ? 'bg-green-500' : voiceConf >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${Math.round(voiceConf * 100)}%` }} />
                    </div>
                    <p className="text-xs text-gray-400 mt-1 text-right">{Math.round(voiceConf * 100)}% confidence</p>
                  </>
                )}
              </div>

              {/* ── Keystroke Dynamics ── */}
              {/* Score is 50 % default when chat is closed; live when chat is open & typing */}
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-yellow-400" />
                    <span className="text-sm text-gray-300">Keystroke Dynamics</span>
                  </div>
                  {keystrokeConf >= 0.5
                    ? <CheckCircle className="h-4 w-4 text-green-400" />
                    : <AlertTriangle className="h-4 w-4 text-red-400" />
                  }
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div className={`h-2 rounded-full transition-all duration-700 ${keystrokeConf >= 0.8 ? 'bg-green-500' : keystrokeConf >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                    style={{ width: `${Math.round(keystrokeConf * 100)}%` }} />
                </div>
                <p className="text-xs text-gray-400 mt-1 text-right">
                  {Math.round(keystrokeConf * 100)}% confidence
                  {keystrokeConf === 0.5 && <span className="text-gray-500"> — open chat to verify</span>}
                </p>
              </div>

              {/* ── Mouse Movement ── */}
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-green-400" />
                    <span className="text-sm text-gray-300">Mouse Movement</span>
                  </div>
                  {mouseConf !== null
                    ? (mouseConf >= 0.5 ? <CheckCircle className="h-4 w-4 text-green-400" /> : <AlertTriangle className="h-4 w-4 text-red-400" />)
                    : <span className="text-xs text-gray-500">🖱️</span>
                  }
                </div>
                {mouseConf === null ? (
                  <p className="text-xs text-gray-500 italic">Move mouse to begin…</p>
                ) : (
                  <>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                      <div className={`h-2 rounded-full transition-all duration-700 ${mouseConf >= 0.8 ? 'bg-green-500' : mouseConf >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${Math.round(mouseConf * 100)}%` }} />
                    </div>
                    <p className="text-xs text-gray-400 mt-1 text-right">{Math.round(mouseConf * 100)}% confidence</p>
                  </>
                )}
              </div>

              {/* Alerts */}
              {alerts.length > 0 && (
                <div>
                  <h4 className="text-white font-semibold mb-2 flex items-center gap-1">
                    <AlertTriangle className="h-4 w-4 text-yellow-400" />Alerts
                  </h4>
                  <div className="space-y-2">
                    {alerts.slice(-5).reverse().map((alert, i) => (
                      <div key={i} className={`p-2 rounded-lg text-xs ${alert.severity === 'high' || alert.severity === 'critical' ? 'bg-red-900/50 border border-red-700' : 'bg-yellow-900/50 border border-yellow-700'}`}>
                        <p className="text-white">{alert.message}</p>
                        <p className="text-gray-400 mt-0.5">{alert.timestamp.toLocaleTimeString()}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="p-3 bg-blue-900/30 border border-blue-700 rounded-lg">
                <p className="text-xs text-blue-300">🔒 All 4 biometrics verified continuously against your registered profile.</p>
              </div>
            </div>
          ) : (
            /* ── Patient: Doctor's Live Biometric Verification ── */
            <div className="overflow-y-auto p-4 space-y-3">
              <h3 className="text-white font-semibold flex items-center gap-2 mb-1">
                <Shield className="h-5 w-5 text-primary-400" />Doctor Verification
              </h3>
              <p className="text-xs text-gray-500 mb-2">Live biometric verification of your doctor's identity</p>

              {/* Face Recognition */}
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Video className="h-4 w-4 text-blue-400" />
                    <span className="text-sm text-gray-300">Face Recognition</span>
                  </div>
                  {doctorScores.face !== null
                    ? (doctorScores.face >= 0.5 ? <CheckCircle className="h-4 w-4 text-green-400" /> : <AlertTriangle className="h-4 w-4 text-red-400" />)
                    : <span className="text-xs text-gray-500">⏳</span>}
                </div>
                {doctorScores.face === null ? (
                  <p className="text-xs text-gray-500 italic">Waiting for doctor's camera…</p>
                ) : (
                  <>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                      <div className={`h-2 rounded-full transition-all duration-700 ${doctorScores.face >= 0.8 ? 'bg-green-500' : doctorScores.face >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${Math.round(doctorScores.face * 100)}%` }} />
                    </div>
                    <p className="text-xs text-gray-400 mt-1 text-right">{Math.round(doctorScores.face * 100)}% confidence</p>
                  </>
                )}
              </div>

              {/* Voice Recognition */}
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Mic className="h-4 w-4 text-purple-400" />
                    <span className="text-sm text-gray-300">Voice Recognition</span>
                  </div>
                  {doctorScores.voice !== null
                    ? (doctorScores.voice >= 0.5 ? <CheckCircle className="h-4 w-4 text-green-400" /> : <AlertTriangle className="h-4 w-4 text-red-400" />)
                    : <span className="text-xs text-gray-500">⏳</span>}
                </div>
                {doctorScores.voice === null ? (
                  <p className="text-xs text-gray-500 italic">Waiting for voice sample…</p>
                ) : (
                  <>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                      <div className={`h-2 rounded-full transition-all duration-700 ${doctorScores.voice >= 0.8 ? 'bg-green-500' : doctorScores.voice >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${Math.round(doctorScores.voice * 100)}%` }} />
                    </div>
                    <p className="text-xs text-gray-400 mt-1 text-right">{Math.round(doctorScores.voice * 100)}% confidence</p>
                  </>
                )}
              </div>

              {/* Keystroke Dynamics */}
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-yellow-400" />
                    <span className="text-sm text-gray-300">Keystroke Dynamics</span>
                  </div>
                  {doctorScores.keystroke !== null
                    ? (doctorScores.keystroke >= 0.5 ? <CheckCircle className="h-4 w-4 text-green-400" /> : <AlertTriangle className="h-4 w-4 text-red-400" />)
                    : <span className="text-xs text-gray-500">⌨️</span>}
                </div>
                {doctorScores.keystroke === null ? (
                  <p className="text-xs text-gray-500 italic">Waiting for keystroke data…</p>
                ) : (
                  <>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                      <div className={`h-2 rounded-full transition-all duration-700 ${doctorScores.keystroke >= 0.8 ? 'bg-green-500' : doctorScores.keystroke >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${Math.round(doctorScores.keystroke * 100)}%` }} />
                    </div>
                    <p className="text-xs text-gray-400 mt-1 text-right">{Math.round(doctorScores.keystroke * 100)}% confidence</p>
                  </>
                )}
              </div>

              {/* Mouse Movement */}
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-green-400" />
                    <span className="text-sm text-gray-300">Mouse Movement</span>
                  </div>
                  {doctorScores.mouse !== null
                    ? (doctorScores.mouse >= 0.5 ? <CheckCircle className="h-4 w-4 text-green-400" /> : <AlertTriangle className="h-4 w-4 text-red-400" />)
                    : <span className="text-xs text-gray-500">🖱️</span>}
                </div>
                {doctorScores.mouse === null ? (
                  <p className="text-xs text-gray-500 italic">Waiting for mouse data…</p>
                ) : (
                  <>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                      <div className={`h-2 rounded-full transition-all duration-700 ${doctorScores.mouse >= 0.8 ? 'bg-green-500' : doctorScores.mouse >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${Math.round(doctorScores.mouse * 100)}%` }} />
                    </div>
                    <p className="text-xs text-gray-400 mt-1 text-right">{Math.round(doctorScores.mouse * 100)}% confidence</p>
                  </>
                )}
              </div>

              {/* Overall Trust Score (patient view) */}
              {(() => {
                const vals = [doctorScores.face, doctorScores.voice, doctorScores.keystroke, doctorScores.mouse].filter(v => v !== null);
                if (vals.length === 0) return null;
                const pct = Math.round(vals.reduce((a, b) => a + b, 0) / vals.length * 100);
                const color = pct >= 70 ? 'text-green-400' : pct >= 50 ? 'text-yellow-400' : 'text-red-400';
                const barColor = pct >= 70 ? 'bg-green-500' : pct >= 50 ? 'bg-yellow-500' : 'bg-red-500';
                const borderColor = pct >= 70 ? 'border-green-700 bg-green-900/30' : pct >= 50 ? 'border-yellow-700 bg-yellow-900/30' : 'border-red-700 bg-red-900/30';
                return (
                  <div className={`rounded-lg p-3 border ${borderColor}`}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Shield className="h-4 w-4 text-primary-400" />
                        <span className="text-sm font-semibold text-white">Overall Trust Score</span>
                      </div>
                      <span className={`text-lg font-bold ${color}`}>{pct}%</span>
                    </div>
                    <div className="w-full bg-gray-600 rounded-full h-3">
                      <div className={`h-3 rounded-full transition-all duration-700 ${barColor}`} style={{ width: `${pct}%` }} />
                    </div>
                    <p className="text-xs text-gray-400 mt-1">Based on {vals.length} active biometric{vals.length > 1 ? 's' : ''}</p>
                  </div>
                );
              })()}

              <div className="p-3 bg-blue-900/30 border border-blue-700 rounded-lg">
                <p className="text-xs text-blue-300">🔒 Doctor's identity is continuously verified against their registered biometric profile in real time.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>

    {/* ── Doctor Lockout Overlay ─────────────────────────────────────────────── */}
    {isDoctor && isLockedOut && (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm">
        <div className="bg-gray-900 border border-red-700 rounded-2xl p-8 max-w-md w-full mx-4 shadow-2xl text-center">

          {/* Header */}
          <div className="flex items-center justify-center gap-3 mb-2">
            <Lock className="h-8 w-8 text-red-400" />
            <h2 className="text-2xl font-bold text-white">Security Lock</h2>
          </div>
          <p className="text-gray-400 text-sm mb-6">
            Your overall trust score has been below 50% for 15 minutes. Mic and camera are disabled until you verify your identity.
          </p>

          {/* Step: OTP */}
          {lockoutStep === 'otp' && (
            <>
              <div className="flex items-center gap-2 justify-center mb-4 text-yellow-400">
                <Mail className="h-5 w-5" />
                <span className="text-sm font-medium">Enter the 6-digit OTP sent to your email</span>
              </div>
              <div className="flex justify-center gap-2 mb-5">
                {lockoutOtp.map((digit, i) => (
                  <input
                    key={i}
                    ref={el => lockoutOtpRefs.current[i] = el}
                    type="text"
                    inputMode="numeric"
                    maxLength={1}
                    value={digit}
                    onChange={e => handleLockoutOtpChange(i, e.target.value)}
                    onKeyDown={e => handleLockoutOtpKeyDown(i, e)}
                    className="w-12 h-14 text-center text-2xl font-bold bg-gray-800 border-2 border-gray-600 focus:border-primary-500 rounded-lg text-white outline-none transition-colors"
                  />
                ))}
              </div>
              <button
                onClick={handleLockoutOtpSubmit}
                disabled={lockoutOtpLoading || lockoutOtp.join('').length < 6}
                className="w-full py-3 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-xl transition-colors flex items-center justify-center gap-2 mb-3"
              >
                {lockoutOtpLoading
                  ? <><Loader2 className="h-5 w-5 animate-spin" /> Verifying…</>
                  : <><ArrowRight className="h-5 w-5" /> Verify OTP</>
                }
              </button>
              <button
                onClick={resendLockoutOtp}
                className="text-sm text-gray-400 hover:text-white underline transition-colors"
              >
                Resend OTP
              </button>
            </>
          )}

          {/* Step: Slide to Restore */}
          {lockoutStep === 'slide' && (
            <>
              <div className="flex items-center gap-2 justify-center mb-2 text-green-400">
                <CheckCircle className="h-5 w-5" />
                <span className="text-sm font-medium">OTP verified! Slide to restore your mic &amp; camera.</span>
              </div>
              <p className="text-xs text-gray-500 mb-6">This confirms you are physically present and in control.</p>

              {/* Slider track */}
              <div
                ref={lockoutSliderRef}
                className="relative w-full h-16 bg-gray-800 border-2 border-green-700 rounded-full overflow-hidden select-none cursor-pointer mb-4"
              >
                {/* Fill */}
                <div
                  className="absolute left-0 top-0 h-full bg-green-700/40 rounded-full transition-none"
                  style={{ width: `${lockoutSliderPos}%` }}
                />
                {/* Label */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <span className="text-white text-sm font-semibold opacity-70">
                    {lockoutSliderPos >= 85 ? '✅ Restoring…' : '← Slide to restore →'}
                  </span>
                </div>
                {/* Handle */}
                <div
                  className="absolute top-1 h-14 w-14 bg-green-500 rounded-full flex items-center justify-center shadow-lg cursor-grab active:cursor-grabbing transition-none"
                  style={{ left: `calc(${lockoutSliderPos}% - ${lockoutSliderPos > 10 ? 56 : 4}px)` }}
                  onMouseDown={handleSliderMouseDown}
                  onTouchStart={handleSliderTouchStart}
                >
                  <ArrowRight className="h-6 w-6 text-white" />
                </div>
              </div>
            </>
          )}

        </div>
      </div>
    )}
    </>
  );
};

export default Meeting;
