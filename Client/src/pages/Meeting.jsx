import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { io } from 'socket.io-client';
import axios from 'axios';
import toast from 'react-hot-toast';
import {
  Video, VideoOff, Mic, MicOff, PhoneOff, Shield,
  Activity, AlertTriangle, CheckCircle, TrendingUp,
  MessageSquare, Send, Users, X
} from 'lucide-react';
import { KeystrokeCapture, MouseCapture } from '../utils/biometricCapture';

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

  // Chat
  const [showChat, setShowChat] = useState(false);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [unreadCount, setUnreadCount] = useState(0);

  // Session state
  const [consultationEnded, setConsultationEnded] = useState(false);

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
  const verificationInterval = useRef(null);

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
      // 1. Get local camera + mic
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        localStreamRef.current = stream;
        if (localVideoRef.current) localVideoRef.current.srcObject = stream;
      } catch {
        toast.error('Failed to access camera/microphone');
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

      // Start biometric monitoring
      keystrokeCapture.current.start();
      mouseCapture.current.start();
      verificationInterval.current = setInterval(() => {
        const kf = keystrokeCapture.current.getFeatures();
        if (kf.some(f => f !== 0)) {
          sock.emit('verify-biometric', { sessionId, doctorId: user.id, type: 'keystroke', payload: kf });
          keystrokeCapture.current.start();
        }
        const me = mouseCapture.current.getEvents();
        if (me.length > 50) {
          sock.emit('verify-biometric', { sessionId, doctorId: user.id, type: 'mouse', payload: me });
          mouseCapture.current.start();
        }
      }, 10000);
    };

    init();

    return () => {
      mounted = false;
      localStreamRef.current?.getTracks().forEach(t => t.stop());
      peerConnectionRef.current?.close();
      socketRef.current?.disconnect();
      if (verificationInterval.current) clearInterval(verificationInterval.current);
    };
  }, [sessionId, user, isDoctor, navigate]);

  // Keep showChatRef in sync so the socket callback can read it without stale closure
  useEffect(() => {
    showChatRef.current = showChat;
    if (showChat) setUnreadCount(0);
  }, [showChat]);

  // Auto-scroll chat to bottom on new message
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const toggleVideo = () => {
    const t = localStreamRef.current?.getVideoTracks()[0];
    if (t) { t.enabled = !t.enabled; setIsVideoOn(t.enabled); }
  };

  const toggleAudio = () => {
    const t = localStreamRef.current?.getAudioTracks()[0];
    if (t) { t.enabled = !t.enabled; setIsAudioOn(t.enabled); }
  };

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
    <div
      className="min-h-screen bg-gray-900 flex flex-col"
      onMouseMove={(e) => mouseCapture.current.handleMouseMove(e)}
      onClick={(e) => mouseCapture.current.handleMouseClick(e)}
      onKeyDown={(e) => keystrokeCapture.current.handleKeyDown(e)}
      onKeyUp={(e) => keystrokeCapture.current.handleKeyUp(e)}
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
                      e.stopPropagation();
                      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
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
          ) : (
            /* Verification Panel */
            <div className="overflow-y-auto p-4">
              <h3 className="text-white font-semibold mb-4 flex items-center">
                <Shield className="h-5 w-5 mr-2 text-primary-400" />Real-time Verification
              </h3>
              <div className="space-y-3 mb-6">
                {[{ label: 'Voice Recognition', w: '95%' }, { label: 'Keystroke Dynamics', w: '92%' }, { label: 'Mouse Movement', w: '88%' }].map(item => (
                  <div key={item.label} className="bg-gray-700 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-300">{item.label}</span>
                      <CheckCircle className="h-4 w-4 text-green-400" />
                    </div>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                      <div className="bg-green-500 h-2 rounded-full" style={{ width: item.w }} />
                    </div>
                  </div>
                ))}
              </div>
              {alerts.length > 0 && (
                <div className="mb-6">
                  <h4 className="text-white font-semibold mb-2 flex items-center">
                    <AlertTriangle className="h-4 w-4 mr-2 text-yellow-400" />Alerts
                  </h4>
                  <div className="space-y-2">
                    {alerts.slice(-5).reverse().map((alert, i) => (
                      <div key={i} className={`p-2 rounded-lg ${alert.severity === 'high' || alert.severity === 'critical' ? 'bg-red-900/50 border border-red-700' : 'bg-yellow-900/50 border border-yellow-700'}`}>
                        <p className="text-xs text-white">{alert.message}</p>
                        <p className="text-xs text-gray-400 mt-1">{alert.timestamp.toLocaleTimeString()}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <h4 className="text-white font-semibold mb-2 flex items-center">
                <TrendingUp className="h-4 w-4 mr-2 text-primary-400" />Recent Verifications
              </h4>
              <div className="space-y-2">
                {verificationLogs.length === 0 && (
                  <div className="text-center py-8 text-gray-500 text-sm">Waiting for verification data…</div>
                )}
                {verificationLogs.slice(-10).reverse().map((log, i) => (
                  <div key={i} className="bg-gray-700 rounded-lg p-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-300 capitalize">{log.type}</span>
                      <span className={`text-xs font-semibold ${log.result.verified ? 'text-green-400' : 'text-red-400'}`}>
                        {log.result.verified ? '✓ Verified' : '✗ Failed'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between mt-1">
                      <span className="text-xs text-gray-400">Confidence: {(log.result.confidence * 100).toFixed(1)}%</span>
                      <span className="text-xs text-gray-500">{log.timestamp.toLocaleTimeString()}</span>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-6 p-3 bg-blue-900/30 border border-blue-700 rounded-lg">
                <p className="text-xs text-blue-300">🔒 Biometric data is continuously verified for secure consultation.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Meeting;
