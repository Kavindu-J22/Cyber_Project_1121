import { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import {
  Shield, Mic, Camera, Keyboard, MousePointer,
  CheckCircle, XCircle, Loader2, Mail, RefreshCw,
  ArrowRight, Lock, ChevronRight, AlertTriangle,
} from 'lucide-react';
import { KeystrokeCapture, MouseCapture, VoiceCapture, FaceCapture } from '../utils/biometricCapture';

// ─── Constants ───────────────────────────────────────────────────────────────
const SECURITY_PHRASE = 'MediConsult Secure Access';
const MIN_PASS = 2;
const CONFIDENCE_THRESHOLD = 0.45;

// ─── Step Indicator ───────────────────────────────────────────────────────────
const StepIndicator = ({ current }) => {
  const steps = [
    { n: 1, label: 'Biometric Check' },
    { n: 2, label: 'OTP Verify' },
    { n: 3, label: 'Human Verify' },
  ];
  return (
    <div className="flex items-center justify-center gap-0 mb-8">
      {steps.map((s, i) => (
        <div key={s.n} className="flex items-center">
          <div className="flex flex-col items-center">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm transition-all duration-300 ${
              current > s.n ? 'bg-green-500 text-white shadow-lg shadow-green-500/30' :
              current === s.n ? 'bg-primary-600 text-white shadow-lg shadow-primary-500/30 ring-4 ring-primary-100' :
              'bg-gray-200 text-gray-500'
            }`}>
              {current > s.n ? <CheckCircle className="h-5 w-5" /> : s.n}
            </div>
            <span className={`text-xs mt-1.5 font-medium whitespace-nowrap ${
              current === s.n ? 'text-primary-600' : current > s.n ? 'text-green-600' : 'text-gray-400'
            }`}>{s.label}</span>
          </div>
          {i < steps.length - 1 && (
            <div className={`w-16 h-0.5 mb-5 mx-1 transition-all duration-500 ${current > s.n ? 'bg-green-400' : 'bg-gray-200'}`} />
          )}
        </div>
      ))}
    </div>
  );
};

// ─── Biometric Card ───────────────────────────────────────────────────────────
const BiometricCard = ({ icon: Icon, label, status, score, hint }) => {
  const statusConfig = {
    idle:    { bg: 'bg-gray-50', border: 'border-gray-200', iconBg: 'bg-gray-100', iconColor: 'text-gray-500', badge: null },
    running: { bg: 'bg-blue-50', border: 'border-blue-200', iconBg: 'bg-blue-100', iconColor: 'text-blue-600', badge: 'Scanning…' },
    pass:    { bg: 'bg-green-50', border: 'border-green-300', iconBg: 'bg-green-100', iconColor: 'text-green-600', badge: 'Passed ✓' },
    fail:    { bg: 'bg-red-50', border: 'border-red-300', iconBg: 'bg-red-100', iconColor: 'text-red-600', badge: 'Failed ✗' },
    skipped: { bg: 'bg-yellow-50', border: 'border-yellow-200', iconBg: 'bg-yellow-100', iconColor: 'text-yellow-600', badge: 'Skipped' },
  };
  const cfg = statusConfig[status] || statusConfig.idle;

  return (
    <div className={`border-2 rounded-xl p-4 transition-all duration-300 ${cfg.bg} ${cfg.border}`}>
      <div className="flex items-center gap-3 mb-2">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${cfg.iconBg}`}>
          {status === 'running'
            ? <Loader2 className={`h-5 w-5 animate-spin ${cfg.iconColor}`} />
            : <Icon className={`h-5 w-5 ${cfg.iconColor}`} />}
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-gray-800 text-sm">{label}</p>
          {cfg.badge && <p className={`text-xs font-medium mt-0.5 ${
            status === 'pass' ? 'text-green-700' : status === 'fail' ? 'text-red-700' :
            status === 'skipped' ? 'text-yellow-700' : 'text-blue-600'
          }`}>{cfg.badge}</p>}
        </div>
        {status === 'pass' && <CheckCircle className="h-6 w-6 text-green-500 flex-shrink-0" />}
        {status === 'fail' && <XCircle className="h-6 w-6 text-red-500 flex-shrink-0" />}
      </div>
      {status !== 'idle' && score !== null && score !== undefined && (
        <div className="mt-2">
          <div className="flex justify-between text-xs text-gray-600 mb-1">
            <span>Confidence</span><span>{Math.round(score * 100)}%</span>
          </div>
          <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-500 ${
              score >= CONFIDENCE_THRESHOLD ? 'bg-green-500' : 'bg-red-500'
            }`} style={{ width: `${Math.round(score * 100)}%` }} />
          </div>
        </div>
      )}
      {hint && status === 'idle' && <p className="text-xs text-gray-500 mt-1">{hint}</p>}
    </div>
  );
};

export default function ConsultationStartVerificationModal({ appointment, doctorEmail, onVerified, onClose }) {
  const [step, setStep] = useState(1);

  // ── Step 1 state ──────────────────────────────────────────────
  const [biometrics, setBiometrics] = useState({
    voice:     { status: 'idle', pass: false, score: null },
    face:      { status: 'idle', pass: false, score: null },
    keystroke: { status: 'idle', pass: false, score: null },
    mouse:     { status: 'idle', pass: false, score: null },
  });
  const [phrase, setPhrase]           = useState('');
  const [scanning, setScanning]       = useState(false);
  const [voiceTimer, setVoiceTimer]   = useState(0);
  const [mouseCount, setMouseCount]   = useState(0);
  const videoRef    = useRef(null);
  const keystrokeCap = useRef(new KeystrokeCapture());
  const mouseCap     = useRef(new MouseCapture());
  const voiceCap     = useRef(new VoiceCapture());
  const faceCap      = useRef(new FaceCapture());
  const mouseCountRef = useRef(0);
  const mountedRef   = useRef(true);

  // ── Step 2 state ──────────────────────────────────────────────
  const [otp, setOtp]               = useState(['', '', '', '', '', '']);
  const [otpLoading, setOtpLoading] = useState(false);
  const [resendTimer, setResendTimer] = useState(60);
  const [canResend, setCanResend]   = useState(false);
  const otpInputRefs = useRef([]);

  // ── Step 3 state ──────────────────────────────────────────────
  const [sliderPos, setSliderPos]     = useState(0);
  const [isSliding, setIsSliding]     = useState(false);
  const [slideVerified, setSlideVerified] = useState(false);
  const sliderRef = useRef(null);

  const token = localStorage.getItem('token');
  const authHeader = { Authorization: `Bearer ${token}` };

  // ── Cleanup on unmount ────────────────────────────────────────
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      faceCap.current.stopCamera();
      mouseCap.current.stop();
    };
  }, []);

  // ── Passive mouse capture ─────────────────────────────────────
  useEffect(() => {
    if (step !== 1) return;
    mouseCap.current.start();
    const handleMove = (e) => {
      mouseCap.current.handleMouseMove(e);
      mouseCountRef.current = mouseCap.current.getEvents().length;
      if (mountedRef.current) setMouseCount(mouseCountRef.current);
    };
    const handleClick = (e) => mouseCap.current.handleMouseClick(e);
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('click', handleClick);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('click', handleClick);
    };
  }, [step]);

  // ── Camera for face ────────────────────────────────────────────
  useEffect(() => {
    if (step !== 1) return;
    let cancelled = false;
    (async () => {
      await new Promise(r => setTimeout(r, 400)); // let modal render
      if (cancelled || !videoRef.current) return;
      await faceCap.current.startCamera(videoRef.current);
    })();
    return () => { cancelled = true; faceCap.current.stopCamera(); };
  }, [step]);

  // ── Keystroke capture ─────────────────────────────────────────
  useEffect(() => {
    if (step !== 1) return;
    keystrokeCap.current.start();
    const kd = (e) => keystrokeCap.current.handleKeyDown(e);
    const ku = (e) => keystrokeCap.current.handleKeyUp(e);
    window.addEventListener('keydown', kd);
    window.addEventListener('keyup', ku);
    return () => {
      window.removeEventListener('keydown', kd);
      window.removeEventListener('keyup', ku);
    };
  }, [step]);

  // ── OTP resend timer ──────────────────────────────────────────
  useEffect(() => {
    if (step !== 2) return;
    if (resendTimer > 0) {
      const t = setTimeout(() => setResendTimer(r => r - 1), 1000);
      return () => clearTimeout(t);
    }
    setCanResend(true);
  }, [resendTimer, step]);

  // ── Send OTP when entering step 2 ────────────────────────────
  useEffect(() => {
    if (step === 2) sendConsultationOtp();
  }, [step]);

  // ── Slider mouse events ───────────────────────────────────────
  useEffect(() => {
    if (!isSliding) return;
    const onMove = (e) => {
      if (!sliderRef.current) return;
      const rect = sliderRef.current.getBoundingClientRect();
      const pct = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
      setSliderPos(pct);
    };
    const onUp = () => {
      setIsSliding(false);
      setSliderPos(p => {
        if (p > 95) { setSlideVerified(true); return 100; }
        return 0;
      });
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
  }, [isSliding]);

  // ── Derived pass count ────────────────────────────────────────
  const passCount = Object.values(biometrics).filter(b => b.pass).length;
  const allDone   = Object.values(biometrics).every(b => b.status !== 'idle' && b.status !== 'running');

  // ── Helper: update one biometric ─────────────────────────────
  const updateBio = (key, patch) =>
    setBiometrics(prev => ({ ...prev, [key]: { ...prev[key], ...patch } }));

  // ── Run biometric verification ────────────────────────────────
  const runVerification = useCallback(async () => {
    if (scanning) return;
    setScanning(true);

    // Mark all as running
    ['voice', 'face', 'keystroke', 'mouse'].forEach(k =>
      updateBio(k, { status: 'running', pass: false, score: null })
    );

    await Promise.allSettled([
      // VOICE
      (async () => {
        try {
          const started = await voiceCap.current.start();
          if (!started) throw new Error('mic unavailable');
          // Count down 5s
          for (let i = 5; i > 0; i--) {
            if (!mountedRef.current) return;
            setVoiceTimer(i);
            await new Promise(r => setTimeout(r, 1000));
          }
          setVoiceTimer(0);
          const wavBlob = await voiceCap.current.stop();
          if (!wavBlob) throw new Error('no audio');
          const fd = new FormData();
          fd.append('voiceSample', wavBlob, 'voice.wav');
          const r = await axios.post('/api/verification/voice', fd, { headers: authHeader });
          const score = r.data.data?.confidence_score ?? r.data.data?.confidence ?? 0;
          const pass  = score >= CONFIDENCE_THRESHOLD;
          updateBio('voice', { status: pass ? 'pass' : 'fail', pass, score });
        } catch {
          updateBio('voice', { status: 'fail', pass: false, score: 0 });
        }
      })(),

      // FACE
      (async () => {
        try {
          await new Promise(r => setTimeout(r, 2000)); // let camera warm up
          if (!videoRef.current) throw new Error('no camera');
          const frame = await faceCap.current.captureFrame(videoRef.current);
          const fd = new FormData();
          fd.append('faceSample', frame, 'face.jpg');
          const r = await axios.post('/api/verification/face', fd, { headers: authHeader });
          const score = r.data.data?.confidence_score ?? r.data.data?.confidence ?? 0;
          const pass  = score >= CONFIDENCE_THRESHOLD;
          updateBio('face', { status: pass ? 'pass' : 'fail', pass, score });
        } catch {
          updateBio('face', { status: 'fail', pass: false, score: 0 });
        }
      })(),

      // KEYSTROKE
      (async () => {
        try {
          await new Promise(r => setTimeout(r, 6000)); // wait for typing window
          const features = keystrokeCap.current.getFeatures();
          const hasData   = keystrokeCap.current.events.length >= 5;
          if (!hasData) throw new Error('not enough typing');
          const r = await axios.post('/api/verification/keystroke',
            { keystrokeSample: features }, { headers: authHeader });
          const score = r.data.data?.confidence ?? 0;
          const pass  = score >= CONFIDENCE_THRESHOLD;
          updateBio('keystroke', { status: pass ? 'pass' : 'fail', pass, score });
        } catch {
          updateBio('keystroke', { status: 'fail', pass: false, score: 0 });
        }
      })(),

      // MOUSE
      (async () => {
        try {
          await new Promise(r => setTimeout(r, 7000)); // collect for 7s
          const events = mouseCap.current.getEvents();
          if (events.length < 20) throw new Error('not enough mouse data');
          const r = await axios.post('/api/verification/mouse',
            { mouseEvents: events }, { headers: authHeader });
          const score = r.data.data?.confidence ?? 0;
          const pass  = score >= CONFIDENCE_THRESHOLD;
          updateBio('mouse', { status: pass ? 'pass' : 'fail', pass, score });
        } catch {
          updateBio('mouse', { status: 'fail', pass: false, score: 0 });
        }
      })(),
    ]);

    setScanning(false);
  }, [scanning]);

  // ── OTP functions ─────────────────────────────────────────────
  const sendConsultationOtp = async () => {
    try {
      setOtpLoading(true);
      await axios.post('/api/otp/consultation/send', {}, { headers: authHeader });
      toast.success('OTP sent to your registered email!');
    } catch (e) {
      toast.error(e.response?.data?.message || 'Failed to send OTP');
    } finally { setOtpLoading(false); }
  };

  const handleOtpChange = (i, val) => {
    if (val && !/^\d$/.test(val)) return;
    const next = [...otp]; next[i] = val; setOtp(next);
    if (val && i < 5) otpInputRefs.current[i + 1]?.focus();
    if (next.every(d => d !== '') && i === 5) verifyOtp(next.join(''));
  };

  const handleOtpKeyDown = (i, e) => {
    if (e.key === 'Backspace' && !otp[i] && i > 0) otpInputRefs.current[i - 1]?.focus();
  };

  const verifyOtp = async (code) => {
    try {
      setOtpLoading(true);
      await axios.post('/api/otp/consultation/verify', { otp: code }, { headers: authHeader });
      toast.success('OTP verified!');
      setStep(3);
    } catch (e) {
      toast.error(e.response?.data?.message || 'Invalid OTP');
      setOtp(['', '', '', '', '', '']);
      otpInputRefs.current[0]?.focus();
    } finally { setOtpLoading(false); }
  };

  const resendOtp = async () => {
    if (!canResend) return;
    try {
      setOtpLoading(true);
      await axios.post('/api/otp/consultation/resend', {}, { headers: authHeader });
      toast.success('OTP resent!');
      setResendTimer(60); setCanResend(false);
      setOtp(['', '', '', '', '', '']);
      otpInputRefs.current[0]?.focus();
    } catch (e) {
      toast.error(e.response?.data?.message || 'Failed to resend');
    } finally { setOtpLoading(false); }
  };

  const maskedEmail = doctorEmail
    ? doctorEmail.replace(/(.{2}).+(@.+)/, '$1•••$2')
    : '•••@•••';

  // ═══════════════════════════════════════════════════════════════
  //  RENDER
  // ═══════════════════════════════════════════════════════════════
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(6px)' }}>
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="bg-gradient-to-r from-primary-700 to-primary-500 rounded-t-2xl px-8 py-6 text-white">
          <div className="flex items-center gap-3 mb-1">
            <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
              <Lock className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold">Identity Verification Required</h2>
              <p className="text-primary-100 text-sm">Complete all 3 steps to start your consultation</p>
            </div>
          </div>
        </div>

        <div className="px-8 pt-6 pb-8">
          <StepIndicator current={step} />

          {/* ══ STEP 1: BIOMETRIC ══════════════════════════════════ */}
          {step === 1 && (
            <div>
              <div className="mb-5 p-4 bg-blue-50 border border-blue-200 rounded-xl flex gap-3">
                <AlertTriangle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-blue-800">
                  <p className="font-semibold">Quick Biometric Scan</p>
                  <p className="mt-0.5">At least <strong>2 of 4</strong> checks must pass. Move your mouse, type the phrase, and allow camera & microphone access.</p>
                </div>
              </div>

              {/* Security phrase input */}
              <div className="mb-4">
                <label className="block text-sm font-semibold text-gray-700 mb-1.5">
                  <Keyboard className="h-4 w-4 inline mr-1.5 text-primary-600" />
                  Type this phrase for keystroke analysis:
                  <span className="ml-2 font-mono text-primary-700 bg-primary-50 px-2 py-0.5 rounded">{SECURITY_PHRASE}</span>
                </label>
                <input
                  type="text"
                  value={phrase}
                  onChange={e => setPhrase(e.target.value)}
                  placeholder="Type the phrase above..."
                  disabled={scanning}
                  className="w-full border-2 border-gray-200 rounded-xl px-4 py-3 text-gray-900 focus:border-primary-400 focus:outline-none transition-colors disabled:bg-gray-50"
                />
              </div>

              {/* Hidden camera preview */}
              <video ref={videoRef} autoPlay muted playsInline className="hidden" />

              {/* Biometric cards grid */}
              <div className="grid grid-cols-2 gap-3 mb-5">
                <BiometricCard icon={Mic} label="Voice" status={biometrics.voice.status}
                  score={biometrics.voice.score}
                  hint="Will auto-record for 5 seconds" />
                {voiceTimer > 0 && biometrics.voice.status === 'running' && (
                  <div className="col-span-2 -mt-2 px-1">
                    <div className="flex items-center gap-2 text-blue-700 text-sm">
                      <Mic className="h-4 w-4 animate-pulse" />
                      <span>Recording voice… {voiceTimer}s remaining</span>
                    </div>
                  </div>
                )}
                <BiometricCard icon={Camera} label="Face Recognition" status={biometrics.face.status}
                  score={biometrics.face.score}
                  hint="Camera captures automatically" />
                <BiometricCard icon={Keyboard} label="Keystroke Dynamics" status={biometrics.keystroke.status}
                  score={biometrics.keystroke.score}
                  hint="Type the phrase above" />
                <BiometricCard icon={MousePointer} label={`Mouse Behavior (${mouseCount} events)`}
                  status={biometrics.mouse.status}
                  score={biometrics.mouse.score}
                  hint="Move your mouse around" />
              </div>

              {/* Pass count badge */}
              {allDone && (
                <div className={`mb-4 p-3 rounded-xl text-center font-semibold text-sm ${
                  passCount >= MIN_PASS ? 'bg-green-50 text-green-800 border border-green-300' : 'bg-red-50 text-red-800 border border-red-300'
                }`}>
                  {passCount >= MIN_PASS
                    ? `✅ ${passCount}/4 checks passed — biometric identity confirmed!`
                    : `❌ Only ${passCount}/4 passed — minimum ${MIN_PASS} required. Please retry.`}
                </div>
              )}

              <div className="flex gap-3">
                <button onClick={onClose} disabled={scanning}
                  className="flex-1 py-3 border-2 border-gray-200 rounded-xl text-gray-700 font-semibold hover:bg-gray-50 transition-colors disabled:opacity-50">
                  Cancel
                </button>
                {!scanning && allDone && passCount < MIN_PASS ? (
                  <button onClick={runVerification}
                    className="flex-1 py-3 bg-orange-500 text-white rounded-xl font-semibold hover:bg-orange-600 transition-colors flex items-center justify-center gap-2">
                    <RefreshCw className="h-4 w-4" /> Retry
                  </button>
                ) : (
                  <button onClick={allDone && passCount >= MIN_PASS ? () => setStep(2) : runVerification}
                    disabled={scanning}
                    className={`flex-1 py-3 rounded-xl font-semibold flex items-center justify-center gap-2 transition-colors ${
                      allDone && passCount >= MIN_PASS
                        ? 'bg-green-600 hover:bg-green-700 text-white'
                        : scanning
                        ? 'bg-primary-400 text-white cursor-not-allowed'
                        : 'bg-primary-600 hover:bg-primary-700 text-white'
                    }`}>
                    {scanning ? <><Loader2 className="h-4 w-4 animate-spin" />Scanning…</>
                      : allDone && passCount >= MIN_PASS
                      ? <><span>Continue to OTP</span><ChevronRight className="h-4 w-4" /></>
                      : <><Shield className="h-4 w-4" />Start Scan</>}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* ══ STEP 2: OTP ════════════════════════════════════════ */}
          {step === 2 && (
            <div className="max-w-md mx-auto">
              <div className="text-center mb-6">
                <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Mail className="h-8 w-8 text-primary-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">Email OTP Verification</h3>
                <p className="text-gray-600 text-sm">
                  A 6-digit OTP has been sent to<br />
                  <span className="font-bold text-gray-900">{maskedEmail}</span>
                </p>
              </div>

              <div className="flex justify-center gap-2.5 mb-6">
                {otp.map((digit, i) => (
                  <input key={i} ref={el => (otpInputRefs.current[i] = el)}
                    type="text" inputMode="numeric" maxLength={1} value={digit}
                    onChange={e => handleOtpChange(i, e.target.value)}
                    onKeyDown={e => handleOtpKeyDown(i, e)}
                    disabled={otpLoading}
                    className="w-12 h-14 text-center text-2xl font-bold border-2 border-gray-300 rounded-xl focus:border-primary-500 focus:ring-2 focus:ring-primary-200 outline-none transition-all disabled:bg-gray-50"
                  />
                ))}
              </div>

              <button onClick={() => verifyOtp(otp.join(''))}
                disabled={otpLoading || otp.some(d => d === '')}
                className="w-full py-3 bg-primary-600 text-white rounded-xl font-semibold hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2 mb-3">
                {otpLoading ? <><Loader2 className="h-4 w-4 animate-spin" />Verifying…</> : 'Verify OTP'}
              </button>

              <div className="text-center">
                <button onClick={resendOtp} disabled={!canResend || otpLoading}
                  className={`inline-flex items-center gap-2 text-sm font-medium ${canResend ? 'text-primary-600 hover:text-primary-700' : 'text-gray-400 cursor-not-allowed'}`}>
                  <RefreshCw className={`h-4 w-4 ${otpLoading ? 'animate-spin' : ''}`} />
                  {canResend ? 'Resend OTP' : `Resend in ${resendTimer}s`}
                </button>
              </div>
            </div>
          )}

          {/* ══ STEP 3: SLIDE TO DRAW ══════════════════════════════ */}
          {step === 3 && (
            <div className="max-w-md mx-auto">
              <div className="text-center mb-6">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Shield className="h-8 w-8 text-green-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">Human Verification</h3>
                <p className="text-gray-600 text-sm">Slide all the way to the right to confirm you are human and start the consultation.</p>
              </div>

              {!slideVerified ? (
                <div ref={sliderRef}
                  className="relative w-full h-16 bg-gray-100 border-2 border-gray-200 rounded-full overflow-hidden cursor-pointer select-none shadow-inner"
                  onMouseDown={() => setIsSliding(true)}
                  onTouchStart={() => setIsSliding(true)}
                  onTouchMove={e => {
                    const touch = e.touches[0];
                    const rect = sliderRef.current.getBoundingClientRect();
                    const pct = Math.max(0, Math.min(100, ((touch.clientX - rect.left) / rect.width) * 100));
                    setSliderPos(pct);
                  }}
                  onTouchEnd={() => {
                    setIsSliding(false);
                    if (sliderPos > 95) { setSlideVerified(true); setSliderPos(100); }
                    else setSliderPos(0);
                  }}>
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <span className="text-sm font-semibold text-gray-500 select-none">
                      {sliderPos < 10 ? '→ Slide to Start Consultation →'
                        : sliderPos < 95 ? '→ Keep sliding →' : '✓ Release!'}
                    </span>
                  </div>
                  <div className="absolute top-0 left-0 h-full bg-gradient-to-r from-green-400 to-green-600 opacity-30 rounded-full transition-all"
                    style={{ width: `${sliderPos}%` }} />
                  <div className="absolute top-1 h-14 w-14 bg-white rounded-full shadow-xl flex items-center justify-center border-2 border-primary-200 transition-all"
                    style={{ left: `calc(${sliderPos}% - 28px + 4px)`, maxLeft: 'calc(100% - 60px)' }}>
                    <ArrowRight className={`h-6 w-6 ${sliderPos > 95 ? 'text-green-600' : 'text-primary-600'}`} />
                  </div>
                </div>
              ) : (
                <div>
                  <div className="p-5 bg-green-50 border-2 border-green-400 rounded-xl flex items-center justify-center gap-3 mb-6">
                    <CheckCircle className="h-7 w-7 text-green-600" />
                    <span className="text-green-800 font-bold text-lg">Human Verified!</span>
                  </div>
                  <button onClick={onVerified}
                    className="w-full py-4 bg-gradient-to-r from-primary-600 to-primary-700 text-white rounded-xl font-bold text-lg hover:from-primary-700 hover:to-primary-800 transition-all shadow-lg shadow-primary-500/30 flex items-center justify-center gap-2">
                    <Camera className="h-5 w-5" />
                    Start Consultation Now
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

