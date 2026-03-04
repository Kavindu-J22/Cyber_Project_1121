import express from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import morgan from 'morgan';
import helmet from 'helmet';
import { createServer } from 'http';
import { Server } from 'socket.io';
import connectDB from './config/db.js';
import authRoutes from './routes/authRoutes.js';
import doctorRoutes from './routes/doctorRoutes.js';
import patientRoutes from './routes/patientRoutes.js';
import sessionRoutes from './routes/sessionRoutes.js';
import verificationRoutes from './routes/verificationRoutes.js';
import otpRoutes from './routes/otpRoutes.js';
import appointmentRoutes from './routes/appointmentRoutes.js';
import consultationRoutes from './routes/consultationRoutes.js';
import Session from './models/Session.js';
import mlService from './services/mlService.js';
import { verifyEmailConfig, sendLockoutTerminatedEmail } from './utils/emailService.js';
import Consultation from './models/Consultation.js';
import Patient from './models/Patient.js';
import Doctor from './models/Doctor.js';

// Load environment variables
dotenv.config();

// Connect to database
connectDB();

// Verify email configuration
verifyEmailConfig();

// Initialize Express app
const app = express();
const httpServer = createServer(app);

// Initialize Socket.IO
const io = new Server(httpServer, {
  cors: {
    origin: process.env.CLIENT_URL || 'http://localhost:5173',
    methods: ['GET', 'POST'],
    credentials: true
  }
});

// Middleware
app.use(helmet());
app.use(cors({
  origin: process.env.CLIENT_URL || 'http://localhost:5173',
  credentials: true
}));
app.use(morgan('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/doctors', doctorRoutes);
app.use('/api/patients', patientRoutes);
app.use('/api/sessions', sessionRoutes);
app.use('/api/verification', verificationRoutes);
app.use('/api/otp', otpRoutes);
app.use('/api/appointments', appointmentRoutes);
app.use('/api/consultations', consultationRoutes);

// Health check
app.get('/api/health', async (req, res) => {
  try {
    const mlHealth = await mlService.checkHealth();
    res.json({
      success: true,
      message: 'Server is running',
      timestamp: new Date().toISOString(),
      mlServices: mlHealth
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Health check failed',
      error: error.message
    });
  }
});

// Root route
app.get('/', (req, res) => {
  res.json({
    message: 'Zero Trust Telehealth Platform API',
    version: '1.0.0',
    endpoints: {
      auth: '/api/auth',
      doctors: '/api/doctors',
      patients: '/api/patients',
      sessions: '/api/sessions',
      verification: '/api/verification',
      otp: '/api/otp',
      appointments: '/api/appointments',
      health: '/api/health'
    }
  });
});

// Track users per room: roomId -> [{ socketId, userId, userRole, userName }]
const roomUsers = new Map();

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log(`✓ Client connected: ${socket.id}`);

  // Join session room
  socket.on('join-session', async (data) => {
    const { sessionId, userId, userRole, userName } = data;
    socket.join(sessionId);

    // Track room membership
    if (!roomUsers.has(sessionId)) roomUsers.set(sessionId, []);
    const users = roomUsers.get(sessionId);
    users.push({ socketId: socket.id, userId, userRole, userName });
    console.log(`${userRole} (${userName}) joined session ${sessionId}`);

    socket.emit('session-joined', { sessionId, socketId: socket.id });

    // Notify existing participants that a new user joined
    socket.to(sessionId).emit('user-joined', {
      socketId: socket.id,
      userId,
      userRole,
      userName
    });
  });

  // ── WebRTC Signaling ──────────────────────────────────────────
  socket.on('offer', ({ offer, targetSocketId, sessionId }) => {
    io.to(targetSocketId).emit('offer', { offer, fromSocketId: socket.id, sessionId });
  });

  socket.on('answer', ({ answer, targetSocketId, sessionId }) => {
    io.to(targetSocketId).emit('answer', { answer, fromSocketId: socket.id, sessionId });
  });

  socket.on('ice-candidate', ({ candidate, targetSocketId, sessionId }) => {
    io.to(targetSocketId).emit('ice-candidate', { candidate, fromSocketId: socket.id, sessionId });
  });

  // ── Session Control ───────────────────────────────────────────
  socket.on('end-session', ({ sessionId }) => {
    console.log(`Session ended by doctor in room: ${sessionId}`);
    // Notify everyone in the room (including the doctor themselves)
    io.to(sessionId).emit('session-ended', {
      message: 'Consultation has been ended by the doctor',
      timestamp: new Date().toISOString()
    });
  });

  // ── Chat ──────────────────────────────────────────────────────
  socket.on('chat-message', (msg) => {
    // Broadcast to all participants in the session room
    io.to(msg.sessionId).emit('chat-message', msg);
  });

  // ── Doctor Biometric Scores → relay to patient ────────────────
  socket.on('doctor-biometric-update', ({ sessionId, scores }) => {
    // Relay to everyone else in the room (patients)
    socket.to(sessionId).emit('doctor-biometric-update', { scores });
  });

  // ── Doctor Lockout Status → relay to patient ──────────────────
  socket.on('doctor-lockout-status', ({ sessionId, isLocked }) => {
    socket.to(sessionId).emit('doctor-lockout-status', { isLocked });
  });

  // ── Lockout max OTP attempts → notify patient via email + end session ──
  socket.on('lockout-max-attempts', async ({ sessionId }) => {
    try {
      const consultation = await Consultation.findOne({ consultationRoomId: sessionId })
        .populate('patientId', 'email fullName')
        .populate('doctorId', 'firstName lastName');

      if (consultation && consultation.patientId) {
        const patient = consultation.patientId;
        const doctor = consultation.doctorId;
        const doctorName = doctor ? `${doctor.firstName} ${doctor.lastName}` : 'Doctor';
        await sendLockoutTerminatedEmail(patient.email, patient.fullName, doctorName);
        console.log(`🔒 Lockout termination email sent for session ${sessionId}`);
      }
    } catch (err) {
      console.error('Failed to send lockout termination email:', err.message);
    }
    // End the session for all participants regardless of email success
    io.to(sessionId).emit('session-ended', {
      message: 'Consultation ended: doctor failed security verification 5 times.',
      timestamp: new Date().toISOString()
    });
  });

  // ── Biometric Verification ────────────────────────────────────
  socket.on('verify-biometric', async (data) => {
    try {
      const { sessionId, doctorId, type, payload } = data;
      let result = null;

      switch (type) {
        case 'voice':
          break;
        case 'keystroke':
          result = await mlService.verifyKeystroke(doctorId, payload);
          break;
        case 'mouse':
          result = await mlService.verifyMouse(doctorId, payload);
          break;
        case 'face':
          result = await mlService.verifyFace(doctorId, payload);
          break;
      }

      if (result) {
        const session = await Session.findOne({ sessionId });
        if (session) {
          session.verificationLogs.push({
            verificationType: type,
            verified: result.verified,
            confidence: result.confidence,
            details: result
          });

          const recentLogs = session.verificationLogs.slice(-10);
          const avgConfidence = recentLogs.reduce((sum, log) =>
            sum + (log.confidence || 0), 0) / recentLogs.length;
          session.overallTrustScore = Math.round(avgConfidence * 100);

          if (avgConfidence < 0.5) {
            session.status = 'suspicious';
            session.alerts.push({
              type: 'low_confidence',
              severity: 'high',
              message: `${type} verification confidence dropped below 50%`,
              details: result
            });
            io.to(sessionId).emit('verification-alert', {
              type: 'low_confidence',
              severity: 'high',
              message: `${type} verification failed`,
              trustScore: session.overallTrustScore
            });
          }
          await session.save();
        }

        io.to(sessionId).emit('verification-result', {
          type,
          result,
          trustScore: session?.overallTrustScore || 100
        });
      }
    } catch (error) {
      console.error('Verification error:', error);
      socket.emit('verification-error', { message: 'Verification failed', error: error.message });
    }
  });

  // ── Disconnect ────────────────────────────────────────────────
  socket.on('disconnect', () => {
    console.log(`✗ Client disconnected: ${socket.id}`);
    // Remove from all rooms and notify peers
    for (const [roomId, users] of roomUsers.entries()) {
      const idx = users.findIndex(u => u.socketId === socket.id);
      if (idx !== -1) {
        users.splice(idx, 1);
        socket.to(roomId).emit('user-left', { socketId: socket.id });
        if (users.length === 0) roomUsers.delete(roomId);
      }
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    message: 'Something went wrong!',
    error: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

// Start server
const PORT = process.env.PORT || 5000;
httpServer.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════════════════╗
║   Zero Trust Telehealth Platform - Backend Server         ║
║   Server running on port ${PORT}                              ║
║   Environment: ${process.env.NODE_ENV || 'development'}                            ║
╚════════════════════════════════════════════════════════════╝
  `);
});

export { io };

