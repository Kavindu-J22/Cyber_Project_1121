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
import Session from './models/Session.js';
import mlService from './services/mlService.js';
import { verifyEmailConfig } from './utils/emailService.js';

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
      sessions: '/api/sessions',
      verification: '/api/verification',
      otp: '/api/otp',
      health: '/api/health'
    }
  });
});

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log(`✓ Client connected: ${socket.id}`);

  // Join session room
  socket.on('join-session', async (data) => {
    const { sessionId, doctorId } = data;
    socket.join(sessionId);
    console.log(`Doctor ${doctorId} joined session ${sessionId}`);
    
    socket.emit('session-joined', { sessionId, socketId: socket.id });
  });

  // Handle real-time verification
  socket.on('verify-biometric', async (data) => {
    try {
      const { sessionId, doctorId, type, payload } = data;
      let result = null;

      // Perform verification based on type
      switch (type) {
        case 'voice':
          // Voice verification would need audio stream handling
          break;
        
        case 'keystroke':
          result = await mlService.verifyKeystroke(doctorId, payload);
          break;
        
        case 'mouse':
          result = await mlService.verifyMouse(doctorId, payload);
          break;
        
        case 'face':
          // Face verification with image file
          result = await mlService.verifyFace(doctorId, payload);
          break;
      }

      if (result) {
        // Update session with verification log
        const session = await Session.findOne({ sessionId });
        if (session) {
          session.verificationLogs.push({
            verificationType: type,
            verified: result.verified,
            confidence: result.confidence,
            details: result
          });

          // Update trust score
          const recentLogs = session.verificationLogs.slice(-10);
          const avgConfidence = recentLogs.reduce((sum, log) => 
            sum + (log.confidence || 0), 0) / recentLogs.length;
          session.overallTrustScore = Math.round(avgConfidence * 100);

          // Check for alerts
          if (avgConfidence < 0.5) {
            session.status = 'suspicious';
            session.alerts.push({
              type: 'low_confidence',
              severity: 'high',
              message: `${type} verification confidence dropped below 50%`,
              details: result
            });

            // Emit alert to client
            io.to(sessionId).emit('verification-alert', {
              type: 'low_confidence',
              severity: 'high',
              message: `${type} verification failed`,
              trustScore: session.overallTrustScore
            });
          }

          await session.save();
        }

        // Emit verification result
        io.to(sessionId).emit('verification-result', {
          type,
          result,
          trustScore: session?.overallTrustScore || 100
        });
      }
    } catch (error) {
      console.error('Verification error:', error);
      socket.emit('verification-error', {
        message: 'Verification failed',
        error: error.message
      });
    }
  });

  socket.on('disconnect', () => {
    console.log(`✗ Client disconnected: ${socket.id}`);
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

