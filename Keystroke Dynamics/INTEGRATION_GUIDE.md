# Integration Guide: Keystroke Dynamics with MERN Stack

## Overview

This guide explains how to integrate the Keystroke Dynamics ML component with the MERN stack telehealth platform for continuous authentication.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MERN Stack Application                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  React Frontend  │────────▶│  Node.js Backend │             │
│  │  (Port 3000)     │         │  (Port 5000)     │             │
│  └──────────────────┘         └──────────────────┘             │
│         │                              │                         │
│         │ Keystroke Events             │ HTTP Requests           │
│         │                              │                         │
│         ▼                              ▼                         │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │ Keystroke        │         │ Keystroke        │             │
│  │ Capture Module   │────────▶│ API Client       │             │
│  └──────────────────┘         └──────────────────┘             │
│                                        │                         │
└────────────────────────────────────────┼─────────────────────────┘
                                         │
                                         │ REST API
                                         │
                        ┌────────────────▼────────────────┐
                        │  Keystroke Dynamics ML API      │
                        │  (Python FastAPI - Port 8002)   │
                        │                                 │
                        │  - Enrollment                   │
                        │  - Verification                 │
                        │  - Continuous Monitoring        │
                        │  - Anomaly Detection            │
                        └─────────────────────────────────┘
```

---

## Part 1: Frontend Integration (React)

### 1.1 Install Dependencies

```bash
cd client
npm install axios
```

### 1.2 Create Keystroke Capture Hook

Create `client/src/hooks/useKeystrokeCapture.js`:

```javascript
import { useState, useEffect, useCallback, useRef } from 'react';

export const useKeystrokeCapture = (userId, enabled = true) => {
  const [keystrokeData, setKeystrokeData] = useState([]);
  const keyDownTimes = useRef({});
  const lastKeyUpTime = useRef(null);
  
  const handleKeyDown = useCallback((e) => {
    if (!enabled) return;
    
    const key = e.key;
    const timestamp = Date.now();
    
    // Record keydown time
    if (!keyDownTimes.current[key]) {
      keyDownTimes.current[key] = timestamp;
    }
  }, [enabled]);
  
  const handleKeyUp = useCallback((e) => {
    if (!enabled) return;
    
    const key = e.key;
    const timestamp = Date.now();
    
    if (keyDownTimes.current[key]) {
      // Calculate hold time (H.key)
      const holdTime = timestamp - keyDownTimes.current[key];
      
      // Calculate UD time (keyup to keydown)
      const udTime = lastKeyUpTime.current 
        ? keyDownTimes.current[key] - lastKeyUpTime.current 
        : 0;
      
      // Store keystroke data
      const keystroke = {
        key,
        holdTime,
        udTime,
        timestamp
      };
      
      setKeystrokeData(prev => [...prev, keystroke]);
      
      // Update references
      delete keyDownTimes.current[key];
      lastKeyUpTime.current = timestamp;
    }
  }, [enabled]);
  
  useEffect(() => {
    if (enabled) {
      window.addEventListener('keydown', handleKeyDown);
      window.addEventListener('keyup', handleKeyUp);
      
      return () => {
        window.removeEventListener('keydown', handleKeyDown);
        window.removeEventListener('keyup', handleKeyUp);
      };
    }
  }, [enabled, handleKeyDown, handleKeyUp]);
  
  const clearKeystrokeData = useCallback(() => {
    setKeystrokeData([]);
  }, []);
  
  return { keystrokeData, clearKeystrokeData };
};
```

### 1.3 Create Keystroke Service

Create `client/src/services/keystrokeService.js`:

```javascript
import axios from 'axios';

const KEYSTROKE_API_URL = 'http://localhost:8002';

class KeystrokeService {
  // Enroll user
  async enrollUser(userId, keystrokeSamples) {
    try {
      const response = await axios.post(`${KEYSTROKE_API_URL}/enroll`, {
        user_id: userId,
        keystroke_samples: keystrokeSamples
      });
      return response.data;
    } catch (error) {
      console.error('Enrollment error:', error);
      throw error;
    }
  }
  
  // Verify user
  async verifyUser(userId, keystrokeSample) {
    try {
      const response = await axios.post(`${KEYSTROKE_API_URL}/verify`, {
        user_id: userId,
        keystroke_sample: keystrokeSample
      });
      return response.data;
    } catch (error) {
      console.error('Verification error:', error);
      throw error;
    }
  }
  
  // Continuous verification
  async continuousVerify(userId, keystrokeStream) {
    try {
      const response = await axios.post(`${KEYSTROKE_API_URL}/continuous-verify`, {
        user_id: userId,
        keystroke_stream: keystrokeStream
      });
      return response.data;
    } catch (error) {
      console.error('Continuous verification error:', error);
      throw error;
    }
  }
  
  // Get statistics
  async getStatistics(userId) {
    try {
      const response = await axios.get(`${KEYSTROKE_API_URL}/statistics/${userId}`);
      return response.data;
    } catch (error) {
      console.error('Statistics error:', error);
      throw error;
    }
  }
  
  // Convert keystroke events to feature vector
  extractFeatures(keystrokeData) {
    if (keystrokeData.length < 10) {
      return null;
    }
    
    // Extract timing features
    const holdTimes = keystrokeData.map(k => k.holdTime / 1000); // Convert to seconds
    const udTimes = keystrokeData.map(k => k.udTime / 1000);
    
    // Calculate DD times (keydown to keydown)
    const ddTimes = [];
    for (let i = 1; i < keystrokeData.length; i++) {
      const ddTime = (keystrokeData[i].timestamp - keystrokeData[i-1].timestamp) / 1000;
      ddTimes.push(ddTime);
    }
    
    // Combine features (simplified - actual implementation should match training)
    const features = [...holdTimes.slice(0, 10), ...ddTimes.slice(0, 10), ...udTimes.slice(0, 11)];
    
    return features;
  }
}

export default new KeystrokeService();
```

### 1.4 Create Continuous Authentication Component

Create `client/src/components/ContinuousAuth.jsx`:

```javascript
import React, { useEffect, useState } from 'react';
import { useKeystrokeCapture } from '../hooks/useKeystrokeCapture';
import keystrokeService from '../services/keystrokeService';
import { Alert, Badge } from 'react-bootstrap';

const ContinuousAuth = ({ userId, onAuthFailure }) => {
  const { keystrokeData, clearKeystrokeData } = useKeystrokeCapture(userId, true);
  const [authStatus, setAuthStatus] = useState('verified');
  const [confidence, setConfidence] = useState(1.0);
  const [lastCheck, setLastCheck] = useState(null);

  useEffect(() => {
    // Verify every 30 seconds
    const interval = setInterval(async () => {
      if (keystrokeData.length >= 20) {
        try {
          // Extract features
          const features = keystrokeService.extractFeatures(keystrokeData);

          if (features) {
            // Verify user
            const result = await keystrokeService.verifyUser(userId, features);

            setConfidence(result.confidence);
            setLastCheck(new Date());

            if (!result.verified) {
              setAuthStatus('failed');

              // Trigger alert
              if (result.critical) {
                onAuthFailure && onAuthFailure(result);
              }
            } else {
              setAuthStatus('verified');
            }
          }

          // Clear old data
          clearKeystrokeData();
        } catch (error) {
          console.error('Verification error:', error);
          setAuthStatus('error');
        }
      }
    }, 30000); // 30 seconds

    return () => clearInterval(interval);
  }, [keystrokeData, userId, clearKeystrokeData, onAuthFailure]);

  const getStatusColor = () => {
    if (authStatus === 'verified') return 'success';
    if (authStatus === 'failed') return 'danger';
    return 'warning';
  };

  const getConfidenceLevel = () => {
    if (confidence >= 0.9) return 'High';
    if (confidence >= 0.75) return 'Medium';
    return 'Low';
  };

  return (
    <div className="continuous-auth-indicator">
      <Badge bg={getStatusColor()}>
        Auth: {authStatus.toUpperCase()}
      </Badge>
      <small className="ms-2">
        Confidence: {(confidence * 100).toFixed(1)}% ({getConfidenceLevel()})
      </small>
      {lastCheck && (
        <small className="ms-2 text-muted">
          Last check: {lastCheck.toLocaleTimeString()}
        </small>
      )}

      {authStatus === 'failed' && (
        <Alert variant="danger" className="mt-2">
          ⚠️ Authentication failed! Your typing pattern doesn't match.
        </Alert>
      )}
    </div>
  );
};

export default ContinuousAuth;
```

### 1.5 Integrate into Video Call Component

Update your video call component:

```javascript
import React, { useState } from 'react';
import ContinuousAuth from './ContinuousAuth';

const VideoCall = ({ doctorId, patientId }) => {
  const [callActive, setCallActive] = useState(true);

  const handleAuthFailure = (result) => {
    // Alert patient
    alert('Doctor authentication failed! Ending call for security.');

    // End call
    setCallActive(false);

    // Notify backend
    notifySecurityBreach(doctorId, result);
  };

  const notifySecurityBreach = async (doctorId, result) => {
    // Send alert to backend
    await fetch('/api/security/breach', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doctorId,
        type: 'keystroke_auth_failure',
        confidence: result.confidence,
        timestamp: new Date().toISOString()
      })
    });
  };

  return (
    <div className="video-call-container">
      {/* Continuous Authentication Indicator */}
      <ContinuousAuth
        userId={doctorId}
        onAuthFailure={handleAuthFailure}
      />

      {/* Video Call UI */}
      {callActive && (
        <div className="video-streams">
          {/* Your video call implementation */}
        </div>
      )}
    </div>
  );
};

export default VideoCall;
```

---

## Part 2: Backend Integration (Node.js/Express)

### 2.1 Install Dependencies

```bash
cd backend
npm install axios
```

### 2.2 Create Keystroke Service

Create `backend/services/keystrokeService.js`:

```javascript
const axios = require('axios');

const KEYSTROKE_API_URL = process.env.KEYSTROKE_API_URL || 'http://localhost:8002';

class KeystrokeService {
  async enrollDoctor(doctorId, keystrokeSamples) {
    try {
      const response = await axios.post(`${KEYSTROKE_API_URL}/enroll`, {
        user_id: doctorId,
        keystroke_samples: keystrokeSamples
      });

      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      console.error('Keystroke enrollment error:', error.message);
      return {
        success: false,
        error: error.message
      };
    }
  }

  async verifyDoctor(doctorId, keystrokeSample) {
    try {
      const response = await axios.post(`${KEYSTROKE_API_URL}/verify`, {
        user_id: doctorId,
        keystroke_sample: keystrokeSample
      });

      return {
        success: true,
        verified: response.data.verified,
        confidence: response.data.confidence,
        alert: response.data.alert,
        critical: response.data.critical
      };
    } catch (error) {
      console.error('Keystroke verification error:', error.message);
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getStatistics(doctorId) {
    try {
      const response = await axios.get(`${KEYSTROKE_API_URL}/statistics/${doctorId}`);
      return response.data;
    } catch (error) {
      console.error('Statistics error:', error.message);
      return null;
    }
  }
}

module.exports = new KeystrokeService();
```

### 2.3 Create API Routes

Create `backend/routes/keystroke.js`:

```javascript
const express = require('express');
const router = express.Router();
const keystrokeService = require('../services/keystrokeService');
const { protect } = require('../middleware/auth');

// @route   POST /api/keystroke/enroll
// @desc    Enroll doctor for keystroke authentication
// @access  Private
router.post('/enroll', protect, async (req, res) => {
  try {
    const { doctorId, keystrokeSamples } = req.body;

    // Verify user is authorized
    if (req.user.id !== doctorId && req.user.role !== 'admin') {
      return res.status(403).json({ message: 'Not authorized' });
    }

    const result = await keystrokeService.enrollDoctor(doctorId, keystrokeSamples);

    if (result.success) {
      res.json(result.data);
    } else {
      res.status(500).json({ message: result.error });
    }
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// @route   POST /api/keystroke/verify
// @desc    Verify doctor keystroke pattern
// @access  Private
router.post('/verify', protect, async (req, res) => {
  try {
    const { doctorId, keystrokeSample } = req.body;

    const result = await keystrokeService.verifyDoctor(doctorId, keystrokeSample);

    // Log verification attempt
    await logVerificationAttempt(doctorId, result);

    // If critical failure, trigger alert
    if (result.critical) {
      await triggerSecurityAlert(doctorId, result);
    }

    res.json(result);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// @route   GET /api/keystroke/statistics/:doctorId
// @desc    Get keystroke statistics
// @access  Private
router.get('/statistics/:doctorId', protect, async (req, res) => {
  try {
    const { doctorId } = req.params;

    const stats = await keystrokeService.getStatistics(doctorId);
    res.json(stats);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

module.exports = router;
```

### 2.4 Update Main Server File

Add to `backend/server.js`:

```javascript
const keystrokeRoutes = require('./routes/keystroke');

// Routes
app.use('/api/keystroke', keystrokeRoutes);
```

---

## Part 3: Database Schema (MongoDB)

### 3.1 Keystroke Verification Log Schema

Create `backend/models/KeystrokeLog.js`:

```javascript
const mongoose = require('mongoose');

const keystrokeLogSchema = new mongoose.Schema({
  doctorId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Doctor',
    required: true
  },
  sessionId: {
    type: String,
    required: true
  },
  verified: {
    type: Boolean,
    required: true
  },
  confidence: {
    type: Number,
    required: true
  },
  confidenceLevel: {
    type: String,
    enum: ['high', 'medium', 'low'],
    required: true
  },
  alert: {
    type: Boolean,
    default: false
  },
  critical: {
    type: Boolean,
    default: false
  },
  timestamp: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('KeystrokeLog', keystrokeLogSchema);
```

---

## Part 4: Deployment

### 4.1 Start Keystroke Dynamics API

```bash
cd "Keystroke Dynamics"

# Install dependencies
pip install -r requirements.txt

# Train model (first time only)
python main.py train

# Start API server
python main.py api
```

The API will run on `http://localhost:8002`

### 4.2 Environment Variables

Add to `backend/.env`:

```env
KEYSTROKE_API_URL=http://localhost:8002
```

### 4.3 Docker Deployment (Optional)

Create `Keystroke Dynamics/Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8002

CMD ["python", "main.py", "api"]
```

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'

services:
  keystroke-api:
    build: ./Keystroke Dynamics
    ports:
      - "8002:8002"
    volumes:
      - ./Keystroke Dynamics/models:/app/models
      - ./Keystroke Dynamics/logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
```

---

## Part 5: Testing Integration

### 5.1 Test Enrollment

```javascript
// In your React component
const handleEnrollment = async () => {
  // Collect 50+ keystroke samples
  const samples = [];

  // ... collect samples ...

  const result = await keystrokeService.enrollUser(doctorId, samples);
  console.log('Enrollment result:', result);
};
```

### 5.2 Test Verification

```javascript
// In your continuous auth component
const testVerification = async () => {
  const features = keystrokeService.extractFeatures(keystrokeData);
  const result = await keystrokeService.verifyUser(doctorId, features);

  console.log('Verification result:', result);
  // { verified: true, confidence: 0.87, alert: false, critical: false }
};
```

---

## Part 6: Security Considerations

### 6.1 TLS/HTTPS

Ensure all communications use HTTPS in production:

```javascript
const KEYSTROKE_API_URL = process.env.NODE_ENV === 'production'
  ? 'https://api.yourdomain.com/keystroke'
  : 'http://localhost:8002';
```

### 6.2 Rate Limiting

The API has built-in rate limiting (100 requests/minute). Adjust in `config.yaml` if needed.

### 6.3 Data Privacy

- Only behavioral embeddings are stored, not raw keystroke data
- All data is encrypted in transit (TLS 1.3)
- GDPR compliant data retention (90 days)

---

## Part 7: Monitoring & Alerts

### 7.1 Real-time Alerts

When verification fails:

1. **Frontend**: Display alert to patient
2. **Backend**: Log security event
3. **Admin**: Send notification to administrators
4. **Action**: Optionally end session automatically

### 7.2 Dashboard Metrics

Track these metrics:

- Total verifications per doctor
- Verification success rate
- Average confidence scores
- Anomaly detection rate
- False positive/negative rates

---

## Troubleshooting

### Issue: API Connection Failed

**Solution**: Ensure Keystroke Dynamics API is running:
```bash
cd "Keystroke Dynamics"
python main.py api
```

### Issue: Low Confidence Scores

**Solution**:
1. Ensure doctor is properly enrolled with 50+ samples
2. Check if typing environment has changed (different keyboard)
3. Retrain with more diverse samples

### Issue: High False Positive Rate

**Solution**: Adjust threshold in `config.yaml`:
```yaml
verification:
  threshold: 0.70  # Lower threshold (was 0.75)
```

---

## Summary

This integration provides:

✅ **Continuous authentication** during telehealth sessions
✅ **Real-time verification** every 30 seconds
✅ **Automatic alerts** on authentication failure
✅ **Privacy-preserving** behavioral biometrics
✅ **Seamless integration** with MERN stack

For questions or issues, refer to the main README.md or contact the development team.

