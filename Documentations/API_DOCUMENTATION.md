# API Documentation - Zero Trust Telehealth Platform

## Base URLs

- **Backend API**: http://localhost:5000
- **Voice API**: http://localhost:8001
- **Keystroke API**: http://localhost:8002
- **Mouse API**: http://localhost:8003

---

## Authentication

All protected endpoints require a JWT token in the Authorization header:

```
Authorization: Bearer <token>
```

---

## Backend API Endpoints

### Authentication

#### Register Doctor
```http
POST /api/auth/register
Content-Type: multipart/form-data

Body:
- firstName: string
- lastName: string
- email: string
- password: string
- medicalLicenseNumber: string
- specialization: string
- yearsOfExperience: number
- voiceSample: file (audio/wav)
- keystrokePattern: string (JSON array)
- mousePattern: string (JSON array)

Response:
{
  "success": true,
  "message": "Doctor registered successfully",
  "data": {
    "doctor": { ... },
    "biometricResults": {
      "voice": true,
      "keystroke": true,
      "mouse": true
    },
    "token": "jwt_token_here"
  }
}
```

#### Login
```http
POST /api/auth/login
Content-Type: application/json

Body:
{
  "email": "doctor@example.com",
  "password": "password123"
}

Response:
{
  "success": true,
  "message": "Login successful",
  "data": {
    "doctor": { ... },
    "token": "jwt_token_here"
  }
}
```

---

### Doctors

#### Get Current Doctor
```http
GET /api/doctors/me
Authorization: Bearer <token>

Response:
{
  "success": true,
  "data": {
    "_id": "...",
    "firstName": "John",
    "lastName": "Smith",
    "email": "john@example.com",
    "medicalLicenseNumber": "MD12345",
    "specialization": "Cardiology",
    "yearsOfExperience": 10,
    "biometricData": {
      "voiceEnrolled": true,
      "keystrokeEnrolled": true,
      "mouseEnrolled": true
    }
  }
}
```

#### Get All Doctors
```http
GET /api/doctors
Authorization: Bearer <token>

Response:
{
  "success": true,
  "count": 5,
  "data": [ ... ]
}
```

---

### Sessions

#### Create Session
```http
POST /api/sessions
Authorization: Bearer <token>
Content-Type: application/json

Body:
{
  "patientId": "patient_123"
}

Response:
{
  "success": true,
  "message": "Session created successfully",
  "data": {
    "sessionId": "uuid-here",
    "doctorId": "...",
    "patientId": "patient_123",
    "status": "active",
    "startTime": "2024-01-01T00:00:00.000Z",
    "overallTrustScore": 100
  }
}
```

#### Get Session
```http
GET /api/sessions/:sessionId
Authorization: Bearer <token>

Response:
{
  "success": true,
  "data": {
    "sessionId": "...",
    "doctorId": { ... },
    "verificationLogs": [ ... ],
    "alerts": [ ... ],
    "overallTrustScore": 95
  }
}
```

#### Add Verification Log
```http
POST /api/sessions/:sessionId/verification
Authorization: Bearer <token>
Content-Type: application/json

Body:
{
  "verificationType": "voice",
  "verified": true,
  "confidence": 0.95,
  "details": { ... }
}

Response:
{
  "success": true,
  "message": "Verification log added",
  "data": { ... }
}
```

---

### Verification

#### Verify Voice
```http
POST /api/verification/voice
Authorization: Bearer <token>
Content-Type: multipart/form-data

Body:
- voiceSample: file (audio)

Response:
{
  "success": true,
  "data": {
    "verified": true,
    "confidence": 0.95,
    "speaker_id": "...",
    "timestamp": "..."
  }
}
```

#### Verify Keystroke
```http
POST /api/verification/keystroke
Authorization: Bearer <token>
Content-Type: application/json

Body:
{
  "keystrokeSample": [0.12, 0.15, ...]
}

Response:
{
  "success": true,
  "data": {
    "verified": true,
    "confidence": 0.92,
    "user_id": "...",
    "timestamp": "..."
  }
}
```

#### Verify Mouse
```http
POST /api/verification/mouse
Authorization: Bearer <token>
Content-Type: application/json

Body:
{
  "mouseEvents": [
    {
      "timestamp": 1234567890.1,
      "x": 100,
      "y": 200,
      "button": "NoButton",
      "state": "Move"
    },
    ...
  ]
}

Response:
{
  "success": true,
  "data": {
    "verified": true,
    "confidence": 0.88,
    "user_id": "...",
    "timestamp": "..."
  }
}
```

---

## WebSocket Events

### Client → Server

#### Join Session
```javascript
socket.emit('join-session', {
  sessionId: 'uuid',
  doctorId: 'doctor_id'
});
```

#### Verify Biometric
```javascript
socket.emit('verify-biometric', {
  sessionId: 'uuid',
  doctorId: 'doctor_id',
  type: 'keystroke', // or 'mouse', 'voice'
  payload: [...]
});
```

### Server → Client

#### Session Joined
```javascript
socket.on('session-joined', (data) => {
  // data: { sessionId, socketId }
});
```

#### Verification Result
```javascript
socket.on('verification-result', (data) => {
  // data: { type, result, trustScore }
});
```

#### Verification Alert
```javascript
socket.on('verification-alert', (data) => {
  // data: { type, severity, message, trustScore }
});
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "success": false,
  "message": "Error description",
  "error": "Detailed error (development only)"
}
```

Common HTTP Status Codes:
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `500` - Server Error

---

## Rate Limiting

- Authentication endpoints: 5 requests per minute
- Other endpoints: 100 requests per minute

---

## Data Models

### Doctor
```typescript
{
  _id: ObjectId,
  firstName: string,
  lastName: string,
  email: string,
  password: string (hashed),
  medicalLicenseNumber: string,
  specialization: string,
  yearsOfExperience: number,
  biometricData: {
    voiceEnrolled: boolean,
    voiceEmbedding: string,
    keystrokeEnrolled: boolean,
    keystrokeProfile: string,
    mouseEnrolled: boolean,
    mouseProfile: string
  },
  isActive: boolean,
  createdAt: Date,
  lastLogin: Date
}
```

### Session
```typescript
{
  _id: ObjectId,
  sessionId: string (UUID),
  doctorId: ObjectId (ref: Doctor),
  patientId: string,
  startTime: Date,
  endTime: Date,
  status: 'active' | 'completed' | 'terminated' | 'suspicious',
  verificationLogs: [{
    timestamp: Date,
    verificationType: 'voice' | 'keystroke' | 'mouse' | 'combined',
    verified: boolean,
    confidence: number,
    details: object
  }],
  alerts: [{
    timestamp: Date,
    type: string,
    severity: 'low' | 'medium' | 'high' | 'critical',
    message: string,
    details: object
  }],
  overallTrustScore: number (0-100),
  metadata: object
}
```

---

**For more details, see the source code in Backend/routes/ and Backend/controllers/**

