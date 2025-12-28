# 🔌 API Usage Examples

## Table of Contents
- [Python Examples](#python-examples)
- [JavaScript Examples](#javascript-examples)
- [cURL Examples](#curl-examples)
- [WebSocket Example](#websocket-example)

---

## Python Examples

### 1. Enroll a Speaker

```python
import requests

API_URL = "http://localhost:8001"

def enroll_speaker(speaker_id, audio_files):
    """Enroll a new speaker"""
    response = requests.post(
        f"{API_URL}/api/v1/enroll",
        json={
            "speaker_id": speaker_id,
            "audio_files": audio_files
        }
    )
    return response.json()

# Example usage
result = enroll_speaker(
    speaker_id="doctor_001",
    audio_files=[
        "path/to/sample1.wav",
        "path/to/sample2.wav",
        "path/to/sample3.wav"
    ]
)

print(f"Enrollment Status: {result['status']}")
print(f"Quality Score: {result['enrollment_quality']}")
```

### 2. Verify Speaker

```python
def verify_speaker(speaker_id, audio_file):
    """Verify speaker identity"""
    response = requests.post(
        f"{API_URL}/api/v1/verify",
        json={
            "speaker_id": speaker_id,
            "audio_file": audio_file
        }
    )
    return response.json()

# Example usage
result = verify_speaker(
    speaker_id="doctor_001",
    audio_file="path/to/verify.wav"
)

print(f"Verified: {result['verified']}")
print(f"Confidence: {result['confidence_score']:.4f}")
print(f"Latency: {result['latency_ms']:.2f} ms")
```

### 3. Upload and Verify

```python
def verify_with_upload(speaker_id, audio_file_path):
    """Verify speaker with file upload"""
    with open(audio_file_path, 'rb') as f:
        files = {'audio_file': f}
        data = {'speaker_id': speaker_id}
        
        response = requests.post(
            f"{API_URL}/api/v1/verify/upload",
            files=files,
            data=data
        )
    
    return response.json()

# Example usage
result = verify_with_upload("doctor_001", "recording.wav")
print(f"Verified: {result['verified']}")
```

### 4. Continuous Verification

```python
def continuous_verify(speaker_id, audio_stream_path):
    """Perform continuous verification on audio stream"""
    with open(audio_stream_path, 'rb') as f:
        files = {'audio_file': f}
        data = {'speaker_id': speaker_id}
        
        response = requests.post(
            f"{API_URL}/api/v1/continuous-verify",
            files=files,
            data=data
        )
    
    return response.json()

# Example usage
result = continuous_verify("doctor_001", "consultation_audio.wav")
print(f"Total Windows: {result['total_windows']}")
print(f"Verified Windows: {result['verified_windows']}")
print(f"Average Confidence: {result['average_confidence']:.4f}")
```

---

## JavaScript Examples

### 1. Enroll a Speaker (React/Node.js)

```javascript
const API_URL = 'http://localhost:8001';

async function enrollSpeaker(speakerId, audioFiles) {
  const response = await fetch(`${API_URL}/api/v1/enroll`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      speaker_id: speakerId,
      audio_files: audioFiles
    })
  });
  
  return await response.json();
}

// Example usage
const result = await enrollSpeaker('doctor_001', [
  'path/to/sample1.wav',
  'path/to/sample2.wav',
  'path/to/sample3.wav'
]);

console.log('Enrollment Status:', result.status);
console.log('Quality Score:', result.enrollment_quality);
```

### 2. Verify with File Upload (React)

```javascript
async function verifyWithUpload(speakerId, audioBlob) {
  const formData = new FormData();
  formData.append('speaker_id', speakerId);
  formData.append('audio_file', audioBlob, 'recording.wav');
  
  const response = await fetch(`${API_URL}/api/v1/verify/upload`, {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

// Example usage with MediaRecorder
let mediaRecorder;
let audioChunks = [];

// Start recording
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };
    
    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      const result = await verifyWithUpload('doctor_001', audioBlob);
      
      console.log('Verified:', result.verified);
      console.log('Confidence:', result.confidence_score);
    };
    
    mediaRecorder.start();
  });

// Stop recording after 3 seconds
setTimeout(() => {
  mediaRecorder.stop();
}, 3000);
```

### 3. Real-time Verification Component (React)

```javascript
import React, { useState } from 'react';

function VoiceVerification({ doctorId }) {
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState(null);
  
  const handleVerify = async (audioBlob) => {
    setIsVerifying(true);
    
    try {
      const formData = new FormData();
      formData.append('speaker_id', doctorId);
      formData.append('audio_file', audioBlob);
      
      const response = await fetch('http://localhost:8001/api/v1/verify/upload', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      setResult(data);
      
      // Trigger alert if verification fails
      if (!data.verified) {
        alert('⚠️ Voice verification failed! Confidence: ' + data.confidence_score);
      }
    } catch (error) {
      console.error('Verification error:', error);
    } finally {
      setIsVerifying(false);
    }
  };
  
  return (
    <div>
      {result && (
        <div className={result.verified ? 'success' : 'error'}>
          <p>Verified: {result.verified ? '✓' : '✗'}</p>
          <p>Confidence: {(result.confidence_score * 100).toFixed(2)}%</p>
          <p>Latency: {result.latency_ms.toFixed(2)} ms</p>
        </div>
      )}
    </div>
  );
}
```

---

## cURL Examples

### 1. Health Check

```bash
curl http://localhost:8001/health
```

### 2. Enroll Speaker

```bash
curl -X POST http://localhost:8001/api/v1/enroll \
  -H "Content-Type: application/json" \
  -d '{
    "speaker_id": "doctor_001",
    "audio_files": [
      "path/to/sample1.wav",
      "path/to/sample2.wav",
      "path/to/sample3.wav"
    ]
  }'
```

### 3. Verify Speaker

```bash
curl -X POST http://localhost:8001/api/v1/verify \
  -H "Content-Type: application/json" \
  -d '{
    "speaker_id": "doctor_001",
    "audio_file": "path/to/verify.wav"
  }'
```

### 4. Upload and Verify

```bash
curl -X POST http://localhost:8001/api/v1/verify/upload \
  -F "speaker_id=doctor_001" \
  -F "audio_file=@recording.wav"
```

### 5. Update Threshold

```bash
curl -X PUT http://localhost:8001/api/v1/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.70}'
```

### 6. Get Enrolled Speakers

```bash
curl http://localhost:8001/api/v1/speakers
```

### 7. Get Speaker Info

```bash
curl http://localhost:8001/api/v1/speakers/doctor_001
```

---

## WebSocket Example

### Python WebSocket Client

```python
import asyncio
import websockets
import json

async def continuous_verification(speaker_id, audio_chunks):
    """Real-time continuous verification via WebSocket"""
    uri = f"ws://localhost:8001/ws/verify/{speaker_id}"
    
    async with websockets.connect(uri) as websocket:
        for chunk in audio_chunks:
            # Send audio chunk
            await websocket.send(chunk)
            
            # Receive verification result
            result = await websocket.recv()
            result_data = json.loads(result)
            
            print(f"Verified: {result_data['verified']}")
            print(f"Confidence: {result_data['confidence_score']:.4f}")
            
            # Alert if verification fails
            if not result_data['verified']:
                print("⚠️ ALERT: Verification failed!")

# Example usage
# asyncio.run(continuous_verification("doctor_001", audio_chunks))
```

### JavaScript WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/verify/doctor_001');

ws.onopen = () => {
  console.log('WebSocket connected');
  
  // Send audio chunks
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      ws.send(event.data);
    }
  };
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  
  console.log('Verified:', result.verified);
  console.log('Confidence:', result.confidence_score);
  
  // Trigger alert if verification fails
  if (!result.verified) {
    alert('⚠️ Voice verification failed!');
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket disconnected');
};
```

---

## Complete Integration Example (MERN Stack)

```javascript
// Backend (Node.js/Express) - Proxy to Python API
const express = require('express');
const axios = require('axios');
const multer = require('multer');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

const VOICEPRINT_API = 'http://localhost:8001';

// Enroll doctor
app.post('/api/doctors/:id/enroll', async (req, res) => {
  try {
    const response = await axios.post(`${VOICEPRINT_API}/api/v1/enroll`, {
      speaker_id: req.params.id,
      audio_files: req.body.audio_files
    });
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Verify during consultation
app.post('/api/consultations/:id/verify', upload.single('audio'), async (req, res) => {
  try {
    const formData = new FormData();
    formData.append('speaker_id', req.body.doctor_id);
    formData.append('audio_file', req.file.buffer, 'audio.wav');
    
    const response = await axios.post(
      `${VOICEPRINT_API}/api/v1/verify/upload`,
      formData
    );
    
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

---

For more examples and detailed API documentation, visit:
**http://localhost:8001/docs** (when the API server is running)

