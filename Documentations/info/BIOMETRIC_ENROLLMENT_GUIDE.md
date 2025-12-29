# Biometric Enrollment Guide

## Overview
This guide explains how to start the ML API servers and use the improved biometric enrollment system.

## Quick Start

### 1. Start All ML API Servers (Recommended)

Simply run the PowerShell script from the project root:

```powershell
.\start-ml-apis.ps1
```

This will automatically:
- Open 3 separate terminal windows
- Activate the virtual environment for each API
- Start Voice Print API on port 8001
- Start Keystroke Dynamics API on port 8002
- Start Mouse Movement API on port 8003

### 2. Manual Start (Alternative)

If you prefer to start each API manually:

#### Voice Print API (Port 8001)
```powershell
cd "Voiceprint Analysis"
.\venv\Scripts\Activate.ps1
python main.py api
```

#### Keystroke Dynamics API (Port 8002)
```powershell
c
.\venv\Scripts\Activate.ps1
python main.py api
```

#### Mouse Movement API (Port 8003)
```powershell
cd "Mouse Movement Analysis"
.\venv\Scripts\Activate.ps1
python main.py api
```

### 3. Verify APIs are Running

Open your browser and check these health endpoints:
- http://localhost:8001/health (Voice Print)
- http://localhost:8002/health (Keystroke Dynamics)
- http://localhost:8003/health (Mouse Movement)

All should return `{"status": "healthy", ...}`

### 4. Start Backend Server

```bash
cd Backend
npm run dev
```

### 5. Start Frontend

```bash
cd Client
npm run dev
```

## Biometric Enrollment Process

### Voice Biometric
1. Click "Start Voice Recording"
2. Speak clearly for 5-10 seconds (e.g., "My name is [Your Name] and I am a medical professional")
3. Recording will auto-stop after 10 seconds or click "Stop Recording"
4. ✅ Voice sample captured!

**Requirements:**
- Minimum: 1 audio sample (5-10 seconds)
- Format: WAV (automatically converted)

### Keystroke Dynamics
1. Click "Capture Sample 1/3"
2. Type the phrase: "The quick brown fox jumps over the lazy dog"
3. Click "Complete Sample"
4. Repeat for samples 2 and 3
5. ✅ All keystroke samples captured!

**Requirements:**
- Minimum: 3 typing samples
- Each sample: ~40+ characters

### Mouse Movement Pattern
1. Click "Start Mouse Recording"
2. Move your mouse naturally in the capture area for 15 seconds
3. Click, drag, and move around naturally
4. Recording will auto-stop after 15 seconds
5. ✅ Mouse pattern captured!

**Requirements:**
- Minimum: 50+ mouse events
- Duration: 15 seconds (auto-capture)

## Configuration Changes Made

### Voice API
- ✅ Added `/api/v1/enroll/upload` endpoint for single file upload
- ✅ Accepts FormData with audio file
- ✅ Automatically processes audio into segments

### Keystroke API
- ✅ Reduced `min_samples` from 50 to 3
- ✅ Reduced `min_few_shot_samples` from 10 to 3
- ✅ Easier onboarding for new users

### Mouse Movement API
- ✅ Reduced `min_samples` from 50 to 1
- ✅ Reduced `min_few_shot_samples` from 50 to 1
- ✅ Single session enrollment supported

### Backend
- ✅ Improved error handling
- ✅ Better logging for debugging
- ✅ Saves doctor record even if biometric enrollment partially fails
- ✅ Detailed error messages for each biometric modality

### Frontend
- ✅ Professional, user-friendly UI
- ✅ Progress indicators with timers
- ✅ Auto-stop timers for voice and mouse
- ✅ Visual feedback for completion
- ✅ Clear instructions for each step
- ✅ Color-coded status (green = complete)

## Troubleshooting

### Voice API Error: "UnicodeDecodeError"
**Fixed!** The new `/api/v1/enroll/upload` endpoint properly handles binary audio data.

### Keystroke API Error: "Need at least 50 samples"
**Fixed!** Configuration updated to accept 3 samples minimum.

### Mouse API Error: "Insufficient samples"
**Fixed!** Configuration updated to accept 1 sample (with 50+ events).

### Database shows biometric fields as null/false
**Fixed!** All issues resolved:
1. Voice API now accepts single file upload
2. Keystroke and Mouse APIs accept fewer samples
3. Backend properly saves enrollment results

## Testing

After starting all services, test the registration flow:

1. Navigate to http://localhost:3000/register
2. Fill in personal information (Step 1)
3. Fill in professional information (Step 2)
4. Complete all three biometric enrollments (Step 3)
5. Click "Complete Registration"
6. Check the database - all biometric fields should be populated!

## Support

If you encounter any issues:
1. Check that all 3 ML APIs are running (health endpoints)
2. Check browser console for errors
3. Check backend terminal for error logs
4. Check ML API terminals for enrollment errors

