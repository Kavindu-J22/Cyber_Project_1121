# Zero Trust Telehealth Platform

A cutting-edge telehealth platform with **continuous biometric authentication** using voice recognition, keystroke dynamics, and mouse movement analysis to ensure secure medical consultations.

## 🎯 Project Overview

This project addresses a critical vulnerability in modern telehealth: **the lack of continuous trust**. We've built a Zero Trust security platform that doesn't just authenticate a doctor at login, but continuously verifies their identity throughout the entire consultation.

### Key Features

- **Multi-Modal Biometric Authentication**
  - 🎤 Voice Recognition (ECAPA-TDNN model)
  - ⌨️ Keystroke Dynamics (Deep Neural Network)
  - 🖱️ Mouse Movement Analysis (Siamese Network)
  - 👤 Face Verification (ResNet50 Triplet Loss)

- **Real-time Verification**
  - Continuous monitoring during video consultations
  - Live trust score calculation
  - Instant alerts for suspicious activity

- **MERN Stack Application**
  - React frontend with Tailwind CSS
  - Node.js/Express backend
  - MongoDB database
  - Socket.IO for real-time communication

## 📁 Project Structure

```
Cyber_Project_1121/
├── Backend/                    # Node.js/Express backend
│   ├── config/                 # Database configuration
│   ├── controllers/            # Route controllers
│   ├── middleware/             # Auth & upload middleware
│   ├── models/                 # MongoDB schemas
│   ├── routes/                 # API routes
│   ├── services/               # ML service integration
│   └── server.js               # Main server file
│
├── Client/                     # React frontend
│   ├── src/
│   │   ├── components/         # Reusable components
│   │   ├── context/            # Auth context
│   │   ├── pages/              # Page components
│   │   ├── utils/              # Biometric capture utilities
│   │   ├── App.jsx             # Main app component
│   │   └── main.jsx            # Entry point
│   └── package.json
│
├── Voiceprint Analysis/        # Python ML - Voice Recognition
│   ├── src/                    # Voice analysis modules
│   ├── main.py                 # CLI interface
│   └── start_api.bat           # API launcher
│
├── Keystroke Dynamics/         # Python ML - Keystroke Analysis
│   ├── src/                    # Keystroke analysis modules
│   ├── main.py                 # CLI interface
│   └── start_api.bat           # API launcher
│
└── Mouse Movement Analysis/    # Python ML - Mouse Analysis
    ├── src/                    # Mouse analysis modules
    ├── main.py                 # CLI interface
    └── start_api.bat           # API launcher
```

## 🚀 Installation & Setup

### Prerequisites

- Node.js (v18 or higher)
- Python 3.8+
- MongoDB Atlas account (or local MongoDB)
- Git

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Cyber_Project_1121
```

### Step 2: Setup Python ML Services

#### Voice Recognition API (Port 8001)
```bash
cd "Voiceprint Analysis"
pip install -r requirements.txt
python main.py api
```

#### Keystroke Dynamics API (Port 8002)
```bash
cd "Keystroke Dynamics"
pip install -r requirements.txt
python main.py api
```

#### Mouse Movement API (Port 8003)
```bash
cd "Mouse Movement Analysis"
pip install -r requirements.txt
python main.py api
```

### Step 3: Setup Backend

```bash
cd Backend
npm install
npm run dev
```

The backend will run on `http://localhost:5000`

### Step 4: Setup Frontend

```bash
cd Client
npm install
npm run dev
```

The frontend will run on `http://localhost:5173`

## 🔧 Configuration

### Backend Environment Variables

Edit `Backend/.env`:

```env
PORT=5000
MONGODB_URI=mongodb+srv://kavindu:kavindu@cluster0.xgvsfnu.mongodb.net/zero-trust-telehealth
JWT_SECRET=your-super-secret-jwt-key
JWT_EXPIRE=7d

# Python ML Service URLs
VOICE_API_URL=http://localhost:8001
KEYSTROKE_API_URL=http://localhost:8002
MOUSE_API_URL=http://localhost:8003

CLIENT_URL=http://localhost:5173
```

## 📖 Usage Guide

### 1. Doctor Registration

1. Navigate to `http://localhost:5173/register`
2. Fill in personal information (name, email, password)
3. Add professional details (license number, specialization, experience)
4. Complete biometric enrollment:
   - **Voice Sample**: Record 5-10 seconds of speech
   - **Keystroke Pattern**: Type the given phrase 3 times
   - **Mouse Pattern**: Move mouse naturally for 10 seconds
5. Submit registration

### 2. Login

1. Navigate to `http://localhost:5173/login`
2. Enter email and password
3. Access dashboard

### 3. Start Consultation

1. From dashboard, click "Start Consultation"
2. Allow camera and microphone access
3. The system will continuously verify your identity using:
   - Voice patterns (when speaking)
   - Typing patterns (when typing)
   - Mouse movements (continuous)
4. Monitor your trust score in real-time
5. End consultation when done

## 🏗️ Architecture

### Frontend (React)
- **Pages**: Login, Register, Dashboard, Meeting
- **Biometric Capture**: Real-time keystroke and mouse event capture
- **WebSocket**: Real-time verification updates
- **State Management**: React Context API

### Backend (Node.js/Express)
- **Authentication**: JWT-based auth
- **Database**: MongoDB with Mongoose
- **Real-time**: Socket.IO for live verification
- **ML Integration**: Axios HTTP client to Python APIs

### ML Services (Python)
- **Voice**: FastAPI server on port 8001
- **Keystroke**: FastAPI server on port 8002
- **Mouse**: FastAPI server on port 8003

## 🔐 Security Features

1. **Continuous Authentication**: Not just login - verify throughout session
2. **Multi-Modal Fusion**: Combine 3 biometric modalities
3. **Real-time Alerts**: Instant notification of suspicious activity
4. **Trust Score**: Dynamic score based on verification confidence
5. **Session Logging**: Complete audit trail of all verifications

## 📊 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new doctor
- `POST /api/auth/login` - Login

### Doctors
- `GET /api/doctors` - Get all doctors
- `GET /api/doctors/me` - Get current doctor
- `GET /api/doctors/:id` - Get doctor by ID

### Sessions
- `POST /api/sessions` - Create new session
- `GET /api/sessions/:sessionId` - Get session details
- `PUT /api/sessions/:sessionId` - Update session
- `POST /api/sessions/:sessionId/verification` - Add verification log

### Verification
- `POST /api/verification/voice` - Verify voice sample
- `POST /api/verification/keystroke` - Verify keystroke pattern
- `POST /api/verification/mouse` - Verify mouse movement
- `GET /api/verification/health` - Check ML services health

## 🧪 Testing

1. Ensure all 3 Python ML APIs are running
2. Start the backend server
3. Start the frontend development server
4. Register a new doctor account
5. Login and start a consultation
6. Observe real-time verification in the meeting interface

## 🤖 Machine Learning Models Documentation

### Comprehensive ML Documentation

We've created detailed documentation explaining each ML component, the models used, and why they were chosen:

1. **[ML_MODELS_TECHNICAL_EXPLANATION.md](ML_MODELS_TECHNICAL_EXPLANATION.md)**
   - Complete technical details of all models
   - Architecture diagrams and specifications
   - Training strategies and loss functions
   - Why deep learning over traditional ML
   - Performance metrics and comparisons

2. **[ML_MODELS_QUICK_REFERENCE.md](ML_MODELS_QUICK_REFERENCE.md)**
   - Quick lookup tables for all models
   - Feature breakdowns
   - Performance metrics at a glance
   - Configuration settings

3. **[ML_MODELS_SUMMARY.md](ML_MODELS_SUMMARY.md)**
   - Executive overview of all ML components
   - Model selection rationale
   - Security features
   - Deployment information

4. **[WHY_DEEP_LEARNING_NOT_RANDOM_FOREST.md](WHY_DEEP_LEARNING_NOT_RANDOM_FOREST.md)**
   - Detailed comparison: Deep Learning vs. Random Forest
   - Real performance data and experiments
   - When to use each approach
   - Technical deep dive on embeddings

### Model Overview

| Component | Model | Accuracy | Speed | Why This Model? |
|-----------|-------|----------|-------|----------------|
| **Voice** | ECAPA-TDNN | 97%+ | 800ms | State-of-the-art speaker recognition with channel attention |
| **Anti-Spoof** | Lightweight CNN | 95%+ | 100ms | Spatial pattern recognition in spectrograms |
| **Keystroke** | Deep Neural Net | 96% | 500ms | Non-linear temporal pattern recognition |
| **Mouse** | Siamese Network | 92%+ | 300ms | Similarity metric learning for behavioral comparison |

### Key Advantages of Deep Learning

✅ **11% Higher Accuracy** - 96% vs. 85% with Random Forest
✅ **Embedding Space** - Creates compact behavioral signatures (128-192 dimensions)
✅ **Few-Shot Learning** - Works with only 3-5 enrollment samples
✅ **Continuous Confidence Scores** - Not just binary yes/no decisions
✅ **No Retraining** - New users can be enrolled instantly
✅ **Compact Storage** - Only 512 bytes to 2KB per user

For complete technical details, see the documentation files listed above.

---

## 📝 Notes

- The MongoDB connection string is already configured in `.env`
- All Python ML models are pre-trained and ready to use
- Voice recording requires HTTPS in production (works on localhost)
- Ensure all 5 services are running simultaneously for full functionality

## 🤝 Contributing

This is an academic project for cybersecurity research. Contributions are welcome!

## 📄 License

This project is for educational and research purposes.

## 👥 Team

Developed as part of a cybersecurity research project on continuous authentication in telehealth.

---

**Built with ❤️ for secure healthcare**

