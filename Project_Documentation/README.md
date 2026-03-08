# MediConsult — Project Documentation Index

> Comprehensive documentation for the MediConsult Zero-Trust Telemedicine Platform

---

## Documentation Files

| # | Document | Description |
|---|----------|-------------|
| 01 | [System Overview](01_SYSTEM_OVERVIEW.md) | Architecture, tech stack, user roles, security model, service map |
| 02 | [Backend API Reference](02_BACKEND_API_REFERENCE.md) | All 38 REST endpoints, 6 database models, middleware, Socket.IO events |
| 03 | [Frontend Architecture](03_FRONTEND_ARCHITECTURE.md) | Pages, components, routing, AuthContext, biometric capture utilities |
| 04 | [Face Verification Model](04_MODEL_FACE_VERIFICATION.md) | ResNet50 + Triplet Loss, 128-D embeddings, enrollment & verification |
| 05 | [Voiceprint Analysis Model](05_MODEL_VOICEPRINT_ANALYSIS.md) | ECAPA-TDNN, 192-D embeddings, anti-spoofing, audio preprocessing |
| 06 | [Keystroke Dynamics Model](06_MODEL_KEYSTROKE_DYNAMICS.md) | 4-layer DNN, 38 features, adaptive templates, timing analysis |
| 07 | [Mouse Movement Model](07_MODEL_MOUSE_MOVEMENT.md) | Siamese Network, ~50 features, anomaly detection, bot detection |
| 08 | [Application Flows & Logic](08_APPLICATION_FLOWS_AND_LOGIC.md) | Registration, login, booking, verification gate, consultation, lockout |
| 09 | [Tools & Dependencies](09_TOOLS_AND_DEPENDENCIES.md) | Every dependency with purpose, port allocation, configuration files |
| 10 | [Controllers Reference](10_CONTROLLERS_REFERENCE.md) | All 36 controller functions: routes, auth, inputs, business logic |
| 11 | [Features Summary](11_FEATURES_SUMMARY.md) | Complete feature list: user-facing, security, real-time, ML/AI |
| 12 | [ML Models Integration](12_ML_MODELS_INTEGRATION.md) | How all 4 models connect to the web app: APIs, timing, data flow |
| 13 | [Environment Setup](13_ENVIRONMENT_SETUP.md) | Every env var, where it's used, default values, external services |
| 14 | [Security Audit](14_SECURITY_AUDIT.md) | Hardcoded credentials, exposed secrets, OWASP findings, fix priorities |
| 15 | [Database Schema](15_DATABASE_SCHEMA.md) | All 6 Mongoose models, fields, types, relationships, data flow |

---

## Quick Reference

### Service Ports

| Service | Port | Tech |
|---------|------|------|
| Backend API | 5000 | Express.js + Socket.IO |
| Frontend | 5173 | Vite + React |
| Voiceprint API | 8001 | FastAPI + ECAPA-TDNN |
| Keystroke API | 8002 | FastAPI + DNN |
| Mouse API | 8003 | FastAPI + Siamese Network |
| Face API | 8004 | FastAPI + ResNet50 |

### Tech Stack Summary

- **Frontend:** React 18, Vite, Tailwind CSS, Socket.IO Client, WebRTC
- **Backend:** Express.js, MongoDB Atlas, Mongoose, JWT, Socket.IO
- **ML Services:** Python 3.10+, FastAPI, PyTorch, SpeechBrain
- **Infrastructure:** Cloudinary, Gmail SMTP, ffmpeg, bcrypt

### How to Start

```bash
# 1. Install dependencies
cd Backend && npm install
cd ../Client && npm install

# 2. Start backend
cd Backend && node server.js

# 3. Start frontend
cd Client && npm run dev

# 4. Start ML services (each in separate terminal)
cd "face verification" && python main.py
cd "Voiceprint Analysis" && python main.py
cd "Keystroke Dynamics" && python main.py
cd "Mouse Movement Analysis" && python main.py
```

---

*Generated from complete codebase analysis of the Cyber_Project_1121 repository.*
