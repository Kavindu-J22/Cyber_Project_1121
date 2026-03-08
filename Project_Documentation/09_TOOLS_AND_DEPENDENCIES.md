# Tools, Libraries & Dependencies Reference

## Why Each Tool Is Used and Where

---

## Backend Dependencies (`Backend/package.json`)

### Server Framework
| Package | Purpose | Where Used |
|---------|---------|------------|
| **express** | HTTP server framework, routing, middleware pipeline | `server.js` — entire backend is built on Express |
| **cors** | Cross-Origin Resource Sharing — allows frontend (port 5173) to call backend (port 5000) | `server.js` — applied globally |
| **helmet** | Sets security HTTP headers (XSS protection, HSTS, no-sniff, clickjacking prevention) | `server.js` — applied globally |
| **morgan** | HTTP request logger — logs method, URL, status, response time | `server.js` — development logging |
| **dotenv** | Loads environment variables from `.env` file | `server.js` — MongoDB URI, JWT secret, email creds |

### Database
| Package | Purpose | Where Used |
|---------|---------|------------|
| **mongoose** | MongoDB ODM — schema definitions, validation, queries, indexing | All model files (`Doctor.js`, `Patient.js`, `Session.js`, etc.) |

### Authentication & Security
| Package | Purpose | Where Used |
|---------|---------|------------|
| **jsonwebtoken** | Create and verify JWT tokens for stateless authentication | `middleware/auth.js` — `generateToken()`, `protect()` |
| **bcryptjs** | Password hashing with salt rounds | `models/Doctor.js`, `models/Patient.js` — pre-save hooks, `comparePassword()` |

### File Handling
| Package | Purpose | Where Used |
|---------|---------|------------|
| **multer** | Multipart form data parser for file uploads | `middleware/upload.js` — voice samples, face images |
| **cloudinary** | Cloud image storage and transformation | `controllers/doctorController.js` — profile image upload/delete |

### Real-Time Communication
| Package | Purpose | Where Used |
|---------|---------|------------|
| **socket.io** | WebSocket server for real-time bidirectional events | `server.js` — WebRTC signaling, chat, biometric updates, lockout |

### Email
| Package | Purpose | Where Used |
|---------|---------|------------|
| **nodemailer** | SMTP email sending via Gmail | `utils/emailService.js` — OTP, approval, rejection, alerts |

### HTTP Client
| Package | Purpose | Where Used |
|---------|---------|------------|
| **axios** | HTTP client for making requests to ML microservices | `services/mlService.js` — all enroll/verify calls to ports 8001-8004 |

### Utilities
| Package | Purpose | Where Used |
|---------|---------|------------|
| **uuid** | Generate unique session IDs | `controllers/sessionController.js` — biometric session IDs |

---

## Frontend Dependencies (`Client/package.json`)

### Core Framework
| Package | Purpose | Where Used |
|---------|---------|------------|
| **react** | UI component library with hooks | Every component file |
| **react-dom** | React DOM renderer | `main.jsx` — app mount point |
| **react-router-dom** | Client-side routing with protected routes | `App.jsx` — all route definitions, `PrivateRoute.jsx` |

### HTTP & Real-Time
| Package | Purpose | Where Used |
|---------|---------|------------|
| **axios** | HTTP client with interceptors for API calls | `AuthContext.jsx` — all API communication |
| **socket.io-client** | WebSocket client for real-time consultation features | `Meeting.jsx` — signaling, chat, biometric events |

### UI
| Package | Purpose | Where Used |
|---------|---------|------------|
| **tailwindcss** | Utility-first CSS framework for all styling | Every component — classnames like `bg-blue-500`, `text-white` |
| **lucide-react** | Icon library (Camera, Mic, Phone, Settings, etc.) | Dashboard, Meeting, modals — UI icons |
| **react-hot-toast** | Toast notification popups | Throughout — success/error feedback |
| **recharts** | Chart/graph library | Available for analytics dashboards |

### Build Tools
| Package | Purpose | Where Used |
|---------|---------|------------|
| **vite** | Fast dev server with HMR + production bundler | Build pipeline, proxy config to backend |
| **postcss** + **autoprefixer** | CSS post-processing for browser compatibility | `postcss.config.js` |

---

## ML Service Dependencies

### Shared Across All Models
| Package | Purpose | Where Used |
|---------|---------|------------|
| **torch / pytorch** | Deep learning framework — model definition, training, inference | All 4 ML services |
| **fastapi** | REST API framework for Python | All 4 ML APIs |
| **uvicorn** | ASGI server to run FastAPI | All 4 ML services — `main.py` |
| **numpy** | Numerical computation — embedding operations | All 4 ML services |
| **pyyaml** | YAML configuration file parsing | All 4 ML services — `config.yaml` |
| **loguru** | Advanced logging with rotation and formatting | All 4 ML services — structured logs |
| **scikit-learn** | Machine learning utilities, metrics, scalers | Keystroke, Mouse — RobustScaler, metrics |
| **pydantic** | Data validation for FastAPI request/response models | All 4 ML services — API schemas |

### Voice-Specific
| Package | Purpose | Where Used |
|---------|---------|------------|
| **speechbrain** | Pre-trained ECAPA-TDNN speaker recognition model | Voice service — embedding extraction |
| **torchaudio** | Audio loading and resampling | Voice service — audio preprocessing |
| **librosa** | Audio analysis and feature extraction | Voice service — mel-spectrogram, LFCC |
| **soundfile** | Audio file I/O | Voice service — WAV reading/writing |
| **noisereduce** | Stationary noise reduction from audio | Voice service — preprocessing |
| **pydub** | Audio format conversion | Voice service — format handling |
| **transformers** | HuggingFace model loading infrastructure | Voice service — SpeechBrain dependency |
| **cryptography** | AES-256 encryption for voiceprint embeddings | Voice service — security layer |

### Face-Specific
| Package | Purpose | Where Used |
|---------|---------|------------|
| **torchvision** | Pre-trained ResNet50 + image transforms | Face service — model backbone, preprocessing |
| **pillow** | Image loading and manipulation | Face service — image I/O |
| **opencv-python** | Face detection via Haar Cascade classifier | Face service — anti-spoofing gate |

### Mouse-Specific
| Package | Purpose | Where Used |
|---------|---------|------------|
| **scipy** | Statistical functions (skew, kurtosis) | Mouse service — feature extraction |
| **pandas** | Data loading from Balabit CSV dataset | Mouse service — training data handling |
| **tensorboard** | Training visualization | Mouse service — loss/metric plots |

### Keystroke-Specific
| Package | Purpose | Where Used |
|---------|---------|------------|
| **openpyxl / xlrd** | Excel file reading for DSL dataset | Keystroke service — training data |
| **pandas** | Data manipulation for training pipeline | Keystroke service — feature engineering |

---

## External Tools

| Tool | Purpose | Where Used |
|------|---------|------------|
| **ffmpeg** | Audio format conversion (any format → WAV 16kHz mono) | Backend `utils/audioConverter.js` — voice enrollment |
| **MongoDB Atlas** | Cloud-hosted MongoDB database | Backend data storage (all models, sessions, etc.) |
| **Gmail SMTP** | Email delivery service with app password | Backend `utils/emailService.js` — all email notifications |
| **Cloudinary** | Cloud image hosting/CDN | Backend `controllers/doctorController.js` — profile images |
| **Google STUN Server** | WebRTC NAT traversal for peer-to-peer connections | Frontend `Meeting.jsx` — `stun:stun.l.google.com:19302` |

---

## Port Allocation

| Service | Port | Technology | Purpose |
|---------|------|------------|---------|
| Frontend | 5173 | Vite (React) | User interface |
| Backend | 5000 | Express.js (Node.js) | REST API + Socket.IO |
| Voice API | 8001 | FastAPI (Python) | ECAPA-TDNN voice verification |
| Keystroke API | 8002 | FastAPI (Python) | DNN keystroke verification |
| Mouse API | 8003 | FastAPI (Python) | Siamese mouse verification |
| Face API | 8004 | FastAPI (Python) | ResNet50 face verification |
| MongoDB | 27017 | MongoDB Atlas (Cloud) | Database |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `Backend/.env` | Environment variables (MongoDB URI, JWT secret, email creds, Cloudinary keys) |
| `Client/vite.config.js` | Vite config with proxy to backend (port 5000) |
| `Client/tailwind.config.js` | Tailwind CSS customization (colors, fonts) |
| `Client/postcss.config.js` | PostCSS plugins (Tailwind, Autoprefixer) |
| `face verification/config.yaml` | Face model settings (threshold, port, embedding dim) |
| `Keystroke Dynamics/config.yaml` | Keystroke model settings (threshold, features, training) |
| `Mouse Movement Analysis/config.yaml` | Mouse model settings (anomaly detection, features) |
| `Voiceprint Analysis/config.yaml` | Voice model settings (anti-spoofing, audio processing) |
