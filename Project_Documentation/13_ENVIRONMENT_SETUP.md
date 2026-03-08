# Environment Variables & Configuration Guide

## Backend (.env)

Create a `.env` file in the `Backend/` folder with these variables:

```env
# ── Server ───────────────────────────────────────────
PORT=5000
NODE_ENV=development          # 'development' or 'production'

# ── Database ─────────────────────────────────────────
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<dbname>

# ── Authentication ───────────────────────────────────
JWT_SECRET=<generate-a-strong-random-string>
JWT_EXPIRE=7d

# ── ML Service URLs ─────────────────────────────────
VOICE_API_URL=http://localhost:8001
KEYSTROKE_API_URL=http://localhost:8002
MOUSE_API_URL=http://localhost:8003
FACE_API_URL=http://localhost:8004

# ── Cloudinary (Image Uploads) ──────────────────────
CLOUDINARY_CLOUD_NAME=<your-cloud-name>
CLOUDINARY_API_KEY=<your-api-key>
CLOUDINARY_API_SECRET=<your-api-secret>

# ── Email (Gmail SMTP) ──────────────────────────────
EMAIL_USER=<your-gmail-address>
EMAIL_PASS=<your-gmail-app-password>

# ── Frontend URL (CORS) ─────────────────────────────
CLIENT_URL=http://localhost:5173

# ── File Upload ──────────────────────────────────────
MAX_FILE_SIZE=10485760        # 10MB in bytes
```

### Required vs Optional

| Variable | Required | Default if Missing |
|----------|----------|--------------------|
| `PORT` | No | `5000` |
| `NODE_ENV` | No | `'development'` |
| `MONGODB_URI` | **Yes** | None — server won't start |
| `JWT_SECRET` | **Yes** | None — auth will break |
| `JWT_EXPIRE` | **Yes** | None — token creation fails |
| `VOICE_API_URL` | No | `http://localhost:8001` |
| `KEYSTROKE_API_URL` | No | `http://localhost:8002` |
| `MOUSE_API_URL` | No | `http://localhost:8003` |
| `FACE_API_URL` | No | `http://localhost:8004` |
| `CLOUDINARY_CLOUD_NAME` | **Yes** | Hardcoded fallback (see security audit) |
| `CLOUDINARY_API_KEY` | **Yes** | Hardcoded fallback |
| `CLOUDINARY_API_SECRET` | **Yes** | Hardcoded fallback |
| `EMAIL_USER` | **Yes** | Hardcoded fallback |
| `EMAIL_PASS` | **Yes** | Hardcoded fallback |
| `CLIENT_URL` | No | `http://localhost:5173` |
| `MAX_FILE_SIZE` | No | `10485760` (10MB) |

### Where Each Variable Is Used

| Variable | File(s) | Purpose |
|----------|---------|---------|
| `PORT` | `server.js` | Express HTTP server port |
| `NODE_ENV` | `server.js` | Controls error detail in responses, logging level |
| `MONGODB_URI` | `config/db.js` | MongoDB Atlas connection string |
| `JWT_SECRET` | `middleware/auth.js` | Signs & verifies JWT tokens |
| `JWT_EXPIRE` | `middleware/auth.js` | Token expiration (e.g., `7d`) |
| `VOICE_API_URL` | `services/mlService.js` | Voiceprint microservice URL |
| `KEYSTROKE_API_URL` | `services/mlService.js` | Keystroke microservice URL |
| `MOUSE_API_URL` | `services/mlService.js` | Mouse analysis microservice URL |
| `FACE_API_URL` | `services/mlService.js` | Face verification microservice URL |
| `CLOUDINARY_*` | `config/cloudinary.js` | Cloud image storage (profile pics) |
| `EMAIL_USER` | `utils/emailService.js` | Gmail sender address (7 email functions) |
| `EMAIL_PASS` | `utils/emailService.js` | Gmail App Password for SMTP |
| `CLIENT_URL` | `server.js` | CORS allowed origin, Socket.IO CORS |
| `MAX_FILE_SIZE` | `middleware/upload.js` | Multer file size limit |

---

## Client (Vite/React)

No `.env` file is currently used. All config is hardcoded:

| Setting | File | Current Value |
|---------|------|---------------|
| Dev server port | `vite.config.js` | `5173` |
| API proxy target | `vite.config.js` | `http://localhost:5000` |
| Socket.IO URL | `src/pages/Meeting.jsx` | `http://localhost:5000` (hardcoded) |

### Recommended: Add `Client/.env`

```env
VITE_API_URL=http://localhost:5000
VITE_SOCKET_URL=http://localhost:5000
```

Then use `import.meta.env.VITE_SOCKET_URL` in `Meeting.jsx` instead of the hardcoded URL.

---

## ML Service Configurations (config.yaml)

Each ML service uses a `config.yaml` file (not environment variables).

### Face Verification (`face verification/config.yaml`)

| Key | Value | Description |
|-----|-------|-------------|
| `api.host` | `0.0.0.0` | Bind address |
| `api.port` | `8004` | Service port |
| `api.cors_origins` | `localhost:3000, :5000, :5173` | Allowed origins |
| `model.architecture` | `resnet50` | Neural network |
| `model.embedding_dim` | `128` | Embedding dimensions |
| `verification.threshold` | `0.8096` | Similarity threshold |
| `database.host` | `localhost` | MongoDB host |
| `database.port` | `27017` | MongoDB port |
| `database.name` | `face_verification_db` | Database name |
| `security.embedding_encryption_key_env` | `EMBEDDING_ENCRYPTION_KEY` | Env var for encryption key (not defined) |
| `security.jwt_secret_env` | `JWT_SECRET` | Env var for JWT |

### Voiceprint Analysis (`Voiceprint Analysis/config.yaml`)

| Key | Value | Description |
|-----|-------|-------------|
| `api.host` | `0.0.0.0` | Bind address |
| `api.port` | `8001` | Service port |
| `api.cors_origins` | `localhost:3000, :5000` | **Missing `:5173`** |
| `model.name` | `ecapa_tdnn` | Neural network |
| `model.embedding_dim` | `192` | Embedding dimensions |
| `verification.threshold` | `0.65` | Similarity threshold |
| `security.embedding_encryption_key_env` | `EMBEDDING_ENCRYPTION_KEY` | Env var for encryption key (not defined) |
| `monitoring.alert_recipients` | `admin@telehealth.com` | Placeholder email |

### Keystroke Dynamics (`Keystroke Dynamics/config.yaml`)

| Key | Value | Description |
|-----|-------|-------------|
| `api.host` | `0.0.0.0` | Bind address |
| `api.port` | `8002` | Service port |
| `api.allowed_origins` | `localhost:3000, :5000` | **Missing `:5173`** |
| `model.input_dim` | `38` | Feature dimensions |
| `model.embedding_dim` | `128` | Embedding dimensions |
| `verification.threshold` | `0.85` | Similarity threshold |

### Mouse Movement Analysis (`Mouse Movement Analysis/config.yaml`)

| Key | Value | Description |
|-----|-------|-------------|
| `api.host` | `0.0.0.0` | Bind address |
| `api.port` | `8003` | Service port |
| `api.allowed_origins` | `localhost:3000, :5000` | **Missing `:5173`** |
| `model.embedding_dim` | `128` | Embedding dimensions |
| `verification.threshold` | `0.85` | Similarity threshold |

---

## Port Allocation Map

```
┌──────────────────────┬──────┬─────────────────────┐
│ Service              │ Port │ Protocol            │
├──────────────────────┼──────┼─────────────────────┤
│ Backend API          │ 5000 │ HTTP + WebSocket    │
│ Frontend (Vite dev)  │ 5173 │ HTTP                │
│ Voiceprint API       │ 8001 │ HTTP (FastAPI)      │
│ Keystroke API        │ 8002 │ HTTP (FastAPI)      │
│ Mouse Movement API   │ 8003 │ HTTP (FastAPI)      │
│ Face Verification API│ 8004 │ HTTP (FastAPI)      │
│ MongoDB              │27017 │ MongoDB wire proto  │
└──────────────────────┴──────┴─────────────────────┘
```

---

## External Service Dependencies

| Service | Purpose | Credentials Needed |
|---------|---------|-------------------|
| **MongoDB Atlas** | Database | Connection string with username/password |
| **Cloudinary** | Image upload/storage | Cloud name, API key, API secret |
| **Gmail SMTP** | Sending emails | Gmail address + App Password (2FA required) |
| **Google STUN servers** | WebRTC NAT traversal | None (free, public) |

### How to Get Gmail App Password

1. Enable 2-Factor Authentication on your Google account
2. Go to <https://myaccount.google.com/apppasswords>
3. Generate a new App Password for "Mail"
4. Use the 16-character password as `EMAIL_PASS`

### How to Get Cloudinary Credentials

1. Sign up at <https://cloudinary.com/>
2. Go to Dashboard → Account Details
3. Copy Cloud Name, API Key, API Secret
