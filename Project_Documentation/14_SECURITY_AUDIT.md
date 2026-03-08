# Security Audit & Findings

> Analysis of security issues, hardcoded credentials, and recommended fixes.

---

## Critical Issues

### 1. Hardcoded Cloudinary Credentials

**File:** `Backend/config/cloudinary.js`

**Problem:** All three Cloudinary credentials (cloud name, API key, API secret) are hardcoded directly in the source code, not pulled from environment variables.

```javascript
// CURRENT (insecure)
cloudinary.config({
  cloud_name: 'dr94xryhl',
  api_key: '364313557416346',
  api_secret: 'HEY0XDnH46SvB1EmhBDJLspwMrA'
});
```

**Fix:** Use environment variables:

```javascript
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET
});
```

**Risk:** Anyone with repository access can use the Cloudinary account, upload/delete images, or exhaust storage quotas.

---

### 2. Hardcoded Gmail App Password

**File:** `Backend/utils/emailService.js`

**Problem:** Gmail address and App Password are hardcoded as fallback defaults, repeated 7 times throughout the file:

```javascript
user: process.env.EMAIL_USER || 'cn3581743@gmail.com',
pass: process.env.EMAIL_PASS || 'jyag rnwm ezvo yqlb'
```

**Fix:** Remove hardcoded fallbacks — require env vars:

```javascript
user: process.env.EMAIL_USER,
pass: process.env.EMAIL_PASS
```

**Risk:** Gmail App Passwords grant full SMTP access. An attacker could send emails from this account, potentially for phishing or spam, leading to account suspension.

---

### 3. Hardcoded Admin Credentials (Plaintext)

**File:** `Backend/controllers/authController.js`

**Problem:** Admin login uses hardcoded plaintext credentials with no hashing:

```javascript
if (email === 'admin@gmail.com' && password === 'admin') {
  // Grant admin access
}
```

**Fix:** Store admin user in MongoDB with bcrypt-hashed password like regular users, or use env vars:

```javascript
if (email === process.env.ADMIN_EMAIL && password === process.env.ADMIN_PASSWORD) { ... }
```

Better: Create an Admin model or add a role field to Doctor/Patient models and authenticate the same way.

**Risk:** Anyone reading the source code knows the admin credentials. Admin may have elevated privileges.

---

### 4. MongoDB Credentials in `.env` (Committed to Git)

**File:** `Backend/.env`

**Problem:** The `.env` file appears to be committed to the repository and contains the MongoDB Atlas connection string with credentials:

```
MONGODB_URI=mongodb+srv://cyberresearch046:Cyber123@cluster0.pbp0on3.mongodb.net/
```

**Fix:**
1. Add `.env` to `.gitignore`
2. Rotate the MongoDB password immediately
3. Create a `.env.example` with placeholder values

**Risk:** Full database access (read, write, delete) to anyone who sees the repository.

---

### 5. Weak JWT Secret

**File:** `Backend/.env`

**Problem:** JWT secret is a descriptive placeholder, not a cryptographically random value:

```
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
```

**Fix:** Generate a strong random secret:

```bash
node -e "console.log(require('crypto').randomBytes(64).toString('hex'))"
```

**Risk:** If the secret is guessable, attackers can forge JWT tokens and impersonate any user (doctor, patient, admin).

---

## High-Severity Issues

### 6. Missing EMBEDDING_ENCRYPTION_KEY

**Files:** `face verification/config.yaml`, `Voiceprint Analysis/config.yaml`

**Problem:** Both configs reference `EMBEDDING_ENCRYPTION_KEY` environment variable, but it is never defined anywhere. The Voiceprint service claims to encrypt embeddings with AES-256, but without this key, it either uses a default or fails silently.

**Fix:** Generate and set the key:

```bash
# Generate a 32-byte (256-bit) key
python -c "import os; print(os.urandom(32).hex())"
```

Set as environment variable before starting ML services.

---

### 7. CORS Mismatch in ML Services

**Files:** `Voiceprint Analysis/config.yaml`, `Keystroke Dynamics/config.yaml`, `Mouse Movement Analysis/config.yaml`

**Problem:** Three ML services list allowed CORS origins as `localhost:3000, localhost:5000` but the Vite frontend runs on port `5173`. Only the Face Verification service includes `:5173`.

**Impact:** Direct browser requests to these ML APIs would be blocked by CORS. This doesn't currently cause issues because all requests go through the Express backend proxy — but could break if the frontend ever calls ML services directly.

**Fix:** Add `http://localhost:5173` to all ML service CORS configs.

---

## Medium-Severity Issues

### 8. Hardcoded Socket.IO URL in Frontend

**File:** `Client/src/pages/Meeting.jsx`

```javascript
const socket = io('http://localhost:5000', { ... });
```

**Impact:** Won't work in production or on any host other than localhost.

**Fix:** Use `import.meta.env.VITE_SOCKET_URL` or derive from `window.location`.

---

### 9. No Rate Limiting on Auth Endpoints

**Files:** `Backend/routes/authRoutes.js`, `Backend/controllers/authController.js`

**Problem:** Login, register, and OTP endpoints have no rate limiting. An attacker can brute-force:
- Doctor/patient login passwords
- OTP codes (used for lockout re-authentication)
- Admin credentials (especially easy since password is `admin`)

**Fix:** Add `express-rate-limit`:

```javascript
import rateLimit from 'express-rate-limit';
const authLimiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 10 });
app.use('/api/auth', authLimiter);
```

---

### 10. No HTTPS Enforcement

**Problem:** All services run on HTTP. WebRTC requires HTTPS in production for camera/microphone access (browsers block `getUserMedia()` on non-HTTPS origins except localhost).

**Impact:** The entire biometric verification system will not work in production without HTTPS.

**Fix:** Use a reverse proxy (nginx) with SSL certificates or deploy behind a service that provides HTTPS (e.g., Heroku, AWS ALB).

---

### 11. File Upload Without Content Validation

**File:** `Backend/middleware/upload.js`

**Problem:** Multer file filter checks MIME type via `file.mimetype`, which is client-provided and easily spoofed. An attacker could upload a malicious file with a fake `image/jpeg` MIME type.

**Fix:** Validate file content (magic bytes) server-side after upload, or use a library like `file-type`.

---

## Low-Severity / Informational

### 12. Admin Email Mismatch

- `.env` / `authController.js` uses `admin@gmail.com`
- Older docs reference `admin@cybersecurity.com`
- These are different — the actual working one is `admin@gmail.com`

### 13. Unused `.env` Variable

- `UPLOAD_PATH=./uploads` is defined in `.env` but never read by any code
- The upload directory is hardcoded in `middleware/upload.js` as `'uploads/'`

### 14. No Input Sanitization on Chat Messages

**File:** Socket.IO handler in `server.js`

- Chat messages are broadcast via Socket.IO without sanitization
- Could allow XSS if messages are rendered with `dangerouslySetInnerHTML` on the frontend
- React's default JSX rendering escapes HTML, so this is low risk unless raw HTML rendering is added

### 15. Monitoring Emails Are Placeholders

**File:** `Voiceprint Analysis/config.yaml`

```yaml
alert_recipients:
  - admin@telehealth.com
  - security@telehealth.com
```

These are placeholder addresses that don't exist, so security alerts from the voiceprint service would silently fail.

---

## Summary by Severity

| Severity | Count | Issues |
|----------|-------|--------|
| **CRITICAL** | 5 | Cloudinary creds, Gmail pwd, admin creds, MongoDB creds, weak JWT |
| **HIGH** | 2 | Missing encryption key, CORS mismatch |
| **MEDIUM** | 4 | Socket.IO URL, no rate limiting, no HTTPS, upload validation |
| **LOW** | 4 | Email mismatch, unused env var, chat XSS risk, placeholder emails |

---

## Recommended Priority Actions

1. **Immediately** — Rotate all exposed credentials (MongoDB password, Gmail App Password, Cloudinary API secret)
2. **Before deployment** — Move all hardcoded secrets to env vars, add `.env` to `.gitignore`
3. **Before deployment** — Generate a strong JWT secret (64+ random bytes)
4. **Before deployment** — Set up HTTPS (required for WebRTC)
5. **Before deployment** — Add rate limiting to auth endpoints
6. **Before deployment** — Replace hardcoded admin with proper DB-backed admin auth
7. **Nice to have** — Define `EMBEDDING_ENCRYPTION_KEY`, fix CORS origins, validate file uploads
