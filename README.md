# AI Radiology Triage Backend API

Automated AI-powered medical scan triage and Doctor Portal Authentication API built with **FastAPI**, **SQLite**, and **TensorFlow / Keras**. Designed to power the **AI Triage Radiology** mobile/web frontend by authenticating clinicians and triaging incoming chest CT / X-ray scans.

## 🚀 Features
- **Medical Portal Authentication (`/register`, `/login`, `/reset-password`, `/me`, `/logout`)**:
  - **Doctor Registration**: Register with `doctor_email`, `new_password`, `password_confirmation`, `full_name`, and medical `role` (`Senior Radiologist`, `Thoracic Oncologist`, `Pulmonologist`, `Resident Radiologist`, `Emergency Physician`, `Medical Officer`, `Radiologic Technologist`).
  - **Doctor Login**: Authenticate with `doctor_email` (or `email`) and `password`, returning doctor profile details and a session token.
  - **Secure Password Storage**: Uses salted `PBKDF2-HMAC-SHA256` password hashing stored in SQLite (`triage_users.db`).
  - **Pre-seeded Demo Account**: Includes the frontend's default account (`doctor@hospital.com` / `password123` — *Dr. Alexander Wright, Senior Radiologist*).
- **Triage AI Model**: Deep Convolutional Neural Network (CNN) trained to distinguish between Normal and Malignant scans (Adenocarcinoma, Large Cell, Squamous Cell carcinoma).
- **FastAPI Endpoints (`/analyze-scan`, `/api/triage`)**: Accepts incoming scan uploads (PNG, JPG, DICOM) or patient metadata, preprocesses images to 224x224 grayscale, and returns JSON triage predictions compatible with the Android frontend.
- **TFLite Support**: Includes `triage_model.tflite` for offline, on-device mobile inference (Android/iOS).

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JoyRo-n/TriageRadiologyBackEnd.git
   cd TriageRadiologyBackEnd
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the API Server:**
   ```bash
   uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access Interactive Docs:**
   - Swagger UI: `http://127.0.0.1:8000/docs`
   - Redoc: `http://127.0.0.1:8000/redoc`

---

## 🔐 Authentication Endpoints

### 1. `POST /register` (or `POST /api/auth/register`)
Registers a new doctor/clinician account matching the frontend registration screen.

- **Request Body (`application/json`)**:
```json
{
  "doctor_email": "dr.sarah@hospital.com",
  "new_password": "password123",
  "password_confirmation": "password123",
  "full_name": "Dr. Sarah Connor",
  "role": "Senior Radiologist"
}
```
*(Also accepts `"email"`, `"password"`, and `"confirm_password"` keys for direct Android `User` model compatibility).*

- **Response (`201 Created`)**:
```json
{
  "status": "success",
  "success": true,
  "message": "Account created successfully!",
  "token": "session_token_string...",
  "user": {
    "id": 2,
    "email": "dr.sarah@hospital.com",
    "full_name": "Dr. Sarah Connor",
    "role": "Senior Radiologist",
    "created_at": "2026-11-11 15:20:00"
  }
}
```

### 2. `POST /login` (or `POST /api/auth/login`)
Signs in a doctor with their email and password.

- **Request Body (`application/json`)**:
```json
{
  "doctor_email": "doctor@hospital.com",
  "password": "password123"
}
```
*(Also accepts `"email"` or `"login"` keys).*

- **Response (`200 OK`)**:
```json
{
  "status": "success",
  "success": true,
  "message": "Welcome back, Dr. Alexander Wright!",
  "token": "session_token_string...",
  "user": {
    "id": 1,
    "email": "doctor@hospital.com",
    "full_name": "Dr. Alexander Wright",
    "role": "Senior Radiologist",
    "created_at": "2026-11-11 15:19:29"
  }
}
```

### 3. `POST /reset-password` (or `POST /api/auth/reset-password`)
Updates a doctor's password using email, new password, and password confirmation.

- **Request Body (`application/json`)**:
```json
{
  "doctor_email": "doctor@hospital.com",
  "new_password": "newpassword123",
  "password_confirmation": "newpassword123"
}
```

### 4. `GET /me` & `POST /logout`
- `GET /me`: Retrieves the active doctor's profile via `Authorization: Bearer <token>` header or `?email=doctor@hospital.com`.
- `POST /logout`: Invalidates the current session token.

---

## 📡 AI Triage Endpoints

### `POST /analyze-scan`
- **Request**: `multipart/form-data` with `file: UploadFile`
- **Response**:
```json
{
  "status": "success",
  "filename": "scan.png",
  "triage_status": "CRITICAL",
  "triage_category": "CRITICAL",
  "confidence_score": 0.937,
  "confidence": 0.937,
  "risk_score": 9.7,
  "primary_finding": "Malignant Pulmonary Lesion / Critical Thoracic Finding",
  "action": "Flagged for immediate Radiologist review!",
  "recommendation": "Flagged for immediate Radiologist review!",
  "processed_at": "2026-11-11 15:20:00"
}
```
