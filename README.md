# AI Radiology Triage Backend API

Automated AI-powered medical scan triage API built with **FastAPI** and **TensorFlow / Keras**. Designed to triage incoming chest CT / X-ray scans by prioritizing malignant/critical findings for immediate radiologist review while routing normal scans to standard queues.

## 🚀 Features
- **Triage AI Model**: Deep Convolutional Neural Network (CNN) trained to distinguish between Normal and Malignant scans (Adenocarcinoma, Large Cell, Squamous Cell carcinoma).
- **FastAPI Endpoint (`/analyze-scan`)**: Accepts incoming scan uploads (PNG, JPG, DICOM), preprocesses images to 224x224 grayscale, and returns JSON triage predictions.
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

## 📡 API Endpoint

### `POST /analyze-scan`
- **Request**: `multipart/form-data` with `file: UploadFile`
- **Response**:
```json
{
  "filename": "scan.png",
  "triage_status": "CRITICAL",
  "confidence_score": 0.937,
  "action": "Flagged for immediate Radiologist review!"
}
```
