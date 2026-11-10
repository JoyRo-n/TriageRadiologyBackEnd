import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np

app = FastAPI(title="AI Radiology Triage API")

# Load model (checks both radiology_triage_model.keras and triage_model.keras)
MODEL_PATH = "radiology_triage_model.keras" if os.path.exists("radiology_triage_model.keras") else "triage_model.keras"
print(f"Loading AI Triage Model from: {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "AI Radiology Triage API",
        "model_file": MODEL_PATH
    }

@app.post("/analyze-scan")
async def analyze_scan(file: UploadFile = File(...)):
    try:
        # 1. Read incoming image bytes (from Android or web client)
        contents = await file.read()
        
        # 2. Decode and preprocess the image (grayscale, 224x224, normalized 0-1)
        raw_img = tf.io.decode_image(contents, channels=1, expand_animations=False)
        resized_img = tf.image.resize(raw_img, (224, 224))
        normalized_img = tf.cast(resized_img, tf.float32) / 255.0
        input_tensor = tf.expand_dims(normalized_img, axis=0)
        
        # 3. Run inference
        prediction = model.predict(input_tensor, verbose=0)
        score = float(prediction[0][0])
        
        # 4. Business triage decision logic
        if score > 0.5:
            triage_status = "CRITICAL"
            action = "Flagged for immediate Radiologist review!"
        else:
            triage_status = "NORMAL"
            action = "Added to standard worklist."
            
        return {
            "filename": file.filename,
            "triage_status": triage_status,
            "confidence_score": round(score, 4),
            "action": action
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process scan: {str(e)}")
