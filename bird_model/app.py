import os
import librosa
import numpy as np
import joblib
import shutil
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Setup templates
templates = Jinja2Templates(directory="templates")

# Try to load model and encoder
MODEL_PATH = "model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

model = None
le = None

if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
else:
    print("Warning: Model or Label Encoder not found. Please run train.py first.")

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}. Details: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if not model or not le:
        return {"error": "Model not trained yet."}
    
    # Save uploaded file temporarily
    temp_file = f"temp_{audio.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
        
    try:
        # Extract features
        features = extract_features(temp_file)
        if features is None:
            return {"error": "Failed to process audio file."}
            
        features = features.reshape(1, -1)
        
        # Predict
        prediction_encoded = model.predict(features)
        prediction_label = le.inverse_transform(prediction_encoded)[0]
        
        # Probabilities
        probabilities = model.predict_proba(features)[0]
        confidence = np.max(probabilities) * 100
        
        return {
            "prediction": prediction_label,
            "confidence": f"{confidence:.2f}%"
        }
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
