# -------------------------28 July-------------------------------
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
import tempfile

# === App Setup ===
app = FastAPI()

# === Allow frontend to access backend ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Models ===
yolo_model = YOLO("C:/Users/Daniyal Rehman/OneDrive/Desktop/Finalfyp/backend/backend/runs/detect/mango_vs_nonmango/weights/best.pt")
cnn_model = load_model("C:/Users/Daniyal Rehman/OneDrive/Desktop/Finalfyp/backend/best_mango_disease_model.keras")
label_map = joblib.load("C:/Users/Daniyal Rehman/OneDrive/Desktop/Finalfyp/backend/label_encoder.pkl")

IMAGE_SIZE = 320

# === Preprocessing for CNN ===
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))

    norm = gray_resized.astype("float32") / 255.0
    norm = np.expand_dims(norm, axis=-1)  # (H, W, 1)
    norm = np.expand_dims(norm, axis=0)   # (1, H, W, 1)

    return norm, gray_resized

# === Estimate severity if anthracnose ===
def estimate_anthracnose_severity(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)
    _, thresh = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY_INV)

    spot_area = np.sum(thresh == 255)
    total_area = gray_img.shape[0] * gray_img.shape[1]
    severity = (spot_area / total_area) * 100
    return round(severity, 2)

# === Main Prediction Route ===

@app.get("/")
def read_root():
    return {"message": "Mango Disease Detection API is running!"}
# In the new backend file (main.py)
@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        img_path = tmp.name

    results = yolo_model(img_path, conf=0.662, verbose=False)[0]
    status = ""
    prediction_text = ""
    disease = ""  # New field for raw disease label
    confidence = 0

    if results.boxes is not None and len(results.boxes.cls) > 0:
        classes = [int(cls) for cls in results.boxes.cls]
        if 0 in classes and 1 not in classes:
            image_array, gray_img = preprocess_image(img_path)
            if image_array is None:
                os.remove(img_path)
                return {"status": "error", "message": "Image read failed"}

            preds = cnn_model.predict(image_array)[0]
            pred_index = np.argmax(preds)
            pred_class = label_map.inverse_transform([pred_index])[0]
            confidence = float(preds[pred_index]) * 100

            disease = pred_class  # Set raw disease label
            if pred_class == "healthy":
                prediction_text = f"🟢 Healthy mango"
            elif pred_class == "sap-burn":
                prediction_text = f"🟡 Healthy mango (with sap burn)"
            elif pred_class == "anthracnose":
                severity = estimate_anthracnose_severity(gray_img)
                prediction_text = f"🔴 Anthracnose detected – severity: {severity}%"
            else:
                prediction_text = f"❓ Unknown class"

            status = "mango"

        elif 1 in classes and 0 not in classes:
            status = "not_mango"
        elif 0 in classes and 1 in classes:
            status = "mixed"
        else:
            status = "unknown"
    else:
        status = "no_object"

    os.remove(img_path)

    return {
        "status": status,
        "disease": disease,  # Add raw disease label
        "prediction": prediction_text,
        "confidence": f"{confidence:.2f}%" if prediction_text else ""
    }