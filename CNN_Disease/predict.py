import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import joblib

# === Load trained model and label encoder ===
model = load_model("best_mango_disease_model.keras")
label_encoder = joblib.load("label_encoder.pkl")

# === Paths ===
test_folder = "test-images"
IMG_SIZE = 320

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Normalize and expand dims
    gray_norm = gray_resized.astype("float32") / 255.0
    gray_norm = np.expand_dims(gray_norm, axis=-1)  # (H, W, 1)
    gray_input = np.expand_dims(gray_norm, axis=0)  # (1, H, W, 1)

    # For severity estimation
    return gray_input, gray_resized

def estimate_anthracnose_severity(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)
    _, thresh = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY_INV)

    spot_area = np.sum(thresh == 255)
    total_area = gray_img.shape[0] * gray_img.shape[1]
    severity = (spot_area / total_area) * 100
    return round(severity, 2)

def predict_class(img_path):
    image_array, gray_img = preprocess_image(img_path)
    if image_array is None:
        return f"❌ Could not read image: {img_path}"

    # Predict
    pred = model.predict(image_array, verbose=0)[0]
    pred_class = label_encoder.inverse_transform([np.argmax(pred)])[0]

    if pred_class == "healthy":
        return f"🟢 {os.path.basename(img_path)} → Healthy mango"

    elif pred_class == "sap-burn":
        return f"🟡 {os.path.basename(img_path)} → Healthy mango (with sap burn)"

    elif pred_class == "anthracnose":
        severity = estimate_anthracnose_severity(gray_img)
        return f"🔴 {os.path.basename(img_path)} → Anthracnose detected – severity: {severity}%"

    else:
        return f"❓ Unknown prediction for {img_path}"

# === Predict all ===
print("=== Predictions ===")
for filename in os.listdir(test_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(test_folder, filename)
        print(predict_class(path))
