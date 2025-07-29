from ultralytics import YOLO
import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model  

# --- Paths and Config ---
YOLO_MODEL_PATH = "C:/Users/Daniyal Rehman/OneDrive/Desktop/Finalfyp/backend/backend/runs/detect/mango_vs_nonmango/weights/best.pt"
CNN_MODEL_PATH = "C:/Users/Daniyal Rehman/OneDrive/Desktop/Finalfyp/backend/best_mango_disease_model.keras"
ENCODER_PATH = "C:/Users/Daniyal Rehman/OneDrive/Desktop/Finalfyp/backend/label_encoder.pkl"
TEST_DIR = "C:/Users/Daniyal Rehman/OneDrive/Desktop/Finalfyp/backend/newtest"
IMAGE_SIZE = 320

# --- Load YOLO model ---
yolo_model = YOLO(YOLO_MODEL_PATH)

# --- Load CNN model & label encoder ---
cnn_model = load_model(CNN_MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
index_to_label = dict(enumerate(label_encoder.classes_))
label_to_index = {v: k for k, v in index_to_label.items()}
known_classes = set(label_encoder.classes_)

# --- Preprocess for CNN ---
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

# --- Estimate severity if anthracnose ---
def estimate_anthracnose_severity(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)
    _, thresh = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY_INV)

    spot_area = np.sum(thresh == 255)
    total_area = gray_img.shape[0] * gray_img.shape[1]
    severity = (spot_area / total_area) * 100
    return round(severity, 2)

# --- Start Processing ---
total = 0
mango_count = 0
non_mango_count = 0
skipped = 0

print("\n🍃 Combined Mango Detector + Disease Classifier\n")

for file in os.listdir(TEST_DIR):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    filepath = os.path.join(TEST_DIR, file)
    total += 1

    # YOLO Prediction (Mango/Not Mango)
    results = yolo_model(filepath, conf=0.5, verbose=False)[0]

    if results.boxes is not None and len(results.boxes.cls) > 0:
        classes = [int(cls) for cls in results.boxes.cls]

        if 0 in classes and 1 not in classes:
            print(f"\n🟢 {file} → Detected: Mango")
            mango_count += 1

            # ➕ Apply CNN Disease Classifier
            image_array, gray_img = preprocess_image(filepath)
            if image_array is None:
                print(f"⚠️ Skipped (Invalid image for CNN): {file}")
                skipped += 1
                continue

            preds = cnn_model.predict(image_array)[0]
            pred_index = np.argmax(preds)
            pred_label = index_to_label[pred_index]
            confidence = preds[pred_index] * 100

            if pred_label == "healthy":
                print(f"    ✅ Prediction: 🟢 Healthy mango ({confidence:.2f}%)")
            elif pred_label == "sap-burn":
                print(f"    ✅ Prediction: 🟡 Healthy mango (with sap burn) ({confidence:.2f}%)")
            elif pred_label == "anthracnose":
                severity = estimate_anthracnose_severity(gray_img)
                print(f"    ✅ Prediction: 🔴 Anthracnose detected ({confidence:.2f}%)")
                print(f"       🔬 Severity Estimate: {severity}%")
            else:
                print(f"    ❌ Unknown prediction label: {pred_label}")

        elif 1 in classes and 0 not in classes:
            print(f"\n🔴 {file} → Detected: Not Mango")
            non_mango_count += 1

        elif 0 in classes and 1 in classes:
            print(f"\n🟡 {file} → Mixed Content (Mango & Non-Mango)")
            skipped += 1

        else:
            print(f"\n⚠️ {file} → Unknown Class Detected")
            skipped += 1
    else:
        print(f"\n🚫 {file} → No Object Detected")
        skipped += 1

# --- Final Summary ---
print("\n📊 Final Summary")
print(f"🔍 Total Images Processed: {total}")
print(f"✅ Mango Only: {mango_count}")
print(f"❌ Not Mango: {non_mango_count}")
print(f"⏭️ Skipped (Invalid/Mixed/No Object): {skipped}")
