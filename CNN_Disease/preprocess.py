import os
import cv2
from tqdm import tqdm

# Input and output folder paths
INPUT_DIR = "disease-data/train"
OUTPUT_DIR = "processed-data/train"
TARGET_SIZE = (320, 320)

# CLAHE setup
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Create output structure
classes = ["anthracnose", "healthy", "sap-burn"]
for cls in classes:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# Process images
for cls in classes:
    input_folder = os.path.join(INPUT_DIR, cls)
    output_folder = os.path.join(OUTPUT_DIR, cls)

    for img_name in tqdm(os.listdir(input_folder), desc=f"Processing {cls}"):
        try:
            img_path = os.path.join(input_folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE
            enhanced = clahe.apply(gray)

            # Resize
            resized = cv2.resize(enhanced, TARGET_SIZE)

            # Save processed image
            save_path = os.path.join(output_folder, img_name)
            cv2.imwrite(save_path, resized)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
