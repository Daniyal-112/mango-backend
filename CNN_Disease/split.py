import os
import shutil
import random

# 📂 Paths
source_folder = "C:/Users/Daniyal Rehman/OneDrive/Desktop/classification/disease-data/train/anthracnose"
valid_folder = "C:/Users/Daniyal Rehman/OneDrive/Desktop/classification/disease-data/valid/anthracnose"

# 🧪 Train/Val Split Ratio
valid_ratio = 0.2  # 20% for validation

# ✅ Make sure validation folder exists
os.makedirs(valid_folder, exist_ok=True)

# 📸 Get all image files from test/healthy
images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(images)

# 🔁 Split
valid_count = int(len(images) * valid_ratio)
valid_images = images[:valid_count]

# 📥 Move selected images to valid/healthy
for img in valid_images:
    shutil.move(os.path.join(source_folder, img), os.path.join(valid_folder, img))

print(f"✅ Split complete: {len(images) - valid_count} remaining in test (train), {valid_count} moved to valid.")
