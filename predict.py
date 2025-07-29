from ultralytics import YOLO
import os
model = YOLO("C:/Users/Daniyal Rehman/OneDrive/Desktop/Finalfyp/backend/backend/runs/detect/mango_vs_nonmango/weights/best.pt")
test_folder = "backend/newtest"
#Counters
total = 0
mango = 0
non_mango = 0
Mixed =0
unknown_class =0
no_object =0

#Loop through test images
for file in os.listdir(test_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(test_folder, file)
        total += 1

        # 🔍 Predict
        results = model(img_path, conf=0.6, verbose=False)[0]

        # Agar kuch detect hua
        if results.boxes is not None and len(results.boxes.cls) > 0:
            classes = [int(cls) for cls in results.boxes.cls]
            if 0 in classes and 1 not in classes:
                print(f"🟢 {file} → Mango")
                mango += 1
            elif 1 in classes and 0 not in classes:
                print(f"🔴 {file} → Not Mango")
                non_mango += 1
            elif 0 in classes and 1 in classes:
                print(f"🟡 {file} → Mixed: Mango & Non-Mango")
                Mixed +=1
            else:
                print(f"⚠️ {file} → Unknown class detected")
                unknown_class +=1
        else:
            print(f"🚫 {file} → No object detected")
            no_object +=1

# 🧾 Final summary
print("\n📊 Final Report")
print(f"Total Images Checked: {total}")
print(f"✅ Mango: {mango}")
print(f"❌ Not Mango: {non_mango}")
print(f"🟡 Mixed: Mango & Non-Mango {Mixed}")
print(f"⚠️ Unknown class detected {unknown_class}")
print(f"🚫 No object detected {no_object}")