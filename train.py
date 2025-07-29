from ultralytics import YOLO

# 🔰 Load a base YOLOv8 model — small & CPU friendly
model = YOLO("backend/yolov8s.pt")  

# 🧠 Train the model with detailed augmentations and settings
model.train(
    data="C:/Users/Daniyal Rehman/OneDrive/Desktop/Finalfyp/backend/data.yaml",         # Path to your dataset YAML
    epochs=150,               # More epochs for better learning
    imgsz=640,                # Image size
    batch=8,                  # Suitable for low-RAM systems
    device="cpu",             # Use 'cuda' if GPU available
    patience=100,              # Early stopping after 20 epochs of no improvement
    optimizer="Adam",         # Better for non-GPU setups
    lr0=0.001,                # Initial learning rate
    lrf=0.01,                 # Final learning rate (lr decay)
    warmup_epochs=3,          # Warmup phase
    weight_decay=0.001,       # Regularization

    # 📸 Data Augmentations
    augment=True,
    mosaic=1.0,
    mixup=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    degrees=10.0,
    scale=0.5,
    shear=2.0,

    # 📁 Logging & Saving
    project="backend/runs/detect",
    name="mango_vs_nonmango",
    exist_ok=True,
    verbose=True
)
