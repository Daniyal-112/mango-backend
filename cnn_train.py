import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving label encoder

# Paths
train_dir = "processed-data/train"
img_size = 320
batch_size = 16
num_classes = 3
epochs = 200

# === Data Generator ===
train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10,
)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True
)

# === Class Weights ===
labels = train_data.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# === Save Label Encoder ===
class_names = list(train_data.class_indices.keys())  # e.g., ['anthracnose', 'healthy', 'sap-burn']
le = LabelEncoder()
le.fit(class_names)
joblib.dump(le, "label_encoder.pkl")
print("✅ label_encoder.pkl saved:", class_names)

# === Model ===
input_layer = Input(shape=(img_size, img_size, 1))
x = tf.keras.layers.Concatenate()([input_layer, input_layer, input_layer])  # Convert to 3 channels

base_model = MobileNetV2(include_top=False, input_tensor=x, weights='imagenet')
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

# === Compile ===
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# === Callbacks ===
callbacks = [
    EarlyStopping(monitor='loss', patience=20, restore_best_weights=True),
    ModelCheckpoint("best_mango_disease_model.keras", save_best_only=True, monitor='loss')
]

# === Train ===
model.fit(
    train_data,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=callbacks
)
