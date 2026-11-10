import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import pydicom

# Set UTF-8 output encoding for Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# 1. Dataset Preprocessing & Loading
# ---------------------------------------------------------------------------
def load_triage_dataset(split, base_dir="archive (1)/Data", img_size=(224, 224), batch_size=32, shuffle=True):
    """
    Loads images from archive (1)/Data for a given split ('train', 'valid', or 'test').
    Maps classes to binary triage labels:
      - 0.0: Normal (Standard queue)
      - 1.0: Critical / Cancer (Adenocarcinoma, Large Cell, Squamous Cell) -> Immediate Review
    """
    split_path = os.path.join(base_dir, split)
    filepaths = []
    labels = []

    for root, _, files in os.walk(split_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, f)
                filepaths.append(full_path)
                folder_name = os.path.basename(root).lower()
                # If folder contains 'normal', label is 0; otherwise 1 (Critical)
                labels.append(0.0 if "normal" in folder_name else 1.0)

    print(f"[{split.upper()}] Found {len(filepaths)} images ({labels.count(0.0)} Normal, {labels.count(1.0)} Critical)")

    def parse_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=1, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0  # Normalize pixels to [0, 1]
        img.set_shape([img_size[0], img_size[1], 1])
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=42)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds, filepaths, labels


# ---------------------------------------------------------------------------
# 2. DICOM Processing Utility (Kept for external DICOM inference)
# ---------------------------------------------------------------------------
def process_dicom(file_path):
    # 1. Read the DICOM file
    dicom_data = pydicom.dcmread(file_path)
    
    # 2. Extract the raw pixel arrays
    image = dicom_data.pixel_array
    
    # 3. Resize to 224x224 (standard size for AI models)
    image_resized = cv2.resize(image, (224, 224))
    
    # 4. Normalize pixel values to be between 0 and 1
    image_normalized = image_resized / np.max(image_resized)
    
    # 5. Reshape for TensorFlow (Batch size, Height, Width, Channels)
    image_tensor = np.reshape(image_normalized, (1, 224, 224, 1))
    
    return image_tensor, image_normalized


# ---------------------------------------------------------------------------
# 3. Build Triage CNN Model Architecture
# ---------------------------------------------------------------------------
def build_triage_model():
    model = models.Sequential([
        # Input layer for our 224x224 grayscale image
        layers.Input(shape=(224, 224, 1)),
        
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 3
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and regularization
        layers.Flatten(),
        layers.Dropout(0.3),  # Prevents overfitting on smaller medical datasets
        
        # Decision layers
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 1 = Critical, 0 = Normal
    ])
    
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model


# ---------------------------------------------------------------------------
# 4. Main Training & Evaluation Pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Loading Dataset from archive (1)/Data...")
    print("=" * 60)
    
    train_ds, train_paths, _ = load_triage_dataset("train", shuffle=True)
    valid_ds, _, _ = load_triage_dataset("valid", shuffle=False)
    test_ds, test_paths, test_labels = load_triage_dataset("test", shuffle=False)
    
    print("\nBuilding Triage Model Architecture...")
    ai_model = build_triage_model()
    ai_model.summary()
    
    # Train the Model
    epochs = 5
    print(f"\nStarting training for {epochs} epochs...")
    history = ai_model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs
    )
    
    # Evaluate on Unseen Test Scans
    print("\n" + "=" * 60)
    print("Evaluating Model on Unseen Test Dataset...")
    print("=" * 60)
    test_loss, test_accuracy = ai_model.evaluate(test_ds)
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Save the Trained Model
    model_save_path = "triage_model.keras"
    ai_model.save(model_save_path)
    print(f"\nModel successfully saved to '{model_save_path}'!")
    
    # Plot and save training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Training curves saved to 'training_history.png'")
    
    # ---------------------------------------------------------------------------
    # 5. Demonstration: Triage Testing with Real Patient Scans
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("--- Testing Triage Business Logic with Real Scans ---")
    print("=" * 60)
    
    # Sample a normal test image and a critical test image
    sample_normal = next(p for p, l in zip(test_paths, test_labels) if l == 0.0)
    sample_critical = next(p for p, l in zip(test_paths, test_labels) if l == 1.0)
    
    def run_triage_inference(image_path, ground_truth):
        raw = tf.io.read_file(image_path)
        img = tf.io.decode_image(raw, channels=1, expand_animations=False)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        img_tensor = tf.expand_dims(img, axis=0)
        
        pred = ai_model.predict(img_tensor, verbose=0)
        score = float(pred[0][0])
        
        print(f"\nFile: {os.path.basename(image_path)}")
        print(f"Ground Truth Label : {'Critical / Malignant' if ground_truth == 1.0 else 'Normal'}")
        print(f"Triage Priority Score (0 to 1): {score:.4f}")
        
        # Apply business logic
        if score > 0.5:
            print("Status: [CRITICAL] Flagged for immediate Radiologist review!")
        else:
            print("Status: [NORMAL] Added to standard worklist.")
            
    run_triage_inference(sample_normal, ground_truth=0.0)
    run_triage_inference(sample_critical, ground_truth=1.0)



ai_model.save('radiology_triage_model.keras')
print("Model successfully saved to disk.")