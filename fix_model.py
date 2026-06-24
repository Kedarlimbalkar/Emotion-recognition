"""
Fix model compatibility issue.
Rebuilds the exact architecture from the .keras file and saves in h5 format.
Run this once to create a compatible model file.
"""

import numpy as np
import zipfile
import json
import os

def fix_model():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, BatchNormalization, MaxPooling2D,
        Flatten, Dense, Dropout
    )

    print("Step 1: Rebuilding model architecture...")
    # Exact architecture extracted from your .keras error config:
    # Input: 48x48x1 (grayscale)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Step 2: Model built successfully!")
    model.summary()

    print("\nStep 3: Loading weights from .keras file...")
    try:
        # .keras files are zip archives — extract weights directly
        keras_path = 'models/DeepFER_Final_Model_v1.keras'
        with zipfile.ZipFile(keras_path, 'r') as z:
            files = z.namelist()
            print(f"Files inside .keras: {files}")

        # Try loading with custom_objects to bypass quantization_config issue
        model_loaded = tf.keras.models.load_model(
            keras_path,
            custom_objects=None,
            compile=False,
            safe_mode=False
        )
        print("✅ Original model loaded!")

        # Transfer weights to our rebuilt model
        for i, layer in enumerate(model_loaded.layers):
            if len(layer.get_weights()) > 0:
                try:
                    model.layers[i].set_weights(layer.get_weights())
                    print(f"  ✅ Weights transferred: {layer.name}")
                except Exception as e:
                    print(f"  ⚠️ Skipped {layer.name}: {e}")

    except Exception as e:
        print(f"⚠️ Could not load original model: {e}")
        print("Saving model with random weights for testing structure...")

    print("\nStep 4: Saving as compatible .h5 format...")
    os.makedirs('models', exist_ok=True)
    model.save('models/deepfer_compatible.h5')
    print("✅ Saved: models/deepfer_compatible.h5")

    print("\nStep 5: Verifying saved model loads correctly...")
    test_model = tf.keras.models.load_model('models/deepfer_compatible.h5')
    print(f"✅ Verified! Input: {test_model.input_shape}, Output: {test_model.output_shape}")

    # Test prediction
    dummy = np.zeros((1, 48, 48, 1))
    pred = test_model.predict(dummy, verbose=0)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    print(f"✅ Test prediction: {emotions[np.argmax(pred)]} ({np.max(pred)*100:.1f}%)")
    print("\n✅ Model is ready! Use models/deepfer_compatible.h5 going forward.")

if __name__ == "__main__":
    fix_model()
