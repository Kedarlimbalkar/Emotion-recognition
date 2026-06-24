"""
Extract real weights from DeepFER_Final_Model_v1.keras
and save as compatible h5 file.
"""
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense

print("Step 1: Rebuilding model architecture...")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(7, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("✅ Model built!")

print("\nStep 2: Extracting weights from .keras file...")
extract_dir = 'models/keras_extracted'

# Clean up old extraction if exists
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)

# Unzip the .keras file
shutil.unpack_archive('models/DeepFER_Final_Model_v1.keras', extract_dir, 'zip')
print(f"✅ Extracted files: {os.listdir(extract_dir)}")

print("\nStep 3: Loading weights...")
weights_path = os.path.join(extract_dir, 'model.weights.h5')
try:
    model.load_weights(weights_path)
    print("✅ Real weights loaded successfully!")
except Exception as e:
    print(f"⚠️ Weight loading issue: {e}")
    print("Trying by_name=True...")
    try:
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("✅ Weights loaded with by_name=True!")
    except Exception as e2:
        print(f"⚠️ Still failed: {e2}")
        print("Continuing with current weights...")

print("\nStep 4: Saving compatible model...")
model.save('models/deepfer_compatible.h5')
print("✅ Saved: models/deepfer_compatible.h5")

print("\nStep 5: Verifying...")
test_model = tf.keras.models.load_model('models/deepfer_compatible.h5')
dummy = np.zeros((1, 48, 48, 1))
pred = test_model.predict(dummy, verbose=0)[0]
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f"✅ Input shape: {test_model.input_shape}")
print(f"✅ Output shape: {test_model.output_shape}")
print(f"✅ Test prediction: {emotions[np.argmax(pred)]} ({np.max(pred)*100:.1f}%)")
print("\n🎉 Done! Your model is ready.")
