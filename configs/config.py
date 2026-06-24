"""
Configuration - Matched to actual DeepFER model (48x48 grayscale CNN)
"""

# Emotions - must match training folder names exactly
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = 7

# Image settings - ACTUAL model input (48x48 grayscale, NOT 224x224)
IMG_SIZE = (48, 48)
IMG_CHANNELS = 1        # grayscale
BATCH_SIZE = 32

# Model paths
MODEL_PATH = "models/deepfer_compatible.h5"   # use this after running fix_model.py
ORIGINAL_MODEL_PATH = "models/DeepFER_Final_Model_v1.keras"
BACKUP_MODEL_PATH = "models/DeepFER_Model.keras"
WEIGHTS_PATH = "models/model_weights.h5"

# Data paths
TRAIN_DIR = "images/train"
VAL_DIR   = "images/validation"

# Training
EPOCHS = 10
LEARNING_RATE = 0.001
