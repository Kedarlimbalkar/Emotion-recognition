"""
Configuration module for Emotion Recognition CNN
"""

import os
from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, AUGMENTED_DATA_DIR, 
                  MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== EMOTION CLASSES ====================
EMOTIONS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

NUM_CLASSES = len(EMOTIONS)
EMOTION_COLORS = {
    'Angry': '#FF6B6B',
    'Disgust': '#95E1D3',
    'Fear': '#FFD93D',
    'Happy': '#6BCB77',
    'Neutral': '#4D96FF',
    'Sad': '#A8E6CF',
    'Surprise': '#FF8B94'
}

# ==================== MODEL CONFIGURATION ====================
# Image parameters
IMAGE_SIZE = (224, 224)
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5

# Model names
MODEL_NAME = "DeepFER_Final_Model"
MODEL_ARCHITECTURE = "MobileNetV2"  # Options: MobileNetV2, ResNet50, EfficientNetB0, VGG16

# Training strategy
USE_TRANSFER_LEARNING = True
FREEZE_BASE_LAYERS = True
FINE_TUNE_FROM_LAYER = 100  # Number of layers to unfreeze for fine-tuning

# ==================== DATA AUGMENTATION ====================
AUGMENTATION_CONFIG = {
    'rotation_range': 30,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'fill_mode': 'nearest'
}

# ==================== TRAINING CALLBACKS ====================
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = 'val_loss'
CHECKPOINT_MONITOR = 'val_accuracy'
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# ==================== DATA SPLIT ====================
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ==================== INFERENCE ====================
CONFIDENCE_THRESHOLD = 0.6
MAX_FACES_PER_IMAGE = 10

# ==================== DEPLOYMENT ====================
STREAMLIT_PORT = 8501
FLASK_PORT = 5000
FLASK_DEBUG = False

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== PATHS TO MODELS ====================
BEST_MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}.keras"
WEIGHTS_PATH = MODEL_DIR / f"{MODEL_NAME}_weights.h5"

# ==================== ENVIRONMENT ====================
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')  # development, production
USE_GPU = True
RANDOM_SEED = 42

# ==================== API ENDPOINTS ====================
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:5000')
ALLOWED_ORIGINS = ['*']  # Update with specific domains in production

print(f"✓ Configuration loaded for {ENVIRONMENT} environment")
print(f"✓ Using model: {MODEL_ARCHITECTURE}")
print(f"✓ Image size: {IMAGE_SIZE}")
print(f"✓ Batch size: {BATCH_SIZE}")
print(f"✓ Number of emotions: {NUM_CLASSES}")
