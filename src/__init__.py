"""
DeepFER - Facial Emotion Recognition using Deep Learning

A comprehensive deep learning project for real-time emotion recognition
from facial images using state-of-the-art CNN architectures.
"""

__version__ = "1.0.0"
__author__ = "Kedar Limbalkar"
__email__ = "kedar@example.com"
__description__ = "Facial Emotion Recognition using Deep Learning"

from src.model import EmotionRecognitionModel, ModelCallbacks
from src.inference import EmotionDetector, BatchPredictor
from src.data_engineering import DataEngineer, create_data_generators

__all__ = [
    'EmotionRecognitionModel',
    'ModelCallbacks',
    'EmotionDetector',
    'BatchPredictor',
    'DataEngineer',
    'create_data_generators'
]
