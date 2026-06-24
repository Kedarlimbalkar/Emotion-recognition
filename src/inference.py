"""
Inference Module - Real-time emotion detection on images and video streams
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionDetector:
    """Real-time emotion detection"""
    
    def __init__(self, model_path=None, config=None):
        self.config = config or sys.modules[__name__]
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if model_path:
            self.load_model(model_path)
        else:
            self.load_model(str(self.config.FINAL_MODEL_PATH))
        
        self.emotions = config.EMOTIONS if config else self.config.EMOTIONS
        self.emotion_colors = config.EMOTION_COLORS if config else self.config.EMOTION_COLORS
    
    def load_model(self, model_path):
        """Load the trained emotion recognition model"""
        logger.info(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        logger.info("✓ Model loaded successfully")
    
    def preprocess_face(self, face_roi):
        """Preprocess face image for prediction"""
        # Resize to model input size
        face_resized = cv2.resize(face_roi, self.config.IMAGE_SIZE)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_resized = clahe.apply(face_resized)
        
        # Normalize
        face_normalized = face_resized / 255.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        face_batch = np.expand_dims(face_batch, axis=-1)  # Add channel dimension
        
        # Repeat for 3 channels if needed
        if self.model.input_shape[-1] == 3:
            face_batch = np.repeat(face_batch, 3, axis=-1)
        
        return face_batch
    
    def predict_emotion(self, face_roi):
        """Predict emotion for a face ROI"""
        face_batch = self.preprocess_face(face_roi)
        
        # Get prediction
        predictions = self.model.predict(face_batch, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        confidence = float(predictions[emotion_idx])
        emotion_name = self.emotions[emotion_idx]
        
        return emotion_name, confidence, predictions
    
    def detect_and_recognize_emotions(self, image_path=None, image_array=None):
        """Detect faces and recognize emotions in an image"""
        
        if image_path:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
        elif image_array is not None:
            image = image_array
        else:
            raise ValueError("Provide either image_path or image_array")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        annotated_image = image.copy()
        
        for (x, y, w, h) in faces[:self.config.MAX_FACES_PER_IMAGE]:
            face_roi = gray[y:y+h, x:x+w]
            
            emotion, confidence, all_predictions = self.predict_emotion(face_roi)
            
            if confidence >= self.config.CONFIDENCE_THRESHOLD:
                results.append({
                    'bbox': (x, y, w, h),
                    'emotion': emotion,
                    'confidence': confidence,
                    'all_predictions': {
                        self.emotions[i]: float(all_predictions[i])
                        for i in range(len(all_predictions))
                    }
                })
                
                # Draw bounding box and emotion label
                color = self._hex_to_bgr(self.emotion_colors[emotion])
                cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
                
                label = f"{emotion} ({confidence:.2f})"
                cv2.putText(
                    annotated_image, label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2
                )
        
        return results, annotated_image
    
    def detect_emotions_video(self, video_path=None, output_path=None, fps=30):
        """Detect emotions in video"""
        
        if video_path:
            cap = cv2.VideoCapture(str(video_path))
        else:
            cap = cv2.VideoCapture(0)  # Webcam
        
        if not cap.isOpened():
            raise ValueError("Cannot open video source")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output video if path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps,
                (frame_width, frame_height)
            )
        
        frame_count = 0
        emotion_counts = {emotion: 0 for emotion in self.emotions.values()}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect emotions
                results, annotated_frame = self.detect_and_recognize_emotions(
                    image_array=frame
                )
                
                # Update emotion counts
                for result in results:
                    emotion_counts[result['emotion']] += 1
                
                # Add frame info
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2
                )
                
                if output_path:
                    out.write(annotated_frame)
                
                # Display
                cv2.imshow('Emotion Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
        
        finally:
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
        
        logger.info(f"Processed {frame_count} frames")
        return emotion_counts
    
    @staticmethod
    def _hex_to_bgr(hex_color):
        """Convert hex color to BGR for OpenCV"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))


class BatchPredictor:
    """Batch prediction on multiple images"""
    
    def __init__(self, model_path=None, config=None):
        self.detector = EmotionDetector(model_path, config)
        self.results = []
    
    def predict_directory(self, directory_path, output_dir=None):
        """Predict emotions for all images in directory"""
        
        directory = Path(directory_path)
        output_dir = Path(output_dir) if output_dir else directory / 'predictions'
        output_dir.mkdir(exist_ok=True)
        
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG')
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory.glob(ext))
        
        logger.info(f"Found {len(image_files)} images")
        
        self.results = []
        
        for image_path in image_files:
            try:
                results, annotated_image = self.detector.detect_and_recognize_emotions(
                    image_path=image_path
                )
                
                # Save annotated image
                output_path = output_dir / f"annotated_{image_path.name}"
                cv2.imwrite(str(output_path), annotated_image)
                
                self.results.append({
                    'image': image_path.name,
                    'detections': results
                })
                
                logger.info(f"✓ Processed {image_path.name}")
            
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
        
        return self.results
    
    def save_results(self, output_path):
        """Save prediction results to JSON"""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    detector = EmotionDetector()
    
    # Test on image
    # results, annotated = detector.detect_and_recognize_emotions(image_path="test_image.jpg")
    # cv2.imwrite("output.jpg", annotated)
    
    # Or test on video
    # emotion_counts = detector.detect_emotions_video(video_path="test_video.mp4", output_path="output.mp4")
    
    print("Inference module loaded successfully!")
