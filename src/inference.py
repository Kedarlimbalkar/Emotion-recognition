"""
Inference - Real-time emotion detection using DeepFER model.
Model input: 48x48 grayscale images
"""

import cv2
import numpy as np
import tensorflow as tf
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import EMOTIONS, IMG_SIZE, MODEL_PATH

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

class EmotionDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)
        self.emotions = EMOTIONS
        print(f"✅ Model loaded: {model_path}")
        print(f"   Input shape: {self.model.input_shape}")

    def preprocess(self, face_img):
        """Convert to grayscale 48x48 — matches training preprocessing."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE)
        normalized = resized.astype('float32') / 255.0
        # Shape: (1, 48, 48, 1)
        return normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

    def predict(self, face_img):
        processed = self.preprocess(face_img)
        preds = self.model.predict(processed, verbose=0)[0]
        idx = np.argmax(preds)
        return self.emotions[idx], float(preds[idx]), preds

    def detect_and_predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        results = []
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            emotion, confidence, all_preds = self.predict(face_img)
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': dict(zip(self.emotions, all_preds.tolist()))
            })
        return results

    def annotate_frame(self, frame, results):
        for r in results:
            x, y, w, h = r['bbox']
            label = f"{r['emotion']} ({r['confidence']*100:.1f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        print("📷 Webcam started. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.detect_and_predict(frame)
            frame = self.annotate_frame(frame, results)
            cv2.imshow('DeepFER - Emotion Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def predict_image(self, image_path):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        results = self.detect_and_predict(frame)
        if not results:
            emotion, confidence, all_preds = self.predict(frame)
            return [{'emotion': emotion, 'confidence': confidence,
                     'all_predictions': dict(zip(self.emotions, all_preds.tolist())),
                     'bbox': None}]
        return results

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.run_webcam()
