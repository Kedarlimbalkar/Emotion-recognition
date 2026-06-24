"""
Evaluation Module - Model evaluation and performance metrics
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import *
from src.data_engineering import create_data_generators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, config, model_path=None):
        self.config = config
        self.model = None
        self.emotions = config.EMOTIONS
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        logger.info(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        logger.info("✓ Model loaded successfully")
        return self.model
    
    def evaluate_on_generator(self, test_generator):
        """Evaluate model on test data generator"""
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        logger.info("Evaluating on test set...")
        
        results = self.model.evaluate(test_generator, verbose=1)
        
        metrics = {
            'loss': float(results[0]),
            'accuracy': float(results[1]),
            'precision': float(results[2]) if len(results) > 2 else None,
            'recall': float(results[3]) if len(results) > 3 else None
        }
        
        return metrics
    
    def get_predictions(self, test_generator):
        """Get predictions on test set"""
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        logger.info("Generating predictions...")
        
        predictions = self.model.predict(test_generator, verbose=1)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = test_generator.classes
        
        return predictions, pred_labels, true_labels
    
    def generate_classification_report(self, y_true, y_pred, save_path=None):
        """Generate detailed classification report"""
        
        emotion_names = list(self.emotions.values())
        
        report = classification_report(
            y_true, y_pred,
            target_names=emotion_names,
            output_dict=True
        )
        
        if save_path is None:
            save_path = self.config.RESULTS_DIR / 'classification_report.json'
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Classification report saved to {save_path}")
        
        # Print report
        print("\n" + "="*80)
        print("📊 CLASSIFICATION REPORT")
        print("="*80)
        
        report_text = classification_report(
            y_true, y_pred,
            target_names=emotion_names
        )
        print(report_text)
        print("="*80 + "\n")
        
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot and save confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        emotion_names = list(self.emotions.values())
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=emotion_names,
            yticklabels=emotion_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.config.RESULTS_DIR / 'confusion_matrix.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Confusion matrix saved to {save_path}")
        plt.close()
        
        return cm
    
    def plot_emotion_distribution(self, y_true, save_path=None):
        """Plot emotion distribution in test set"""
        
        emotion_names = list(self.emotions.values())
        unique, counts = np.unique(y_true, return_counts=True)
        
        plt.figure(figsize=(12, 6))
        colors = list(self.config.EMOTION_COLORS.values())
        
        bars = plt.bar(
            [emotion_names[i] for i in unique],
            counts,
            color=colors
        )
        
        plt.title('Emotion Distribution in Test Set', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xlabel('Emotion', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.config.RESULTS_DIR / 'emotion_distribution.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Emotion distribution plot saved to {save_path}")
        plt.close()
    
    def plot_per_class_metrics(self, y_true, y_pred, save_path=None):
        """Plot precision, recall, f1 for each emotion"""
        
        emotion_names = list(self.emotions.values())
        
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        x = np.arange(len(emotion_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(emotion_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.config.RESULTS_DIR / 'per_class_metrics.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Per-class metrics plot saved to {save_path}")
        plt.close()
    
    def generate_evaluation_summary(self, y_true, y_pred, metrics, save_path=None):
        """Generate complete evaluation summary"""
        
        emotion_names = list(self.emotions.values())
        
        summary = {
            'overall_metrics': metrics,
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'weighted_precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'weighted_recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        if save_path is None:
            save_path = self.config.RESULTS_DIR / 'evaluation_summary.json'
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Evaluation summary saved to {save_path}")
        
        print("\n" + "="*60)
        print("📈 EVALUATION SUMMARY")
        print("="*60)
        print(f"Overall Accuracy: {summary['accuracy']:.4f}")
        print(f"Macro Precision: {summary['macro_precision']:.4f}")
        print(f"Macro Recall: {summary['macro_recall']:.4f}")
        print(f"Macro F1-Score: {summary['macro_f1']:.4f}")
        print("="*60 + "\n")
        
        return summary


def main():
    """Main evaluation function"""
    
    logger.info("="*60)
    logger.info("🔍 EMOTION RECOGNITION MODEL EVALUATION")
    logger.info("="*60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        sys.modules[__name__],
        model_path=FINAL_MODEL_PATH
    )
    
    # Prepare test data
    test_dir = PROCESSED_DATA_DIR / 'splits' / 'test'
    
    if not test_dir.exists():
        logger.error(f"Test data not found at {test_dir}")
        return
    
    logger.info(f"Loading test data from {test_dir}")
    test_datagen = create_data_generators(
        str(test_dir), str(test_dir), sys.modules[__name__]
    )[1]  # Get val_datagen for test
    
    # Evaluate
    metrics = evaluator.evaluate_on_generator(test_datagen)
    logger.info(f"Test Results: {metrics}")
    
    # Get predictions
    predictions, pred_labels, true_labels = evaluator.get_predictions(test_datagen)
    
    # Generate reports
    evaluator.generate_classification_report(true_labels, pred_labels)
    evaluator.plot_confusion_matrix(true_labels, pred_labels)
    evaluator.plot_emotion_distribution(true_labels)
    evaluator.plot_per_class_metrics(true_labels, pred_labels)
    evaluator.generate_evaluation_summary(true_labels, pred_labels, metrics)
    
    logger.info("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
