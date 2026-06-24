"""
Training Module - Complete training pipeline for emotion recognition model
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import *
from src.model import EmotionRecognitionModel, ModelCallbacks
from src.data_engineering import create_data_generators

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Main training pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.history = None
        self.model = None
        
    def prepare_data(self, train_dir, val_dir):
        """Prepare data generators"""
        logger.info("Preparing data generators...")
        
        train_generator, val_generator = create_data_generators(
            train_dir, val_dir, self.config
        )
        
        logger.info(f"✓ Train samples: {train_generator.samples}")
        logger.info(f"✓ Val samples: {val_generator.samples}")
        
        return train_generator, val_generator
    
    def build_model(self):
        """Build the model"""
        logger.info(f"Building {self.config.MODEL_ARCHITECTURE} model...")
        
        model_builder = EmotionRecognitionModel(self.config)
        self.model = model_builder.build_model()
        model_builder.compile_model()
        
        logger.info("✓ Model built and compiled")
        return self.model
    
    def train(self, train_generator, val_generator, epochs=None):
        """Train the model"""
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        epochs = epochs or self.config.EPOCHS
        
        logger.info("="*60)
        logger.info("🚀 STARTING TRAINING")
        logger.info("="*60)
        
        # Get callbacks
        callbacks = ModelCallbacks.get_callbacks(self.config)
        
        # Train
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("="*60)
        logger.info("✓ TRAINING COMPLETE")
        logger.info("="*60)
        
        return self.history
    
    def fine_tune(self, train_generator, val_generator, epochs=10):
        """Fine-tune the model by unfreezing base layers"""
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        logger.info("="*60)
        logger.info("🔧 STARTING FINE-TUNING")
        logger.info("="*60)
        
        # Unfreeze base model
        model_builder = EmotionRecognitionModel(self.config)
        model_builder.model = self.model
        model_builder.unfreeze_base_model(
            num_layers=self.config.FINE_TUNE_FROM_LAYER,
            new_learning_rate=self.config.FINE_TUNE_LEARNING_RATE
        )
        
        logger.info(f"Learning rate reduced to {self.config.FINE_TUNE_LEARNING_RATE}")
        
        # Get callbacks for fine-tuning
        callbacks = ModelCallbacks.get_callbacks(self.config)
        
        # Fine-tune
        fine_tune_history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            initial_epoch=self.config.EPOCHS,
            verbose=1
        )
        
        logger.info("="*60)
        logger.info("✓ FINE-TUNING COMPLETE")
        logger.info("="*60)
        
        return fine_tune_history
    
    def save_model(self):
        """Save trained model"""
        
        if self.model is None:
            raise ValueError("Model not trained. Train first.")
        
        # Save in Keras format
        self.model.save(str(self.config.FINAL_MODEL_PATH))
        logger.info(f"✓ Model saved to {self.config.FINAL_MODEL_PATH}")
        
        # Save weights
        self.model.save_weights(str(self.config.WEIGHTS_PATH))
        logger.info(f"✓ Weights saved to {self.config.WEIGHTS_PATH}")
    
    def save_training_summary(self):
        """Save training summary to JSON"""
        
        if self.history is None:
            logger.warning("No training history to save")
            return
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_architecture': self.config.MODEL_ARCHITECTURE,
            'epochs': len(self.history.history['loss']),
            'batch_size': self.config.BATCH_SIZE,
            'learning_rate': float(self.config.LEARNING_RATE),
            'final_metrics': {
                'train_loss': float(self.history.history['loss'][-1]),
                'train_accuracy': float(self.history.history['accuracy'][-1]),
                'val_loss': float(self.history.history['val_loss'][-1]),
                'val_accuracy': float(self.history.history['val_accuracy'][-1])
            },
            'best_metrics': {
                'best_train_accuracy': float(max(self.history.history['accuracy'])),
                'best_val_accuracy': float(max(self.history.history['val_accuracy']))
            }
        }
        
        summary_path = self.config.RESULTS_DIR / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Training summary saved to {summary_path}")
        
        print("\n" + "="*60)
        print("📊 TRAINING SUMMARY")
        print("="*60)
        print(f"Epochs trained: {summary['epochs']}")
        print(f"Final Train Accuracy: {summary['final_metrics']['train_accuracy']:.4f}")
        print(f"Final Val Accuracy: {summary['final_metrics']['val_accuracy']:.4f}")
        print(f"Best Val Accuracy: {summary['best_metrics']['best_val_accuracy']:.4f}")
        print("="*60 + "\n")
        
        return summary
    
    def plot_training_history(self, save_path=None):
        """Plot and save training history"""
        
        if self.history is None:
            logger.warning("No training history to plot")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.config.RESULTS_DIR / 'training_history.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Training history plot saved to {save_path}")
        plt.close()


def main():
    """Main training function"""
    
    logger.info("="*60)
    logger.info("🎯 EMOTION RECOGNITION CNN TRAINING")
    logger.info("="*60)
    
    # Initialize pipeline
    pipeline = TrainingPipeline(sys.modules[__name__])
    
    # Check data
    train_dir = PROCESSED_DATA_DIR / 'splits' / 'train'
    val_dir = PROCESSED_DATA_DIR / 'splits' / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        logger.error(f"Data not found at {PROCESSED_DATA_DIR}")
        logger.error("Please run data preparation first: python src/prepare_data.py")
        return
    
    # Prepare data
    train_gen, val_gen = pipeline.prepare_data(str(train_dir), str(val_dir))
    
    # Build model
    pipeline.build_model()
    
    # Train
    pipeline.train(train_gen, val_gen, epochs=EPOCHS)
    
    # Fine-tune (optional)
    # pipeline.fine_tune(train_gen, val_gen, epochs=10)
    
    # Save model
    pipeline.save_model()
    
    # Save summary and plots
    pipeline.save_training_summary()
    pipeline.plot_training_history()
    
    logger.info("✓ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
