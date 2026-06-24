"""
Model Architecture Module - DeepFER CNN for Emotion Recognition
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50, EfficientNetB0, VGG16
)
import logging

logger = logging.getLogger(__name__)


class EmotionRecognitionModel:
    """Build and manage emotion recognition models"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def build_mobilenetv2_model(self, input_shape=(224, 224, 3), num_classes=7):
        """Build MobileNetV2 transfer learning model"""
        
        logger.info("Building MobileNetV2 model...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build the model
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Preprocessing
            layers.Rescaling(1./127.5, offset=-1),
            
            # Base model
            base_model,
            
            # Custom top layers
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(256, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu', name='dense_3'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        self.model = model
        logger.info("✓ MobileNetV2 model built successfully")
        return model
    
    def build_resnet50_model(self, input_shape=(224, 224, 3), num_classes=7):
        """Build ResNet50 transfer learning model"""
        
        logger.info("Building ResNet50 model...")
        
        base_model = ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Rescaling(1./255),
            
            base_model,
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        self.model = model
        logger.info("✓ ResNet50 model built successfully")
        return model
    
    def build_efficientnetb0_model(self, input_shape=(224, 224, 3), num_classes=7):
        """Build EfficientNetB0 transfer learning model"""
        
        logger.info("Building EfficientNetB0 model...")
        
        base_model = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Rescaling(1./255),
            
            base_model,
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        self.model = model
        logger.info("✓ EfficientNetB0 model built successfully")
        return model
    
    def build_custom_cnn(self, input_shape=(224, 224, 3), num_classes=7):
        """Build custom CNN from scratch"""
        
        logger.info("Building custom CNN model...")
        
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=input_shape, name='conv1_1'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool4'),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu', name='dense1'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu', name='dense2'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        self.model = model
        logger.info("✓ Custom CNN model built successfully")
        return model
    
    def build_model(self, architecture=None, input_shape=None, num_classes=None):
        """Build model based on specified architecture"""
        
        architecture = architecture or self.config.MODEL_ARCHITECTURE
        input_shape = input_shape or (self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.CHANNELS)
        num_classes = num_classes or self.config.NUM_CLASSES
        
        if architecture == "MobileNetV2":
            return self.build_mobilenetv2_model(input_shape, num_classes)
        elif architecture == "ResNet50":
            return self.build_resnet50_model(input_shape, num_classes)
        elif architecture == "EfficientNetB0":
            return self.build_efficientnetb0_model(input_shape, num_classes)
        elif architecture == "Custom":
            return self.build_custom_cnn(input_shape, num_classes)
        else:
            logger.warning(f"Unknown architecture: {architecture}. Using MobileNetV2")
            return self.build_mobilenetv2_model(input_shape, num_classes)
    
    def compile_model(self, learning_rate=None):
        """Compile the model"""
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        learning_rate = learning_rate or self.config.LEARNING_RATE
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        logger.info(f"✓ Model compiled with learning rate: {learning_rate}")
        return self.model
    
    def unfreeze_base_model(self, num_layers=None, new_learning_rate=None):
        """Unfreeze base model layers for fine-tuning"""
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        # Find the base model layer
        for layer in self.model.layers:
            if hasattr(layer, 'trainable'):
                # Check if it's a pre-trained base model
                if isinstance(layer, (MobileNetV2, ResNet50, EfficientNetB0, VGG16)):
                    num_layers = num_layers or self.config.FINE_TUNE_FROM_LAYER
                    
                    # Unfreeze last N layers
                    layer.trainable = True
                    for l in layer.layers[:-num_layers]:
                        l.trainable = False
                    
                    logger.info(f"✓ Unfroze last {num_layers} layers for fine-tuning")
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=new_learning_rate or self.config.FINE_TUNE_LEARNING_RATE)
    
    def get_model(self):
        """Get the built model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        return self.model
    
    def model_summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        return self.model.summary()
    
    def save_model(self, filepath):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        self.model.save(filepath)
        logger.info(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"✓ Model loaded from {filepath}")
        return self.model


class ModelCallbacks:
    """Create training callbacks"""
    
    @staticmethod
    def get_callbacks(config):
        """Return list of callbacks for training"""
        
        from tensorflow.keras.callbacks import (
            ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
            TensorBoard, CSVLogger
        )
        
        callbacks = [
            ModelCheckpoint(
                filepath=str(config.BEST_MODEL_PATH),
                monitor=config.CHECKPOINT_MONITOR,
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            EarlyStopping(
                monitor=config.EARLY_STOPPING_MONITOR,
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            ReduceLROnPlateau(
                monitor=config.EARLY_STOPPING_MONITOR,
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),
            
            TensorBoard(
                log_dir=str(config.LOGS_DIR),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            
            CSVLogger(
                filename=str(config.RESULTS_DIR / 'training_log.csv'),
                append=True
            )
        ]
        
        return callbacks


if __name__ == "__main__":
    print("Model architecture module loaded successfully!")
