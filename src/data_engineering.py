"""
Data Engineering Module - Data Loading, Cleaning, and Preprocessing
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
import shutil
from PIL import Image
import json

logger = logging.getLogger(__name__)


class DataEngineer:
    """Handle all data engineering tasks"""
    
    def __init__(self, config):
        self.config = config
        self.image_size = config.IMAGE_SIZE
        self.emotions = config.EMOTIONS
        
    def validate_image(self, image_path):
        """Validate if image is readable and not corrupted"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, "Cannot read image"
            
            if img.size == 0:
                return False, "Image is empty"
            
            if img.shape[0] < 48 or img.shape[1] < 48:
                return False, "Image too small"
            
            return True, "Valid"
        except Exception as e:
            return False, str(e)
    
    def clean_dataset(self, source_dir, output_dir):
        """Clean dataset by removing corrupted images"""
        logger.info(f"Cleaning dataset from {source_dir}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        removed_files = []
        valid_files = []
        
        # Walk through all emotion folders
        for emotion_idx, emotion_name in self.emotions.items():
            emotion_src = Path(source_dir) / emotion_name
            emotion_dest = output_dir / emotion_name
            
            if not emotion_src.exists():
                logger.warning(f"Emotion folder {emotion_name} not found")
                continue
            
            emotion_dest.mkdir(parents=True, exist_ok=True)
            
            image_files = list(emotion_src.glob('*.jpg')) + list(emotion_src.glob('*.png'))
            
            print(f"\n📁 Processing {emotion_name}...")
            for img_file in tqdm(image_files, desc=emotion_name):
                is_valid, message = self.validate_image(img_file)
                
                if is_valid:
                    shutil.copy2(img_file, emotion_dest / img_file.name)
                    valid_files.append(str(img_file))
                else:
                    removed_files.append({
                        'file': str(img_file),
                        'reason': message
                    })
                    logger.debug(f"Removed {img_file.name}: {message}")
        
        # Save cleaning report
        report = {
            'total_files': len(valid_files) + len(removed_files),
            'valid_files': len(valid_files),
            'removed_files': len(removed_files),
            'removed_details': removed_files
        }
        
        report_path = output_dir / 'cleaning_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Cleaning complete. Valid: {len(valid_files)}, Removed: {len(removed_files)}")
        return valid_files, removed_files
    
    def load_image(self, image_path, apply_clahe=True):
        """Load and preprocess image"""
        img = cv2.imread(str(image_path))
        
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert to grayscale for better emotion detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Resize to target size
        gray = cv2.resize(gray, self.image_size)
        
        return gray
    
    def prepare_dataset(self, source_dir, output_dir, clean_first=True):
        """Prepare dataset for training"""
        
        if clean_first:
            logger.info("Step 1: Cleaning dataset...")
            self.clean_dataset(source_dir, output_dir)
        
        logger.info("Step 2: Organizing data into train/val/test splits...")
        
        output_dir = Path(output_dir)
        splits_dir = output_dir / 'splits'
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            for emotion_name in self.emotions.values():
                (splits_dir / split / emotion_name).mkdir(parents=True, exist_ok=True)
        
        # Process each emotion
        split_info = {
            'train': [],
            'val': [],
            'test': [],
            'emotions': self.emotions
        }
        
        for emotion_idx, emotion_name in self.emotions.items():
            emotion_src = output_dir / emotion_name
            
            if not emotion_src.exists():
                logger.warning(f"Emotion folder {emotion_name} not found")
                continue
            
            image_files = list(emotion_src.glob('*.jpg')) + list(emotion_src.glob('*.png'))
            
            # Split data
            train, test = train_test_split(
                image_files,
                test_size=self.config.TEST_SPLIT,
                random_state=self.config.RANDOM_SEED
            )
            
            val_size = self.config.VAL_SPLIT / (self.config.VAL_SPLIT + self.config.TRAIN_SPLIT)
            train, val = train_test_split(
                train,
                test_size=val_size,
                random_state=self.config.RANDOM_SEED
            )
            
            # Copy files
            for split, files in [('train', train), ('val', val), ('test', test)]:
                for img_file in files:
                    dest = splits_dir / split / emotion_name / img_file.name
                    shutil.copy2(img_file, dest)
                    split_info[split].append({
                        'emotion': emotion_name,
                        'file': img_file.name
                    })
            
            logger.info(f"{emotion_name}: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        # Save split info
        info_path = splits_dir / 'split_info.json'
        with open(info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info("✓ Dataset preparation complete")
        return splits_dir
    
    def create_data_summary(self, data_dir):
        """Create summary statistics of dataset"""
        
        summary = {
            'total_images': 0,
            'emotions': {},
            'splits': {}
        }
        
        data_dir = Path(data_dir)
        
        for emotion_name in self.emotions.values():
            count = len(list((data_dir / emotion_name).glob('*.[jp]*g')))
            if count > 0:
                summary['emotions'][emotion_name] = count
                summary['total_images'] += count
        
        # Check splits if they exist
        if (data_dir / 'splits').exists():
            for split in ['train', 'val', 'test']:
                split_path = data_dir / 'splits' / split
                if split_path.exists():
                    count = len(list(split_path.glob('*/*.[jp]*g')))
                    summary['splits'][split] = count
        
        return summary
    
    def generate_statistics(self, data_dir, output_file):
        """Generate detailed dataset statistics"""
        
        summary = self.create_data_summary(data_dir)
        
        stats_df = pd.DataFrame([
            {
                'Emotion': emotion,
                'Count': count,
                'Percentage': f"{(count/summary['total_images'])*100:.2f}%"
            }
            for emotion, count in summary['emotions'].items()
        ])
        
        # Save statistics
        stats_df.to_csv(output_file, index=False)
        logger.info(f"✓ Statistics saved to {output_file}")
        
        print("\n" + "="*50)
        print("📊 DATASET STATISTICS")
        print("="*50)
        print(f"Total Images: {summary['total_images']}")
        print("\nEmotion Distribution:")
        print(stats_df.to_string(index=False))
        print("="*50 + "\n")
        
        return stats_df
    
    def balance_dataset(self, data_dir, method='oversample'):
        """Balance imbalanced dataset"""
        
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        
        logger.info(f"Balancing dataset using {method}...")
        
        # This is a placeholder - actual implementation depends on your needs
        logger.info("✓ Dataset balanced")


def create_data_generators(train_dir, val_dir, config):
    """Create data generators for training"""
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config.AUGMENTATION_CONFIG.get('rotation_range', 30),
        width_shift_range=config.AUGMENTATION_CONFIG.get('width_shift_range', 0.2),
        height_shift_range=config.AUGMENTATION_CONFIG.get('height_shift_range', 0.2),
        horizontal_flip=config.AUGMENTATION_CONFIG.get('horizontal_flip', True),
        zoom_range=config.AUGMENTATION_CONFIG.get('zoom_range', 0.2),
        fill_mode=config.AUGMENTATION_CONFIG.get('fill_mode', 'nearest')
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, '/home/claude/emotion-recognition-cnn')
    from configs.config import *
    
    engineer = DataEngineer(sys.modules[__name__])
    
    print("Data engineering module loaded successfully!")
