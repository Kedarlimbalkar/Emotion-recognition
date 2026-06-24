"""
Data Preparation Script
Cleans, validates, and organizes data for training
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from configs.config import *
from src.data_engineering import DataEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main data preparation function"""
    
    parser = argparse.ArgumentParser(
        description='Prepare data for emotion recognition model training'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='data/raw',
        help='Source directory containing raw emotion folders'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for cleaned and split data'
    )
    
    parser.add_argument(
        '--skip-clean',
        action='store_true',
        help='Skip cleaning step'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("🔧 DATA PREPARATION PIPELINE")
    logger.info("="*60)
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    # Check source directory
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        logger.error("Please provide a directory with emotion folders (Angry, Happy, etc.)")
        return
    
    # Initialize data engineer
    engineer = DataEngineer(sys.modules[__name__])
    
    try:
        # Step 1: Clean dataset
        if not args.skip_clean:
            logger.info("\nStep 1: Cleaning dataset...")
            logger.info("-" * 60)
            
            valid_files, removed_files = engineer.clean_dataset(
                source_dir=str(source_dir),
                output_dir=str(output_dir)
            )
            
            if removed_files:
                logger.warning(f"⚠️ Removed {len(removed_files)} corrupted images")
                for item in removed_files[:5]:  # Show first 5
                    logger.debug(f"  - {item['file']}: {item['reason']}")
        
        # Step 2: Prepare train/val/test splits
        logger.info("\nStep 2: Creating train/val/test splits...")
        logger.info("-" * 60)
        
        splits_dir = engineer.prepare_dataset(
            source_dir=str(output_dir),
            output_dir=str(output_dir),
            clean_first=False
        )
        
        # Step 3: Generate statistics
        logger.info("\nStep 3: Generating dataset statistics...")
        logger.info("-" * 60)
        
        stats_df = engineer.generate_statistics(
            data_dir=str(output_dir),
            output_file=str(RESULTS_DIR / 'dataset_statistics.csv')
        )
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("✅ DATA PREPARATION COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        logger.info(f"\n📊 Dataset Summary:")
        logger.info(f"  ✓ Source: {source_dir}")
        logger.info(f"  ✓ Output: {output_dir}")
        logger.info(f"  ✓ Total images: {stats_df['Count'].sum()}")
        logger.info(f"\n📁 Data locations:")
        logger.info(f"  ✓ Training data: {PROCESSED_DATA_DIR / 'splits' / 'train'}")
        logger.info(f"  ✓ Validation data: {PROCESSED_DATA_DIR / 'splits' / 'val'}")
        logger.info(f"  ✓ Test data: {PROCESSED_DATA_DIR / 'splits' / 'test'}")
        logger.info(f"\n🚀 Next steps:")
        logger.info(f"  1. Review data statistics in results/dataset_statistics.csv")
        logger.info(f"  2. Run training: python src/train.py")
        logger.info(f"  3. Evaluate model: python src/evaluate.py")
        
    except Exception as e:
        logger.error(f"❌ Error during data preparation: {e}")
        raise


if __name__ == "__main__":
    main()
