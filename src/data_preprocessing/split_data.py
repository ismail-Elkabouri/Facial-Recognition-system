import json
import logging
import os
import random
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'data_splitting_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('DataSplitting')


class DatasetConfig:
    """Configuration for dataset splitting parameters."""

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'data')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_dataset')
    AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, 'augmented_dataset')
    TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'training_data')
    METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
    NON_CLASS_DIRS = ['debug_rejected', 'debug_split_rejected', 'metadata', 'training_data', 'logs']

    IMAGE_SIZE = (224, 224)
    TRAIN_RATIO = 0.8
    VALIDATION_RATIO = 0.1
    TEST_RATIO = 0.1
    MIN_SPLIT_SIZE = 2
    MIN_IMAGES_PER_CLASS = 5
    MAX_IMAGES_PER_CLASS = 1000


class DatasetSplitter:
    """Class for splitting dataset into training, validation, and test sets."""

    def __init__(self, config: DatasetConfig):
        """Initialize the dataset splitter with configuration.

        Args:
            config: Dataset configuration object.
        """
        self.config = config
        self.stats = {
            'total_images': 0,
            'classes': 0,
            'train_images': 0,
            'validation_images': 0,
            'test_images': 0,
            'class_distribution': {},
            'removed_classes': [],
        }

    def combine_datasets(self) -> Dict[str, List[str]]:
        """Combine processed and augmented datasets.

        Returns:
            Dictionary mapping class names to lists of image paths.
        """
        combined_data = {}
        for source_dir in [self.config.PROCESSED_DATA_DIR, self.config.AUGMENTED_DATA_DIR]:
            if not os.path.exists(source_dir):
                logger.warning(f'Directory not found: {source_dir}')
                continue
            for class_name in os.listdir(source_dir):
                if class_name in self.config.NON_CLASS_DIRS or not os.path.isdir(
                    os.path.join(source_dir, class_name)
                ):
                    continue
                class_path = os.path.join(source_dir, class_name)
                if class_name not in combined_data:
                    combined_data[class_name] = []
                image_paths = [
                    os.path.join(class_path, img)
                    for img in os.listdir(class_path)
                    if img.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                combined_data[class_name].extend(image_paths)
        return combined_data

    def prepare_class_data(self, class_name: str, image_paths: List[str]) -> Tuple[Optional[List[str]], str]:
        """Prepare class data by filtering based on size constraints.

        Args:
            class_name: Name of the class.
            image_paths: List of image paths for the class.

        Returns:
            Tuple of (filtered image paths or None, status message).
        """
        if len(image_paths) < self.config.MIN_IMAGES_PER_CLASS:
            return None, f'Too few images: {len(image_paths)}'
        if len(image_paths) > self.config.MAX_IMAGES_PER_CLASS:
            image_paths = random.sample(image_paths, self.config.MAX_IMAGES_PER_CLASS)
        return image_paths, 'Pass'

    def split_data(self) -> Dict[str, any]:
        """Split dataset into training, validation, and test sets.

        Returns:
            Dictionary with splitting statistics.
        """
        start_time = datetime.now()
        output_dir = self.config.TRAINING_DATA_DIR
        combined_data = self.combine_datasets()

        # Create output directories
        for split in ['train', 'validation', 'test', 'logs']:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)

        valid_classes = []
        for class_name, image_paths in tqdm(combined_data.items(), desc='Splitting classes'):
            image_paths, status = self.prepare_class_data(class_name, image_paths)
            if image_paths is None:
                logger.warning(f'Skipping class "{class_name}": {status}')
                self.stats['removed_classes'].append({'class': class_name, 'reason': status})
                continue
            valid_classes.append((class_name, image_paths))

        for class_name, image_paths in valid_classes:
            if len(image_paths) < self.config.MIN_IMAGES_PER_CLASS + 2 * self.config.MIN_SPLIT_SIZE:
                train_images = image_paths
                val_images = []
                test_images = []
                logger.warning(
                    f'Class "{class_name}" has too few images ({len(image_paths)}). All assigned to train.'
                )
            else:
                train_images, temp_images = train_test_split(
                    image_paths,
                    train_size=self.config.TRAIN_RATIO,
                    random_state=42,
                    shuffle=True,
                )
                val_ratio = self.config.VALIDATION_RATIO / (self.config.VALIDATION_RATIO + self.config.TEST_RATIO)
                val_images, test_images = train_test_split(
                    temp_images,
                    train_size=val_ratio,
                    random_state=42,
                    shuffle=True,
                )
                if len(val_images) < self.config.MIN_SPLIT_SIZE:
                    val_images.extend(
                        random.sample(
                            train_images,
                            min(self.config.MIN_SPLIT_SIZE - len(val_images), len(train_images)),
                        )
                    )
                    train_images = [x for x in train_images if x not in val_images]
                if len(test_images) < self.config.MIN_SPLIT_SIZE:
                    test_images.extend(
                        random.sample(
                            train_images,
                            min(self.config.MIN_SPLIT_SIZE - len(test_images), len(train_images)),
                        )
                    )
                    train_images = [x for x in train_images if x not in test_images]

            splits = {'train': train_images, 'validation': val_images, 'test': test_images}
            for split_name, images in splits.items():
                split_dir = os.path.join(output_dir, split_name, class_name)
                os.makedirs(split_dir, exist_ok=True)
                for img_path in images:
                    dest_path = os.path.join(split_dir, os.path.basename(img_path))
                    shutil.copy2(img_path, dest_path)
                self.stats[f'{split_name}_images'] += len(images)
                if class_name not in self.stats['class_distribution']:
                    self.stats['class_distribution'][class_name] = {}
                self.stats['class_distribution'][class_name][split_name] = len(images)

        self.stats['total_images'] = sum(
            [
                self.stats['train_images'],
                self.stats['validation_images'],
                self.stats['test_images'],
            ]
        )
        self.stats['classes'] = len(valid_classes)

        # Save class distribution plot
        plot_path = os.path.join(output_dir, 'logs')
        os.makedirs(plot_path, exist_ok=True)
        plt.figure(figsize=(15, 8))
        class_names = list(self.stats['class_distribution'].keys())
        splits = ['train', 'validation', 'test']
        data = []
        for split in splits:
            data.append([self.stats['class_distribution'][class_name].get(split, 0) for class_name in class_names])
        x = np.arange(len(class_names))
        width = 0.25
        for i, (split, split_data) in enumerate(zip(splits, data)):
            plt.bar(x + i * width, split_data, width, label=split)
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution Across Splits')
        plt.xticks(x + width, class_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, 'class_distribution.png'))
        plt.close()

        # Save metadata
        try:
            os.makedirs(self.config.METADATA_DIR, exist_ok=True)
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'config': {k: v for k, v in vars(DatasetConfig).items() if not k.startswith('_')},
                'stats': self.stats,
            }
            metadata_path = os.path.join(
                self.config.METADATA_DIR,
                f'split_metadata_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            )
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f'Failed to save metadata: {str(e)}')
            raise

        logger.info(f'Dataset splitting completed in {(datetime.now() - start_time).total_seconds():.2f} seconds')
        logger.info(f'Total images: {self.stats["total_images"]}')
        logger.info(f'Number of classes: {self.stats["classes"]}')
        logger.info(f'Classes skipped: {len(self.stats["removed_classes"])}')
        logger.info(f'Output saved to {self.config.TRAINING_DATA_DIR}')
        logger.info(f'Class distribution plot saved to {plot_path}/class_distribution.png')
        logger.info(f'Metadata saved to {metadata_path}')

        return self.stats


def main() -> None:
    """Main function to split the dataset."""
    try:
        logger.info('Starting dataset splitting...')
        config = DatasetConfig()
        splitter = DatasetSplitter(config)

        if os.path.exists(config.TRAINING_DATA_DIR):
            shutil.rmtree(config.TRAINING_DATA_DIR)
        os.makedirs(config.TRAINING_DATA_DIR)
        os.makedirs(config.METADATA_DIR, exist_ok=True)

        stats = splitter.split_data()

    except Exception as e:
        logger.error(f'Error in dataset splitting: {str(e)}', exc_info=True)
        raise


if __name__ == '__main__':
    main()