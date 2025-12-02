import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f"data_augmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('EnhancedDataAugmentation')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'data')
INPUT_DATA_DIR = os.path.join(DATA_DIR, 'processed_dataset')
OUTPUT_AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, 'augmented_dataset')
VISUALIZATION_DIR = os.path.join(OUTPUT_AUGMENTED_DATA_DIR, 'visualizations')
DEBUG_DIR = os.path.join(OUTPUT_AUGMENTED_DATA_DIR, 'debug_rejected')


class AugmentationConfig:
    """Configuration for enhanced data augmentation parameters."""

    # Image size
    TARGET_SIZE = (224, 224)

    # Base augmentation parameters
    GEOMETRIC_TRANSFORMS = {
        'rotation_range': (-15, 15),
        'width_shift_range': (-0.05, 0.05),
        'height_shift_range': (-0.05, 0.05),
        'shear_range': (-5, 5),
        'zoom_range': (0.95, 1.05),
        'horizontal_flip': True,
    }

    # Advanced augmentation parameters
    COLOR_TRANSFORMS = {
        'brightness_range': (0.8, 1.2),
        'contrast_range': (0.9, 1.1),
        'saturation_range': (0.9, 1.1),
        'hue_shift_range': (-0.03, 0.03),
    }

    NOISE_PARAMS = {
        'gaussian_noise_range': (3, 15),
        'jpeg_quality_range': (90, 100),
    }

    WEATHER_EFFECTS = {
        'shadow_range': (0.4, 0.6),
        'blur_range': (0.3, 0.7),
    }

    # Augmentation strategy
    AUGMENTATION_TYPES = [
        'geometric',
        'photometric',
        'noise',
        'weather',
        'mixed',
    ]

    # Number of augmentations per strategy
    AUGMENTATIONS_PER_TYPE = {
        'default': 4,
        'underrepresented': 8,
        'priority': 12,
    }

    # Threshold for underrepresented classes
    UNDERREPRESENTED_THRESHOLD = 20

    # Priority classes for extra augmentation
    PRIORITY_CLASSES = ['samih', 'hamid', 'zainab']

    # Quality thresholds
    QUALITY_THRESHOLDS = {
        'min_face_size': 50,
        'min_brightness': 25,
        'max_brightness': 255,
        'min_contrast': 12,
    }


class FaceAugmenter:
    """Class for applying and verifying image augmentations."""

    def __init__(self):
        """Initialize augmentation pipelines."""
        self.geometric_aug = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT,
                p=0.6,
            ),
            A.Perspective(scale=(0.03, 0.07), p=0.2),
            A.HorizontalFlip(p=0.5),
        ])

        self.photometric_aug = A.Compose([
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.03,
                p=0.7,
            ),
            A.RandomGamma(gamma_limit=(90, 110), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4,
            ),
        ])

        self.noise_aug = A.Compose([
            A.GaussNoise(var_limit=(3, 15), p=0.4),
            A.ImageCompression(quality_lower=90, quality_upper=100, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])

        self.weather_aug = A.Compose([
            A.RandomShadow(p=0.2),
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, p=0.1),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0,
                p=0.2,
            ),
        ])

    def apply_augmentation(self, image: np.ndarray, aug_type: str) -> np.ndarray:
        """Apply specific augmentation type to the image.

        Args:
            image: Input image as a NumPy array.
            aug_type: Type of augmentation to apply ('geometric', 'photometric', 'noise', 'weather', 'mixed').

        Returns:
            Augmented image as a NumPy array.
        """
        try:
            if aug_type == 'geometric':
                return self.geometric_aug(image=image)['image']
            elif aug_type == 'photometric':
                return self.photometric_aug(image=image)['image']
            elif aug_type == 'noise':
                return self.noise_aug(image=image)['image']
            elif aug_type == 'weather':
                return self.weather_aug(image=image)['image']
            elif aug_type == 'mixed':
                augs = [self.geometric_aug, self.photometric_aug, self.noise_aug, self.weather_aug]
                selected_augs = random.sample(augs, k=2)
                img = image.copy()
                for aug in selected_augs:
                    img = aug(image=img)['image']
                return img
        except Exception as e:
            logger.warning(f'Augmentation failed: {str(e)}')
            return image

    def verify_quality(self, image: np.ndarray) -> Tuple[bool, str]:
        """Verify if the augmented image meets quality standards.

        Args:
            image: Input image as a NumPy array.

        Returns:
            Tuple of (is_valid, reason), where is_valid is True if the image meets quality standards,
            and reason is an empty string or the rejection reason.
        """
        try:
            # Convert to grayscale for checks
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Check brightness
            brightness = np.mean(gray)
            if brightness < AugmentationConfig.QUALITY_THRESHOLDS['min_brightness']:
                return False, 'Low brightness'
            if brightness > AugmentationConfig.QUALITY_THRESHOLDS['max_brightness']:
                return False, 'High brightness'

            # Check contrast
            contrast = np.std(gray)
            if contrast < AugmentationConfig.QUALITY_THRESHOLDS['min_contrast']:
                return False, 'Low contrast'

            # Check face presence and size using MTCNN
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend='mtcnn',
                align=False,
                enforce_detection=False,
            )
            if not faces or not faces[0].get('facial_area'):
                return False, 'No face detected'
            x, y, w, h = (
                faces[0]['facial_area']['x'],
                faces[0]['facial_area']['y'],
                faces[0]['facial_area']['w'],
                faces[0]['facial_area']['h'],
            )
            if min(w, h) < AugmentationConfig.QUALITY_THRESHOLDS['min_face_size']:
                return False, 'Face too small'

            return True, ''
        except Exception as e:
            logger.warning(f'Quality verification failed: {str(e)}')
            return False, f'Verification error: {str(e)}'

    def save_rejected_image(self, image: np.ndarray, input_path: str, aug_type: str, reason: str) -> None:
        """Save rejected image for debugging.

        Args:
            image: Rejected image as a NumPy array.
            input_path: Path to the original image.
            aug_type: Type of augmentation applied.
            reason: Reason for rejection.
        """
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filename = f'rejected_{os.path.splitext(os.path.basename(input_path))[0]}_{aug_type}_{timestamp}.jpg'
            output_path = os.path.join(DEBUG_DIR, filename)
            img_copy = image.copy()
            cv2.putText(img_copy, reason, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            Image.fromarray(img_copy).save(output_path, quality=95, optimize=True)
            logger.debug(f'Saved rejected image: {output_path}, reason: {reason}')
        except Exception as e:
            logger.error(f'Error saving rejected image: {str(e)}')


def process_single_image(args: Tuple[str, str, FaceAugmenter, int, bool]) -> Dict[str, any]:
    """Process a single image with augmentations.

    Args:
        args: Tuple containing (input_path, output_dir, augmenter, aug_count, copy_original).

    Returns:
        Dictionary with processing results (path, saved count, rejected count, rejection reasons).
    """
    input_path, output_dir, augmenter, aug_count, copy_original = args
    try:
        # Load and preprocess image
        image = cv2.imread(input_path)
        if image is None:
            logger.error(f'Failed to load image: {input_path}')
            return {'path': input_path, 'saved': 0, 'rejected': 0, 'reasons': []}
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, AugmentationConfig.TARGET_SIZE)

        augmented_images = []
        quality_failures = []

        # Copy original image if requested
        if copy_original:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filename = f'original_{os.path.splitext(os.path.basename(input_path))[0]}_{timestamp}.jpg'
            output_path = os.path.join(output_dir, filename)
            Image.fromarray(image).save(output_path, quality=95, optimize=True)
            augmented_images.append((image, 'original'))

        # Apply each augmentation type
        for aug_type in AugmentationConfig.AUGMENTATION_TYPES:
            for _ in range(aug_count):
                augmented = augmenter.apply_augmentation(image, aug_type)

                # Verify quality
                is_valid, reason = augmenter.verify_quality(augmented)
                if is_valid:
                    augmented_images.append((augmented, aug_type))
                else:
                    quality_failures.append({'type': aug_type, 'reason': reason})
                    augmenter.save_rejected_image(augmented, input_path, aug_type, reason)

        # Save augmented images
        saved_count = 0
        for idx, (augmented, aug_type) in enumerate(augmented_images):
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filename = f'{os.path.splitext(os.path.basename(input_path))[0]}_{aug_type}_{timestamp}_{idx}.jpg'
            output_path = os.path.join(output_dir, filename)
            Image.fromarray(augmented).save(output_path, quality=95, optimize=True)
            saved_count += 1

        return {
            'path': input_path,
            'saved': saved_count,
            'rejected': len(quality_failures),
            'reasons': quality_failures,
        }
    except Exception as e:
        logger.error(f'Error processing {input_path}: {str(e)}')
        return {'path': input_path, 'saved': 0, 'rejected': 0, 'reasons': []}


def visualize_augmentations(image_path: str, augmenter: FaceAugmenter) -> None:
    """Create visualization of different augmentation types.

    Args:
        image_path: Path to the input image.
        augmenter: FaceAugmenter instance.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f'Failed to load image for visualization: {image_path}')
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Augmentation Examples', fontsize=16)

        # Original
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        # Show each augmentation type
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for aug_type, pos in zip(AugmentationConfig.AUGMENTATION_TYPES, positions):
            augmented = augmenter.apply_augmentation(image, aug_type)
            axes[pos[0], pos[1]].imshow(augmented)
            axes[pos[0], pos[1]].set_title(f'{aug_type.capitalize()}')
            axes[pos[0], pos[1]].axis('off')

        plt.tight_layout()
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'augmentation_examples.png'))
        plt.close()
    except Exception as e:
        logger.error(f'Visualization failed: {str(e)}')


def main() -> None:
    """Main function to perform enhanced data augmentation."""
    try:
        logger.info('Starting enhanced data augmentation process...')

        # Create output directories
        os.makedirs(OUTPUT_AUGMENTED_DATA_DIR, exist_ok=True)
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        os.makedirs(DEBUG_DIR, exist_ok=True)

        # Initialize augmenter
        augmenter = FaceAugmenter()

        # Process each class
        total_augmented = 0
        total_rejected = 0
        class_stats = {}
        rejection_reasons = {}

        class_dirs = [
            d for d in os.listdir(INPUT_DATA_DIR)
            if os.path.isdir(os.path.join(INPUT_DATA_DIR, d))
        ]

        for class_name in tqdm(class_dirs, desc='Processing classes'):
            input_class_dir = os.path.join(INPUT_DATA_DIR, class_name)
            output_class_dir = os.path.join(OUTPUT_AUGMENTED_DATA_DIR, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            # Get all images in class
            image_files = [
                f for f in os.listdir(input_class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            # Determine augmentation count based on class size
            if class_name in AugmentationConfig.PRIORITY_CLASSES:
                aug_count = AugmentationConfig.AUGMENTATIONS_PER_TYPE['priority']
            elif len(image_files) < AugmentationConfig.UNDERREPRESENTED_THRESHOLD:
                aug_count = AugmentationConfig.AUGMENTATIONS_PER_TYPE['underrepresented']
            else:
                aug_count = AugmentationConfig.AUGMENTATIONS_PER_TYPE['default']

            # Prepare arguments for parallel processing
            process_args = [
                (os.path.join(input_class_dir, img), output_class_dir, augmenter, aug_count, True)
                for img in image_files
            ]

            # Process images in parallel
            class_augmented = 0
            class_rejected = 0
            class_reasons = []
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_single_image, process_args))

            # Update statistics
            for result in results:
                if result['saved'] > 0:
                    class_augmented += result['saved']
                class_rejected += result['rejected']
                for reason in result['reasons']:
                    reason_key = f"{reason['type']}: {reason['reason']}"
                    class_reasons.append(reason_key)
                    rejection_reasons[reason_key] = rejection_reasons.get(reason_key, 0) + 1

            total_augmented += class_augmented
            total_rejected += class_rejected
            class_stats[class_name] = {
                'original': len(image_files),
                'augmented': class_augmented,
                'rejected': class_rejected,
                'rejection_reasons': class_reasons,
            }

            # Create visualization for first class
            if class_name == class_dirs[0] and image_files:
                visualize_augmentations(os.path.join(input_class_dir, image_files[0]), augmenter)

        # Log final statistics
        logger.info('\nAugmentation Statistics:')
        logger.info(f'Total augmented images: {total_augmented}')
        logger.info(f'Total rejected images: {total_rejected}')
        logger.info('\nPer-class Statistics:')
        for class_name, stats in class_stats.items():
            logger.info(
                f"{class_name}: {stats['original']} original → "
                f"{stats['augmented']} augmented, {stats['rejected']} rejected"
            )
        logger.info('\nRejection Reasons:')
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            logger.info(f'{reason}: {count} occurrences')

        logger.info(f'\nOutput saved to: {OUTPUT_AUGMENTED_DATA_DIR}')
        logger.info(f'Visualizations saved to: {VISUALIZATION_DIR}')
        logger.info(f'Rejected images saved to: {DEBUG_DIR}')

    except Exception as e:
        logger.error(f'Critical error in augmentation pipeline: {str(e)}', exc_info=True)
        raise


if __name__ == '__main__':
    main()