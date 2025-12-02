import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
from tqdm import tqdm

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('EnhancedPreprocessing')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'models')
FACE_LABELS_FILENAME = os.path.join(MODEL_DIR, 'face-labels.pickle')
METADATA_FILENAME = os.path.join(MODEL_DIR, 'preprocessing_metadata.pickle')
DEBUG_DIR = os.path.join(DATA_DIR, 'debug_rejected')
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Quality thresholds (relaxed)
FACE_CONFIDENCE_THRESHOLD = 0.9
BLUR_THRESHOLD = 50
BRIGHTNESS_RANGE = (30, 255)
CONTRAST_THRESHOLD = 15
FACE_MIN_SIZE = 80


class FaceQualityError(Exception):
    """Custom exception for face quality issues."""


def initialize_face_detector() -> None:
    """Initialize face detection backend."""
    logger.info('Initializing face detection backend (MTCNN via DeepFace)...')
    try:
        # Warm up the detector
        test_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        DeepFace.extract_faces(
            img_path=test_img,
            detector_backend='mtcnn',
            enforce_detection=False,
        )
        logger.info('Face detector initialized successfully')
    except Exception as e:
        logger.error(f'Error initializing face detector: {e}')
        raise


def assess_image_quality(image: np.ndarray) -> Tuple[Optional[Dict[str, float]], bool, str]:
    """Assess various quality metrics of an image.

    Args:
        image: Input image as a NumPy array.

    Returns:
        Tuple of (quality_metrics, quality_passed, quality_error), where:
        - quality_metrics: Dictionary of quality metrics (blur_score, brightness, contrast, face_size).
        - quality_passed: True if the image passes quality checks.
        - quality_error: Reason for failure (if any).
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate quality metrics
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # Check face size using MTCNN
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend='mtcnn',
            enforce_detection=False,
        )
        face_size = 0
        if faces and faces[0].get('facial_area'):
            w, h = faces[0]['facial_area']['w'], faces[0]['facial_area']['h']
            face_size = min(w, h)

        quality_metrics = {
            'blur_score': blur_score,
            'brightness': brightness,
            'contrast': contrast,
            'face_size': face_size,
        }

        # Evaluate quality
        if blur_score < BLUR_THRESHOLD:
            return quality_metrics, False, 'Low blur score'
        if not (BRIGHTNESS_RANGE[0] <= brightness <= BRIGHTNESS_RANGE[1]):
            return quality_metrics, False, 'Brightness out of range'
        if contrast < CONTRAST_THRESHOLD:
            return quality_metrics, False, 'Low contrast'
        if face_size < FACE_MIN_SIZE:
            return quality_metrics, False, 'Face too small'

        return quality_metrics, True, ''

    except Exception as e:
        logger.error(f'Error in quality assessment: {e}')
        return None, False, f'Quality assessment error: {str(e)}'


def enhance_face(face_image: np.ndarray) -> np.ndarray:
    """Enhance face image quality through various preprocessing techniques.

    Args:
        face_image: Input face image as a NumPy array.

    Returns:
        Enhanced face image as a NumPy array.
    """
    try:
        # Convert to float32 for processing
        face_float = face_image.astype(np.float32)

        # Normalize lighting using LAB color space
        lab = cv2.cvtColor(face_float, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l.astype(np.uint8))
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        # Adjust contrast
        enhanced_rgb = cv2.normalize(
            enhanced_rgb,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        return enhanced_rgb

    except Exception as e:
        logger.error(f'Error in face enhancement: {e}')
        return face_image


def detect_and_align_face(image: np.ndarray) -> Tuple[Optional[np.ndarray], float, Optional[dict]]:
    """Detect and align face using MTCNN.

    Args:
        image: Input image as a NumPy array.

    Returns:
        Tuple of (aligned_face, face_confidence, landmarks), where:
        - aligned_face: Aligned face image or None if detection fails.
        - face_confidence: Confidence score of the detected face.
        - landmarks: Facial landmarks or None if detection fails.
    """
    try:
        face_objs = DeepFace.extract_faces(
            img_path=image,
            detector_backend='mtcnn',
            enforce_detection=False,
            align=True,
        )

        if not face_objs or len(face_objs) == 0:
            return None, 0, None

        # Select face with highest confidence
        face_obj = max(face_objs, key=lambda x: x.get('confidence', 0))
        confidence = face_obj.get('confidence', 0)

        if confidence < FACE_CONFIDENCE_THRESHOLD:
            logger.warning(f'Face detection confidence too low: {confidence}')
            return None, confidence, None

        face = face_obj['face']
        landmarks = face_obj.get('landmarks', None)

        # Convert to uint8 and pad to ensure size
        face = (face * 255).astype(np.uint8)
        pad = int(max(face.shape[0], face.shape[1]) * 0.1)
        face = cv2.copyMakeBorder(face, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

        return face, confidence, landmarks

    except Exception as e:
        logger.warning(f'Face detection failed: {e}')
        return None, 0, None


def save_rejected_image(image: np.ndarray, image_path: str, reason: str) -> None:
    """Save rejected image for debugging.

    Args:
        image: Image to save as a NumPy array.
        image_path: Original image path.
        reason: Reason for rejection.
    """
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename = f'rejected_{os.path.splitext(os.path.basename(image_path))[0]}_{timestamp}.jpg'
        output_path = os.path.join(DEBUG_DIR, filename)
        Image.fromarray(image).save(output_path, quality=95, optimize=True)
        logger.debug(f'Saved rejected image: {output_path}, reason: {reason}')
    except Exception as e:
        logger.error(f'Error saving rejected image: {str(e)}')


def process_single_image(image_path: str, output_dir: str, class_name: str) -> Dict[str, any]:
    """Process a single image through the pipeline.

    Args:
        image_path: Path to input image.
        output_dir: Base output directory.
        class_name: Class/identity name.

    Returns:
        Dictionary with processing results and metrics.
    """
    result = {
        'path': image_path,
        'class': class_name,
        'success': False,
        'error': None,
        'quality_metrics': None,
    }

    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError('Could not read image')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect and align face
        face, confidence, landmarks = detect_and_align_face(img)
        if face is None:
            raise FaceQualityError('No face detected or low confidence')

        # Assess and enhance face quality
        quality_metrics, quality_passed, quality_error = assess_image_quality(face)
        if not quality_passed:
            save_rejected_image(face, image_path, quality_error)
            raise FaceQualityError(quality_error)

        enhanced_face = enhance_face(face)

        # Resize to target dimensions
        resized_face = cv2.resize(
            enhanced_face,
            (IMG_WIDTH, IMG_HEIGHT),
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Save processed image
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        output_filename = os.path.join(
            class_output_dir,
            f'processed_{os.path.basename(image_path)}',
        )

        Image.fromarray(resized_face).save(output_filename)

        result.update({
            'success': True,
            'quality_metrics': quality_metrics,
            'confidence': confidence,
            'output_path': output_filename,
        })

    except FaceQualityError as e:
        result['error'] = f'Quality error: {str(e)}'
    except Exception as e:
        result['error'] = f'Processing error: {str(e)}'
        logger.error(f'Error processing {image_path}: {e}', exc_info=True)

    return result


def process_dataset(input_dir: str, output_dir: str, max_workers: int = 4) -> Dict[str, any]:
    """Process entire dataset with parallel execution.

    Args:
        input_dir: Input dataset directory.
        output_dir: Output directory for processed images.
        max_workers: Maximum number of parallel workers.

    Returns:
        Dictionary with processing statistics and metadata.
    """
    stats = {
        'total_images': 0,
        'successful': 0,
        'failed': 0,
        'quality_metrics': [],
        'class_distribution': {},
        'errors': [],
    }

    # Collect all image paths
    image_tasks = []
    for root, _, files in os.walk(input_dir):
        if root == input_dir:
            continue

        class_name = os.path.basename(root)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_tasks.append((image_path, output_dir, class_name))

    logger.info(f'Found {len(image_tasks)} images to process')

    # Process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(process_single_image, *task): task[0]
            for task in image_tasks
        }

        for future in tqdm(as_completed(future_to_path), total=len(image_tasks)):
            image_path = future_to_path[future]
            try:
                result = future.result()
                stats['total_images'] += 1

                if result['success']:
                    stats['successful'] += 1
                    class_name = result['class']
                    stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1

                    if result['quality_metrics']:
                        stats['quality_metrics'].append(result['quality_metrics'])
                else:
                    stats['failed'] += 1
                    stats['errors'].append({
                        'path': image_path,
                        'error': result['error'],
                    })

            except Exception as e:
                stats['failed'] += 1
                stats['errors'].append({
                    'path': image_path,
                    'error': str(e),
                })

    return stats


def analyze_dataset_statistics(stats: Dict[str, any]) -> None:
    """Analyze and log dataset statistics.

    Args:
        stats: Dataset processing statistics.
    """
    logger.info('\nDataset Processing Statistics:')
    logger.info(f'Total images processed: {stats["total_images"]}')
    logger.info(f'Successfully processed: {stats["successful"]}')
    logger.info(f'Failed processing: {stats["failed"]}')

    if stats['quality_metrics']:
        blur_scores = [m['blur_score'] for m in stats['quality_metrics']]
        brightness_values = [m['brightness'] for m in stats['quality_metrics']]
        contrast_values = [m['contrast'] for m in stats['quality_metrics']]
        face_sizes = [m['face_size'] for m in stats['quality_metrics']]

        logger.info('\nQuality Metrics Summary:')
        logger.info(
            f'Blur scores - Mean: {np.mean(blur_scores):.2f}, '
            f'Min: {np.min(blur_scores):.2f}, '
            f'Max: {np.max(blur_scores):.2f}'
        )
        logger.info(
            f'Brightness - Mean: {np.mean(brightness_values):.2f}, '
            f'Min: {np.min(brightness_values):.2f}, '
            f'Max: {np.max(brightness_values):.2f}'
        )
        logger.info(
            f'Contrast - Mean: {np.mean(contrast_values):.2f}, '
            f'Min: {np.min(contrast_values):.2f}, '
            f'Max: {np.max(contrast_values):.2f}'
        )
        logger.info(
            f'Face sizes - Mean: {np.mean(face_sizes):.2f}, '
            f'Min: {np.min(face_sizes):.2f}, '
            f'Max: {np.max(face_sizes):.2f}'
        )

    logger.info('\nClass Distribution:')
    for class_name, count in stats['class_distribution'].items():
        logger.info(f'{class_name}: {count} images')

    if stats['errors']:
        logger.info('\nTop Processing Errors:')
        error_counts = {}
        for error in stats['errors']:
            error_type = error['error'].split(':')[0]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f'{error_type}: {count} occurrences')


def main() -> None:
    """Main function to preprocess images and prepare dataset."""
    try:
        logger.info('Starting enhanced face preprocessing pipeline...')
        logger.info(f'Processing started by user: {os.getenv("USER", "unknown")}')

        # Initialize face detector
        initialize_face_detector()

        # Setup directories
        input_dir = os.path.join(DATA_DIR, 'main_dataset')
        processed_dir = os.path.join(DATA_DIR, 'processed_dataset')
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(DEBUG_DIR, exist_ok=True)

        # Process dataset
        stats = process_dataset(input_dir, processed_dir)

        # Analyze and log statistics
        analyze_dataset_statistics(stats)

        # Save processing metadata
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'user': os.getenv('USER', 'unknown'),
            'stats': stats,
            'config': {
                'img_size': (IMG_WIDTH, IMG_HEIGHT),
                'face_confidence_threshold': FACE_CONFIDENCE_THRESHOLD,
                'blur_threshold': BLUR_THRESHOLD,
                'brightness_range': BRIGHTNESS_RANGE,
                'contrast_threshold': CONTRAST_THRESHOLD,
                'face_min_size': FACE_MIN_SIZE,
            },
        }

        with open(METADATA_FILENAME, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f'Processing metadata saved to {METADATA_FILENAME}')
        logger.info('Face preprocessing completed successfully')

    except Exception as e:
        logger.error(f'Critical error in preprocessing pipeline: {str(e)}', exc_info=True)
        raise


if __name__ == '__main__':
    main()