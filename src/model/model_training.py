"""
Enhanced model training module with improved architecture and methodology.

This module provides a more effective approach to training face recognition models
using specialized architectures and advanced training techniques.
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('ModelTraining')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'data', 'training_data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'models')
FACE_LABELS_FILENAME = os.path.join(MODEL_DIR, 'face-labels.pickle')
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 70
LEARNING_RATE = 1e-4
FINE_TUNING_LEARNING_RATE = 1e-5
L2_REGULARIZATION = 0.01
LABEL_SMOOTHING = 0.1
DROPOUT_RATE_1 = 0.5
DROPOUT_RATE_2 = 0.3

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, 'logs'), exist_ok=True)


def create_data_generators() -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator, Dict[int, float]]:
    """Create data generators with appropriate augmentation for face recognition.

    Returns:
        Tuple of (train_generator, validation_generator, test_generator, class_weights).
    """
    logger.info('Creating data generators with augmentation...')

    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect',
    )

    # Validation/test data generators with minimal processing
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet.preprocess_input,
    )

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
    )

    # Load validation data
    validation_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'validation'),
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    # Load test data
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    # Compute class weights for imbalanced datasets
    class_counts = np.bincount(train_generator.classes)
    total_samples = np.sum(class_counts)
    num_classes = len(class_counts)
    class_weights = {
        i: total_samples / (num_classes * class_counts[i]) if class_counts[i] > 0 else 1.0
        for i in range(num_classes)
    }

    logger.info(f'Found {train_generator.samples} training images in {len(train_generator.class_indices)} classes')
    logger.info(f'Found {validation_generator.samples} validation images')
    logger.info(f'Found {test_generator.samples} test images')
    logger.info(f'Class weights: {class_weights}')

    return train_generator, validation_generator, test_generator, class_weights


def create_model(num_classes: int) -> Tuple[Model, Model]:
    """Create a face recognition model using transfer learning with ResNet50.

    Args:
        num_classes: Number of classes (individuals) to recognize.

    Returns:
        Tuple of (compiled Keras model, base ResNet50 model).
    """
    logger.info(f'Creating model with {num_classes} output classes...')

    # Load ResNet50 model without top layers
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    )

    # Freeze base model layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom top layers with regularization
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE_1)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE_2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile model with label smoothing and gradient clipping
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipvalue=1.0),
        loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy'],
    )

    logger.info(f'Model created with {model.count_params()} parameters')
    return model, base_model


def train_model(
    model: Model,
    train_generator: ImageDataGenerator,
    validation_generator: ImageDataGenerator,
    class_weights: Dict[int, float],
    epochs: int = EPOCHS,
) -> tf.keras.callbacks.History:
    """Train the model with enhanced overfitting prevention.

    Args:
        model: Keras model to train.
        train_generator: Training data generator.
        validation_generator: Validation data generator.
        class_weights: Class weights for imbalanced data.
        epochs: Maximum number of epochs.

    Returns:
        Training history object.
    """
    logger.info(f'Starting model training for up to {epochs} epochs...')

    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(
                MODEL_DIR,
                'checkpoints',
                'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5',
            ),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        TensorBoard(
            log_dir=os.path.join(MODEL_DIR, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S')),
            histogram_freq=1,
            update_freq='epoch',
        ),
    ]

    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
    validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    logger.info('Initial training completed')
    return history


def fine_tune_model(
    model: Model,
    base_model: Model,
    train_generator: ImageDataGenerator,
    validation_generator: ImageDataGenerator,
    class_weights: Dict[int, float],
    epochs: int = 0,
) -> tf.keras.callbacks.History:
    """Fine-tune the model with enhanced overfitting prevention.

    Args:
        model: Keras model to fine-tune.
        base_model: Base ResNet50 model with layers to unfreeze.
        train_generator: Training data generator.
        validation_generator: Validation data generator.
        class_weights: Class weights for imbalanced data.
        epochs: Maximum number of epochs for fine-tuning.

    Returns:
        Fine-tuning history object.
    """
    logger.info('Starting model fine-tuning...')

    # Unfreeze the last 30 layers of the base model
    for layer in base_model.layers[-30:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNING_LEARNING_RATE, clipvalue=0.5),
        loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy'],
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=0.0005,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
    validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    logger.info('Fine-tuning completed')
    return fine_tune_history


def evaluate_model(model: Model, test_generator: ImageDataGenerator) -> Tuple[float, float]:
    """Evaluate the model on the test set.

    Args:
        model: Trained Keras model.
        test_generator: Test data generator.

    Returns:
        Tuple of (loss, accuracy) on test set.
    """
    logger.info('Evaluating model on test set...')

    steps = max(1, test_generator.samples // BATCH_SIZE)
    loss, accuracy = model.evaluate(test_generator, steps=steps)

    logger.info(f'Test loss: {loss:.4f}')
    logger.info(f'Test accuracy: {accuracy:.4f}')

    return loss, accuracy


def plot_training_history(
    history: tf.keras.callbacks.History, fine_tune_history: Optional[tf.keras.callbacks.History] = None
) -> None:
    """Plot and save training history graphs.

    Args:
        history: Initial training history.
        fine_tune_history: Fine-tuning history (optional).
    """
    logger.info('Plotting training history...')

    plt.figure(figsize=(12, 8))

    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    if fine_tune_history:
        epochs_initial = len(history.history['accuracy'])
        fine_tune_epochs = range(epochs_initial, epochs_initial + len(fine_tune_history.history['accuracy']))
        plt.plot(fine_tune_epochs, fine_tune_history.history['accuracy'], label='Fine-tuning Training Accuracy')
        plt.plot(fine_tune_epochs, fine_tune_history.history['val_accuracy'], label='Fine-tuning Validation Accuracy')

    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    if fine_tune_history:
        plt.plot(fine_tune_epochs, fine_tune_history.history['loss'], label='Fine-tuning Training Loss')
        plt.plot(fine_tune_epochs, fine_tune_history.history['val_loss'], label='Fine-tuning Validation Loss')

    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    logger.info(f'Training history plot saved to {os.path.join(MODEL_DIR, "training_history.png")}')
    plt.close()


def save_model_and_labels(model: Model, train_generator: ImageDataGenerator) -> None:
    """Save the trained model and class labels.

    Args:
        model: Trained Keras model.
        train_generator: Training data generator with class indices.
    """
    logger.info('Saving model and class labels...')

    model_filename = os.path.join(MODEL_DIR, 'face_recognition_model.h5')
    model.save(model_filename)
    logger.info(f'Model saved to {model_filename}')

    class_dictionary = {v: k for k, v in train_generator.class_indices.items()}
    with open(FACE_LABELS_FILENAME, 'wb') as f:
        pickle.dump(class_dictionary, f)

    logger.info(f'Class dictionary saved to {FACE_LABELS_FILENAME}')
    logger.info(f'Class dictionary: {class_dictionary}')


def main() -> None:
    """Main function to train and evaluate the face recognition model."""
    try:
        logger.info('Starting face recognition model training...')

        # Create data generators and get class weights
        train_generator, validation_generator, test_generator, class_weights = create_data_generators()

        # Create model
        num_classes = len(train_generator.class_indices)
        model, base_model = create_model(num_classes)

        # Initial training
        history = train_model(model, train_generator, validation_generator, class_weights)

        # Fine-tuning
        fine_tune_history = fine_tune_model(model, base_model, train_generator, validation_generator, class_weights)

        # Evaluate model
        evaluate_model(model, test_generator)

        # Plot training history
        plot_training_history(history, fine_tune_history)

        # Save model and labels
        save_model_and_labels(model, train_generator)

        logger.info('Model training completed successfully')

    except Exception as e:
        logger.error(f'Error in model training: {str(e)}', exc_info=True)
        raise


if __name__ == '__main__':
    main()