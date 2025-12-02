import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, auc
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('models', 'evaluation_log.txt')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FacialRecognitionEvaluation")

# Configuration
MODEL_DIR = 'models'
DATA_DIR = 'data/training_data'
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32

# Ensure output directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def plot_confusion_matrix(cm, class_labels, save_path):
    """Plot and save the confusion matrix with generic class labels."""
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_precision_recall_curve(y_true_bin, y_pred, num_classes, save_path):
    """Plot and save macro-averaged Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_pred.ravel())
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Macro-averaged Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Precision-Recall curve saved to {save_path}")
    return pr_auc


def plot_metrics_summary(metrics, save_path):
    """Plot and save a bar chart for accuracy, precision, recall, and F1 score."""
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['blue', 'green', 'orange', 'red'])
    plt.ylim([0, 1])
    plt.ylabel('Score')
    plt.title('Summary of Classification Metrics')
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.grid(True, axis='y')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Metrics summary plot saved to {save_path}")


def evaluate_identification_metrics(model, test_generator, model_dir):
    """Evaluate the facial recognition system with reasonable metrics."""
    logger.info("Starting evaluation of facial recognition system...")

    # Get class indices and labels
    class_indices = test_generator.class_indices
    num_classes = len(class_indices)
    class_labels = [f'Class {i}' for i in range(num_classes)]  # Generic labels

    # Get true labels and predictions
    y_true = test_generator.classes
    y_pred = model.predict(test_generator, steps=max(1, test_generator.samples // test_generator.batch_size + 1))
    if y_pred.shape[0] > test_generator.samples:
        y_pred = y_pred[:test_generator.samples]

    # Binarize y_true for Precision-Recall curve
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    # Compute standard classification metrics
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_true == y_pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='macro')
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Macro-averaged Precision: {precision:.4f}")
    logger.info(f"Macro-averaged Recall: {recall:.4f}")
    logger.info(f"Macro-averaged F1 Score: {f1:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(cm, class_labels, os.path.join(model_dir, 'confusion_matrix.png'))

    # Plot Precision-Recall curve
    pr_auc = plot_precision_recall_curve(y_true_bin, y_pred, num_classes,
                                         os.path.join(model_dir, 'precision_recall_curve.png'))

    # Plot metrics summary
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    plot_metrics_summary(metrics, os.path.join(model_dir, 'metrics_summary.png'))

    # Save metrics to file
    with open(os.path.join(model_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro-averaged Precision: {precision:.4f}\n")
        f.write(f"Macro-averaged Recall: {recall:.4f}\n")
        f.write(f"Macro-averaged F1 Score: {f1:.4f}\n")
        f.write(f"Area Under Precision-Recall Curve (AUC): {pr_auc:.4f}\n")
    logger.info(f"Metrics saved to {os.path.join(model_dir, 'evaluation_metrics.txt')}")


def main():
    """Main function to load model and evaluate metrics."""
    try:
        # Load the model
        model = load_model(os.path.join(MODEL_DIR, 'face_recognition_model.h5'))
        logger.info("Model loaded successfully.")

        # Prepare test data
        val_test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(DATA_DIR, 'test'),
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        logger.info(f"Found {test_generator.samples} test images in {len(test_generator.class_indices)} classes")

        # Evaluate
        evaluate_identification_metrics(model, test_generator, MODEL_DIR)

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()