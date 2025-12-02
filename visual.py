import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Paths
MODEL_PATH = 'models/face_recognition_model.h5'
LABELS_PATH = 'models/face-labels.pickle'

TEST_DIR = 'data/images/'

# Load model and class names
model = load_model(MODEL_PATH)
with open(LABELS_PATH, 'rb') as f:
    class_list = pickle.load(f)

detector = MTCNN()
image_width, image_height = 224, 224

def detect_and_predict(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_img)

    if len(results) != 1:
        print('Not exactly 1 face detected; skipped.')
        return None, None, None

    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)

    # Draw bounding box
    img_disp = rgb_img.copy()
    cv2.rectangle(img_disp, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # Extract face ROI and resize only the ROI for the model
    face_roi = rgb_img[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (image_width, image_height))

    # Preprocess for model
    x_input = image.img_to_array(face_resized)
    x_input = np.expand_dims(x_input, axis=0)
    x_input = preprocess_input(x_input)

    # Predict
    predicted_prob = model.predict(x_input)
    predicted_idx = np.argmax(predicted_prob[0])
    predicted_label = class_list[predicted_idx]

    return img_disp, predicted_label, None  # No confidence needed

def display_image(img_disp, pred_label, actual_label):
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.imshow(img_disp)
    # Set axis scales with ticks every 100 units
    ax.set_xticks(np.arange(0, img_disp.shape[1], 400))
    ax.set_yticks(np.arange(0, img_disp.shape[0], 500))
    info = (
        f"Predicted face: {pred_label}\n"
        f"Actual face: {actual_label}\n"
        f"{'='*20}"
    )
    # Place text below the image with more space
    ax.text(10, img_disp.shape[0] + 250, info, fontsize=12, family='monospace', ha='left', va='top')
    # Adjust layout to make space for text
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Add extra space at the bottom
    plt.show()
    input(f"Press Enter to continue...")

def main():
    # List of all 35 classes from the training log
    all_classes = [
        'abobakr', 'ahmed', 'ismail', 'samih', 'zainab'
    ]

    for class_folder in all_classes:
        folder_path = os.path.join(TEST_DIR, class_folder)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist; skipping.")
            continue
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    # Detect face and predict
                    img_disp, pred_label, _ = detect_and_predict(img)
                    if img_disp is not None:
                        # Display the image with predicted and actual labels
                        display_image(img_disp, pred_label, class_folder)
                break  # Process only the first image per class

    print("Finished displaying images from all classes.")

if __name__ == "__main__":
    main()