import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from deepface import DeepFace
import os
from datetime import datetime
import logging

# --- Configure Logging ---
logger = logging.getLogger("FaceRecognition_MTCNN")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    log_file_path = "face_recognition_mtcnn.log"
    if os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
        except Exception as e:
            logger.warning(f"Could not remove old log file: {log_file_path} - {e}")

    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {os.path.abspath(log_file_path)}")


def normalize_name(name):
    """Normalize names to ensure consistency."""
    if not name:
        return ""
    return str(name).lower().strip().replace('_', ' ').replace('-', ' ').replace('  ', ' ')


class FaceRecognition:
    def __init__(self, auth_manager, database):
        logger.info("Initializing FaceRecognition (VGG16+MTCNN backend)...")
        self.auth_manager = auth_manager
        self.database = database
        self.current_class_id = None
        self.recognized_students = set()

        # Paths
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
        MODEL_DIR = os.path.join(PROJECT_DIR, 'models')

        # Load trained model
        model_path = os.path.join(MODEL_DIR, 'face_recognition_model.h5')
        if not os.path.exists(model_path):
            logger.critical(f"Model file NOT FOUND at {model_path}")
            raise FileNotFoundError(f"Cannot find {model_path}")
        self.model = load_model(model_path)
        logger.info("Successfully loaded VGG16 model.")

        # Load labels
        labels_path = os.path.join(MODEL_DIR, 'face-labels.pickle')
        if not os.path.exists(labels_path):
            logger.critical(f"Labels file NOT FOUND at {labels_path}")
            raise FileNotFoundError(f"Cannot find {labels_path}")
        with open(labels_path, 'rb') as f:
            og_labels = pickle.load(f)
            self.labels = {key: normalize_name(value) for key, value in og_labels.items()}
        logger.info(f"Loaded label mappings: {dict(list(self.labels.items())[:3])}")

        # Model input size
        self.image_width, self.image_height = 224, 224

        # Student name-to-ID mapping
        self.name_to_id = {}
        students = self.auth_manager.get_all_users(role='student')
        for student in students:
            normalized_name = normalize_name(student['name'])
            self.name_to_id[normalized_name] = student['id']
        logger.info(f"Loaded {len(self.name_to_id)} student name-to-ID mappings.")
        if not self.name_to_id:
            logger.warning("Name-to-ID map is EMPTY. Attendance recording will fail.")

    def detect_faces(self, frame):
        """Detect faces using MTCNN backend."""
        logger.debug(f"detect_faces: Frame shape: {frame.shape if frame is not None else 'None'}")
        detected_boxes_info = []

        try:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            face_objects = DeepFace.extract_faces(
                img_path=frame, detector_backend='mtcnn',
                enforce_detection=False, align=True
            )
            for face_obj in face_objects:
                if isinstance(face_obj, dict) and 'facial_area' in face_obj and isinstance(face_obj['facial_area'], dict):
                    fa = face_obj['facial_area']
                    if all(k in fa for k in ('x', 'y', 'w', 'h')) and fa['w'] > 0 and fa['h'] > 0:
                        detected_boxes_info.append({
                            'box': (fa['x'], fa['y'], fa['w'], fa['h']),
                            'detection_confidence': face_obj.get('confidence', 0.0)
                        })
            logger.debug(f"detect_faces: Detected {len(detected_boxes_info)} faces.")
        except Exception as e:
            logger.error(f"Error in MTCNN detection: {e}", exc_info=True)

        return [{'box': det['box']} for det in detected_boxes_info]

    def preprocess_frame(self, frame):
        """Resize and normalize frame for model input."""
        resized_frame = cv2.resize(frame, (self.image_width, self.image_height))
        normalized_frame = preprocess_input(resized_frame)
        return np.expand_dims(normalized_frame, axis=0)

    def recognize_faces(self, frame):
        """Detect and recognize faces in the frame, and record attendance."""
        faces = self.detect_faces(frame)
        results = []

        for face in faces:
            (x, y, w, h) = face['box']
            x, y = max(0, x), max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            if w > 0 and h > 0:
                face_region = frame[y:y+h, x:x+w]
                preprocessed = self.preprocess_frame(face_region)

                predicted_prob = self.model.predict(preprocessed, verbose=0)
                predicted_class = np.argmax(predicted_prob[0])
                confidence = predicted_prob[0][predicted_class]

                name = self.labels.get(predicted_class, "Unknown")
                normalized_name = normalize_name(name)
                student_id = self.name_to_id.get(normalized_name)

                logger.info(f"ATTENDANCE_TRACE: Predicted '{name}' (normalized '{normalized_name}') - Confidence: {confidence:.4f}, ID: {student_id}")

                if student_id and confidence > 0.5:
                    if student_id not in self.recognized_students and self.current_class_id:
                        self.record_attendance(self.current_class_id, student_id, 'present')
                        self.recognized_students.add(student_id)
                    elif not self.current_class_id:
                        logger.warning(f"Student {student_id} recognized, but no active class session.")
                elif not student_id:
                    logger.warning(f"Name '{normalized_name}' not found in name_to_id.")
                else:
                    logger.debug(f"Confidence {confidence:.4f} below threshold.")

                # Append result with only name and box (no confidence)
                results.append((name, (x, y, w, h)))

        if results:
            logger.info(f"Recognized results: {results}")

        return results

    def start_attendance_session(self, class_id):
        """Start attendance session for a specific class."""
        self.current_class_id = class_id
        self.recognized_students.clear()
        logger.info(f"Started attendance session for class ID {class_id}")

    def record_attendance(self, class_id, student_id, status):
        """Record attendance for a student in the database."""
        try:
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Recording attendance: class_id={class_id}, student_id={student_id}, date={date}, status={status}")
            self.database.record_attendance(class_id, student_id, date, status)
        except Exception as e:
            logger.error(f"Error recording attendance: {e}", exc_info=True)

    def stop_attendance_session(self):
        """Stop the current attendance session."""
        logger.info(f"Stopped attendance session for class ID: {self.current_class_id}. Recognized students: {list(self.recognized_students)}")
        self.current_class_id = None
        self.recognized_students.clear()
