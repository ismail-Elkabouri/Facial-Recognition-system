import os
import numpy as np
import tensorflow as tf
from deepface import DeepFace
import sqlite3
import json

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'data', 'augmented_data')  # Updated path
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'models')
IMG_WIDTH, IMG_HEIGHT = 224, 224  # DeepFace will handle resizing internally

# Check if the augmented_data directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: Directory {DATA_DIR} does not exist. Please ensure the augmented_data folder is present.")
    exit()

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Delete the existing database to start fresh
db_path = os.path.join(MODEL_DIR, 'face_embeddings.db')
if os.path.exists(db_path):
    try:
        os.remove(db_path)
        print(f"Successfully deleted existing database: {db_path}")
    except Exception as e:
        print(f"Error deleting database {db_path}: {e}")
        exit()
else:
    print(f"No existing database found at {db_path}")

# Generate and store embeddings in a database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, embedding TEXT)')  # Store as TEXT

# Use DeepFace to generate embeddings
for class_name in os.listdir(DATA_DIR):  # Updated to use DATA_DIR directly
    class_dir = os.path.join(DATA_DIR, class_name)

    # Skip if not a directory
    if not os.path.isdir(class_dir):
        print(f"Skipping {class_name}: Not a directory")
        continue

    print(f"\nProcessing student: {class_name}")
    embeddings = []
    image_count = 0

    for img_name in os.listdir(class_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {img_name}")
            continue

        img_path = os.path.join(class_dir, img_name)
        try:
            # Generate embedding using DeepFace (FaceNet model)
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",  # Ensure we're using FaceNet
                enforce_detection=False
            )[0]["embedding"]
            embedding = np.array(embedding, dtype=np.float32)
            print(f"Embedding for {img_name} in {class_name}: shape {embedding.shape}")
            embeddings.append(embedding)
            image_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if embeddings:
        print(f"Processed {image_count} images for {class_name}")
        # Average embeddings for the student
        avg_embedding = np.mean(embeddings, axis=0)
        print(f"Average embedding for {class_name}: shape {avg_embedding.shape}")

        # Store as JSON string
        embedding_json = json.dumps(avg_embedding.tolist())
        cursor.execute("INSERT INTO students (name, embedding) VALUES (?, ?)",
                       (class_name, embedding_json))
        print(f"Stored embedding for {class_name}")
    else:
        print(f"No embeddings generated for {class_name} (processed {image_count} images)")

conn.commit()

# Verify the stored embeddings
cursor.execute("SELECT name, embedding FROM students")
for name, emb_json in cursor.fetchall():
    emb_array = np.array(json.loads(emb_json), dtype=np.float32)
    print(f"Verified stored embedding for {name}: shape {emb_array.shape}")

conn.close()
print("Embeddings stored in face_embeddings.db")