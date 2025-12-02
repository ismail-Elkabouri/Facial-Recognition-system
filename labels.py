import pickle
import os

# --- IMPORTANT: Adjust this path to point to your actual labels file ---
# Option 1: If this script is in the same directory as your main script (e.g., src/recognition)
# and your models folder is at the project root (e.g., facial_recognition_project/models)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets dir of inspect_labels.py
    # Go up two levels to project root, then into models
    LABELS_PATH = os.path.join(BASE_DIR, "models", "face-labels.pickle")
except NameError: # If running in an interactive console where __file__ is not defined
    # Assume models is a subdir of current working directory or use an absolute path
    LABELS_PATH = os.path.join(os.getcwd(), "models", "face-labels.pickle")
    # Or provide an absolute path directly:
    # LABELS_PATH = "C:/Users/Administrator/Documents/facial_recognition_project/models/face-labels.pickle"


print(f"Attempting to load labels from: {os.path.abspath(LABELS_PATH)}")

if not os.path.exists(LABELS_PATH):
    print(f"ERROR: Labels file not found at {LABELS_PATH}")
else:
    try:
        with open(LABELS_PATH, 'rb') as f:
            labels_data = pickle.load(f)

        print("\n--- Contents of face-labels.pickle ---")
        print(f"Type of loaded data: {type(labels_data)}")

        if isinstance(labels_data, dict):
            print("Format: {class_index: 'Name'}")
            for class_index, name in labels_data.items():
                print(f"  Index {class_index}: '{name}' (Type of index: {type(class_index)}, Type of name: {type(name)})")
            # Example: If your model has 5 output classes for 5 people
            # Expected output:
            #   Index 0: 'Person_A'
            #   Index 1: 'Person_B'
            #   ...
        elif isinstance(labels_data, list):
            print("Format: List of names ['Name1', 'Name2', ...]")
            print("This means the index of the list implies the class_index.")
            for i, name in enumerate(labels_data):
                 print(f"  Implicit Index {i}: '{name}' (Type of name: {type(name)})")
            print("\nNOTE: If your labels are a list, your Keras model's output (predicted_class_index) directly corresponds to the list index.")
        else:
            print("Data is not a dictionary or a list. Contents:")
            print(labels_data)

        print("\n--- End of Contents ---")

    except pickle.UnpicklingError as e:
        print(f"ERROR: Could not unpickle the file. It might be corrupted or not a pickle file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")