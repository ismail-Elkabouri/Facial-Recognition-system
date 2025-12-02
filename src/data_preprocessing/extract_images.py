import os
import shutil
from PIL import Image
import io

# Path to the downloaded LFW dataset folder
dataset_path = "lfw_dataset"  # Update if the path is different (e.g., "lfw_dataset/lfw-deepfunneled/lfw-deepfunneled")

# Output folder for filtered images
output_dir = "data/organized_dataset"

# Image count range
MIN_IMAGES = 20
MAX_IMAGES = 30

# Minimum resolution for "good quality" (width, height)
MIN_RESOLUTION = (50, 50)


# Function to check image quality
def is_good_quality(image_path):
    try:
        with Image.open(image_path) as image:
            # Verify image is not corrupted
            image.verify()
            # Reopen for size check (verify() closes the file)
        with Image.open(image_path) as image:
            width, height = image.size
            return width >= MIN_RESOLUTION[0] and height >= MIN_RESOLUTION[1]
    except Exception as e:
        print(f"Quality check failed for {image_path}: {e}")
        return False


# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Counter for saved images and valid subfolders
saved_images = 0
valid_subfolders = 0

# Iterate through subfolders in dataset
print("Scanning subfolders...")
for person_folder in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_folder)

    # Skip if not a directory
    if not os.path.isdir(person_path):
        continue

    # Count images in the subfolder
    image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_count = len(image_files)

    # Check if subfolder has 20–30 images
    if MIN_IMAGES <= image_count <= MAX_IMAGES:
        print(f"Processing {person_folder} with {image_count} images...")

        # Check quality of all images in the subfolder
        valid_images = []
        for img_file in image_files:
            img_path = os.path.join(person_path, img_file)
            if is_good_quality(img_path):
                valid_images.append(img_file)
            else:
                print(f"Skipping low-quality image: {img_path}")

        # Only proceed if there are still 20–30 valid images after quality check
        if MIN_IMAGES <= len(valid_images) <= MAX_IMAGES:
            # Create person-specific output subfolder
            person_output_dir = os.path.join(output_dir, person_folder)
            os.makedirs(person_output_dir, exist_ok=True)

            # Copy valid images
            for img_file in valid_images:
                src_path = os.path.join(person_path, img_file)
                dst_path = os.path.join(person_output_dir, img_file)
                try:
                    shutil.copy2(src_path, dst_path)
                    saved_images += 1
                    if saved_images % 100 == 0:
                        print(f"Saved {saved_images} images...")
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")
                    continue

            valid_subfolders += 1
            print(f"Saved {len(valid_images)} images for {person_folder}")
        else:
            print(f"Skipped {person_folder}: Only {len(valid_images)} valid images after quality check")
    else:
        print(f"Skipped {person_folder}: {image_count} images (outside {MIN_IMAGES}–{MAX_IMAGES} range)")

print(f"✅ Done! Saved {saved_images} images across {valid_subfolders} subfolders to {output_dir}.")