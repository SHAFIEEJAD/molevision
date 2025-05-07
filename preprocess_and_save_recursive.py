import os
import cv2
import pandas as pd
from tqdm import tqdm

def apply_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    gaussian_blur = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian_blur, -0.5, 0)
    return sharpened

# Paths
source_root = "/work/ws-tmp/g062484-melo/images/images"
target_root = "/work/ws-tmp/g062484-melo/images/preprocessed_512"
metadata_path = "/work/ws-tmp/g062484-melo/images/cleaned_dataset.xlsx"

os.makedirs(target_root, exist_ok=True)
df = pd.read_excel(metadata_path)[['image_id', 'type']].dropna()

for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing images"):
    relative_path = row['image_id']  # includes subdirectory, e.g., NEL/NEL025.JPG
    input_path = os.path.join(source_root, relative_path)
    output_path = os.path.join(target_root, relative_path)

    # Skip if already processed
    if os.path.exists(output_path):
        continue

    # Read image
    image = cv2.imread(input_path)
    if image is None:
        print(f"⚠ Failed to read: {input_path}")
        continue

    # Preprocess and resize
    image = apply_preprocessing(image)
    image = cv2.resize(image, (512, 512))

    # Create subdirectory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save processed image
    cv2.imwrite(output_path, image)
