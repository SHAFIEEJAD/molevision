import os
import cv2
import pandas as pd
from tqdm import tqdm

# Custom preprocessing function
def apply_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    gaussian_blur = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian_blur, -0.5, 0)
    return sharpened

# Paths
original_dir = "/work/ws-tmp/g062484-melo/images/images"
output_dir = "/work/ws-tmp/g062484-melo/images/preprocessed_512"
os.makedirs(output_dir, exist_ok=True)
metadata_path = "/work/ws-tmp/g062484-melo/images/cleaned_dataset.xlsx"

# Load metadata
df = pd.read_excel(metadata_path)[['image_id', 'type']].dropna()

# Process and save
for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing images"):
    image_path = os.path.join(original_dir, row['image_id'])
    output_path = os.path.join(output_dir, row['image_id'])

    if os.path.exists(output_path):
        continue  # Skip if already processed

    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠ Failed to load: {image_path}")
        continue

    image = apply_preprocessing(image)
    image = cv2.resize(image, (512, 512))  # Set target resolution
    cv2.imwrite(output_path, image)

