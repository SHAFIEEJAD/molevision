import os
import cv2
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def apply_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    gaussian_blur = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian_blur, -0.5, 0)
    return sharpened

def process_image(row):
    image_id = row['image_id']
    input_path = os.path.join(IMAGE_DIR, image_id)
    output_path = os.path.join(OUTPUT_DIR, image_id)

    if os.path.exists(output_path):
        return

    image = cv2.imread(input_path)
    if image is None:
        print(f"⚠ Failed to read {input_path}")
        return

    image = apply_preprocessing(image)
    image = cv2.resize(image, (512, 512))
    cv2.imwrite(output_path, image)

# Global paths
IMAGE_DIR = "/work/ws-tmp/g062484-melo/images/images"
OUTPUT_DIR = "/work/ws-tmp/g062484-melo/images/preprocessed_512"
os.makedirs(OUTPUT_DIR, exist_ok=True)
metadata_path = "/work/ws-tmp/g062484-melo/images/cleaned_dataset.xlsx"

df = pd.read_excel(metadata_path)[['image_id', 'type']].dropna()

print(f"Using {cpu_count()} CPU cores...")
with Pool(processes=cpu_count()) as pool:
    list(tqdm(pool.imap_unordered(process_image, [row for _, row in df.iterrows()]), total=len(df)))

