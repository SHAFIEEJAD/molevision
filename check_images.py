import os
import pandas as pd

# Paths
directory = "/home/shafieej/Desktop/Project_work/DataSet/images/images"
xlsx_path = '/home/shafieej/Desktop/Project_work/DataSet/images/macroscopic_data_combined.xlsx'

# Load data
df = pd.read_excel(xlsx_path)

# Normalize column names just in case
df.columns = df.columns.str.strip().str.lower()

# Initialize lists
images_found = []
images_not_found = []

# Check image existence
for _, row in df.iterrows():
    image_name = row['image_id']
    image_path = os.path.join(directory, image_name)
    if os.path.isfile(image_path):
        images_found.append(row)
    else:
        images_not_found.append(image_name)

# Convert to DataFrames
found_df = pd.DataFrame(images_found)

# Optional: encode 'type' column (benign=0, malignant=1)
label_map = {'benign': 0, 'malignant': 1}
found_df['type'] = found_df['type'].str.strip().str.lower().map(label_map)

# Save cleaned data
found_df.to_excel('cleaned_dataset.xlsx', index=False)
print("✅ Cleaned dataset saved to 'cleaned_dataset.xlsx'.")

# Save missing image list (optional)
if images_not_found:
    pd.DataFrame(images_not_found, columns=['missing_image_id']).to_excel('missing_images.xlsx', index=False)
    print("⚠ Missing image list saved to 'missing_images.xlsx'.")

# Summary
print(f"📊 Total rows in original dataset: {len(df)}")
print(f"✅ Images found and retained: {len(found_df)}")
print(f"❌ Images not found and removed: {len(images_not_found)}")

