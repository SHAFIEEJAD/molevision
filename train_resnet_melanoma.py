import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import datetime


# Custom preprocessing filter
def apply_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    blur_metric = cv2.Laplacian(enhanced, cv2.CV_64F).var()
    if blur_metric < 100:
        print("⚠ Warning: Captured image may be blurry!")
    gaussian_blur = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian_blur, -0.5, 0)
    return sharpened

# Custom dataset with preprocessing and augmentation
class MelanomaDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_id'])
        image = cv2.imread(image_path)
        # image = apply_preprocessing(image)
        # image = cv2.resize(image, (224, 224))
        # image = np.stack([image]*3, axis=-1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
	if self.transform:
            image = self.transform(image)
        label = int(row['type'])
        return image, label

# Paths
image_dir = "/work/ws-tmp/g062484-melo/images/preprocessed_512"
metadata_path = "/work/ws-tmp/g062484-melo/images/cleaned_dataset.xlsx"
df = pd.read_excel(metadata_path)[['image_id', 'type']].dropna()

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['type'], random_state=42)

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Simpler transform for testing
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset and DataLoader
train_dataset = MelanomaDataset(train_df, image_dir, transform=train_transform)
test_dataset = MelanomaDataset(test_df, image_dir, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, pin_memory=True)

torch.backends.cudnn.benchmark = True

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda edevice:")
print(torch.cuda.get_device_name(0))
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Early stopping settings
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')
patience = 3
counter = 0

# Training loop with early stopping
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Early stopping validation
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Load best weights
model.load_state_dict(best_model_wts)

# Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# Metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['benign', 'malignant']))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Create output directory
plot_dir = "/work/ws-tmp/g062484-melo/images/plots"
os.makedirs(plot_dir, exist_ok=True)

# Generate timestamped filename
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plot_path = os.path.join(plot_dir, f"confusion_matrix_{timestamp}.png")



plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0, 1], ['benign', 'malignant'])
plt.yticks([0, 1], ['benign', 'malignant'])
plt.tight_layout()
plt.savefig(plot_path)
print(f"Saved confusion matrix to {plot_path}")




