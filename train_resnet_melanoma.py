import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.ops import sigmoid_focal_loss

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
import optuna
from optuna.trial import Trial


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
        # Warning if unreadable or too small
        if image is None or image.shape[0] < 64 or image.shape[1] < 64:
            raise ValueError(f"Corrupted or unreadable image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # minor jitter
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),            # slight affine
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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=8, pin_memory=True)

torch.backends.cudnn.benchmark = True

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda device:")
print(torch.cuda.get_device_name(0))

weights = MobileNet_V3_Large_Weights.DEFAULT
model = mobilenet_v3_large(weights=weights)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)

# model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
# model.fc = nn.Linear(model.fc.in_features, 2)

for name, param in model.named_parameters():
    if "classifier" in name or "features.15" in name:  # last feature block
        param.requires_grad = True



model = model.to(device)

# Training setup
class_counts = train_df['type'].value_counts().to_dict()
total = sum(class_counts.values())
weights = [total / class_counts[i] for i in range(2)]

# the new sigmoid_focal_loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=2).float()
        return sigmoid_focal_loss(inputs, targets_one_hot, alpha=self.alpha, gamma=self.gamma, reduction="mean")

criterion = FocalLoss().to(device)

# criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# Early stopping settings
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')
patience = 3
counter = 0


# Load previous checkpoint if available: The next 4 lines + start_epoch in range.
# start_epoch = 17  # <-- Set to the last completed epoch number
# # Load previous checkpoint
# model.load_state_dict(torch.load("/work/ws-tmp/g062484-melo/images/latest_model_epoch17.pth"))
# print("✅ Loaded checkpoint from epoch", start_epoch)

# ----------------- OPTUNA OBJECTIVE FUNCTION -----------------
def objective(trial: Trial):
    # Sample hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)
    step_size = trial.suggest_int("step_size", 3, 10)

    # Model
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    for name, param in model.named_parameters():
        if "classifier" in name or "features.15" in name:
            param.requires_grad = True
    model = model.to(device)

    # Focal Loss
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.75, gamma=2):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=2).float()
            return sigmoid_focal_loss(inputs, targets_one_hot, alpha=self.alpha, gamma=self.gamma, reduction="mean")

    criterion = FocalLoss().to(device)

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scaler = torch.cuda.amp.GradScaler()

    # Training (short for tuning)
    model.train()
    for epoch in range(5):  # Short run per trial
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp


# Training loop with early stopping
# epochs = 20
# scaler = torch.cuda.amp.GradScaler()

# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
    

#     for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         with torch.cuda.amp.autocast():
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#         # Backward pass & optimizer update (AMP compatible)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         running_loss += loss.item()

#     avg_loss = running_loss / len(train_loader)
#     print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
#     torch.save(model.state_dict(), f"latest_model_epoch{epoch+1}.pth")

#     # Step the scheduler
#     scheduler.step()
    
#     # Early stopping validation
#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         best_model_wts = copy.deepcopy(model.state_dict())
#         counter = 0
#     else:
#         counter += 1
#         if counter >= patience:
#             print("Early stopping triggered.")
#             break

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

# Export to ONNX
onnx_export_path = f"/work/ws-tmp/g062484-melo/images/melanoma_mobilenetv3_{timestamp}.onnx"
dummy_input = torch.randn(1, 3, 512, 512).to(device)

torch.onnx.export(
    model,
    dummy_input,
    onnx_export_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print(f"✅ Exported ONNX model to {onnx_export_path}")

torch.save(model.state_dict(), f"/work/ws-tmp/g062484-melo/images/melanoma_model_final.pth")
print("✅ Saved final PyTorch model for GUI usage")

# ----------------- RUN OPTUNA -----------------
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    print("✅ Best Trial:")
    print(f"F1 Score: {study.best_trial.value}")
    print("Params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
# ------------------------------------------------
