import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import os, cv2, datetime, copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna

# Dataset
df = pd.read_excel("/work/ws-tmp/g062484-melo/images/cleaned_dataset.xlsx")[['image_id', 'type']].dropna()
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['type'], random_state=42)
image_dir = "/work/ws-tmp/g062484-melo/images/preprocessed_512"

# Data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.RandomAffine(degrees=0, translate=(0.03,0.03)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# Dataset class
class MelanomaDataset(Dataset):
    def __init__(self, df, dir, transform):
        self.df, self.dir, self.transform = df, dir, transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.df.iloc[idx]['image_id'])
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return self.transform(img), int(self.df.iloc[idx]['type'])

train_loader = DataLoader(MelanomaDataset(train_df, image_dir, train_transform), batch_size=32, shuffle=True, num_workers=8)
test_loader = DataLoader(MelanomaDataset(test_df, image_dir, test_transform), batch_size=32, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Focal loss class (✅ Recommended)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        one_hot = torch.nn.functional.one_hot(targets, num_classes=2).float()
        return sigmoid_focal_loss(inputs, one_hot, alpha=self.alpha, gamma=self.gamma, reduction='mean')

# Optuna hyperparameter tuning (✅ Recommended)
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    wd = trial.suggest_loguniform('wd', 1e-6, 1e-3)
    gamma = trial.suggest_float('gamma', 0.1, 0.9)
    step = trial.suggest_int('step', 3, 10)

    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features,2)
    model.to(device)

    criterion = FocalLoss().to(device)
    optimiz = optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    schedul = optim.lr_scheduler.StepLR(optimiz,step,gamma)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for _ in range(5):
        for X,y in train_loader:
            X,y = X.to(device),y.to(device)
            optimiz.zero_grad()
            with torch.cuda.amp.autocast():
                loss = criterion(model(X),y)
            scaler.scale(loss).backward()
            scaler.step(optimiz)
            scaler.update()
        schedul.step()

    model.eval()
    preds,true = [],[]
    with torch.no_grad():
        for X,y in test_loader:
            preds += model(X.to(device)).argmax(1).cpu().tolist()
            true += y.tolist()

    f1 = f1_score(true, preds)
    print(f"Trial {trial.number}: F1={f1:.4f}, Params={trial.params}")
    return f1

study=optuna.create_study(direction='maximize')
study.optimize(objective,n_trials=10)
print("Best trial:",study.best_trial.params)

# Save Optuna results explicitly
study.trials_dataframe().to_csv('optuna_results.csv', index=False)
print("✅ Optuna results saved to optuna_results.csv")

# Use best params
params = study.best_trial.params
model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features,2)
model.to(device)
criterion = FocalLoss().to(device)
optimizer = optim.Adam(model.parameters(),lr=params['lr'],weight_decay=params['wd'])
scheduler = optim.lr_scheduler.StepLR(optimizer,params['step'],params['gamma'])
scaler = torch.cuda.amp.GradScaler()

# Final training (✅ Recommended)
for epoch in range(20):
    model.train()
    for X,y in tqdm(train_loader):
        X,y=X.to(device),y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss=criterion(model(X),y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()

# Final save
stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M")
torch.save(model.state_dict(),f"final_{stamp}.pth")
torch.onnx.export(model,torch.randn(1,3,512,512).to(device),f"final_{stamp}.onnx",input_names=['input'],output_names=['output'],opset_version=11)
print("✅ All done")
