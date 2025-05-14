import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# --- Config ---
CSV_PATH   = 'final_train_data.csv'
IMAGE_BASE = '/home/data_shares/purrlab/CheXpert/'
IMG_SIZE   = (128, 128)
BATCH_SIZE = 128
EPOCHS     = 10
N_FOLDS    = 5
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Calibration metrics (same logic as before) ---
def compute_calibration_metrics(y_true, y_prob, n_bins=5):
    bin_edges = np.linspace(0, 1, n_bins+1)
    ece = mce = 0.0
    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)
    accs = (pred == y_true).astype(float)
    idx = np.digitize(conf, bin_edges, right=True) - 1
    idx = np.clip(idx, 0, n_bins-1)
    for i in range(n_bins):
        mask = idx == i
        if not mask.any(): continue
        bin_acc  = accs[mask].mean()
        bin_conf = conf[mask].mean()
        weight   = mask.sum() / len(y_true)
        ece += weight * abs(bin_acc - bin_conf)
        mce = max(mce, abs(bin_acc - bin_conf))
    brier = np.mean([
        brier_score_loss((y_true == i).astype(int), y_prob[:, i])
        for i in range(y_prob.shape[1])
    ])
    return ece, mce, brier

# --- Dataset ---
class PleuralDataset(Dataset):
    def __init__(self, paths, tabular, labels, base_path, transform):
        self.paths     = paths
        self.tabular   = tabular
        self.labels    = labels
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.base_path, self.paths[idx])).convert('L')
        img = self.transform(img)               # [1, H, W], floats [0,1]
        tab = torch.from_numpy(self.tabular[idx])
        lbl = int(self.labels[idx])
        return img, tab, lbl

# --- Model ---
class MultiInputResNet50(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        # Load a ResNet50 backbone
        self.cnn = models.resnet50(pretrained=False)
        # Adapt first conv to 1-channel input
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove final fc
        self.cnn.fc = nn.Identity()

        # Tabular MLP
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

    def forward(self, img, tab):
        # img → features
        x_img = self.cnn(img)                    # [B, 2048]
        # tab → features
        x_tab = self.tab_mlp(tab)                # [B, 64]
        # concat & classify
        x = torch.cat([x_img, x_tab], dim=1)     # [B, 2112]
        return self.classifier(x)                # [B, 3]

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),  # scales to [0,1] and gives [1,H,W]
])

# --- Load & preprocess CSV ---
df = pd.read_csv(CSV_PATH)
df = df[df['Pleural Effusion'].notna()]
df = pd.get_dummies(df, columns=['Sex','Frontal/Lateral','AP/PA'])

feature_cols = (
    ['Age','Sex_Male','Sex_Female','Sex_Unknown']
    + [c for c in df if c.startswith('corner')]
    + [c for c in df if c.startswith('hist')]
)
tab_array = df[feature_cols].values.astype(np.float32)
tab_array = StandardScaler().fit_transform(tab_array)

labels = df['Pleural Effusion'].values
labels = np.where(labels == -1, 2, labels).astype(np.int64)
paths  = df['Path'].astype(str).values

# --- 5-fold Stratified CV ---
kf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
results = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(paths, labels), start=1):
    print(f"\n=== Fold {fold}/{N_FOLDS} ===")

    # Split data
    p_tr, p_va = paths[tr_idx], paths[va_idx]
    t_tr, t_va = tab_array[tr_idx], tab_array[va_idx]
    y_tr, y_va = labels[tr_idx], labels[va_idx]

    # DataLoaders
    train_ds = PleuralDataset(p_tr, t_tr, y_tr, IMAGE_BASE, transform)
    val_ds   = PleuralDataset(p_va, t_va, y_va, IMAGE_BASE, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model, loss, optimizer
    model     = MultiInputResNet50(tab_dim=tab_array.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    start = time.time()
    model.train()
    for epoch in range(EPOCHS):
        for imgs, tabs, labs in train_loader:
            imgs = imgs.to(DEVICE)
            tabs = tabs.to(DEVICE)
            labs = labs.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs, tabs)
            loss   = criterion(logits, labs)
            loss.backward()
            optimizer.step()
    elapsed = (time.time() - start) / 60

    # Evaluate
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, tabs, labs in val_loader:
            imgs = imgs.to(DEVICE)
            tabs = tabs.to(DEVICE)
            logits = model(imgs, tabs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labs.numpy())

    y_prob = np.vstack(all_probs)
    y_pred = np.hstack(all_preds)
    y_true = np.hstack(all_labels)

    # Metrics
    auc  = roc_auc_score(y_true, y_prob, multi_class='ovo')
    acc  = accuracy_score(y_true, y_pred)
    ece, mce, brier = compute_calibration_metrics(y_true, y_prob)

    results.append({
        'Fold':         fold,
        'AUC':          round(auc,3),
        'Accuracy':     round(acc,3),
        'ECE':          round(ece,4),
        'MCE':          round(mce,4),
        'Brier Score':  round(brier,4),
        'Time (min)':   round(elapsed,2)
    })

# --- Save results ---
results_df = pd.DataFrame(results)
results_df.to_csv("results_exp1_pytorch.csv", index=False)
print("Saved results to results_exp1_pytorch.csv")
