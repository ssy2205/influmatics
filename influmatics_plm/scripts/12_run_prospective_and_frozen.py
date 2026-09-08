import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from models import DistanceHead
from losses import CombinedDistanceLoss

class StandardPairDataset(Dataset):
    def __init__(self, edge_df, embeddings_dict):
        self.df = edge_df.reset_index(drop=True)
        self.embeddings = embeddings_dict
        self.v1 = self.df['node_A'].values
        self.v2 = self.df['node_B'].values
        self.targets = self.df['distance'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        emb1 = self.embeddings[self.v1[idx]]
        emb2 = self.embeddings[self.v2[idx]]
        dist = torch.tensor(self.targets[idx], dtype=torch.float32)
        return emb1, emb2, dist

def parse_year_cohort2(name):
    parts = str(name).strip().split('/')
    m = re.search(r'(\d+)$', parts[-1])
    if m:
        yy = int(m.group(1))
        if yy >= 1000:
            return yy
        return 2000 + yy if yy < 50 else 1900 + yy
    return None

def train_and_eval(train_df, test_df, model, criterion, optimizer, device, epochs=1):
    train_ds = StandardPairDataset(train_df, embeddings)
    test_ds = StandardPairDataset(test_df, embeddings)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    model.train()
    for ep in range(epochs):
        for emb1, emb2, dist in train_loader:
            emb1, emb2, dist = emb1.to(device), emb2.to(device), dist.to(device)
            optimizer.zero_grad()
            preds = model(emb1, emb2)
            loss = criterion(preds, dist)
            loss.backward()
            optimizer.step()
            
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for emb1, emb2, dist in test_loader:
            emb1, emb2 = emb1.to(device), emb2.to(device)
            preds = model(emb1, emb2)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(dist.numpy())
            
    mae = mean_absolute_error(val_targets, val_preds)
    return float(mae)

base_dir = os.path.abspath(os.path.join(script_dir, '..'))
emb_path = os.path.join(base_dir, 'data/processed/embeddings_esm2_650m.pt')
embeddings = torch.load(emb_path, map_location='cpu')

df = pd.read_csv(os.path.join(base_dir, 'data/processed/cohort2_pairwise.csv.gz'))
df['year_A'] = df['virus1'].map(parse_year_cohort2)
df = df.dropna(subset=['year_A'])
df = df[df['virus1'].isin(embeddings.keys()) & df['virus2'].isin(embeddings.keys())].reset_index(drop=True)
df['node_A'] = df['virus1']
df['node_B'] = df['virus2']

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

rolling_results = {}
for test_year in [2019, 2020, 2021, 2022]:
    train_df = df[df['year_A'] < test_year]
    test_df = df[df['year_A'] == test_year]
    if len(test_df) == 0: continue
    model = DistanceHead(emb_dim=1280, hidden_dim=512).to(device)
    criterion = CombinedDistanceLoss(alpha=0.3, margin=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mae = train_and_eval(train_df, test_df, model, criterion, optimizer, device, epochs=1)
    rolling_results[str(test_year)] = round(mae, 4)

frozen_results = {}
train_frozen_df = df[df['year_A'] <= 2018]
frozen_model = DistanceHead(emb_dim=1280, hidden_dim=512).to(device)
criterion = CombinedDistanceLoss(alpha=0.3, margin=0.1)
optimizer = torch.optim.AdamW(frozen_model.parameters(), lr=1e-3)

train_ds = StandardPairDataset(train_frozen_df, embeddings)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
frozen_model.train()
for ep in range(1):
    for emb1, emb2, dist in train_loader:
        emb1, emb2, dist = emb1.to(device), emb2.to(device), dist.to(device)
        optimizer.zero_grad()
        preds = frozen_model(emb1, emb2)
        loss = criterion(preds, dist)
        loss.backward()
        optimizer.step()

frozen_model.eval()
for test_year in [2019, 2020, 2021, 2022]:
    test_df = df[df['year_A'] == test_year]
    if len(test_df) == 0: continue
    test_ds = StandardPairDataset(test_df, embeddings)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    val_preds, val_targets = [], []
    with torch.no_grad():
        for emb1, emb2, dist in test_loader:
            emb1, emb2 = emb1.to(device), emb2.to(device)
            preds = frozen_model(emb1, emb2)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(dist.numpy())
    if len(val_targets) > 0:
        mae = mean_absolute_error(val_targets, val_preds)
        frozen_results[str(test_year)] = round(float(mae), 4)

results = {
    "Rolling_Prospective_MAE": rolling_results,
    "Frozen_Future_MAE": frozen_results
}

reports_dir = os.path.join(base_dir, 'reports')
with open(os.path.join(reports_dir, 'prospective_frozen_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("Saved Prospective and Frozen results.")
