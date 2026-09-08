import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from scripts.models import DistanceHead
from scripts.losses import CombinedDistanceLoss

class SimpleDataset(Dataset):
    def __init__(self, df, embeddings_dict):
        self.df = df
        self.v1 = df['node_A'].values
        self.v2 = df['node_B'].values
        self.targets = df['distance'].values
        self.embs = embeddings_dict
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.embs[self.v1[idx]], self.embs[self.v2[idx]], torch.tensor(self.targets[idx], dtype=torch.float32)

class HeadR0(nn.Module):
    # R0: Scalar Euclidean (||u - v||_2)
    def __init__(self, hidden_dim=2346):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, u, v):
        dist = torch.norm(u - v, p=2, dim=-1, keepdim=True)
        return self.mlp(dist).squeeze(-1)

class HeadR1(nn.Module):
    # R1: Symmetric ([|u - v|, u * v] + Softplus)
    def __init__(self, emb_dim=1280, hidden_dim=932):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, u, v):
        diff = torch.abs(u - v)
        mult = u * v
        x = torch.cat([diff, mult], dim=-1)
        return self.mlp(x).squeeze(-1)

class HeadR2(nn.Module):
    # R2: Directional ([u, v, u - v, u * v])
    def __init__(self, emb_dim=1280, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, u, v):
        diff = u - v
        mult = u * v
        x = torch.cat([u, v, diff, mult], dim=-1)
        return self.mlp(x).squeeze(-1)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_nn(model, dataloader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for b in dataloader:
            emb1, emb2, dist = b
            emb1, emb2 = emb1.to(device), emb2.to(device)
            p1 = model(emb1, emb2)
            p2 = model(emb2, emb1)
            p = torch.clamp((p1 + p2) / 2.0, min=0.0)
            preds.extend(p.cpu().numpy())
            tgts.extend(dist.numpy())
    preds = np.array(preds)
    tgts = np.array(tgts)
    mae = mean_absolute_error(tgts, preds)
    rho, _ = spearmanr(tgts, preds)
    return float(mae), float(rho)

def train_nn(model, train_loader, val_loader, test_loader, epochs, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    
    best_val_mae = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for emb1, emb2, dist in train_loader:
            emb1, emb2, dist = emb1.to(device), emb2.to(device), dist.to(device)
            optimizer.zero_grad()
            preds = model(emb1, emb2)
            loss = criterion(preds, dist)
            loss.backward()
            optimizer.step()
            
        val_mae, val_rho = evaluate_nn(model, val_loader, device)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_mae, test_rho = evaluate_nn(model, test_loader, device)
    return test_mae, test_rho

def run_classical_baselines(train_df, test_df, virus_to_seq):
    # C0-a: Full sequence mutation count
    # C0-b: 7 key positions mutation count
    koel_sites = {145, 155, 156, 158, 159, 189, 193}
    
    def get_features(df):
        X_a = []
        X_b = []
        Y = []
        valid_idx = []
        for idx, row in df.iterrows():
            vA = row['node_A']
            vB = row['node_B']
            if vA in virus_to_seq and vB in virus_to_seq:
                seqA = virus_to_seq[vA]
                seqB = virus_to_seq[vB]
                min_len = min(len(seqA), len(seqB))
                
                # C0-a
                mut_all = sum(1 for i in range(min_len) if seqA[i] != seqB[i])
                
                # C0-b
                mut_key = sum(1 for p in koel_sites if p-1 < min_len and seqA[p-1] != seqB[p-1])
                
                X_a.append([mut_all])
                X_b.append([mut_key])
                Y.append(row['distance'])
                valid_idx.append(idx)
        return np.array(X_a), np.array(X_b), np.array(Y)
    
    X_train_a, X_train_b, Y_train = get_features(train_df)
    X_test_a, X_test_b, Y_test = get_features(test_df)
    
    reg_a = Ridge(alpha=1.0)
    reg_a.fit(X_train_a, Y_train)
    preds_a = reg_a.predict(X_test_a)
    mae_a = mean_absolute_error(Y_test, preds_a)
    rho_a, _ = spearmanr(Y_test, preds_a)
    
    reg_b = Ridge(alpha=1.0)
    reg_b.fit(X_train_b, Y_train)
    preds_b = reg_b.predict(X_test_b)
    mae_b = mean_absolute_error(Y_test, preds_b)
    rho_b, _ = spearmanr(Y_test, preds_b)
    
    return (mae_a, rho_a), (mae_b, rho_b)

def run_c1_baseline(train_df, test_df, mean_embeddings):
    def get_features(df):
        X = []
        Y = []
        for idx, row in df.iterrows():
            vA = row['node_A']
            vB = row['node_B']
            if vA in mean_embeddings and vB in mean_embeddings:
                embA = mean_embeddings[vA].numpy()
                embB = mean_embeddings[vB].numpy()
                X.append(np.concatenate([embA, embB, np.abs(embA-embB), embA*embB]))
                Y.append(row['distance'])
        return np.array(X), np.array(Y)
    
    X_train, Y_train = get_features(train_df)
    X_test, Y_test = get_features(test_df)
    
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, Y_train)
    preds = reg.predict(X_test)
    mae = mean_absolute_error(Y_test, preds)
    rho, _ = spearmanr(Y_test, preds)
    
    return mae, rho

def main():
    device = torch.device('cpu') 
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    splits_dir = os.path.join(base_dir, 'data', 'splits')
    emb_path = os.path.join(base_dir, 'data', 'processed', 'embeddings_esm2_650m.pt')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(splits_dir, 'cohort2_train.csv.gz'))
    val_df = pd.read_csv(os.path.join(splits_dir, 'cohort2_val.csv.gz'))
    test_df = pd.read_csv(os.path.join(splits_dir, 'cohort2_test.csv.gz'))
    
    sys.path.append(os.path.join(base_dir, 'scripts'))
    import importlib.util
    spec = importlib.util.spec_from_file_location("extract", os.path.join(base_dir, 'scripts', '04_extract_embeddings.py'))
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)
    virus_to_seq = extract.get_virus_to_seq_mapping()
    
    mean_embeddings = torch.load(emb_path, map_location='cpu')
    
    common = set(mean_embeddings.keys()).intersection(set(virus_to_seq.keys()))
    train_df = train_df[train_df['node_A'].isin(common) & train_df['node_B'].isin(common)].reset_index(drop=True)
    val_df = val_df[val_df['node_A'].isin(common) & val_df['node_B'].isin(common)].reset_index(drop=True)
    test_df = test_df[test_df['node_A'].isin(common) & test_df['node_B'].isin(common)].reset_index(drop=True)
    
    results = {}
    
    print("Running Stage 0: Classical Baselines...")
    (mae_a, rho_a), (mae_b, rho_b) = run_classical_baselines(train_df, test_df, virus_to_seq)
    mae_c1, rho_c1 = run_c1_baseline(train_df, test_df, mean_embeddings)
    
    results["C0-a"] = {"Trainable_Parameters": 1, "Test_MAE": round(mae_a, 4), "Test_Spearman_rho": round(rho_a, 4)}
    results["C0-b"] = {"Trainable_Parameters": 1, "Test_MAE": round(mae_b, 4), "Test_Spearman_rho": round(rho_b, 4)}
    results["C1"] = {"Trainable_Parameters": 5120, "Test_MAE": round(mae_c1, 4), "Test_Spearman_rho": round(rho_c1, 4)}
    
    print("Running Stage A: Pair Representation Ablation...")
    train_ds = SimpleDataset(train_df, mean_embeddings)
    val_ds = SimpleDataset(val_df, mean_embeddings)
    test_ds = SimpleDataset(test_df, mean_embeddings)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    models = {
        "R0": HeadR0(),
        "R1": HeadR1(),
        "R2": HeadR2()
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        params = count_params(model)
        mae, rho = train_nn(model, train_loader, val_loader, test_loader, epochs=8, device=device)
        results[name] = {"Trainable_Parameters": params, "Test_MAE": round(mae, 4), "Test_Spearman_rho": round(rho, 4)}
    
    out_path = os.path.join(reports_dir, 'stage_0_and_A_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Execution completed. Output saved to", out_path)

if __name__ == "__main__":
    main()
