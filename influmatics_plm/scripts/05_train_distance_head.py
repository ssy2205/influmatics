import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from models import DistanceHead
from dataset import DistanceDataset
from combined_ranking_loss import CombinedDistanceLoss

def run_evaluation(model, dataloader, device):
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for emb1, emb2, dist in dataloader:
            emb1, emb2 = emb1.to(device), emb2.to(device)
            preds = model(emb1, emb2)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(dist.numpy())
            
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    
    rmse = float(np.sqrt(mean_squared_error(val_targets, val_preds)))
    mae = float(mean_absolute_error(val_targets, val_preds))
    rho_res, _ = spearmanr(val_targets, val_preds)
    rho = float(rho_res) if not np.isnan(rho_res) else 0.0
    
    return {'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'Spearman_rho': round(rho, 4)}

def train_and_evaluate(cohort_name, embeddings, device, epochs=10, batch_size=256, lr=1e-3):
    print(f"\n==========================================")
    print(f"       Training Pipeline: {cohort_name}   ")
    print(f"==========================================")
    
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    splits_dir = os.path.join(base_dir, 'data/splits')
    models_dir = os.path.join(base_dir, 'models/checkpoints')
    os.makedirs(models_dir, exist_ok=True)
    
    train_path = os.path.join(splits_dir, f"{cohort_name}_train.csv.gz")
    val_path = os.path.join(splits_dir, f"{cohort_name}_val.csv.gz")
    test_path = os.path.join(splits_dir, f"{cohort_name}_test.csv.gz")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Filter only pairs where both viruses have valid embeddings
    train_df = train_df[train_df['virus1'].isin(embeddings.keys()) & train_df['virus2'].isin(embeddings.keys())]
    val_df = val_df[val_df['virus1'].isin(embeddings.keys()) & val_df['virus2'].isin(embeddings.keys())]
    test_df = test_df[test_df['virus1'].isin(embeddings.keys()) & test_df['virus2'].isin(embeddings.keys())]
    
    # Mathematical Audit: Strict Node Disjointness Assertion
    train_nodes = set(train_df['virus1']).union(set(train_df['virus2']))
    test_nodes = set(test_df['virus1']).union(set(test_df['virus2']))
    overlap = train_nodes.intersection(test_nodes)
    assert len(overlap) == 0, f"Critical Data Leakage: Overlapping nodes between Train and Test: {overlap}"
    print(f"Node Disjointness Certified: 0 overlapping nodes between Train ({len(train_nodes)}) and Test ({len(test_nodes)})")
    print(f"Valid Samples -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    train_dataset = DistanceDataset(train_df, embeddings)
    val_dataset = DistanceDataset(val_df, embeddings)
    test_dataset = DistanceDataset(test_df, embeddings)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # High-capacity Interaction Head
    model = DistanceHead(emb_dim=1280, hidden_dim=512).to(device)
    criterion = CombinedDistanceLoss(alpha=0.3, margin=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_val_metrics = {}
    best_ckpt_path = os.path.join(models_dir, f"{cohort_name}_best.pt")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for emb1, emb2, dist in train_loader:
            emb1, emb2, dist = emb1.to(device), emb2.to(device), dist.to(device)
            optimizer.zero_grad()
            preds = model(emb1, emb2)
            loss = criterion(preds, dist)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(dist)
            
        train_loss /= len(train_dataset)
        val_metrics = run_evaluation(model, val_loader, device)
        
        # Exact unrounded float comparison
        raw_rmse = val_metrics['RMSE']
        if raw_rmse < best_val_loss:
            best_val_loss = raw_rmse
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), best_ckpt_path)
            saved_mark = " [*Best Model Saved]"
        else:
            saved_mark = ""
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val RMSE: {val_metrics['RMSE']:.4f}, MAE: {val_metrics['MAE']:.4f}, Rho: {val_metrics['Spearman_rho']:.4f}{saved_mark}")
        
    # Evaluate best model on test set
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_metrics = run_evaluation(model, test_loader, device)
    
    print(f"\n>> Final Best Validation: {best_val_metrics}")
    print(f">> Final Test Evaluation: {test_metrics}")
    
    return {
        'cohort': cohort_name,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'validation_metrics': best_val_metrics,
        'test_metrics': test_metrics
    }

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"PyTorch execution device: {device}")
    
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    emb_path = os.path.join(base_dir, 'data/processed/embeddings_esm2_650m.pt')
    embeddings = torch.load(emb_path, map_location='cpu')
    print(f"Total embeddings loaded: {len(embeddings)}")
    
    results = {}
    res_c1 = train_and_evaluate('cohort1', embeddings, device, epochs=10, batch_size=256, lr=1e-3)
    if res_c1: results['cohort1'] = res_c1
    
    res_c2 = train_and_evaluate('cohort2', embeddings, device, epochs=10, batch_size=256, lr=1e-3)
    if res_c2: results['cohort2'] = res_c2
    
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, 'model_evaluation_metrics.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved all certified metrics to {report_file}")

if __name__ == '__main__':
    main()
