import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))
plm_dir = os.path.abspath(os.path.join(script_dir, '..'))
if plm_dir not in sys.path:
    sys.path.append(plm_dir)

from scripts.combined_ranking_loss import CombinedRankingLoss

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class HeadR1(nn.Module):
    """
    Standard Symmetric Head R1.
    Input: [|u - v|, u * v] (dim: emb_dim * 2)
    """
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

class HeadR1WithGly(nn.Module):
    """
    Stage C: Symmetric Head R1 augmented with 3D Glycosylation features.
    Input: [|u - v|, u * v, gly_features] (dim: emb_dim * 2 + 3)
    gly_features = [gain, loss, shared]
    """
    def __init__(self, emb_dim=1280, hidden_dim=932, gly_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + gly_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, u, v, gly_feat):
        diff = torch.abs(u - v)
        mult = u * v
        x = torch.cat([diff, mult, gly_feat], dim=-1)
        return self.mlp(x).squeeze(-1)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class IndexPairDataset(Dataset):
    def __init__(self, df, virus_to_idx):
        self.idx_A = torch.tensor([virus_to_idx[v] for v in df['node_A']], dtype=torch.long)
        self.idx_B = torch.tensor([virus_to_idx[v] for v in df['node_B']], dtype=torch.long)
        self.targets = torch.tensor(df['distance'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.idx_A[idx], self.idx_B[idx], self.targets[idx]

# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------
def evaluate_p0(model, emb_matrix_device, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for idx_A, idx_B, dist in loader:
            idx_A, idx_B = idx_A.to(device), idx_B.to(device)
            emb1 = emb_matrix_device[idx_A]
            emb2 = emb_matrix_device[idx_B]
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

def evaluate_p0_gly(model, emb_matrix_device, gly_matrix_device, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for idx_A, idx_B, dist in loader:
            idx_A, idx_B = idx_A.to(device), idx_B.to(device)
            emb1 = emb_matrix_device[idx_A]
            emb2 = emb_matrix_device[idx_B]
            gly_AB = gly_matrix_device[idx_A, idx_B]
            gly_BA = gly_matrix_device[idx_B, idx_A]
            p1 = model(emb1, emb2, gly_AB)
            p2 = model(emb2, emb1, gly_BA)
            p = torch.clamp((p1 + p2) / 2.0, min=0.0)
            preds.extend(p.cpu().numpy())
            tgts.extend(dist.numpy())
    preds = np.array(preds)
    tgts = np.array(tgts)
    mae = mean_absolute_error(tgts, preds)
    rho, _ = spearmanr(tgts, preds)
    return float(mae), float(rho)

# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------
def train_and_eval_p0(emb_matrix_device, train_df, val_df, test_df, virus_to_idx, epochs, device, criterion):
    ds_train = IndexPairDataset(train_df, virus_to_idx)
    ds_val = IndexPairDataset(val_df, virus_to_idx)
    loader_train = DataLoader(ds_train, batch_size=256, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=256, shuffle=False)

    model = HeadR1(emb_dim=1280, hidden_dim=932).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_mae = float('inf')
    best_state = None
    best_val_rho = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for idx_A, idx_B, dist in loader_train:
            idx_A, idx_B, dist = idx_A.to(device), idx_B.to(device), dist.to(device)
            emb1 = emb_matrix_device[idx_A]
            emb2 = emb_matrix_device[idx_B]
            optimizer.zero_grad()
            preds = model(emb1, emb2)
            loss = criterion(preds, dist)
            loss.backward()
            optimizer.step()

        val_mae, val_rho = evaluate_p0(model, emb_matrix_device, loader_val, device)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rho = val_rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_mae, test_rho = None, None
    if test_df is not None:
        ds_test = IndexPairDataset(test_df, virus_to_idx)
        loader_test = DataLoader(ds_test, batch_size=256, shuffle=False)
        test_mae, test_rho = evaluate_p0(model, emb_matrix_device, loader_test, device)

    return best_val_mae, best_val_rho, test_mae, test_rho

def train_and_eval_p0_gly(emb_matrix_device, gly_matrix_device, train_df, val_df, test_df, virus_to_idx, epochs, device):
    ds_train = IndexPairDataset(train_df, virus_to_idx)
    ds_val = IndexPairDataset(val_df, virus_to_idx)
    loader_train = DataLoader(ds_train, batch_size=256, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=256, shuffle=False)

    model = HeadR1WithGly(emb_dim=1280, hidden_dim=932, gly_dim=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    best_val_mae = float('inf')
    best_state = None
    best_val_rho = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for idx_A, idx_B, dist in loader_train:
            idx_A, idx_B, dist = idx_A.to(device), idx_B.to(device), dist.to(device)
            emb1 = emb_matrix_device[idx_A]
            emb2 = emb_matrix_device[idx_B]
            gly_AB = gly_matrix_device[idx_A, idx_B]
            optimizer.zero_grad()
            preds = model(emb1, emb2, gly_AB)
            loss = criterion(preds, dist)
            loss.backward()
            optimizer.step()

        val_mae, val_rho = evaluate_p0_gly(model, emb_matrix_device, gly_matrix_device, loader_val, device)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rho = val_rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_mae, test_rho = None, None
    if test_df is not None:
        ds_test = IndexPairDataset(test_df, virus_to_idx)
        loader_test = DataLoader(ds_test, batch_size=256, shuffle=False)
        test_mae, test_rho = evaluate_p0_gly(model, emb_matrix_device, gly_matrix_device, loader_test, device)

    return best_val_mae, best_val_rho, test_mae, test_rho

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device for Stage C and Stage D Execution: {device}", flush=True)

    data_dir = os.path.join(plm_dir, 'data')
    splits_dir = os.path.join(data_dir, 'splits')
    processed_dir = os.path.join(data_dir, 'processed')
    reports_dir = os.path.join(plm_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    cache_path = os.path.join(processed_dir, 'stage_B_cache.pt')
    glyco_path = os.path.join(processed_dir, 'glyco_counts.json')

    print("Loading cached embeddings and glycosylation annotations...", flush=True)
    cache = torch.load(cache_path, weights_only=False)
    with open(glyco_path, 'r') as f:
        glyco = json.load(f)

    unique_viruses = sorted(list(cache['global'].keys()))
    virus_to_idx = {v: i for i, v in enumerate(unique_viruses)}
    N = len(unique_viruses)
    print(f"Loaded {N} unique viruses.", flush=True)

    # Global Mean Embeddings on Device
    mat_P0_device = torch.stack([cache['global'][v] for v in unique_viruses], dim=0).to(device)

    # Build Pairwise Glycosylation Matrix: (N, N, 3)
    # Channel 0: gly_gain_count (|S_B \ S_A|)
    # Channel 1: gly_loss_count (|S_A \ S_B|)
    # Channel 2: shared_gly_count (|S_A & S_B|)
    site_sets = [set(glyco[v]['sites']) for v in unique_viruses]
    glyco_pair_np = np.zeros((N, N, 3), dtype=np.float32)
    for i in range(N):
        s_i = site_sets[i]
        for j in range(N):
            s_j = site_sets[j]
            glyco_pair_np[i, j, 0] = len(s_j - s_i)
            glyco_pair_np[i, j, 1] = len(s_i - s_j)
            glyco_pair_np[i, j, 2] = len(s_i & s_j)
    gly_matrix_device = torch.tensor(glyco_pair_np, dtype=torch.float32, device=device)
    print("Precomputed pairwise glycosylation tensor (N, N, 3) on device.", flush=True)

    def filter_df(df):
        if 'node_A' not in df.columns:
            df['node_A'] = df['virus1']
        if 'node_B' not in df.columns:
            df['node_B'] = df['virus2']
        return df[df['node_A'].isin(virus_to_idx) & df['node_B'].isin(virus_to_idx)].reset_index(drop=True)

    folds_data = []
    for f in [1, 2, 3]:
        f_tr = filter_df(pd.read_csv(os.path.join(splits_dir, f'cohort2_fold{f}_train.csv.gz')))
        f_va = filter_df(pd.read_csv(os.path.join(splits_dir, f'cohort2_fold{f}_val.csv.gz')))
        folds_data.append((f_tr, f_va))

    train_full = filter_df(pd.read_csv(os.path.join(splits_dir, 'cohort2_train.csv.gz')))
    val_full = filter_df(pd.read_csv(os.path.join(splits_dir, 'cohort2_val.csv.gz')))
    test_full = filter_df(pd.read_csv(os.path.join(splits_dir, 'cohort2_test.csv.gz')))

    epochs = 8

    results = {
        "Stage_C": {
            "P0_Baseline": {
                "Description": "P0: Frozen ESM-2 Global Mean + R1 Symmetric Head + Smooth L1",
                "Trainable_Parameters": count_params(HeadR1(1280, 932)),
                "Temporal_CV": {},
                "Test_Evaluation": {}
            },
            "P0_Plus_Gly": {
                "Description": "P0 + 3D Glycosylation Features ([gain, loss, shared]) + R1 Head + Smooth L1",
                "Trainable_Parameters": count_params(HeadR1WithGly(1280, 932, 3)),
                "Temporal_CV": {},
                "Test_Evaluation": {}
            }
        },
        "Stage_D": {}
    }

    # =========================================================================
    # STAGE C: N-Glycosylation Feature Ablation
    # =========================================================================
    print("\n=======================================================", flush=True)
    print("   STAGE C: N-Glycosylation Feature Ablation", flush=True)
    print("=======================================================", flush=True)

    # 1. Evaluate P0 Baseline across folds
    smooth_l1 = nn.SmoothL1Loss()
    for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
        v_mae, v_rho, _, _ = train_and_eval_p0(
            mat_P0_device, f_tr, f_va, None, virus_to_idx, epochs, device, smooth_l1
        )
        results["Stage_C"]["P0_Baseline"]["Temporal_CV"][f"Fold_{f_idx}"] = {
            "MAE": round(v_mae, 4), "Spearman_rho": round(v_rho, 4)
        }
        print(f"Stage C | Fold {f_idx} | P0 Baseline -> Val MAE: {v_mae:.4f}, Rho: {v_rho:.4f}", flush=True)

    macro_mae_p0 = np.mean([results["Stage_C"]["P0_Baseline"]["Temporal_CV"][f"Fold_{i}"]["MAE"] for i in [1, 2, 3]])
    macro_rho_p0 = np.mean([results["Stage_C"]["P0_Baseline"]["Temporal_CV"][f"Fold_{i}"]["Spearman_rho"] for i in [1, 2, 3]])
    results["Stage_C"]["P0_Baseline"]["Temporal_CV"]["Macro_MAE"] = round(float(macro_mae_p0), 4)
    results["Stage_C"]["P0_Baseline"]["Temporal_CV"]["Macro_Spearman_rho"] = round(float(macro_rho_p0), 4)
    print(f">> Stage C | P0 Baseline Macro CV -> MAE: {macro_mae_p0:.4f}, Rho: {macro_rho_p0:.4f}", flush=True)

    # P0 Baseline Full Test
    v_mae, v_rho, t_mae, t_rho = train_and_eval_p0(
        mat_P0_device, train_full, val_full, test_full, virus_to_idx, epochs, device, smooth_l1
    )
    results["Stage_C"]["P0_Baseline"]["Test_Evaluation"] = {
        "MAE": round(t_mae, 4), "Spearman_rho": round(t_rho, 4)
    }
    print(f">> Stage C | P0 Baseline Test -> MAE: {t_mae:.4f}, Rho: {t_rho:.4f}", flush=True)

    # 2. Evaluate P0 + Gly across folds
    for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
        v_mae, v_rho, _, _ = train_and_eval_p0_gly(
            mat_P0_device, gly_matrix_device, f_tr, f_va, None, virus_to_idx, epochs, device
        )
        results["Stage_C"]["P0_Plus_Gly"]["Temporal_CV"][f"Fold_{f_idx}"] = {
            "MAE": round(v_mae, 4), "Spearman_rho": round(v_rho, 4)
        }
        print(f"Stage C | Fold {f_idx} | P0 + Gly -> Val MAE: {v_mae:.4f}, Rho: {v_rho:.4f}", flush=True)

    macro_mae_gly = np.mean([results["Stage_C"]["P0_Plus_Gly"]["Temporal_CV"][f"Fold_{i}"]["MAE"] for i in [1, 2, 3]])
    macro_rho_gly = np.mean([results["Stage_C"]["P0_Plus_Gly"]["Temporal_CV"][f"Fold_{i}"]["Spearman_rho"] for i in [1, 2, 3]])
    results["Stage_C"]["P0_Plus_Gly"]["Temporal_CV"]["Macro_MAE"] = round(float(macro_mae_gly), 4)
    results["Stage_C"]["P0_Plus_Gly"]["Temporal_CV"]["Macro_Spearman_rho"] = round(float(macro_rho_gly), 4)
    print(f">> Stage C | P0 + Gly Macro CV -> MAE: {macro_mae_gly:.4f}, Rho: {macro_rho_gly:.4f}", flush=True)

    # P0 + Gly Full Test
    v_mae, v_rho, t_mae, t_rho = train_and_eval_p0_gly(
        mat_P0_device, gly_matrix_device, train_full, val_full, test_full, virus_to_idx, epochs, device
    )
    results["Stage_C"]["P0_Plus_Gly"]["Test_Evaluation"] = {
        "MAE": round(t_mae, 4), "Spearman_rho": round(t_rho, 4)
    }
    print(f">> Stage C | P0 + Gly Test -> MAE: {t_mae:.4f}, Rho: {t_rho:.4f}", flush=True)


    # =========================================================================
    # STAGE D: Ranking Loss Lambda Exploration
    # =========================================================================
    print("\n=======================================================", flush=True)
    print("   STAGE D: Ranking Loss Lambda Exploration", flush=True)
    print("=======================================================", flush=True)

    lambdas = [0.0, 0.1, 0.3, 0.5, 1.0]

    for lam in lambdas:
        lam_key = f"lambda_{lam}"
        results["Stage_D"][lam_key] = {
            "Lambda": lam,
            "Description": f"P0 + CombinedRankingLoss (lambda={lam}, margin=0.1)",
            "Trainable_Parameters": count_params(HeadR1(1280, 932)),
            "Temporal_CV": {},
            "Test_Evaluation": {}
        }

        criterion = CombinedRankingLoss(lambda_param=lam, margin=0.1)

        for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
            v_mae, v_rho, _, _ = train_and_eval_p0(
                mat_P0_device, f_tr, f_va, None, virus_to_idx, epochs, device, criterion
            )
            results["Stage_D"][lam_key]["Temporal_CV"][f"Fold_{f_idx}"] = {
                "MAE": round(v_mae, 4), "Spearman_rho": round(v_rho, 4)
            }
            print(f"Stage D | lambda={lam} | Fold {f_idx} -> Val MAE: {v_mae:.4f}, Rho: {v_rho:.4f}", flush=True)

        m_mae = np.mean([results["Stage_D"][lam_key]["Temporal_CV"][f"Fold_{i}"]["MAE"] for i in [1, 2, 3]])
        m_rho = np.mean([results["Stage_D"][lam_key]["Temporal_CV"][f"Fold_{i}"]["Spearman_rho"] for i in [1, 2, 3]])
        results["Stage_D"][lam_key]["Temporal_CV"]["Macro_MAE"] = round(float(m_mae), 4)
        results["Stage_D"][lam_key]["Temporal_CV"]["Macro_Spearman_rho"] = round(float(m_rho), 4)
        print(f">> Stage D | lambda={lam} Macro CV -> MAE: {m_mae:.4f}, Rho: {m_rho:.4f}", flush=True)

        # Full Test Evaluation
        v_mae, v_rho, t_mae, t_rho = train_and_eval_p0(
            mat_P0_device, train_full, val_full, test_full, virus_to_idx, epochs, device, criterion
        )
        results["Stage_D"][lam_key]["Test_Evaluation"] = {
            "MAE": round(t_mae, 4), "Spearman_rho": round(t_rho, 4)
        }
        print(f">> Stage D | lambda={lam} Test -> MAE: {t_mae:.4f}, Rho: {t_rho:.4f}", flush=True)

    # Save to JSON
    out_json = os.path.join(reports_dir, 'stage_C_and_D_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"\nSuccessfully saved Stage C and Stage D results to {out_json}", flush=True)

if __name__ == '__main__':
    main()
