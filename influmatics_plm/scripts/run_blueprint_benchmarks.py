#!/usr/bin/env python3
"""
run_blueprint_benchmarks.py
Comprehensive Remedial & Blueprint Benchmark Suite for Influmatics H3N2 PLM.
Fully compliant with 'Influmatics_PLM_실험_계획서.md'.

Includes:
1. Multi-Seed 3-Fold Temporal Expanding CV (Main Table 1):
   - Stage 0: C0-a (Hamming Ridge), C0-b (Koel 7-site Ridge)
   - Stage A: R0 (Scalar L2), R1 (Symmetric abs*mult), R2 (Directional)
   - Stage B: P0 (Global Mean), P1 (Global+Epi), P2 (Global+Epi+Key), P3 (Attn 5.2M), P3_matched (Attn 2.82M)
   - Stage C: Glycosylation Feature (P0 + Gly [gain, loss, shared])
   - Stage D: Ranking Loss lambda exploration [0.0, 0.1, 0.3, 0.5, 1.0]
2. Multi-Seed Rolling Prospective Test (Main Table 2 - Primary):
   - Prospective years 2019, 2020, 2021, 2022 under Deployment-like policy
   - Models: B0, B1, B2, B2_matched, B4
   - Metrics: MAE, RMSE, Spearman rho, 2-AU AUROC, AUPRC, Sensitivity, Specificity
3. Multi-Seed Frozen Future Test (Main Table 3 - Secondary):
   - Train on <=2018 once, evaluate without retraining on 2019, 2020, 2021, 2022
   - Forecast degradation decay curve (gap 1 to 4 years)
4. Multi-Seed Strict Cold-Start Test (Supplementary Table):
   - Both-unseen node pairs in prospective years 2019-2022
5. Unbiased Landmark MDS & Cartography Audit:
   - 15 anchors, 47 held-out test nodes
   - Biased disparity (anchor copy cheat) vs Unbiased held-out disparity & 2D pairwise distance RMSE
6. Statistical Significance Tests (Paired t-tests)
"""

import os
import sys
import json
import time
import math
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, precision_recall_curve, auc
from scipy.stats import spearmanr, pearsonr, ttest_rel
from scipy.spatial import procrustes
from scipy.optimize import minimize
from sklearn.manifold import MDS

plm_dir = "/Users/shiftyellow/Documents/project-local/influmatics_project/influmatics_code/influmatics_plm"
if plm_dir not in sys.path:
    sys.path.append(plm_dir)

from scripts.combined_ranking_loss import CombinedRankingLoss

# Global Seeds and Hardware
SEEDS = [42, 17, 29]
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Epitope and Key site definitions (Koel et al. 2013)
KOEL_7_SITES = [145, 155, 156, 158, 159, 189, 193]
EPITOPE_SITES = sorted(list(set([
    122, 124, 126, 131, 133, 135, 137, 142, 143, 144, 145, 146, # Epitope A
    128, 129, 155, 156, 157, 158, 159, 160, 163, 164, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197, 198, # Epitope B
    44, 45, 46, 47, 48, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294, 297, 299, 300, 304, 305, 307, 308, 309, 310, 311, 312, # Epitope C
    96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 220, 222, 223, 224, 225, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248, # Epitope D
    57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265 # Epitope E
])))

EPI_INDICES = [pos - 1 for pos in EPITOPE_SITES if pos - 1 < 328]
KEY_INDICES = [pos - 1 for pos in KOEL_7_SITES if pos - 1 < 328]

def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# Architecture Modules
# ---------------------------------------------------------------------------
class HeadR0(nn.Module):
    """Stage A: R0 Scalar Distance r = ||u - v||_2 with MLP"""
    def __init__(self, hidden_dim=932):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, u, v):
        d = torch.norm(u - v, p=2, dim=-1, keepdim=True)
        return self.mlp(d).squeeze(-1)

class HeadR1(nn.Module):
    """Stage A: R1 Symmetric [|u - v|, u * v]"""
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
    """Stage A: R2 Directional [u, v, u - v, u * v] with budget-matched hidden dim"""
    def __init__(self, emb_dim=1280, hidden_dim=476):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, u, v):
        diff = u - v
        mult = u * v
        x = torch.cat([u, v, diff, mult], dim=-1)
        return self.mlp(x).squeeze(-1)

class HeadR1WithGly(nn.Module):
    """Stage C: R1 + Glycosylation features [gain, loss, shared]"""
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

class BiologyGuidedAttentionPooling(nn.Module):
    """Stage B: P3 Learnable Attention s_i = g(h_i) + beta_E * I_epi + beta_K * I_key"""
    def __init__(self, emb_dim=1280, seq_len=328):
        super().__init__()
        self.g = nn.Linear(emb_dim, 1)
        self.beta_E = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.beta_K = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
        I_epi = torch.zeros(seq_len, 1, dtype=torch.float32)
        I_key = torch.zeros(seq_len, 1, dtype=torch.float32)
        I_epi[EPI_INDICES] = 1.0
        I_key[KEY_INDICES] = 1.0
        self.register_buffer('I_epi', I_epi)
        self.register_buffer('I_key', I_key)

    def forward(self, h):
        scores = self.g(h) + self.beta_E * self.I_epi + self.beta_K * self.I_key
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(h * weights, dim=1)
        return pooled

class P3Model(nn.Module):
    def __init__(self, emb_dim=1280, hidden_dim=932):
        super().__init__()
        self.pool = BiologyGuidedAttentionPooling(emb_dim=emb_dim, seq_len=328)
        self.head = HeadR1(emb_dim=emb_dim * 2, hidden_dim=hidden_dim)

    def forward(self, u_global, u_attn, v_global, v_attn):
        u = torch.cat([u_global, u_attn], dim=-1)
        v = torch.cat([v_global, v_attn], dim=-1)
        return self.head(u, v)

# ---------------------------------------------------------------------------
# Dataset & Metric Helpers
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

def compute_comprehensive_metrics(tgts, preds):
    mae = float(mean_absolute_error(tgts, preds))
    rmse = float(np.sqrt(mean_squared_error(tgts, preds)))
    rho, _ = spearmanr(tgts, preds)
    rho = float(rho) if not np.isnan(rho) else 0.0
    r, _ = pearsonr(tgts, preds)
    r = float(r) if not np.isnan(r) else 0.0

    # 2 AU decision classification metrics
    binary_true = (tgts >= 2.0).astype(int)
    if len(np.unique(binary_true)) > 1:
        auroc = float(roc_auc_score(binary_true, preds))
        prec, rec, _ = precision_recall_curve(binary_true, preds)
        auprc = float(auc(rec, prec))
        
        binary_pred = (preds >= 2.0).astype(int)
        tp = np.sum((binary_true == 1) & (binary_pred == 1))
        fp = np.sum((binary_true == 0) & (binary_pred == 1))
        fn = np.sum((binary_true == 1) & (binary_pred == 0))
        tn = np.sum((binary_true == 0) & (binary_pred == 0))
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        auroc, auprc, sensitivity, specificity = float('nan'), float('nan'), float('nan'), float('nan')

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Spearman_rho": round(rho, 4),
        "Pearson_r": round(r, 4),
        "AUROC": round(auroc, 4) if not np.isnan(auroc) else None,
        "AUPRC": round(auprc, 4) if not np.isnan(auprc) else None,
        "Sensitivity": round(sensitivity, 4) if not np.isnan(sensitivity) else None,
        "Specificity": round(specificity, 4) if not np.isnan(specificity) else None
    }

def eval_head(model, emb_matrix_device, loader, device, model_type='R1', gly_matrix_device=None):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for idx_A, idx_B, dist in loader:
            idx_A, idx_B = idx_A.to(device), idx_B.to(device)
            emb1 = emb_matrix_device[idx_A]
            emb2 = emb_matrix_device[idx_B]
            if model_type == 'R1WithGly':
                gly_AB = gly_matrix_device[idx_A, idx_B]
                p1 = model(emb1, emb2, gly_AB)
                p2 = model(emb2, emb1, gly_AB)
            else:
                p1 = model(emb1, emb2)
                p2 = model(emb2, emb1)
            p = torch.clamp((p1 + p2) / 2.0, min=0.0)
            preds.extend(p.cpu().numpy())
            tgts.extend(dist.numpy())
    return np.array(tgts), np.array(preds)

def eval_model_p3(model, emb_global_device, all_tokens_device, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        all_attn_device = model.pool(all_tokens_device)
        for idx_A, idx_B, dist in loader:
            idx_A, idx_B = idx_A.to(device), idx_B.to(device)
            u_glob = emb_global_device[idx_A]
            v_glob = emb_global_device[idx_B]
            u_attn = all_attn_device[idx_A]
            v_attn = all_attn_device[idx_B]
            p1 = model(u_glob, u_attn, v_glob, v_attn)
            p2 = model(v_glob, v_attn, u_glob, u_attn)
            p = torch.clamp((p1 + p2) / 2.0, min=0.0)
            preds.extend(p.cpu().numpy())
            tgts.extend(dist.numpy())
    return np.array(tgts), np.array(preds)

# ---------------------------------------------------------------------------
# Training Engines (Fast MPS Staged)
# ---------------------------------------------------------------------------
def train_head_model(emb_matrix_device, train_df, val_df, virus_to_idx, epochs, device, criterion, head_class=HeadR1, head_kwargs=None, gly_matrix_device=None):
    if head_kwargs is None:
        head_kwargs = {'emb_dim': 1280, 'hidden_dim': 932}
    ds_train = IndexPairDataset(train_df, virus_to_idx)
    loader_train = DataLoader(ds_train, batch_size=1024, shuffle=True)
    model = head_class(**head_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_mae = float('inf')
    best_state = None

    for ep in range(epochs):
        model.train()
        for idx_A, idx_B, dist in loader_train:
            idx_A, idx_B, dist = idx_A.to(device), idx_B.to(device), dist.to(device)
            emb1 = emb_matrix_device[idx_A]
            emb2 = emb_matrix_device[idx_B]
            optimizer.zero_grad()
            if gly_matrix_device is not None:
                gly_AB = gly_matrix_device[idx_A, idx_B]
                p = model(emb1, emb2, gly_AB)
            else:
                p = model(emb1, emb2)
            loss = criterion(p, dist)
            loss.backward()
            optimizer.step()

        if val_df is not None:
            loader_val = DataLoader(IndexPairDataset(val_df, virus_to_idx), batch_size=2048, shuffle=False)
            m_type = 'R1WithGly' if gly_matrix_device is not None else 'Standard'
            tgts, preds = eval_head(model, emb_matrix_device, loader_val, device, model_type=m_type, gly_matrix_device=gly_matrix_device)
            val_mae = mean_absolute_error(tgts, preds)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if device.type == 'mps':
            torch.mps.empty_cache()

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model

def train_p3_model(emb_global_device, all_tokens_device, train_df, val_df, virus_to_idx, epochs, device, hidden_dim=932):
    ds_train = IndexPairDataset(train_df, virus_to_idx)
    loader_train = DataLoader(ds_train, batch_size=1024, shuffle=True)
    model = P3Model(emb_dim=1280, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss()

    best_val_mae = float('inf')
    best_state = None

    idx_A_all = ds_train.idx_A
    idx_B_all = ds_train.idx_B
    targets_all = ds_train.targets
    n_train = len(ds_train)

    for ep in range(epochs):
        model.train()
        # Staged pool update
        sample_size = min(4096, n_train)
        sample_indices = torch.randint(0, n_train, (sample_size,))
        s_A = idx_A_all[sample_indices].to(device)
        s_B = idx_B_all[sample_indices].to(device)
        s_dist = targets_all[sample_indices].to(device)

        optimizer.zero_grad()
        all_attn = model.pool(all_tokens_device)
        u_attn = all_attn[s_A]
        v_attn = all_attn[s_B]
        u_glob = emb_global_device[s_A]
        v_glob = emb_global_device[s_B]
        p = model(u_glob, u_attn, v_glob, v_attn)
        loss = criterion(p, s_dist)
        loss.backward()
        optimizer.step()

        # Mini-batch update with detached all_attn
        with torch.no_grad():
            all_attn_det = model.pool(all_tokens_device)

        for idx_A, idx_B, dist in loader_train:
            idx_A_dev, idx_B_dev, dist = idx_A.to(device), idx_B.to(device), dist.to(device)
            u_g = emb_global_device[idx_A_dev]
            v_g = emb_global_device[idx_B_dev]
            u_a = all_attn_det[idx_A_dev]
            v_a = all_attn_det[idx_B_dev]

            optimizer.zero_grad()
            p = model(u_g, u_a, v_g, v_a)
            loss = criterion(p, dist)
            loss.backward()
            optimizer.step()

        if val_df is not None:
            loader_val = DataLoader(IndexPairDataset(val_df, virus_to_idx), batch_size=2048, shuffle=False)
            tgts, preds = eval_model_p3(model, emb_global_device, all_tokens_device, loader_val, device)
            val_mae = mean_absolute_error(tgts, preds)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if device.type == 'mps':
            torch.mps.empty_cache()

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model

# Classical Baseline Extractors
def extract_mut_features(df, virus_to_seq, mode='key'):
    X = []
    for _, row in df.iterrows():
        seqA = virus_to_seq[row['node_A']]
        seqB = virus_to_seq[row['node_B']]
        min_len = min(len(seqA), len(seqB))
        if mode == 'key':
            m = sum(1 for p in KOEL_7_SITES if p - 1 < min_len and seqA[p - 1] != seqB[p - 1])
        elif mode == 'hamming':
            m = sum(1 for i in range(min_len) if seqA[i] != seqB[i])
        X.append([m])
    return np.array(X, dtype=np.float32), df['distance'].values.astype(np.float32)

# Multilateration for Landmark MDS
def triangulate_strain(predicted_dists_to_anchors, anchor_coords):
    def loss(p):
        dists = np.linalg.norm(anchor_coords - p, axis=1)
        return np.sum((dists - predicted_dists_to_anchors) ** 2)
    x0 = np.mean(anchor_coords, axis=0)
    res = minimize(loss, x0, method='BFGS')
    return res.x

# ---------------------------------------------------------------------------
# Main Orchestration Loop
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("="*70, flush=True)
    print(" Influmatics PLM Comprehensive Blueprint Benchmark Suite", flush=True)
    print("="*70, flush=True)

    data_dir = os.path.join(plm_dir, 'data')
    splits_dir = os.path.join(data_dir, 'splits')
    reports_dir = os.path.join(plm_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load mappings & tokens
    print("Loading representations and mappings to MPS...", flush=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("extract", os.path.join(plm_dir, 'scripts', '04_extract_embeddings.py'))
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)
    virus_to_seq = extract.get_virus_to_seq_mapping()

    cache_path = os.path.join(data_dir, 'processed', 'stage_B_cache.pt')
    cache = torch.load(cache_path, weights_only=False)
    common = sorted(list(set(cache['global'].keys()).intersection(set(virus_to_seq.keys()))))
    virus_to_idx = {v: i for i, v in enumerate(common)}
    N_viruses = len(common)

    mat_P0_device = torch.stack([cache['global'][v] for v in common], dim=0).to(device)
    mat_P1_device = torch.cat([mat_P0_device, torch.stack([cache['epi'][v] for v in common], dim=0).to(device)], dim=-1)
    mat_P2_device = torch.cat([mat_P1_device, torch.stack([cache['key'][v] for v in common], dim=0).to(device)], dim=-1)
    all_tokens_device = torch.stack([cache['tokens'][v] for v in common], dim=0).to(device).float()
    print(f"Loaded {N_viruses:,} global & token embeddings directly onto {device}!", flush=True)

    # Glycosylation tensor
    glyco_path = os.path.join(data_dir, 'processed', 'glyco_counts.json')
    with open(glyco_path, 'r', encoding='utf-8') as f:
        glyco = json.load(f)
    site_sets = [set(glyco.get(v, {}).get('sites', [])) for v in common]
    glyco_pair_np = np.zeros((N_viruses, N_viruses, 3), dtype=np.float32)
    for i in range(N_viruses):
        s_i = site_sets[i]
        for j in range(N_viruses):
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

    # Load splits
    folds_data = []
    for f in [1, 2, 3]:
        f_tr = filter_df(pd.read_csv(os.path.join(splits_dir, f'cohort2_fold{f}_train.csv.gz')))
        f_va = filter_df(pd.read_csv(os.path.join(splits_dir, f'cohort2_fold{f}_val.csv.gz')))
        folds_data.append((f_tr, f_va))

    train_full = filter_df(pd.read_csv(os.path.join(splits_dir, 'cohort2_train.csv.gz')))
    val_full = filter_df(pd.read_csv(os.path.join(splits_dir, 'cohort2_val.csv.gz')))
    dev_full = pd.concat([train_full, val_full], ignore_index=True)
    test_full = filter_df(pd.read_csv(os.path.join(splits_dir, 'cohort2_test.csv.gz')))

    df_all = pd.concat([dev_full, test_full], ignore_index=True)
    df_all['pair_year'] = df_all[['year_A', 'year_B']].max(axis=1)
    test_full['pair_year'] = test_full[['year_A', 'year_B']].max(axis=1)

    epochs = 8
    smooth_l1 = nn.SmoothL1Loss()

    # Results dictionary skeleton
    results = {
        "metadata": {
            "title": "Influmatics PLM Comprehensive Blueprint Benchmark Results",
            "protocol": "Influmatics_PLM_실험_계획서.md",
            "seeds": SEEDS,
            "device": str(device)
        },
        "Main_Table_1_Temporal_CV": {},
        "Main_Table_2_Rolling_Prospective_Primary": {},
        "Main_Table_3_Frozen_Future_Secondary": {},
        "Supplementary_Table_Strict_Cold_Start": {},
        "Unbiased_Landmark_Cartography": {},
        "Statistical_Significance_Tests": {}
    }

    # =========================================================================
    # PART 1: MAIN TABLE 1 — TEMPORAL EXPANDING CV (Stage 0 ~ Stage D)
    # =========================================================================
    print("\n" + "="*70, flush=True)
    print(" PART 1: Main Table 1 — Multi-Seed Temporal Expanding CV", flush=True)
    print("="*70, flush=True)

    cv_models = {
        "C0-a_Hamming": "Stage 0: Full HA1 Hamming distance + Ridge",
        "C0-b_Koel7": "Stage 0: Koel 7 Key-site mutation count + Ridge",
        "R0_Scalar": "Stage A: Scalar L2 distance ||u - v||_2 + MLP",
        "R1_Symmetric": "Stage A: Symmetric [|u - v|, u * v] + HeadR1 (B1 baseline)",
        "R2_Directional": "Stage A: Directional [u, v, u - v, u * v] + HeadR2 (budget-matched)",
        "P0_Global": "Stage B: Global Mean (1280d)",
        "P1_Epitope": "Stage B: Global + Epitope (2560d)",
        "P2_KeySite": "Stage B: Global + Epitope + Key-site (3840d)",
        "P3_BiologyAttn_5.2M": "Stage B: Biology-guided Attention (5.21M params)",
        "P3_BiologyAttn_2.82M": "Stage B: Biology-guided Attention Parameter-Matched (2.82M params)",
        "P0_Plus_Gly": "Stage C: P0 + Glycosylation [gain, loss, shared] (2.82M params)",
        "Ranking_lambda_0.0": "Stage D: Ranking Loss lambda=0.0",
        "Ranking_lambda_0.1": "Stage D: Ranking Loss lambda=0.1",
        "Ranking_lambda_0.3": "Stage D: Ranking Loss lambda=0.3 (B4 baseline)",
        "Ranking_lambda_0.5": "Stage D: Ranking Loss lambda=0.5",
        "Ranking_lambda_1.0": "Stage D: Ranking Loss lambda=1.0"
    }

    for m in cv_models:
        results["Main_Table_1_Temporal_CV"][m] = {"seeds": {}, "summary": {}}

    for seed in SEEDS:
        print(f"\n--- Running Temporal CV with Seed {seed} ---", flush=True)
        set_seeds(seed)

        # Stage 0: Classical Baselines
        for c_name, c_mode in [("C0-a_Hamming", "hamming"), ("C0-b_Koel7", "key")]:
            f_maes, f_rhos = [], []
            for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
                X_tr, Y_tr = extract_mut_features(f_tr, virus_to_seq, mode=c_mode)
                X_va, Y_va = extract_mut_features(f_va, virus_to_seq, mode=c_mode)
                reg = Ridge(alpha=1.0)
                reg.fit(X_tr, Y_tr)
                preds = reg.predict(X_va)
                m = compute_comprehensive_metrics(Y_va, preds)
                f_maes.append(m['MAE'])
                f_rhos.append(m['Spearman_rho'])
            results["Main_Table_1_Temporal_CV"][c_name]["seeds"][str(seed)] = {
                "Macro_MAE": float(np.mean(f_maes)), "Macro_rho": float(np.mean(f_rhos))
            }
            print(f"  [{c_name}] Macro-MAE: {np.mean(f_maes):.4f}", flush=True)

        # Stage A: Representation Ablation
        # R0
        f_maes, f_rhos = [], []
        for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
            model = train_head_model(mat_P0_device, f_tr, f_va, virus_to_idx, epochs, device, smooth_l1, head_class=HeadR0, head_kwargs={'hidden_dim': 932})
            loader_va = DataLoader(IndexPairDataset(f_va, virus_to_idx), batch_size=2048, shuffle=False)
            tgts, preds = eval_head(model, mat_P0_device, loader_va, device)
            m = compute_comprehensive_metrics(tgts, preds)
            f_maes.append(m['MAE']); f_rhos.append(m['Spearman_rho'])
        results["Main_Table_1_Temporal_CV"]["R0_Scalar"]["seeds"][str(seed)] = {"Macro_MAE": float(np.mean(f_maes)), "Macro_rho": float(np.mean(f_rhos))}
        print(f"  [R0_Scalar] Macro-MAE: {np.mean(f_maes):.4f}", flush=True)

        # R1 (Standard P0)
        f_maes, f_rhos = [], []
        for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
            model = train_head_model(mat_P0_device, f_tr, f_va, virus_to_idx, epochs, device, smooth_l1, head_class=HeadR1, head_kwargs={'emb_dim': 1280, 'hidden_dim': 932})
            loader_va = DataLoader(IndexPairDataset(f_va, virus_to_idx), batch_size=2048, shuffle=False)
            tgts, preds = eval_head(model, mat_P0_device, loader_va, device)
            m = compute_comprehensive_metrics(tgts, preds)
            f_maes.append(m['MAE']); f_rhos.append(m['Spearman_rho'])
        results["Main_Table_1_Temporal_CV"]["R1_Symmetric"]["seeds"][str(seed)] = {"Macro_MAE": float(np.mean(f_maes)), "Macro_rho": float(np.mean(f_rhos))}
        results["Main_Table_1_Temporal_CV"]["P0_Global"]["seeds"][str(seed)] = {"Macro_MAE": float(np.mean(f_maes)), "Macro_rho": float(np.mean(f_rhos))}
        print(f"  [R1_Symmetric / P0_Global] Macro-MAE: {np.mean(f_maes):.4f}", flush=True)

        # R2 Directional
        f_maes, f_rhos = [], []
        for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
            model = train_head_model(mat_P0_device, f_tr, f_va, virus_to_idx, epochs, device, smooth_l1, head_class=HeadR2, head_kwargs={'emb_dim': 1280, 'hidden_dim': 476})
            loader_va = DataLoader(IndexPairDataset(f_va, virus_to_idx), batch_size=2048, shuffle=False)
            tgts, preds = eval_head(model, mat_P0_device, loader_va, device)
            m = compute_comprehensive_metrics(tgts, preds)
            f_maes.append(m['MAE']); f_rhos.append(m['Spearman_rho'])
        results["Main_Table_1_Temporal_CV"]["R2_Directional"]["seeds"][str(seed)] = {"Macro_MAE": float(np.mean(f_maes)), "Macro_rho": float(np.mean(f_rhos))}
        print(f"  [R2_Directional] Macro-MAE: {np.mean(f_maes):.4f}", flush=True)

        # Stage B: Pooling Ablation (P1, P2, P3, P3_matched)
        # P1
        f_maes, f_rhos = [], []
        for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
            model = train_head_model(mat_P1_device, f_tr, f_va, virus_to_idx, epochs, device, smooth_l1, head_class=HeadR1, head_kwargs={'emb_dim': 2560, 'hidden_dim': 932})
            loader_va = DataLoader(IndexPairDataset(f_va, virus_to_idx), batch_size=2048, shuffle=False)
            tgts, preds = eval_head(model, mat_P1_device, loader_va, device)
            m = compute_comprehensive_metrics(tgts, preds)
            f_maes.append(m['MAE']); f_rhos.append(m['Spearman_rho'])
        results["Main_Table_1_Temporal_CV"]["P1_Epitope"]["seeds"][str(seed)] = {"Macro_MAE": float(np.mean(f_maes)), "Macro_rho": float(np.mean(f_rhos))}
        print(f"  [P1_Epitope] Macro-MAE: {np.mean(f_maes):.4f}", flush=True)

        # P2
        f_maes, f_rhos = [], []
        for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
            model = train_head_model(mat_P2_device, f_tr, f_va, virus_to_idx, epochs, device, smooth_l1, head_class=HeadR1, head_kwargs={'emb_dim': 3840, 'hidden_dim': 932})
            loader_va = DataLoader(IndexPairDataset(f_va, virus_to_idx), batch_size=2048, shuffle=False)
            tgts, preds = eval_head(model, mat_P2_device, loader_va, device)
            m = compute_comprehensive_metrics(tgts, preds)
            f_maes.append(m['MAE']); f_rhos.append(m['Spearman_rho'])
        results["Main_Table_1_Temporal_CV"]["P2_KeySite"]["seeds"][str(seed)] = {"Macro_MAE": float(np.mean(f_maes)), "Macro_rho": float(np.mean(f_rhos))}
        print(f"  [P2_KeySite] Macro-MAE: {np.mean(f_maes):.4f}", flush=True)

        # P3 (5.2M)
        f_maes, f_rhos = [], []
        for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
            model = train_p3_model(mat_P0_device, all_tokens_device, f_tr, f_va, virus_to_idx, epochs, device, hidden_dim=932)
            loader_va = DataLoader(IndexPairDataset(f_va, virus_to_idx), batch_size=2048, shuffle=False)
            tgts, preds = eval_model_p3(model, mat_P0_device, all_tokens_device, loader_va, device)
            m = compute_comprehensive_metrics(tgts, preds)
            f_maes.append(m['MAE']); f_rhos.append(m['Spearman_rho'])
        results["Main_Table_1_Temporal_CV"]["P3_BiologyAttn_5.2M"]["seeds"][str(seed)] = {"Macro_MAE": float(np.mean(f_maes)), "Macro_rho": float(np.mean(f_rhos))}
        print(f"  [P3_BiologyAttn_5.2M] Macro-MAE: {np.mean(f_maes):.4f}", flush=True)

        # P3_matched (2.82M)
        f_maes, f_rhos = [], []
        for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
            model = train_p3_model(mat_P0_device, all_tokens_device, f_tr, f_va, virus_to_idx, epochs, device, hidden_dim=524)
            loader_va = DataLoader(IndexPairDataset(f_va, virus_to_idx), batch_size=2048, shuffle=False)
            tgts, preds = eval_model_p3(model, mat_P0_device, all_tokens_device, loader_va, device)
            m = compute_comprehensive_metrics(tgts, preds)
            f_maes.append(m['MAE']); f_rhos.append(m['Spearman_rho'])
        results["Main_Table_1_Temporal_CV"]["P3_BiologyAttn_2.82M"]["seeds"][str(seed)] = {"Macro_MAE": float(np.mean(f_maes)), "Macro_rho": float(np.mean(f_rhos))}
        print(f"  [P3_BiologyAttn_2.82M] Macro-MAE: {np.mean(f_maes):.4f}", flush=True)

        # Stage C: Glycosylation Feature
        f_maes, f_rhos = [], []
        for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
            model = train_head_model(mat_P0_device, f_tr, f_va, virus_to_idx, epochs, device, smooth_l1, head_class=HeadR1WithGly, head_kwargs={'emb_dim': 1280, 'hidden_dim': 932, 'gly_dim': 3}, gly_matrix_device=gly_matrix_device)
            loader_va = DataLoader(IndexPairDataset(f_va, virus_to_idx), batch_size=2048, shuffle=False)
            tgts, preds = eval_head(model, mat_P0_device, loader_va, device, model_type='R1WithGly', gly_matrix_device=gly_matrix_device)
            m = compute_comprehensive_metrics(tgts, preds)
            f_maes.append(m['MAE']); f_rhos.append(m['Spearman_rho'])
        results["Main_Table_1_Temporal_CV"]["P0_Plus_Gly"]["seeds"][str(seed)] = {"Macro_MAE": float(np.mean(f_maes)), "Macro_rho": float(np.mean(f_rhos))}
        print(f"  [P0_Plus_Gly] Macro-MAE: {np.mean(f_maes):.4f}", flush=True)

        # Stage D: Ranking Loss lambda exploration [0.0, 0.1, 0.3, 0.5, 1.0]
        for l_val in [0.0, 0.1, 0.3, 0.5, 1.0]:
            l_key = f"Ranking_lambda_{l_val}"
            rank_loss = CombinedRankingLoss(lambda_param=l_val, margin=0.1) if l_val > 0 else smooth_l1
            f_maes, f_rhos = [], []
            for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
                model = train_head_model(mat_P0_device, f_tr, f_va, virus_to_idx, epochs, device, rank_loss, head_class=HeadR1, head_kwargs={'emb_dim': 1280, 'hidden_dim': 932})
                loader_va = DataLoader(IndexPairDataset(f_va, virus_to_idx), batch_size=2048, shuffle=False)
                tgts, preds = eval_head(model, mat_P0_device, loader_va, device)
                m = compute_comprehensive_metrics(tgts, preds)
                f_maes.append(m['MAE']); f_rhos.append(m['Spearman_rho'])
            results["Main_Table_1_Temporal_CV"][l_key]["seeds"][str(seed)] = {"Macro_MAE": float(np.mean(f_maes)), "Macro_rho": float(np.mean(f_rhos))}
            print(f"  [{l_key}] Macro-MAE: {np.mean(f_maes):.4f}", flush=True)

    # Compute CV Summaries across seeds
    for m in cv_models:
        maes = [results["Main_Table_1_Temporal_CV"][m]["seeds"][str(s)]["Macro_MAE"] for s in SEEDS]
        rhos = [results["Main_Table_1_Temporal_CV"][m]["seeds"][str(s)]["Macro_rho"] for s in SEEDS]
        results["Main_Table_1_Temporal_CV"][m]["summary"] = {
            "mean_Macro_MAE": round(float(np.mean(maes)), 4),
            "std_Macro_MAE": round(float(np.std(maes)), 4),
            "mean_Macro_rho": round(float(np.mean(rhos)), 4),
            "std_Macro_rho": round(float(np.std(rhos)), 4)
        }

    # =========================================================================
    # PART 2 & 3: MAIN TABLE 2 (Rolling) & SUPPLEMENTARY (Strict)
    # =========================================================================
    print("\n" + "="*70, flush=True)
    print(" PART 2 & 3: Rolling Prospective Test & Strict Cold-Start", flush=True)
    print("="*70, flush=True)

    backtest_windows = [(2018, 2019), (2019, 2020), (2020, 2021), (2021, 2022)]
    rolling_models = ["B0", "B1", "B2", "B2_matched", "B4"]
    rank_loss_03 = CombinedRankingLoss(lambda_param=0.3, margin=0.1)

    for m in rolling_models:
        results["Main_Table_2_Rolling_Prospective_Primary"][m] = {"seeds": {}, "summary": {}}
        results["Supplementary_Table_Strict_Cold_Start"][m] = {"seeds": {}, "summary": {}}

    for seed in SEEDS:
        print(f"\n--- Running Rolling Test with Seed {seed} ---", flush=True)
        for m in rolling_models:
            results["Main_Table_2_Rolling_Prospective_Primary"][m]["seeds"][str(seed)] = {}
            results["Supplementary_Table_Strict_Cold_Start"][m]["seeds"][str(seed)] = {}

        for train_max_yr, test_yr in backtest_windows:
            window_key = f"<={train_max_yr} -> {test_yr}"
            tr_df = df_all[df_all['pair_year'] <= train_max_yr].reset_index(drop=True)
            te_df_dep = df_all[df_all['pair_year'] == test_yr].reset_index(drop=True)

            train_nodes = set(tr_df['node_A']).union(set(tr_df['node_B']))
            te_df_str = te_df_dep[~te_df_dep['node_A'].isin(train_nodes) & ~te_df_dep['node_B'].isin(train_nodes)].reset_index(drop=True)

            loader_dep = DataLoader(IndexPairDataset(te_df_dep, virus_to_idx), batch_size=2048, shuffle=False)
            loader_str = DataLoader(IndexPairDataset(te_df_str, virus_to_idx), batch_size=2048, shuffle=False)

            set_seeds(seed)

            # B0
            X_tr, Y_tr = extract_mut_features(tr_df, virus_to_seq, mode='key')
            reg = Ridge(alpha=1.0)
            reg.fit(X_tr, Y_tr)
            X_dep, Y_dep = extract_mut_features(te_df_dep, virus_to_seq, mode='key')
            X_str, Y_str = extract_mut_features(te_df_str, virus_to_seq, mode='key')
            m_dep = compute_comprehensive_metrics(Y_dep, reg.predict(X_dep))
            m_str = compute_comprehensive_metrics(Y_str, reg.predict(X_str))
            results["Main_Table_2_Rolling_Prospective_Primary"]["B0"]["seeds"][str(seed)][window_key] = m_dep
            results["Supplementary_Table_Strict_Cold_Start"]["B0"]["seeds"][str(seed)][window_key] = m_str

            # B1
            model_b1 = train_head_model(mat_P0_device, tr_df, None, virus_to_idx, epochs, device, smooth_l1, head_class=HeadR1)
            t_d, p_d = eval_head(model_b1, mat_P0_device, loader_dep, device)
            t_s, p_s = eval_head(model_b1, mat_P0_device, loader_str, device)
            results["Main_Table_2_Rolling_Prospective_Primary"]["B1"]["seeds"][str(seed)][window_key] = compute_comprehensive_metrics(t_d, p_d)
            results["Supplementary_Table_Strict_Cold_Start"]["B1"]["seeds"][str(seed)][window_key] = compute_comprehensive_metrics(t_s, p_s)

            # B2
            model_b2 = train_p3_model(mat_P0_device, all_tokens_device, tr_df, None, virus_to_idx, epochs, device, hidden_dim=932)
            t_d, p_d = eval_model_p3(model_b2, mat_P0_device, all_tokens_device, loader_dep, device)
            t_s, p_s = eval_model_p3(model_b2, mat_P0_device, all_tokens_device, loader_str, device)
            results["Main_Table_2_Rolling_Prospective_Primary"]["B2"]["seeds"][str(seed)][window_key] = compute_comprehensive_metrics(t_d, p_d)
            results["Supplementary_Table_Strict_Cold_Start"]["B2"]["seeds"][str(seed)][window_key] = compute_comprehensive_metrics(t_s, p_s)

            # B2_matched
            model_b2m = train_p3_model(mat_P0_device, all_tokens_device, tr_df, None, virus_to_idx, epochs, device, hidden_dim=524)
            t_d, p_d = eval_model_p3(model_b2m, mat_P0_device, all_tokens_device, loader_dep, device)
            t_s, p_s = eval_model_p3(model_b2m, mat_P0_device, all_tokens_device, loader_str, device)
            results["Main_Table_2_Rolling_Prospective_Primary"]["B2_matched"]["seeds"][str(seed)][window_key] = compute_comprehensive_metrics(t_d, p_d)
            results["Supplementary_Table_Strict_Cold_Start"]["B2_matched"]["seeds"][str(seed)][window_key] = compute_comprehensive_metrics(t_s, p_s)

            # B4
            model_b4 = train_head_model(mat_P0_device, tr_df, None, virus_to_idx, epochs, device, rank_loss_03, head_class=HeadR1)
            t_d, p_d = eval_head(model_b4, mat_P0_device, loader_dep, device)
            t_s, p_s = eval_head(model_b4, mat_P0_device, loader_str, device)
            results["Main_Table_2_Rolling_Prospective_Primary"]["B4"]["seeds"][str(seed)][window_key] = compute_comprehensive_metrics(t_d, p_d)
            results["Supplementary_Table_Strict_Cold_Start"]["B4"]["seeds"][str(seed)][window_key] = compute_comprehensive_metrics(t_s, p_s)

            print(f"  {window_key} -> Dep MAE: B0={results['Main_Table_2_Rolling_Prospective_Primary']['B0']['seeds'][str(seed)][window_key]['MAE']:.4f}, "
                  f"B1={results['Main_Table_2_Rolling_Prospective_Primary']['B1']['seeds'][str(seed)][window_key]['MAE']:.4f}, "
                  f"B2m={results['Main_Table_2_Rolling_Prospective_Primary']['B2_matched']['seeds'][str(seed)][window_key]['MAE']:.4f}, "
                  f"B4={results['Main_Table_2_Rolling_Prospective_Primary']['B4']['seeds'][str(seed)][window_key]['MAE']:.4f}", flush=True)

        for m in rolling_models:
            dep_maes = [results["Main_Table_2_Rolling_Prospective_Primary"][m]["seeds"][str(seed)][f"<={y-1} -> {y}"]["MAE"] for y in [2019, 2020, 2021, 2022]]
            str_maes = [results["Supplementary_Table_Strict_Cold_Start"][m]["seeds"][str(seed)][f"<={y-1} -> {y}"]["MAE"] for y in [2019, 2020, 2021, 2022]]
            results["Main_Table_2_Rolling_Prospective_Primary"][m]["seeds"][str(seed)]["Season_Macro_MAE"] = float(np.mean(dep_maes))
            results["Supplementary_Table_Strict_Cold_Start"][m]["seeds"][str(seed)]["Season_Macro_MAE"] = float(np.mean(str_maes))

    for m in rolling_models:
        dep_macros = [results["Main_Table_2_Rolling_Prospective_Primary"][m]["seeds"][str(s)]["Season_Macro_MAE"] for s in SEEDS]
        str_macros = [results["Supplementary_Table_Strict_Cold_Start"][m]["seeds"][str(s)]["Season_Macro_MAE"] for s in SEEDS]
        results["Main_Table_2_Rolling_Prospective_Primary"][m]["summary"] = {
            "mean_Season_Macro_MAE": round(float(np.mean(dep_macros)), 4),
            "std_Season_Macro_MAE": round(float(np.std(dep_macros)), 4)
        }
        results["Supplementary_Table_Strict_Cold_Start"][m]["summary"] = {
            "mean_Season_Macro_MAE": round(float(np.mean(str_macros)), 4),
            "std_Season_Macro_MAE": round(float(np.std(str_macros)), 4)
        }

    # =========================================================================
    # PART 4: MAIN TABLE 3 — FROZEN FUTURE TEST & DECAY CURVE
    # =========================================================================
    print("\n" + "="*70, flush=True)
    print(" PART 4: Main Table 3 — Frozen Future Test (Decay Curve)", flush=True)
    print("="*70, flush=True)

    tr_frozen_df = df_all[df_all['pair_year'] <= 2018].reset_index(drop=True)
    forecast_years = [2019, 2020, 2021, 2022]

    for m in ["B0", "B1", "B2_matched", "B4"]:
        results["Main_Table_3_Frozen_Future_Secondary"][m] = {"seeds": {}, "summary": {}}

    for seed in SEEDS:
        print(f"\n--- Running Frozen Future Test with Seed {seed} ---", flush=True)
        set_seeds(seed)

        # Train each model ONCE on <=2018
        # B0
        X_tr, Y_tr = extract_mut_features(tr_frozen_df, virus_to_seq, mode='key')
        reg_b0 = Ridge(alpha=1.0)
        reg_b0.fit(X_tr, Y_tr)

        # B1
        model_b1_frz = train_head_model(mat_P0_device, tr_frozen_df, None, virus_to_idx, epochs, device, smooth_l1, head_class=HeadR1)

        # B2_matched
        model_b2m_frz = train_p3_model(mat_P0_device, all_tokens_device, tr_frozen_df, None, virus_to_idx, epochs, device, hidden_dim=524)

        # B4
        model_b4_frz = train_head_model(mat_P0_device, tr_frozen_df, None, virus_to_idx, epochs, device, rank_loss_03, head_class=HeadR1)

        for m in ["B0", "B1", "B2_matched", "B4"]:
            results["Main_Table_3_Frozen_Future_Secondary"][m]["seeds"][str(seed)] = {}

        for y in forecast_years:
            gap = y - 2018
            te_df_y = df_all[df_all['pair_year'] == y].reset_index(drop=True)
            loader_y = DataLoader(IndexPairDataset(te_df_y, virus_to_idx), batch_size=2048, shuffle=False)

            # B0 eval
            X_y, Y_y = extract_mut_features(te_df_y, virus_to_seq, mode='key')
            p_b0 = reg_b0.predict(X_y)
            results["Main_Table_3_Frozen_Future_Secondary"]["B0"]["seeds"][str(seed)][f"Gap_{gap}yr_{y}"] = compute_comprehensive_metrics(Y_y, p_b0)

            # B1 eval
            t, p = eval_head(model_b1_frz, mat_P0_device, loader_y, device)
            results["Main_Table_3_Frozen_Future_Secondary"]["B1"]["seeds"][str(seed)][f"Gap_{gap}yr_{y}"] = compute_comprehensive_metrics(t, p)

            # B2_matched eval
            t, p = eval_model_p3(model_b2m_frz, mat_P0_device, all_tokens_device, loader_y, device)
            results["Main_Table_3_Frozen_Future_Secondary"]["B2_matched"]["seeds"][str(seed)][f"Gap_{gap}yr_{y}"] = compute_comprehensive_metrics(t, p)

            # B4 eval
            t, p = eval_head(model_b4_frz, mat_P0_device, loader_y, device)
            results["Main_Table_3_Frozen_Future_Secondary"]["B4"]["seeds"][str(seed)][f"Gap_{gap}yr_{y}"] = compute_comprehensive_metrics(t, p)

            print(f"  Gap {gap}yr ({y}) MAE -> B0: {results['Main_Table_3_Frozen_Future_Secondary']['B0']['seeds'][str(seed)][f'Gap_{gap}yr_{y}']['MAE']:.4f}, "
                  f"B1: {results['Main_Table_3_Frozen_Future_Secondary']['B1']['seeds'][str(seed)][f'Gap_{gap}yr_{y}']['MAE']:.4f}, "
                  f"B2m: {results['Main_Table_3_Frozen_Future_Secondary']['B2_matched']['seeds'][str(seed)][f'Gap_{gap}yr_{y}']['MAE']:.4f}", flush=True)

    # Compute Frozen summaries
    for m in ["B0", "B1", "B2_matched", "B4"]:
        for y in forecast_years:
            gap = y - 2018
            g_key = f"Gap_{gap}yr_{y}"
            g_maes = [results["Main_Table_3_Frozen_Future_Secondary"][m]["seeds"][str(s)][g_key]["MAE"] for s in SEEDS]
            results["Main_Table_3_Frozen_Future_Secondary"][m]["summary"][g_key] = {
                "mean_MAE": round(float(np.mean(g_maes)), 4),
                "std_MAE": round(float(np.std(g_maes)), 4)
            }

    # =========================================================================
    # PART 5: STATISTICAL SIGNIFICANCE TESTS
    # =========================================================================
    b1_all, b4_all, b2m_all, b0_all = [], [], [], []
    for s in SEEDS:
        for y in [2019, 2020, 2021, 2022]:
            b1_all.append(results["Main_Table_2_Rolling_Prospective_Primary"]["B1"]["seeds"][str(s)][f"<={y-1} -> {y}"]["MAE"])
            b4_all.append(results["Main_Table_2_Rolling_Prospective_Primary"]["B4"]["seeds"][str(s)][f"<={y-1} -> {y}"]["MAE"])
            b2m_all.append(results["Main_Table_2_Rolling_Prospective_Primary"]["B2_matched"]["seeds"][str(s)][f"<={y-1} -> {y}"]["MAE"])
            b0_all.append(results["Main_Table_2_Rolling_Prospective_Primary"]["B0"]["seeds"][str(s)][f"<={y-1} -> {y}"]["MAE"])

    t_14, p_14 = ttest_rel(b1_all, b4_all)
    t_12m, p_12m = ttest_rel(b1_all, b2m_all)
    t_10, p_10 = ttest_rel(b1_all, b0_all)

    results["Statistical_Significance_Tests"] = {
        "B1_vs_B4_Paired_TTest": {
            "N": len(b1_all), "mean_diff_B4_minus_B1": round(float(np.mean(np.array(b4_all) - np.array(b1_all))), 4),
            "t_stat": round(float(t_14), 4), "p_value": round(float(p_14), 5), "significant_at_005": bool(p_14 < 0.05)
        },
        "B1_vs_B2m_Paired_TTest": {
            "N": len(b1_all), "mean_diff_B2m_minus_B1": round(float(np.mean(np.array(b2m_all) - np.array(b1_all))), 4),
            "t_stat": round(float(t_12m), 4), "p_value": round(float(p_12m), 5), "significant_at_005": bool(p_12m < 0.05)
        },
        "B1_vs_B0_Paired_TTest": {
            "N": len(b1_all), "mean_diff_B0_minus_B1": round(float(np.mean(np.array(b0_all) - np.array(b1_all))), 4),
            "t_stat": round(float(t_10), 4), "p_value": round(float(p_10), 5), "significant_at_005": bool(p_10 < 0.05)
        }
    }

    # =========================================================================
    # PART 6: UNBIASED LANDMARK CARTOGRAPHY AUDIT
    # =========================================================================
    print("\n" + "="*70, flush=True)
    print(" PART 6: Landmark Cartography Audit (No Anchor Cheat)", flush=True)
    print("="*70, flush=True)

    pairs = set()
    for _, r in test_full.iterrows():
        pairs.add(tuple(sorted([r['virus1'], r['virus2']])))

    c = Counter(test_full['virus1']).copy()
    c.update(Counter(test_full['virus2']))
    candidates = [v for v, _ in c.most_common()]
    clique = []
    for cand in candidates:
        if all(tuple(sorted([cand, member])) in pairs for member in clique):
            clique.append(cand)

    n_clique = len(clique)
    v2idx = {v: i for i, v in enumerate(clique)}
    D_true = np.zeros((n_clique, n_clique), dtype=np.float64)
    sub_df = test_full[test_full['virus1'].isin(v2idx) & test_full['virus2'].isin(v2idx)]
    for _, row in sub_df.iterrows():
        i = v2idx[row['virus1']]; j = v2idx[row['virus2']]
        D_true[i, j] = row['distance']; D_true[j, i] = row['distance']

    model_fixed_b4 = train_head_model(mat_P0_device, dev_full, None, virus_to_idx, epochs, device, rank_loss_03, head_class=HeadR1)
    D_pred = np.zeros((n_clique, n_clique), dtype=np.float64)
    with torch.no_grad():
        for i in range(n_clique):
            u_emb = mat_P0_device[virus_to_idx[clique[i]]].unsqueeze(0)
            for j in range(i + 1, n_clique):
                v_emb = mat_P0_device[virus_to_idx[clique[j]]].unsqueeze(0)
                p1 = model_fixed_b4(u_emb, v_emb).item()
                p2 = model_fixed_b4(v_emb, u_emb).item()
                sym_d = max(0.0, (p1 + p2) / 2.0)
                D_pred[i, j] = sym_d; D_pred[j, i] = sym_d

    mds = MDS(n_components=2, metric=True, dissimilarity='precomputed', random_state=42, n_init=15, max_iter=1000)
    true_coords = mds.fit_transform(D_true)

    num_anchors = 15
    anchor_indices = np.linspace(0, n_clique - 1, num_anchors, dtype=int)
    anchor_coords = true_coords[anchor_indices]
    held_out_indices = np.array([i for i in range(n_clique) if i not in anchor_indices], dtype=int)

    pred_coords = np.zeros_like(true_coords)
    for i in range(n_clique):
        predicted_dists = D_pred[i, anchor_indices]
        pred_coords[i] = triangulate_strain(predicted_dists, anchor_coords)

    # 1. Biased disparity with anchor copy cheat
    biased_pred = pred_coords.copy()
    biased_pred[anchor_indices] = true_coords[anchor_indices]
    _, _, biased_disparity = procrustes(true_coords, biased_pred)

    # 2. Unbiased held-out disparity
    true_heldout = true_coords[held_out_indices]
    pred_heldout = pred_coords[held_out_indices]
    _, _, unbiased_heldout_disparity = procrustes(true_heldout, pred_heldout)

    # 3. 2D pairwise distance RMSE
    D_2d_true = np.sqrt(np.sum((true_heldout[:, None, :] - true_heldout[None, :, :])**2, axis=-1))
    D_2d_pred = np.sqrt(np.sum((pred_heldout[:, None, :] - pred_heldout[None, :, :])**2, axis=-1))
    heldout_triu = np.triu_indices(len(held_out_indices), k=1)
    heldout_2d_rmse = float(np.sqrt(np.mean((D_2d_true[heldout_triu] - D_2d_pred[heldout_triu])**2)))

    results["Unbiased_Landmark_Cartography"] = {
        "n_clique": n_clique,
        "n_anchors": num_anchors,
        "n_heldout": len(held_out_indices),
        "biased_all_points_disparity": round(float(biased_disparity), 4),
        "unbiased_heldout_disparity": round(float(unbiased_heldout_disparity), 4),
        "disparity_inflation_ratio": round(float(unbiased_heldout_disparity / biased_disparity), 4),
        "heldout_2d_pairwise_distance_rmse": round(heldout_2d_rmse, 4)
    }

    print(f"Biased Disparity (with anchor cheat): {biased_disparity:.4f}", flush=True)
    print(f"Unbiased Held-out Disparity: {unbiased_heldout_disparity:.4f} (Inflation: {unbiased_heldout_disparity/biased_disparity:.2f}x)", flush=True)
    print(f"Held-out 2D Pairwise RMSE: {heldout_2d_rmse:.4f} AU", flush=True)

    # Save to JSON
    out_json = os.path.join(reports_dir, 'blueprint_benchmark_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"\nSuccessfully executed all Blueprint benchmarks and saved results to {out_json} in {time.time() - t_start:.2f}s!", flush=True)

if __name__ == '__main__':
    main()
