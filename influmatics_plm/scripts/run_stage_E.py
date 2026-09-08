import os
import sys
import json
import time
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import procrustes
from scipy.optimize import minimize
from sklearn.manifold import MDS
from collections import Counter

# ---------------------------------------------------------------------------
# 1. Strict Reproducibility Seeds
# ---------------------------------------------------------------------------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seeds(42)

script_dir = os.path.dirname(os.path.abspath(__file__))
plm_dir = os.path.abspath(os.path.join(script_dir, '..'))
if plm_dir not in sys.path:
    sys.path.append(plm_dir)

from scripts.combined_ranking_loss import CombinedRankingLoss

KOEL_7_SITES = {145, 155, 156, 158, 159, 189, 193}
EPITOPE_A = {122, 124, 126, 131, 133, 135, 137, 142, 143, 144, 145, 146}
EPITOPE_B = {155, 156, 158, 159, 160, 186, 187, 188, 189, 190, 192, 193, 196, 197}
EPITOPE_C = {45, 46, 47, 48, 50, 51, 53, 54, 275, 276, 278, 279, 280, 297, 299, 300, 305, 307, 308, 309, 310, 311, 312}
EPITOPE_D = {96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248}
EPITOPE_E = {57, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265}
ALL_EPITOPES = EPITOPE_A | EPITOPE_B | EPITOPE_C | EPITOPE_D | EPITOPE_E

EPI_INDICES = sorted([pos - 1 for pos in ALL_EPITOPES if pos - 1 < 328])
KEY_INDICES = sorted([pos - 1 for pos in KOEL_7_SITES if pos - 1 < 328])

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class HeadR1(nn.Module):
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

class BiologyGuidedAttentionPooling(nn.Module):
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

def compute_metrics(tgts, preds):
    mae = float(mean_absolute_error(tgts, preds))
    rho, _ = spearmanr(tgts, preds)
    rho = float(rho) if not np.isnan(rho) else 0.0
    
    # 2 AU binary threshold for AUROC
    binary_true = (tgts >= 2.0).astype(int)
    if len(np.unique(binary_true)) > 1:
        auroc = float(roc_auc_score(binary_true, preds))
    else:
        auroc = float('nan')
    return round(mae, 4), round(rho, 4), round(auroc, 4)

# ---------------------------------------------------------------------------
# Training Helpers
# ---------------------------------------------------------------------------
def eval_model_p0(model, emb_matrix_device, loader, device):
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
    return np.array(tgts), np.array(preds)

def train_p0_family(emb_matrix_device, train_df, val_df, virus_to_idx, epochs, device, criterion):
    ds_train = IndexPairDataset(train_df, virus_to_idx)
    ds_val = IndexPairDataset(val_df, virus_to_idx) if val_df is not None else None
    
    loader_train = DataLoader(ds_train, batch_size=256, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=256, shuffle=False) if ds_val is not None else None

    model = HeadR1(emb_dim=1280, hidden_dim=932).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_mae = float('inf')
    best_state = None

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

        if loader_val is not None:
            tgts, preds = eval_model_p0(model, emb_matrix_device, loader_val, device)
            val_mae = mean_absolute_error(tgts, preds)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model

def eval_model_p3(model, global_matrix_device, all_tokens_cpu, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        pooled_attn_list = []
        for i in range(0, all_tokens_cpu.shape[0], 64):
            chunk = all_tokens_cpu[i:i+64].to(device).float()
            pooled_attn_list.append(model.pool(chunk))
        all_attn_device = torch.cat(pooled_attn_list, dim=0)

        for idx_A, idx_B, dist in loader:
            idx_A, idx_B = idx_A.to(device), idx_B.to(device)
            u_glob = global_matrix_device[idx_A]
            v_glob = global_matrix_device[idx_B]
            u_attn = all_attn_device[idx_A]
            v_attn = all_attn_device[idx_B]

            p1 = model(u_glob, u_attn, v_glob, v_attn)
            p2 = model(v_glob, v_attn, u_glob, u_attn)
            p = torch.clamp((p1 + p2) / 2.0, min=0.0)
            preds.extend(p.cpu().numpy())
            tgts.extend(dist.numpy())
    return np.array(tgts), np.array(preds)

def train_p3_model(global_matrix_device, all_tokens_cpu, train_df, val_df, virus_to_idx, epochs, device):
    ds_train = IndexPairDataset(train_df, virus_to_idx)
    ds_val = IndexPairDataset(val_df, virus_to_idx) if val_df is not None else None

    loader_train = DataLoader(ds_train, batch_size=256, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=256, shuffle=False) if ds_val is not None else None

    model = P3Model(emb_dim=1280, hidden_dim=932).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    best_val_mae = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for idx_A, idx_B, dist in loader_train:
            idx_A, idx_B, dist = idx_A.to(device), idx_B.to(device), dist.to(device)
            u_uniq, inv_u = torch.unique(idx_A, return_inverse=True)
            v_uniq, inv_v = torch.unique(idx_B, return_inverse=True)

            tok_u = all_tokens_cpu[u_uniq.cpu()].to(device).float()
            tok_v = all_tokens_cpu[v_uniq.cpu()].to(device).float()

            u_attn = model.pool(tok_u)[inv_u]
            v_attn = model.pool(tok_v)[inv_v]
            u_glob = global_matrix_device[idx_A]
            v_glob = global_matrix_device[idx_B]

            optimizer.zero_grad()
            preds = model(u_glob, u_attn, v_glob, v_attn)
            loss = criterion(preds, dist)
            loss.backward()
            optimizer.step()

        if loader_val is not None:
            tgts, preds = eval_model_p3(model, global_matrix_device, all_tokens_cpu, loader_val, device)
            val_mae = mean_absolute_error(tgts, preds)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model

# ---------------------------------------------------------------------------
# Classical Baseline Helper (B0: C0-b)
# ---------------------------------------------------------------------------
def extract_mut_key(df, virus_to_seq):
    X, Y = [], []
    for _, row in df.iterrows():
        seqA = virus_to_seq[row['node_A']]
        seqB = virus_to_seq[row['node_B']]
        min_len = min(len(seqA), len(seqB))
        mut_key = sum(1 for p in KOEL_7_SITES if p - 1 < min_len and seqA[p - 1] != seqB[p - 1])
        X.append([mut_key])
        Y.append(row['distance'])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# ---------------------------------------------------------------------------
# Triangulation & SVG Map Functions
# ---------------------------------------------------------------------------
def triangulate_strain(predicted_distances, anchor_coords):
    def loss(pos):
        dists = np.sqrt(np.sum((anchor_coords - pos)**2, axis=1))
        return np.mean((dists - predicted_distances)**2)
    init_pos = np.mean(anchor_coords, axis=0)
    res = minimize(loss, init_pos, method='BFGS')
    return res.x

def generate_landmark_svg(true_coords, pred_coords, proc_true, proc_pred, years, disparity, out_svg_path):
    width, height = 1500, 520
    panel_w, panel_h = 460, 420
    color_map = {2019: "#3b82f6", 2020: "#10b981", 2021: "#f59e0b", 2022: "#ef4444"}
    
    def scale_coords(coords, target_box):
        min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
        min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
        span_x = max(max_x - min_x, 1e-5)
        span_y = max(max_y - min_y, 1e-5)
        bx, by, bw, bh = target_box
        margin = 35
        sx = (coords[:, 0] - min_x) / span_x * (bw - 2 * margin) + bx + margin
        sy = (coords[:, 1] - min_y) / span_y * (bh - 2 * margin) + by + margin
        return sx, sy

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<style>',
        '  .title { font-family: -apple-system, sans-serif; font-size: 14px; font-weight: bold; fill: #1e3a8a; }',
        '  .legend { font-family: -apple-system, sans-serif; font-size: 11px; fill: #334155; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>'
    ]

    panels = [
        ("(A) Uncontaminated Ground-Truth MDS (100% Observed Test Clique)", (30, 60, panel_w, panel_h), true_coords, False),
        ("(B) Landmark Triangulation Map (Anchored MDS, 15 Anchors)", (520, 60, panel_w, panel_h), pred_coords, False),
        (f"(C) Procrustes Superposition Overlay (Disparity: {disparity:.4f})", (1010, 60, panel_w, panel_h), None, True)
    ]

    for p_title, (bx, by, bw, bh), coords, is_overlay in panels:
        svg_parts.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="#f8fafc" stroke="#e2e8f0" rx="8"/>')
        svg_parts.append(f'<text x="{bx+15}" y="{by-15}" class="title">{p_title}</text>')
        for gx in range(bx + 40, bx + bw - 20, 80):
            svg_parts.append(f'<line x1="{gx}" y1="{by+20}" x2="{gx}" y2="{by+bh-20}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        for gy in range(by + 40, by + bh - 20, 80):
            svg_parts.append(f'<line x1="{bx+20}" y1="{gy}" x2="{bx+bw-20}" y2="{gy}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')

        if not is_overlay:
            sx, sy = scale_coords(coords, (bx, by, bw, bh))
            for i in range(len(coords)):
                col = color_map.get(years[i], "#64748b")
                svg_parts.append(f'<circle cx="{sx[i]:.1f}" cy="{sy[i]:.1f}" r="5" fill="{col}" fill-opacity="0.85" stroke="#ffffff" stroke-width="1"/>')
        else:
            all_pts = np.vstack([proc_true, proc_pred])
            all_sx, all_sy = scale_coords(all_pts, (bx, by, bw, bh))
            n = len(proc_true)
            t_sx, t_sy = all_sx[:n], all_sy[:n]
            p_sx, p_sy = all_sx[n:], all_sy[n:]
            for i in range(n):
                svg_parts.append(f'<line x1="{t_sx[i]:.1f}" y1="{t_sy[i]:.1f}" x2="{p_sx[i]:.1f}" y2="{p_sy[i]:.1f}" stroke="#64748b" stroke-width="0.8" stroke-opacity="0.4"/>')
            for i in range(n):
                svg_parts.append(f'<circle cx="{t_sx[i]:.1f}" cy="{t_sy[i]:.1f}" r="4" fill="#94a3b8" fill-opacity="0.6"/>')
            for i in range(n):
                col = color_map.get(years[i], "#64748b")
                svg_parts.append(f'<circle cx="{p_sx[i]:.1f}" cy="{p_sy[i]:.1f}" r="5" fill="{col}" fill-opacity="0.9" stroke="#ffffff" stroke-width="1"/>')

    legend_y = 500
    svg_parts.append(f'<text x="50" y="{legend_y}" class="legend" font-weight="bold">Strain Year:</text>')
    lx = 140
    for yr, col in color_map.items():
        svg_parts.append(f'<circle cx="{lx}" cy="{legend_y-4}" r="5" fill="{col}"/>')
        svg_parts.append(f'<text x="{lx+10}" y="{legend_y}" class="legend">{yr}</text>')
        lx += 75
    svg_parts.append('</svg>')
    
    with open(out_svg_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(svg_parts))
    print(f"Landmark MDS SVG successfully saved to {out_svg_path}", flush=True)

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device for Stage E Execution: {device}", flush=True)

    data_dir = os.path.join(plm_dir, 'data')
    splits_dir = os.path.join(data_dir, 'splits')
    reports_dir = os.path.join(plm_dir, 'reports')
    figures_dir = os.path.join(plm_dir, 'figures')
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load mappings & cached representations
    print("Loading sequences and mapping...", flush=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("extract", os.path.join(plm_dir, 'scripts', '04_extract_embeddings.py'))
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)
    virus_to_seq = extract.get_virus_to_seq_mapping()

    cache_path = os.path.join(data_dir, 'processed', 'stage_B_cache.pt')
    cache = torch.load(cache_path, weights_only=False)
    common = sorted(list(set(cache['global'].keys()).intersection(set(virus_to_seq.keys()))))
    virus_to_idx = {v: i for i, v in enumerate(common)}
    N = len(common)
    print(f"Common verified viruses: {N:,}", flush=True)

    mat_P0_device = torch.stack([cache['global'][v] for v in common], dim=0).to(device)
    all_tokens_cpu = torch.stack([cache['tokens'][v] for v in common], dim=0)

    def filter_df(df):
        if 'node_A' not in df.columns:
            df['node_A'] = df['virus1']
        if 'node_B' not in df.columns:
            df['node_B'] = df['virus2']
        return df[df['node_A'].isin(virus_to_idx) & df['node_B'].isin(virus_to_idx)].reset_index(drop=True)

    # 2. Load splits
    folds_data = []
    for f in [1, 2, 3]:
        f_tr = filter_df(pd.read_csv(os.path.join(splits_dir, f'cohort2_fold{f}_train.csv.gz')))
        f_va = filter_df(pd.read_csv(os.path.join(splits_dir, f'cohort2_fold{f}_val.csv.gz')))
        folds_data.append((f_tr, f_va))

    train_full = filter_df(pd.read_csv(os.path.join(splits_dir, 'cohort2_train.csv.gz')))
    val_full = filter_df(pd.read_csv(os.path.join(splits_dir, 'cohort2_val.csv.gz')))
    dev_full = pd.concat([train_full, val_full], ignore_index=True)
    test_full = filter_df(pd.read_csv(os.path.join(splits_dir, 'cohort2_test.csv.gz')))

    # Combine all data to support exact yearly rolling backtests
    df_all = pd.concat([dev_full, test_full], ignore_index=True)
    df_all['pair_year'] = df_all[['year_A', 'year_B']].max(axis=1)
    test_full['pair_year'] = test_full[['year_A', 'year_B']].max(axis=1)

    print(f"Total dataset pairs: {len(df_all):,} (Dev <= 2018: {len(dev_full):,}, Test 2019-2022: {len(test_full):,})", flush=True)

    # Output structure
    results = {
        "metadata": {
            "evaluation_type": "Stage E Final Comprehensive Benchmark",
            "random_seed": 42,
            "terminology_mandate": "Retrospective Rolling Backtest (Formerly labeled prospective)",
            "selection_bias_disclosure": "The 2019~2022 test partition is a fixed historical holdout reused for longitudinal monitoring. Hyperparameter selection was strictly frozen on the 2003~2018 3-Fold Temporal CV.",
            "models": {
                "B0": "C0-b: Koel 7 Key-Site Mutation Count + Ridge Regression",
                "B1": "P0: ESM-2 650M Global Mean + R1 Symmetric Head + Smooth L1",
                "B2": "P3: Biology-guided Attention + R1 Symmetric Head + Smooth L1",
                "B4": "Stage D Winner: ESM-2 650M Global Mean + R1 Head + CombinedRankingLoss (lambda=0.3)"
            }
        },
        "Part1_Temporal_Expanding_Window_CV": {},
        "Part2_Retrospective_Rolling_Backtest": {},
        "Part3_Fixed_Dev_Time_Decay": {},
        "Part4_Landmark_MDS_Triangulation": {}
    }

    epochs = 8

    # =========================================================================
    # PART 1: 2003~2018 3-Fold Temporal Expanding-Window CV
    # =========================================================================
    print("\n=======================================================", flush=True)
    print("   PART 1: 2003~2018 3-Fold Temporal Expanding-Window CV", flush=True)
    print("=======================================================", flush=True)

    # 1. B0 (C0-b: Ridge)
    set_seeds(42)
    b0_cv_results = {}
    for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
        X_tr, Y_tr = extract_mut_key(f_tr, virus_to_seq)
        X_va, Y_va = extract_mut_key(f_va, virus_to_seq)
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, Y_tr)
        preds = reg.predict(X_va)
        mae, rho, auroc = compute_metrics(Y_va, preds)
        b0_cv_results[f"Fold_{f_idx}"] = {"MAE": mae, "Spearman_rho": rho}
        print(f"Part 1 | Fold {f_idx} | B0 -> Val MAE: {mae:.4f}, Rho: {rho:.4f}", flush=True)

    macro_mae = np.mean([b0_cv_results[f"Fold_{i}"]["MAE"] for i in [1, 2, 3]])
    macro_rho = np.mean([b0_cv_results[f"Fold_{i}"]["Spearman_rho"] for i in [1, 2, 3]])
    b0_cv_results["Macro_MAE"] = round(float(macro_mae), 4)
    b0_cv_results["Macro_Spearman_rho"] = round(float(macro_rho), 4)
    results["Part1_Temporal_Expanding_Window_CV"]["B0"] = b0_cv_results
    print(f">> Part 1 | B0 Macro CV -> MAE: {macro_mae:.4f}, Rho: {macro_rho:.4f}", flush=True)

    # 2. B1 (P0: Pure Smooth L1)
    set_seeds(42)
    b1_cv_results = {}
    smooth_l1 = nn.SmoothL1Loss()
    for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
        model = train_p0_family(mat_P0_device, f_tr, f_va, virus_to_idx, epochs, device, smooth_l1)
        ds_va = IndexPairDataset(f_va, virus_to_idx)
        loader_va = DataLoader(ds_va, batch_size=256, shuffle=False)
        tgts, preds = eval_model_p0(model, mat_P0_device, loader_va, device)
        mae, rho, auroc = compute_metrics(tgts, preds)
        b1_cv_results[f"Fold_{f_idx}"] = {"MAE": mae, "Spearman_rho": rho}
        print(f"Part 1 | Fold {f_idx} | B1 -> Val MAE: {mae:.4f}, Rho: {rho:.4f}", flush=True)

    macro_mae = np.mean([b1_cv_results[f"Fold_{i}"]["MAE"] for i in [1, 2, 3]])
    macro_rho = np.mean([b1_cv_results[f"Fold_{i}"]["Spearman_rho"] for i in [1, 2, 3]])
    b1_cv_results["Macro_MAE"] = round(float(macro_mae), 4)
    b1_cv_results["Macro_Spearman_rho"] = round(float(macro_rho), 4)
    results["Part1_Temporal_Expanding_Window_CV"]["B1"] = b1_cv_results
    print(f">> Part 1 | B1 Macro CV -> MAE: {macro_mae:.4f}, Rho: {macro_rho:.4f}", flush=True)

    # 3. B2 (P3: Biology-guided Attention)
    set_seeds(42)
    b2_cv_results = {}
    for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
        model = train_p3_model(mat_P0_device, all_tokens_cpu, f_tr, f_va, virus_to_idx, epochs, device)
        ds_va = IndexPairDataset(f_va, virus_to_idx)
        loader_va = DataLoader(ds_va, batch_size=256, shuffle=False)
        tgts, preds = eval_model_p3(model, mat_P0_device, all_tokens_cpu, loader_va, device)
        mae, rho, auroc = compute_metrics(tgts, preds)
        b2_cv_results[f"Fold_{f_idx}"] = {"MAE": mae, "Spearman_rho": rho}
        print(f"Part 1 | Fold {f_idx} | B2 -> Val MAE: {mae:.4f}, Rho: {rho:.4f}", flush=True)

    macro_mae = np.mean([b2_cv_results[f"Fold_{i}"]["MAE"] for i in [1, 2, 3]])
    macro_rho = np.mean([b2_cv_results[f"Fold_{i}"]["Spearman_rho"] for i in [1, 2, 3]])
    b2_cv_results["Macro_MAE"] = round(float(macro_mae), 4)
    b2_cv_results["Macro_Spearman_rho"] = round(float(macro_rho), 4)
    results["Part1_Temporal_Expanding_Window_CV"]["B2"] = b2_cv_results
    print(f">> Part 1 | B2 Macro CV -> MAE: {macro_mae:.4f}, Rho: {macro_rho:.4f}", flush=True)

    # 4. B4 (CombinedRankingLoss lambda=0.3)
    set_seeds(42)
    b4_cv_results = {}
    rank_loss_03 = CombinedRankingLoss(lambda_param=0.3, margin=0.1)
    for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
        model = train_p0_family(mat_P0_device, f_tr, f_va, virus_to_idx, epochs, device, rank_loss_03)
        ds_va = IndexPairDataset(f_va, virus_to_idx)
        loader_va = DataLoader(ds_va, batch_size=256, shuffle=False)
        tgts, preds = eval_model_p0(model, mat_P0_device, loader_va, device)
        mae, rho, auroc = compute_metrics(tgts, preds)
        b4_cv_results[f"Fold_{f_idx}"] = {"MAE": mae, "Spearman_rho": rho}
        print(f"Part 1 | Fold {f_idx} | B4 -> Val MAE: {mae:.4f}, Rho: {rho:.4f}", flush=True)

    macro_mae = np.mean([b4_cv_results[f"Fold_{i}"]["MAE"] for i in [1, 2, 3]])
    macro_rho = np.mean([b4_cv_results[f"Fold_{i}"]["Spearman_rho"] for i in [1, 2, 3]])
    b4_cv_results["Macro_MAE"] = round(float(macro_mae), 4)
    b4_cv_results["Macro_Spearman_rho"] = round(float(macro_rho), 4)
    results["Part1_Temporal_Expanding_Window_CV"]["B4"] = b4_cv_results
    print(f">> Part 1 | B4 Macro CV -> MAE: {macro_mae:.4f}, Rho: {macro_rho:.4f}", flush=True)

    # =========================================================================
    # PART 2: Retrospective Rolling Backtest (<=T -> T+1)
    # =========================================================================
    print("\n=======================================================", flush=True)
    print("   PART 2: Retrospective Rolling Backtest (<=T -> T+1)", flush=True)
    print("=======================================================", flush=True)
    backtest_windows = [
        (2018, 2019),
        (2019, 2020),
        (2020, 2021),
        (2021, 2022)
    ]

    for m_name in ["B0", "B1", "B2", "B4"]:
        results["Part2_Retrospective_Rolling_Backtest"][m_name] = {}

    for train_max_yr, test_yr in backtest_windows:
        window_key = f"<={train_max_yr} -> {test_yr}"
        print(f"\n--- Backtest Window: {window_key} ---", flush=True)
        tr_df = df_all[df_all['pair_year'] <= train_max_yr].reset_index(drop=True)
        te_df = df_all[df_all['pair_year'] == test_yr].reset_index(drop=True)
        ds_te = IndexPairDataset(te_df, virus_to_idx)
        loader_te = DataLoader(ds_te, batch_size=256, shuffle=False)

        # B0
        set_seeds(42)
        X_tr, Y_tr = extract_mut_key(tr_df, virus_to_seq)
        X_te, Y_te = extract_mut_key(te_df, virus_to_seq)
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, Y_tr)
        preds_b0 = reg.predict(X_te)
        mae, rho, auroc = compute_metrics(Y_te, preds_b0)
        results["Part2_Retrospective_Rolling_Backtest"]["B0"][window_key] = {
            "MAE": mae, "Spearman_rho": rho, "AUROC_at_2AU": auroc
        }
        print(f"Backtest {window_key} | B0 -> MAE: {mae:.4f}, Rho: {rho:.4f}, AUROC: {auroc}", flush=True)

        # B1
        set_seeds(42)
        model_b1 = train_p0_family(mat_P0_device, tr_df, None, virus_to_idx, epochs, device, smooth_l1)
        tgts_b1, preds_b1 = eval_model_p0(model_b1, mat_P0_device, loader_te, device)
        mae, rho, auroc = compute_metrics(tgts_b1, preds_b1)
        results["Part2_Retrospective_Rolling_Backtest"]["B1"][window_key] = {
            "MAE": mae, "Spearman_rho": rho, "AUROC_at_2AU": auroc
        }
        print(f"Backtest {window_key} | B1 -> MAE: {mae:.4f}, Rho: {rho:.4f}, AUROC: {auroc}", flush=True)

        # B2
        set_seeds(42)
        model_b2 = train_p3_model(mat_P0_device, all_tokens_cpu, tr_df, None, virus_to_idx, epochs, device)
        tgts_b2, preds_b2 = eval_model_p3(model_b2, mat_P0_device, all_tokens_cpu, loader_te, device)
        mae, rho, auroc = compute_metrics(tgts_b2, preds_b2)
        results["Part2_Retrospective_Rolling_Backtest"]["B2"][window_key] = {
            "MAE": mae, "Spearman_rho": rho, "AUROC_at_2AU": auroc
        }
        print(f"Backtest {window_key} | B2 -> MAE: {mae:.4f}, Rho: {rho:.4f}, AUROC: {auroc}", flush=True)

        # B4
        set_seeds(42)
        model_b4 = train_p0_family(mat_P0_device, tr_df, None, virus_to_idx, epochs, device, rank_loss_03)
        tgts_b4, preds_b4 = eval_model_p0(model_b4, mat_P0_device, loader_te, device)
        mae, rho, auroc = compute_metrics(tgts_b4, preds_b4)
        results["Part2_Retrospective_Rolling_Backtest"]["B4"][window_key] = {
            "MAE": mae, "Spearman_rho": rho, "AUROC_at_2AU": auroc
        }
        print(f"Backtest {window_key} | B4 -> MAE: {mae:.4f}, Rho: {rho:.4f}, AUROC: {auroc}", flush=True)

    # Compute Macro Backtest Metrics
    for m_name in ["B0", "B1", "B2", "B4"]:
        m_maes = [results["Part2_Retrospective_Rolling_Backtest"][m_name][f"<={y-1} -> {y}"]["MAE"] for y in [2019, 2020, 2021, 2022]]
        m_rhos = [results["Part2_Retrospective_Rolling_Backtest"][m_name][f"<={y-1} -> {y}"]["Spearman_rho"] for y in [2019, 2020, 2021, 2022]]
        m_aurocs = [results["Part2_Retrospective_Rolling_Backtest"][m_name][f"<={y-1} -> {y}"]["AUROC_at_2AU"] for y in [2019, 2020, 2021, 2022]]
        results["Part2_Retrospective_Rolling_Backtest"][m_name]["Macro_Average"] = {
            "MAE": round(float(np.mean(m_maes)), 4),
            "Spearman_rho": round(float(np.mean(m_rhos)), 4),
            "AUROC_at_2AU": round(float(np.mean(m_aurocs)), 4)
        }

    # =========================================================================
    # PART 3: Fixed Base Model (<=2018) Time Decay Extrapolation
    # =========================================================================
    print("\n=======================================================", flush=True)
    print("   PART 3: Fixed Base Model (<=2018) Time Decay Extrapolation", flush=True)
    print("=======================================================", flush=True)
    dev_train_df = dev_full # all <= 2018

    # Train once on full dev
    set_seeds(42)
    X_dev, Y_dev = extract_mut_key(dev_train_df, virus_to_seq)
    reg_fixed_b0 = Ridge(alpha=1.0)
    reg_fixed_b0.fit(X_dev, Y_dev)

    set_seeds(42)
    model_fixed_b1 = train_p0_family(mat_P0_device, dev_train_df, None, virus_to_idx, epochs, device, smooth_l1)
    set_seeds(42)
    model_fixed_b2 = train_p3_model(mat_P0_device, all_tokens_cpu, dev_train_df, None, virus_to_idx, epochs, device)
    set_seeds(42)
    model_fixed_b4 = train_p0_family(mat_P0_device, dev_train_df, None, virus_to_idx, epochs, device, rank_loss_03)

    for m_name in ["B0", "B1", "B2", "B4"]:
        results["Part3_Fixed_Dev_Time_Decay"][m_name] = {}

    for horizon, yr in enumerate([2019, 2020, 2021, 2022], 1):
        yr_df = test_full[test_full['pair_year'] == yr].reset_index(drop=True)
        ds_yr = IndexPairDataset(yr_df, virus_to_idx)
        loader_yr = DataLoader(ds_yr, batch_size=256, shuffle=False)

        # B0
        X_yr, Y_yr = extract_mut_key(yr_df, virus_to_seq)
        preds_b0 = reg_fixed_b0.predict(X_yr)
        mae, rho, auroc = compute_metrics(Y_yr, preds_b0)
        results["Part3_Fixed_Dev_Time_Decay"]["B0"][f"Year_{yr}_(T+{horizon})"] = {
            "MAE": mae, "Spearman_rho": rho
        }

        # B1
        t_b1, p_b1 = eval_model_p0(model_fixed_b1, mat_P0_device, loader_yr, device)
        mae, rho, auroc = compute_metrics(t_b1, p_b1)
        results["Part3_Fixed_Dev_Time_Decay"]["B1"][f"Year_{yr}_(T+{horizon})"] = {
            "MAE": mae, "Spearman_rho": rho
        }

        # B2
        t_b2, p_b2 = eval_model_p3(model_fixed_b2, mat_P0_device, all_tokens_cpu, loader_yr, device)
        mae, rho, auroc = compute_metrics(t_b2, p_b2)
        results["Part3_Fixed_Dev_Time_Decay"]["B2"][f"Year_{yr}_(T+{horizon})"] = {
            "MAE": mae, "Spearman_rho": rho
        }

        # B4
        t_b4, p_b4 = eval_model_p0(model_fixed_b4, mat_P0_device, loader_yr, device)
        mae, rho, auroc = compute_metrics(t_b4, p_b4)
        results["Part3_Fixed_Dev_Time_Decay"]["B4"][f"Year_{yr}_(T+{horizon})"] = {
            "MAE": mae, "Spearman_rho": rho
        }
        print(f"Time Decay T+{horizon} ({yr}) -> B0: {results['Part3_Fixed_Dev_Time_Decay']['B0'][f'Year_{yr}_(T+{horizon})']['MAE']} | B1: {results['Part3_Fixed_Dev_Time_Decay']['B1'][f'Year_{yr}_(T+{horizon})']['MAE']} | B2: {results['Part3_Fixed_Dev_Time_Decay']['B2'][f'Year_{yr}_(T+{horizon})']['MAE']} | B4: {results['Part3_Fixed_Dev_Time_Decay']['B4'][f'Year_{yr}_(T+{horizon})']['MAE']}", flush=True)

    # =========================================================================
    # PART 4: Post-hoc Landmark Triangulation (Landmark MDS) 2D Map
    # =========================================================================
    print("\n=======================================================", flush=True)
    print("   PART 4: Post-hoc Landmark Triangulation (Landmark MDS) 2D Map", flush=True)
    print("=======================================================", flush=True)

    # 1. Extract 100% Fully Observed Clique from Test Set
    pairs = set()
    for _, r in test_full.iterrows():
        u, v = sorted([r['virus1'], r['virus2']])
        pairs.add((u, v))

    c = Counter(test_full['virus1']).copy()
    c.update(Counter(test_full['virus2']))
    candidates = [v for v, cnt in c.most_common()]

    clique = []
    for cand in candidates:
        if all(tuple(sorted([cand, member])) in pairs for member in clique):
            clique.append(cand)

    n_clique = len(clique)
    print(f"Identified Fully-Observed Test Clique: {n_clique} strains.", flush=True)
    v2idx = {v: i for i, v in enumerate(clique)}

    D_true = np.zeros((n_clique, n_clique), dtype=np.float64)
    sub_df = test_full[test_full['virus1'].isin(v2idx) & test_full['virus2'].isin(v2idx)]
    for _, row in sub_df.iterrows():
        i = v2idx[row['virus1']]
        j = v2idx[row['virus2']]
        D_true[i, j] = row['distance']
        D_true[j, i] = row['distance']

    # Predict Pairwise Matrix using B4 (Top Ranking Loss Winner)
    D_pred = np.zeros((n_clique, n_clique), dtype=np.float64)
    with torch.no_grad():
        for i in range(n_clique):
            u_emb = mat_P0_device[virus_to_idx[clique[i]]].unsqueeze(0)
            for j in range(i + 1, n_clique):
                v_emb = mat_P0_device[virus_to_idx[clique[j]]].unsqueeze(0)
                p1 = model_fixed_b4(u_emb, v_emb).item()
                p2 = model_fixed_b4(v_emb, u_emb).item()
                sym_d = max(0.0, (p1 + p2) / 2.0)
                D_pred[i, j] = sym_d
                D_pred[j, i] = sym_d

    # Ground Truth Metric MDS
    mds = MDS(n_components=2, metric=True, dissimilarity='precomputed', random_state=42, n_init=15, max_iter=1000)
    true_coords = mds.fit_transform(D_true)

    # 15 Anchor Landmark Triangulation
    num_anchors = min(15, n_clique)
    anchor_indices = np.linspace(0, n_clique - 1, num_anchors, dtype=int)
    anchor_coords = true_coords[anchor_indices]

    pred_coords = np.zeros_like(true_coords)
    for i in range(n_clique):
        if i in anchor_indices:
            pred_coords[i] = true_coords[i]
        else:
            predicted_dists_to_anchors = D_pred[i, anchor_indices]
            pred_coords[i] = triangulate_strain(predicted_dists_to_anchors, anchor_coords)

    # Procrustes Superposition
    proc_true, proc_pred, disparity = procrustes(true_coords, pred_coords)
    r_axis1, _ = pearsonr(proc_true[:, 0], proc_pred[:, 0])
    r_axis2, _ = pearsonr(proc_true[:, 1], proc_pred[:, 1])
    rho_axis1, _ = spearmanr(proc_true[:, 0], proc_pred[:, 0])
    rho_axis2, _ = spearmanr(proc_true[:, 1], proc_pred[:, 1])

    triu_idx = np.triu_indices(n_clique, k=1)
    clique_mae = float(mean_absolute_error(D_true[triu_idx], D_pred[triu_idx]))
    clique_rho, _ = spearmanr(D_true[triu_idx], D_pred[triu_idx])

    def parse_year(name):
        parts = str(name).strip().split('/')
        m = re.search(r'(\d+)$', parts[-1])
        if m:
            yy = int(m.group(1))
            return yy if yy >= 1000 else (2000 + yy if yy < 50 else 1900 + yy)
        return 2020

    clique_years = [parse_year(v) for v in clique]

    results["Part4_Landmark_MDS_Triangulation"] = {
        "model_used": "B4 (P0 + CombinedRankingLoss lambda=0.3)",
        "num_strains_in_clique": n_clique,
        "num_anchors": num_anchors,
        "clique_pairwise_mae": round(clique_mae, 4),
        "clique_pairwise_rho": round(float(clique_rho), 4),
        "procrustes_disparity": round(float(disparity), 4),
        "axis_1_pearson_r": round(float(r_axis1), 4),
        "axis_1_spearman_rho": round(float(rho_axis1), 4),
        "axis_2_pearson_r": round(float(r_axis2), 4),
        "axis_2_spearman_rho": round(float(rho_axis2), 4)
    }
    print(f">> Part 4 | Landmark MDS Disparity: {disparity:.4f}, Axis1 r: {r_axis1:.4f}, Axis2 r: {r_axis2:.4f}", flush=True)

    out_svg = os.path.join(figures_dir, 'stage_E_landmark_mds.svg')
    generate_landmark_svg(true_coords, pred_coords, proc_true, proc_pred, clique_years, disparity, out_svg)

    # 5. Save final report JSON
    out_json = os.path.join(reports_dir, 'stage_E_final_benchmarks.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"\nSuccessfully saved full Stage E final benchmarks to {out_json}", flush=True)

if __name__ == '__main__':
    main()
