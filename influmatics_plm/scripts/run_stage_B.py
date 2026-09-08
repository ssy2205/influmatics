import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))
plm_dir = os.path.abspath(os.path.join(script_dir, '..'))
if plm_dir not in sys.path:
    sys.path.append(plm_dir)

# Antigenic site definitions (Koel et al., 2013 and canonical sites A-E)
KOEL_7_SITES = {145, 155, 156, 158, 159, 189, 193}
EPITOPE_A = {122, 124, 126, 131, 133, 135, 137, 142, 143, 144, 145, 146}
EPITOPE_B = {155, 156, 158, 159, 160, 186, 187, 188, 189, 190, 192, 193, 196, 197}
EPITOPE_C = {45, 46, 47, 48, 50, 51, 53, 54, 275, 276, 278, 279, 280, 297, 299, 300, 305, 307, 308, 309, 310, 311, 312}
EPITOPE_D = {96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248}
EPITOPE_E = {57, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265}
ALL_EPITOPES = EPITOPE_A | EPITOPE_B | EPITOPE_C | EPITOPE_D | EPITOPE_E

# 0-indexed residue coordinates for 328 AA HA1 domain
EPI_INDICES = sorted([pos - 1 for pos in ALL_EPITOPES if pos - 1 < 328])
KEY_INDICES = sorted([pos - 1 for pos in KOEL_7_SITES if pos - 1 < 328])

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class HeadR1(nn.Module):
    """
    Stage A Winning Pair Architecture (R1: Symmetric).
    Input: [|u - v|, u * v]
    Activation: Softplus
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

class BiologyGuidedAttentionPooling(nn.Module):
    """
    P3: Learnable biology-guided attention pooling module.
    s_i = g(h_i) + beta_E * I_epi + beta_K * I_key
    alpha = softmax(s)
    z = sum_i alpha_i * h_i
    """
    def __init__(self, emb_dim=1280, seq_len=328):
        super().__init__()
        self.g = nn.Linear(emb_dim, 1)
        self.beta_E = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.beta_K = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
        # Indicator vectors (seq_len, 1)
        I_epi = torch.zeros(seq_len, 1, dtype=torch.float32)
        I_key = torch.zeros(seq_len, 1, dtype=torch.float32)
        I_epi[EPI_INDICES] = 1.0
        I_key[KEY_INDICES] = 1.0
        self.register_buffer('I_epi', I_epi)
        self.register_buffer('I_key', I_key)

    def forward(self, h):
        # h: (batch_size, seq_len, emb_dim)
        scores = self.g(h) + self.beta_E * self.I_epi + self.beta_K * self.I_key
        weights = torch.softmax(scores, dim=1) # (batch_size, seq_len, 1)
        pooled = torch.sum(h * weights, dim=1) # (batch_size, emb_dim)
        return pooled

class P3Model(nn.Module):
    """
    P3 Model with 2560-dim representation: [z_global; z_attn]
    """
    def __init__(self, emb_dim=1280, hidden_dim=932):
        super().__init__()
        self.pool = BiologyGuidedAttentionPooling(emb_dim=emb_dim, seq_len=328)
        self.head = HeadR1(emb_dim=emb_dim * 2, hidden_dim=hidden_dim)

    def forward(self, u_global, u_attn, v_global, v_attn):
        u = torch.cat([u_global, u_attn], dim=-1)
        v = torch.cat([v_global, v_attn], dim=-1)
        return self.head(u, v)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# Memory-Efficient Datasets (Store single tensor of indices, zero duplication)
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
# Training and Evaluation Helpers
# ---------------------------------------------------------------------------
def evaluate_static(model, emb_matrix_device, loader, device):
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

def train_and_eval_static(name, model_fn, emb_matrix_device, train_df, val_df, test_df, virus_to_idx, epochs, device):
    ds_train = IndexPairDataset(train_df, virus_to_idx)
    ds_val = IndexPairDataset(val_df, virus_to_idx)
    
    loader_train = DataLoader(ds_train, batch_size=256, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=256, shuffle=False)
    
    model = model_fn().to(device)
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
            optimizer.zero_grad()
            preds = model(emb1, emb2)
            loss = criterion(preds, dist)
            loss.backward()
            optimizer.step()

        val_mae, val_rho = evaluate_static(model, emb_matrix_device, loader_val, device)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rho = val_rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    test_mae, test_rho = None, None
    if test_df is not None:
        ds_test = IndexPairDataset(test_df, virus_to_idx)
        loader_test = DataLoader(ds_test, batch_size=256, shuffle=False)
        test_mae, test_rho = evaluate_static(model, emb_matrix_device, loader_test, device)

    return best_val_mae, best_val_rho, test_mae, test_rho

def evaluate_p3(model, global_matrix_device, all_tokens_cpu, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        # Pre-pool all unique viruses in chunks of 64
        pooled_attn_list = []
        for i in range(0, all_tokens_cpu.shape[0], 64):
            chunk = all_tokens_cpu[i:i+64].to(device).float()
            pooled_attn_list.append(model.pool(chunk))
        all_attn_device = torch.cat(pooled_attn_list, dim=0) # (1464, 1280) on device

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
    preds = np.array(preds)
    tgts = np.array(tgts)
    mae = mean_absolute_error(tgts, preds)
    rho, _ = spearmanr(tgts, preds)
    return float(mae), float(rho)

def train_and_eval_p3(name, global_matrix_device, all_tokens_cpu, train_df, val_df, test_df, virus_to_idx, epochs, device):
    ds_train = IndexPairDataset(train_df, virus_to_idx)
    ds_val = IndexPairDataset(val_df, virus_to_idx)
    
    loader_train = DataLoader(ds_train, batch_size=256, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=256, shuffle=False)

    model = P3Model(emb_dim=1280, hidden_dim=932).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    best_val_mae = float('inf')
    best_state = None
    best_val_rho = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for idx_A, idx_B, dist in loader_train:
            idx_A, idx_B, dist = idx_A.to(device), idx_B.to(device), dist.to(device)

            u_uniq, inv_u = torch.unique(idx_A, return_inverse=True)
            v_uniq, inv_v = torch.unique(idx_B, return_inverse=True)

            # Move only mini-batch unique virus tokens to device
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

        val_mae, val_rho = evaluate_p3(model, global_matrix_device, all_tokens_cpu, loader_val, device)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rho = val_rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_mae, test_rho = None, None
    if test_df is not None:
        ds_test = IndexPairDataset(test_df, virus_to_idx)
        loader_test = DataLoader(ds_test, batch_size=256, shuffle=False)
        test_mae, test_rho = evaluate_p3(model, global_matrix_device, all_tokens_cpu, loader_test, device)

    beta_E = float(model.pool.beta_E.item())
    beta_K = float(model.pool.beta_K.item())
    return best_val_mae, best_val_rho, test_mae, test_rho, beta_E, beta_K

# ---------------------------------------------------------------------------
# Classical Baselines Helpers (C0-a, C0-b)
# ---------------------------------------------------------------------------
def extract_mutations(df, virus_to_seq):
    X_a, X_b, Y = [], [], []
    for _, row in df.iterrows():
        vA, vB = row['node_A'], row['node_B']
        seqA, seqB = virus_to_seq[vA], virus_to_seq[vB]
        min_len = min(len(seqA), len(seqB))
        mut_all = sum(1 for i in range(min_len) if seqA[i] != seqB[i])
        mut_key = sum(1 for p in KOEL_7_SITES if p-1 < min_len and seqA[p-1] != seqB[p-1])
        X_a.append([mut_all])
        X_b.append([mut_key])
        Y.append(row['distance'])
    return np.array(X_a), np.array(X_b), np.array(Y)

def eval_classical(X_train, Y_train, X_val, Y_val):
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, Y_train)
    preds = reg.predict(X_val)
    mae = float(mean_absolute_error(Y_val, preds))
    rho, _ = spearmanr(Y_val, preds)
    return round(mae, 4), round(float(rho), 4)

# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device for Stage B Execution: {device}", flush=True)

    splits_dir = os.path.join(plm_dir, 'data', 'splits')
    cache_path = os.path.join(plm_dir, 'data', 'processed', 'stage_B_cache.pt')
    reports_dir = os.path.join(plm_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    print("Loading sequences and mapping...", flush=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("extract", os.path.join(plm_dir, 'scripts', '04_extract_embeddings.py'))
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)
    virus_to_seq = extract.get_virus_to_seq_mapping()

    print(f"Loading cached Stage B embeddings from {cache_path}...", flush=True)
    cache = torch.load(cache_path, map_location='cpu')
    common = set(cache['global'].keys()).intersection(set(virus_to_seq.keys()))
    print(f"Common verified nodes: {len(common):,}", flush=True)

    def filter_df(df):
        if 'node_A' not in df.columns:
            df['node_A'] = df['virus1']
        if 'node_B' not in df.columns:
            df['node_B'] = df['virus2']
        return df[df['node_A'].isin(common) & df['node_B'].isin(common)].reset_index(drop=True)

    # 1. Load 3-Fold Temporal Splits
    folds_data = []
    for f in [1, 2, 3]:
        f_tr = filter_df(pd.read_csv(os.path.join(splits_dir, f'cohort2_fold{f}_train.csv.gz')))
        f_va = filter_df(pd.read_csv(os.path.join(splits_dir, f'cohort2_fold{f}_val.csv.gz')))
        folds_data.append((f_tr, f_va))
        print(f"Fold {f}: Train pairs={len(f_tr):,}, Val pairs={len(f_va):,}", flush=True)

    # 2. Load Full Dev and Test Splits
    train_full = filter_df(pd.read_csv(os.path.join(splits_dir, 'cohort2_train.csv.gz')))
    val_full = filter_df(pd.read_csv(os.path.join(splits_dir, 'cohort2_val.csv.gz')))
    test_full = filter_df(pd.read_csv(os.path.join(splits_dir, 'cohort2_test.csv.gz')))
    print(f"Full Dev Set (Train={len(train_full):,}, Val={len(val_full):,}), Test Set={len(test_full):,}", flush=True)

    # Prepare shared compact matrices
    unique_viruses = sorted(list(common))
    virus_to_idx = {v: i for i, v in enumerate(unique_viruses)}

    mat_global_cpu = torch.stack([cache['global'][v] for v in unique_viruses], dim=0) # (1464, 1280)
    mat_epi_cpu = torch.stack([cache['epi'][v] for v in unique_viruses], dim=0)       # (1464, 1280)
    mat_key_cpu = torch.stack([cache['key'][v] for v in unique_viruses], dim=0)       # (1464, 1280)
    all_tokens_cpu = torch.stack([cache['tokens'][v] for v in unique_viruses], dim=0) # (1464, 328, 1280) half on CPU

    mat_P0_device = mat_global_cpu.to(device)
    mat_P1_device = torch.cat([mat_global_cpu, mat_epi_cpu], dim=-1).to(device)
    mat_P2_device = torch.cat([mat_global_cpu, mat_epi_cpu, mat_key_cpu], dim=-1).to(device)
    print("Shared embedding matrices successfully prepared on device.", flush=True)

    results = {
        "C0-a": {"Description": "전장 서열 변이 수(Hamming) + Ridge", "Trainable_Parameters": 1, "Temporal_CV": {}, "Test_Evaluation": {}},
        "C0-b": {"Description": "7대 핵심 위치 변이 수 + Ridge", "Trainable_Parameters": 1, "Temporal_CV": {}, "Test_Evaluation": {}},
        "P0": {"Description": "Global Mean (329 AA 단순 평균, 1280-dim)", "Embedding_Dim": 1280, "Trainable_Parameters": count_params(HeadR1(1280, 932)), "Temporal_CV": {}, "Test_Evaluation": {}},
        "P1": {"Description": "Global + Canonical Epitope ([z_global; z_epi], 2560-dim)", "Embedding_Dim": 2560, "Trainable_Parameters": count_params(HeadR1(2560, 932)), "Temporal_CV": {}, "Test_Evaluation": {}},
        "P2": {"Description": "Global + Epitope + Key Sites ([z_global; z_epi; z_key], 3840-dim)", "Embedding_Dim": 3840, "Trainable_Parameters": count_params(HeadR1(3840, 932)), "Temporal_CV": {}, "Test_Evaluation": {}},
        "P3": {"Description": "Learnable Attention (s_i = g(h_i) + beta_E * I_epi + beta_K * I_key 학습, 2560-dim)", "Embedding_Dim": 2560, "Trainable_Parameters": count_params(P3Model(1280, 932)), "Temporal_CV": {}, "Test_Evaluation": {}}
    }

    epochs = 8

    # =========================================================================
    # PART 1: Classical Baselines Evaluation across 3-Fold Temporal CV & Test
    # =========================================================================
    print("\n=======================================================", flush=True)
    print("   Evaluating Classical Baselines (C0-a, C0-b)", flush=True)
    print("=======================================================", flush=True)
    for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
        X_tr_a, X_tr_b, Y_tr = extract_mutations(f_tr, virus_to_seq)
        X_va_a, X_va_b, Y_va = extract_mutations(f_va, virus_to_seq)

        mae_a, rho_a = eval_classical(X_tr_a, Y_tr, X_va_a, Y_va)
        mae_b, rho_b = eval_classical(X_tr_b, Y_tr, X_va_b, Y_va)

        results["C0-a"]["Temporal_CV"][f"Fold_{f_idx}"] = {"MAE": mae_a, "Spearman_rho": rho_a}
        results["C0-b"]["Temporal_CV"][f"Fold_{f_idx}"] = {"MAE": mae_b, "Spearman_rho": rho_b}
        print(f"Fold {f_idx} | C0-a (MAE: {mae_a:.4f}, Rho: {rho_a:.4f}) | C0-b (MAE: {mae_b:.4f}, Rho: {rho_b:.4f})", flush=True)

    # Macro-average for C0-a, C0-b
    for c_name in ["C0-a", "C0-b"]:
        macro_mae = np.mean([results[c_name]["Temporal_CV"][f"Fold_{i}"]["MAE"] for i in [1, 2, 3]])
        macro_rho = np.mean([results[c_name]["Temporal_CV"][f"Fold_{i}"]["Spearman_rho"] for i in [1, 2, 3]])
        results[c_name]["Temporal_CV"]["Macro_MAE"] = round(float(macro_mae), 4)
        results[c_name]["Temporal_CV"]["Macro_Spearman_rho"] = round(float(macro_rho), 4)
        print(f">> {c_name} Temporal CV Macro -> MAE: {macro_mae:.4f}, Spearman Rho: {macro_rho:.4f}", flush=True)

    # Test evaluation for C0-a, C0-b (Trained on full dev train, evaluated on 2019-2022 test)
    X_full_a, X_full_b, Y_full = extract_mutations(train_full, virus_to_seq)
    X_test_a, X_test_b, Y_test = extract_mutations(test_full, virus_to_seq)

    t_mae_a, t_rho_a = eval_classical(X_full_a, Y_full, X_test_a, Y_test)
    t_mae_b, t_rho_b = eval_classical(X_full_b, Y_full, X_test_b, Y_test)

    results["C0-a"]["Test_Evaluation"] = {"MAE": t_mae_a, "Spearman_rho": t_rho_a}
    results["C0-b"]["Test_Evaluation"] = {"MAE": t_mae_b, "Spearman_rho": t_rho_b}
    print(f">> C0-a Test -> MAE: {t_mae_a:.4f}, Spearman Rho: {t_rho_a:.4f}", flush=True)
    print(f">> C0-b Test -> MAE: {t_mae_b:.4f}, Spearman Rho: {t_rho_b:.4f}", flush=True)

    # =========================================================================
    # PART 2: Pooling Models (P0, P1, P2, P3) 3-Fold Temporal CV
    # =========================================================================
    print("\n=======================================================", flush=True)
    print("   Evaluating 4 Pooling Models across 3-Fold Temporal CV", flush=True)
    print("=======================================================", flush=True)
    static_configs = [
        ("P0", lambda: HeadR1(emb_dim=1280, hidden_dim=932), mat_P0_device),
        ("P1", lambda: HeadR1(emb_dim=2560, hidden_dim=932), mat_P1_device),
        ("P2", lambda: HeadR1(emb_dim=3840, hidden_dim=932), mat_P2_device),
    ]

    for f_idx, (f_tr, f_va) in enumerate(folds_data, 1):
        print(f"\n--- Running Fold {f_idx}/3 ---", flush=True)
        # Static models (P0, P1, P2)
        for p_name, model_fn, mat_device in static_configs:
            v_mae, v_rho, _, _ = train_and_eval_static(
                f"{p_name}_Fold{f_idx}", model_fn, mat_device, f_tr, f_va, None, virus_to_idx, epochs, device
            )
            results[p_name]["Temporal_CV"][f"Fold_{f_idx}"] = {"MAE": round(v_mae, 4), "Spearman_rho": round(v_rho, 4)}
            print(f"Fold {f_idx} | {p_name} -> Val MAE: {v_mae:.4f}, Rho: {v_rho:.4f}", flush=True)

        # Learnable Attention (P3)
        v_mae_p3, v_rho_p3, _, _, beta_e, beta_k = train_and_eval_p3(
            f"P3_Fold{f_idx}", mat_P0_device, all_tokens_cpu, f_tr, f_va, None, virus_to_idx, epochs, device
        )
        results["P3"]["Temporal_CV"][f"Fold_{f_idx}"] = {
            "MAE": round(v_mae_p3, 4), "Spearman_rho": round(v_rho_p3, 4),
            "Learned_beta_E": round(beta_e, 4), "Learned_beta_K": round(beta_k, 4)
        }
        print(f"Fold {f_idx} | P3 -> Val MAE: {v_mae_p3:.4f}, Rho: {v_rho_p3:.4f} (beta_E: {beta_e:.4f}, beta_K: {beta_k:.4f})", flush=True)

    # Compute Macro-Averages for P0, P1, P2, P3
    for p_name in ["P0", "P1", "P2", "P3"]:
        macro_mae = np.mean([results[p_name]["Temporal_CV"][f"Fold_{i}"]["MAE"] for i in [1, 2, 3]])
        macro_rho = np.mean([results[p_name]["Temporal_CV"][f"Fold_{i}"]["Spearman_rho"] for i in [1, 2, 3]])
        results[p_name]["Temporal_CV"]["Macro_MAE"] = round(float(macro_mae), 4)
        results[p_name]["Temporal_CV"]["Macro_Spearman_rho"] = round(float(macro_rho), 4)
        print(f">> {p_name} Temporal CV Macro -> MAE: {macro_mae:.4f}, Spearman Rho: {macro_rho:.4f}", flush=True)

    # =========================================================================
    # PART 3: Independent 2019~2022 Test Evaluation for P0, P1, P2, P3
    # =========================================================================
    print("\n=======================================================", flush=True)
    print("   Evaluating 4 Pooling Models on Independent Test Set", flush=True)
    print("=======================================================", flush=True)
    for p_name, model_fn, mat_device in static_configs:
        v_mae, v_rho, t_mae, t_rho = train_and_eval_static(
            f"{p_name}_Test", model_fn, mat_device, train_full, val_full, test_full, virus_to_idx, epochs, device
        )
        results[p_name]["Test_Evaluation"] = {
            "MAE": round(t_mae, 4),
            "Spearman_rho": round(t_rho, 4)
        }
        print(f">> {p_name} Final Test -> MAE: {t_mae:.4f}, Spearman Rho: {t_rho:.4f}", flush=True)

    v_mae_p3, v_rho_p3, t_mae_p3, t_rho_p3, beta_e, beta_k = train_and_eval_p3(
        "P3_Test", mat_P0_device, all_tokens_cpu, train_full, val_full, test_full, virus_to_idx, epochs, device
    )
    results["P3"]["Test_Evaluation"] = {
        "MAE": round(t_mae_p3, 4),
        "Spearman_rho": round(t_rho_p3, 4),
        "Learned_beta_E": round(beta_e, 4),
        "Learned_beta_K": round(beta_k, 4)
    }
    print(f">> P3 Final Test -> MAE: {t_mae_p3:.4f}, Spearman Rho: {t_rho_p3:.4f} (beta_E: {beta_e:.4f}, beta_K: {beta_k:.4f})", flush=True)

    # Save to final reports JSON
    out_json = os.path.join(reports_dir, 'stage_B_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"\nSuccessfully saved full unified Stage B results to {out_json}", flush=True)

if __name__ == '__main__':
    main()
