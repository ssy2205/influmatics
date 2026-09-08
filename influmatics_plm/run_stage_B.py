import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from transformers import AutoTokenizer, EsmModel

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

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
    def __init__(self, emb_dim=1280, hidden_dim=932):
        super().__init__()
        self.pool = BiologyGuidedAttentionPooling(emb_dim=emb_dim, seq_len=328)
        self.head = HeadR1(emb_dim=emb_dim, hidden_dim=hidden_dim)

    def forward_pairs(self, u_pooled, v_pooled):
        return self.head(u_pooled, v_pooled)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class StaticVectorDataset(Dataset):
    def __init__(self, df, emb_dict):
        self.v1 = df['node_A'].values
        self.v2 = df['node_B'].values
        self.targets = df['distance'].values
        self.embs = emb_dict

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.embs[self.v1[idx]], self.embs[self.v2[idx]], torch.tensor(self.targets[idx], dtype=torch.float32)

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
def evaluate_static(model, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for emb1, emb2, dist in loader:
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

def train_static_model(name, model, train_loader, val_loader, test_loader, epochs, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    best_val_mae = float('inf')
    best_state = None
    best_val_rho = 0.0

    print(f"\n--- Training {name} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for emb1, emb2, dist in train_loader:
            emb1, emb2, dist = emb1.to(device), emb2.to(device), dist.to(device)
            optimizer.zero_grad()
            preds = model(emb1, emb2)
            loss = criterion(preds, dist)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(dist)
            n_samples += len(dist)

        val_mae, val_rho = evaluate_static(model, val_loader, device)
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_val_rho = val_rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        mark = " [*Best]" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/n_samples:.4f} | Val MAE: {val_mae:.4f}, Rho: {val_rho:.4f}{mark}")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_mae, test_rho = evaluate_static(model, test_loader, device)
    print(f">> {name} Final Test -> MAE: {test_mae:.4f}, Spearman Rho: {test_rho:.4f}")
    return test_mae, test_rho, best_val_mae, best_val_rho

def evaluate_p3(model, all_tokens_device, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        # Pre-pool all unique viruses once using current attention weights
        pooled_all = model.pool(all_tokens_device.float()) # (N_viruses, 1280)
        for idx_A, idx_B, dist in loader:
            u = pooled_all[idx_A]
            v = pooled_all[idx_B]
            p1 = model.head(u, v)
            p2 = model.head(v, u)
            p = torch.clamp((p1 + p2) / 2.0, min=0.0)
            preds.extend(p.cpu().numpy())
            tgts.extend(dist.numpy())
    preds = np.array(preds)
    tgts = np.array(tgts)
    mae = mean_absolute_error(tgts, preds)
    rho, _ = spearmanr(tgts, preds)
    return float(mae), float(rho)

def train_p3_model(model, all_tokens_device, train_loader, val_loader, test_loader, epochs, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    best_val_mae = float('inf')
    best_state = None
    best_val_rho = 0.0

    print(f"\n--- Training P3 (Biology-guided Learnable Attention) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for idx_A, idx_B, dist in train_loader:
            idx_A, idx_B, dist = idx_A.to(device), idx_B.to(device), dist.to(device)

            # Extract unique viruses in this mini-batch to pool efficiently
            u_uniq, inv_u = torch.unique(idx_A, return_inverse=True)
            v_uniq, inv_v = torch.unique(idx_B, return_inverse=True)

            u_pool = model.pool(all_tokens_device[u_uniq].float())
            v_pool = model.pool(all_tokens_device[v_uniq].float())

            u = u_pool[inv_u]
            v = v_pool[inv_v]

            optimizer.zero_grad()
            preds = model.head(u, v)
            loss = criterion(preds, dist)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(dist)
            n_samples += len(dist)

        val_mae, val_rho = evaluate_p3(model, all_tokens_device, val_loader, device)
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_val_rho = val_rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        mark = " [*Best]" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/n_samples:.4f} | Val MAE: {val_mae:.4f}, Rho: {val_rho:.4f}{mark} (beta_E: {model.pool.beta_E.item():.4f}, beta_K: {model.pool.beta_K.item():.4f})")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_mae, test_rho = evaluate_p3(model, all_tokens_device, test_loader, device)
    print(f">> P3 Final Test -> MAE: {test_mae:.4f}, Spearman Rho: {test_rho:.4f} (Learned beta_E: {model.pool.beta_E.item():.4f}, beta_K: {model.pool.beta_K.item():.4f})")
    return test_mae, test_rho, best_val_mae, best_val_rho

# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device for Stage B: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    splits_dir = os.path.join(base_dir, 'data', 'splits')
    cache_path = os.path.join(base_dir, 'data', 'processed', 'stage_B_cache.pt')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    print("Loading split data...")
    train_df = pd.read_csv(os.path.join(splits_dir, 'cohort2_train.csv.gz'))
    val_df = pd.read_csv(os.path.join(splits_dir, 'cohort2_val.csv.gz'))
    test_df = pd.read_csv(os.path.join(splits_dir, 'cohort2_test.csv.gz'))

    import importlib.util
    spec = importlib.util.spec_from_file_location("extract", os.path.join(base_dir, 'scripts', '04_extract_embeddings.py'))
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)
    virus_to_seq = extract.get_virus_to_seq_mapping()

    emb_mean_existing = torch.load(os.path.join(base_dir, 'data', 'processed', 'embeddings_esm2_650m.pt'), map_location='cpu')
    common = set(emb_mean_existing.keys()).intersection(set(virus_to_seq.keys()))

    def filter_df(df):
        return df[df['node_A'].isin(common) & df['node_B'].isin(common)].reset_index(drop=True)

    train_df = filter_df(train_df)
    val_df = filter_df(val_df)
    test_df = filter_df(test_df)
    print(f"Dataset split pairs: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    unique_viruses = sorted(list(
        set(train_df['node_A']).union(set(train_df['node_B']))
        .union(set(val_df['node_A'])).union(set(val_df['node_B']))
        .union(set(test_df['node_A'])).union(set(test_df['node_B']))
    ))
    print(f"Total unique viruses across splits: {len(unique_viruses):,}")

    # Check or generate cached embeddings
    if os.path.exists(cache_path):
        print(f"Loading cached Stage B embeddings from {cache_path}...")
        cache = torch.load(cache_path, map_location='cpu')
    else:
        print(f"Extracting Stage B per-residue and pooled embeddings for {len(unique_viruses)} viruses on {device}...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").half().to(device)
        esm_model.eval()

        global_dict = {}
        epi_dict = {}
        key_dict = {}
        tokens_dict = {}

        batch_size = 48
        t0 = time.time()
        with torch.inference_mode():
            for i in range(0, len(unique_viruses), batch_size):
                b_viruses = unique_viruses[i:i+batch_size]
                b_seqs = [virus_to_seq[v] for v in b_viruses]

                inputs = tokenizer(b_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = esm_model(**inputs)
                hidden = out.last_hidden_state # (B, L, 1280)
                mask = inputs['attention_mask']

                for b, v_name in enumerate(b_viruses):
                    seq_len = mask[b].sum().item()
                    valid_h = hidden[b, 1:seq_len-1, :] # (328, 1280)

                    z_glob = valid_h.mean(dim=0).float().cpu()
                    z_epi = valid_h[EPI_INDICES].mean(dim=0).float().cpu()
                    z_key = valid_h[KEY_INDICES].mean(dim=0).float().cpu()
                    toks = valid_h.half().cpu()

                    global_dict[v_name] = z_glob
                    epi_dict[v_name] = z_epi
                    key_dict[v_name] = z_key
                    tokens_dict[v_name] = toks

                if (i // batch_size) % 5 == 0 or (i + batch_size) >= len(unique_viruses):
                    elapsed = time.time() - t0
                    print(f"Extracted {min(i+batch_size, len(unique_viruses))}/{len(unique_viruses)} viruses ({elapsed:.1f}s)...")

        cache = {
            'global': global_dict,
            'epi': epi_dict,
            'key': key_dict,
            'tokens': tokens_dict
        }
        torch.save(cache, cache_path)
        print(f"Successfully saved Stage B cache to {cache_path} ({os.path.getsize(cache_path)/1e9:.2f} GB)")

    # Prepare representations for P0, P1, P2
    print("\nPreparing pooled representations for P0, P1, P2...")
    rep_P0 = {v: cache['global'][v] for v in unique_viruses}
    rep_P1 = {v: torch.cat([cache['global'][v], cache['epi'][v]], dim=-1) for v in unique_viruses}
    rep_P2 = {v: torch.cat([cache['global'][v], cache['epi'][v], cache['key'][v]], dim=-1) for v in unique_viruses}

    # Index tensor for P3
    virus_to_idx = {v: i for i, v in enumerate(unique_viruses)}
    all_tokens_tensor = torch.stack([cache['tokens'][v] for v in unique_viruses], dim=0).to(device) # (N, 328, 1280) half

    results = {}
    epochs = 8

    # -------------------------------------------------------------
    # 1. P0: Global Mean (기준선, 329 AA 단순 평균)
    # -------------------------------------------------------------
    ds_train_P0 = StaticVectorDataset(train_df, rep_P0)
    ds_val_P0 = StaticVectorDataset(val_df, rep_P0)
    ds_test_P0 = StaticVectorDataset(test_df, rep_P0)

    loader_train_P0 = DataLoader(ds_train_P0, batch_size=256, shuffle=True)
    loader_val_P0 = DataLoader(ds_val_P0, batch_size=256, shuffle=False)
    loader_test_P0 = DataLoader(ds_test_P0, batch_size=256, shuffle=False)

    model_P0 = HeadR1(emb_dim=1280, hidden_dim=932)
    params_P0 = count_params(model_P0)
    t_mae_P0, t_rho_P0, v_mae_P0, v_rho_P0 = train_static_model(
        "P0: Global Mean", model_P0, loader_train_P0, loader_val_P0, loader_test_P0, epochs, device
    )
    results["P0"] = {
        "Description": "Global Mean (329 AA simple average)",
        "Embedding_Dim": 1280,
        "Trainable_Parameters": params_P0,
        "Val_MAE": round(v_mae_P0, 4),
        "Val_Spearman_rho": round(v_rho_P0, 4),
        "Test_MAE": round(t_mae_P0, 4),
        "Test_Spearman_rho": round(t_rho_P0, 4)
    }

    # -------------------------------------------------------------
    # 2. P1: Global + Canonical Epitopes ([z_global; z_epi])
    # -------------------------------------------------------------
    ds_train_P1 = StaticVectorDataset(train_df, rep_P1)
    ds_val_P1 = StaticVectorDataset(val_df, rep_P1)
    ds_test_P1 = StaticVectorDataset(test_df, rep_P1)

    loader_train_P1 = DataLoader(ds_train_P1, batch_size=256, shuffle=True)
    loader_val_P1 = DataLoader(ds_val_P1, batch_size=256, shuffle=False)
    loader_test_P1 = DataLoader(ds_test_P1, batch_size=256, shuffle=False)

    model_P1 = HeadR1(emb_dim=2560, hidden_dim=932)
    params_P1 = count_params(model_P1)
    t_mae_P1, t_rho_P1, v_mae_P1, v_rho_P1 = train_static_model(
        "P1: Global + Canonical Epitopes", model_P1, loader_train_P1, loader_val_P1, loader_test_P1, epochs, device
    )
    results["P1"] = {
        "Description": "Global + Canonical Epitopes ([z_global; z_epi])",
        "Embedding_Dim": 2560,
        "Trainable_Parameters": params_P1,
        "Val_MAE": round(v_mae_P1, 4),
        "Val_Spearman_rho": round(v_rho_P1, 4),
        "Test_MAE": round(t_mae_P1, 4),
        "Test_Spearman_rho": round(t_rho_P1, 4)
    }

    # -------------------------------------------------------------
    # 3. P2: Global + Epitope + Key Sites ([z_global; z_epi; z_key])
    # -------------------------------------------------------------
    ds_train_P2 = StaticVectorDataset(train_df, rep_P2)
    ds_val_P2 = StaticVectorDataset(val_df, rep_P2)
    ds_test_P2 = StaticVectorDataset(test_df, rep_P2)

    loader_train_P2 = DataLoader(ds_train_P2, batch_size=256, shuffle=True)
    loader_val_P2 = DataLoader(ds_val_P2, batch_size=256, shuffle=False)
    loader_test_P2 = DataLoader(ds_test_P2, batch_size=256, shuffle=False)

    model_P2 = HeadR1(emb_dim=3840, hidden_dim=932)
    params_P2 = count_params(model_P2)
    t_mae_P2, t_rho_P2, v_mae_P2, v_rho_P2 = train_static_model(
        "P2: Global + Epitope + Key Sites", model_P2, loader_train_P2, loader_val_P2, loader_test_P2, epochs, device
    )
    results["P2"] = {
        "Description": "Global + Epitope + Key Sites ([z_global; z_epi; z_key])",
        "Embedding_Dim": 3840,
        "Trainable_Parameters": params_P2,
        "Val_MAE": round(v_mae_P2, 4),
        "Val_Spearman_rho": round(v_rho_P2, 4),
        "Test_MAE": round(t_mae_P2, 4),
        "Test_Spearman_rho": round(t_rho_P2, 4)
    }

    # -------------------------------------------------------------
    # 4. P3: Biology-guided Learnable Attention
    # -------------------------------------------------------------
    ds_train_P3 = IndexPairDataset(train_df, virus_to_idx)
    ds_val_P3 = IndexPairDataset(val_df, virus_to_idx)
    ds_test_P3 = IndexPairDataset(test_df, virus_to_idx)

    loader_train_P3 = DataLoader(ds_train_P3, batch_size=256, shuffle=True)
    loader_val_P3 = DataLoader(ds_val_P3, batch_size=256, shuffle=False)
    loader_test_P3 = DataLoader(ds_test_P3, batch_size=256, shuffle=False)

    model_P3 = P3Model(emb_dim=1280, hidden_dim=932)
    params_P3 = count_params(model_P3)
    t_mae_P3, t_rho_P3, v_mae_P3, v_rho_P3 = train_p3_model(
        model_P3, all_tokens_tensor, loader_train_P3, loader_val_P3, loader_test_P3, epochs, device
    )
    results["P3"] = {
        "Description": "Biology-guided Learnable Attention (s_i = g(h_i) + beta_E * I_epi + beta_K * I_key)",
        "Embedding_Dim": 1280,
        "Trainable_Parameters": params_P3,
        "Val_MAE": round(v_mae_P3, 4),
        "Val_Spearman_rho": round(v_rho_P3, 4),
        "Test_MAE": round(t_mae_P3, 4),
        "Test_Spearman_rho": round(t_rho_P3, 4),
        "Learned_beta_E": round(float(model_P3.pool.beta_E.item()), 4),
        "Learned_beta_K": round(float(model_P3.pool.beta_K.item()), 4)
    }

    out_json = os.path.join(reports_dir, 'stage_B_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved Stage B results to {out_json}")
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    main()
