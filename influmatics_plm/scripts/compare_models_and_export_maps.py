import os
import sys
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import MDS
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '..'))

emb_path = os.path.join(base_dir, 'data/processed/embeddings_esm2_650m.pt')
embs = torch.load(emb_path, map_location='cpu')

t1 = pd.read_csv(os.path.join(base_dir, 'data/splits/cohort1_train.csv.gz'))
t2 = pd.read_csv(os.path.join(base_dir, 'data/splits/cohort2_train.csv.gz'))
v2 = pd.read_csv(os.path.join(base_dir, 'data/splits/cohort2_val.csv.gz'))
test2 = pd.read_csv(os.path.join(base_dir, 'data/splits/cohort2_test.csv.gz'))

t2 = t2[t2['virus1'].isin(embs) & t2['virus2'].isin(embs)].reset_index(drop=True)
v2 = v2[v2['virus1'].isin(embs) & v2['virus2'].isin(embs)].reset_index(drop=True)
test2 = test2[test2['virus1'].isin(embs) & test2['virus2'].isin(embs)].reset_index(drop=True)

train_comb = pd.concat([t1, t2], ignore_index=True)
train_comb = train_comb[train_comb['virus1'].isin(embs) & train_comb['virus2'].isin(embs)].reset_index(drop=True)

# 1. Model Definitions
class Model1_Original(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1280*4, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, u, v):
        x = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
        return self.mlp(x).squeeze(-1)

class Model2_Symmetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1280*3, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1), nn.Softplus()
        )
    def forward(self, u, v):
        diff = torch.abs(u - v)
        mult = u * v
        sq = (u - v) ** 2
        x = torch.cat([diff, mult, sq], dim=-1)
        return self.mlp(x).squeeze(-1)

class Model3_MetricProjection(nn.Module):
    def __init__(self, proj_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, proj_dim)
        )
    def forward(self, u, v):
        pu = self.proj(u)
        pv = self.proj(v)
        diff = pu - pv
        return torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-12)

# Fast In-Memory Dataset
class FastDS:
    def __init__(self, df):
        self.u = torch.stack([embs[v] for v in df['virus1']])
        self.v = torch.stack([embs[v] for v in df['virus2']])
        self.d = torch.tensor(df['distance'].values, dtype=torch.float32)
    def __len__(self): return len(self.d)

print("Preparing in-memory datasets...")
ds_train_c2 = FastDS(t2)
ds_train_comb = FastDS(train_comb)
ds_val_c2 = FastDS(v2)
ds_test_c2 = FastDS(test2)

# 2. Extract 100% Fully-Observed Clique of 62 strains
pairs = set()
for _, r in test2.iterrows():
    u, v = sorted([r['virus1'], r['virus2']])
    pairs.add((u, v))

c = Counter(test2['virus1']).copy()
c.update(Counter(test2['virus2']))
candidates = [v for v, cnt in c.most_common()]

clique = []
for cand in candidates:
    if all(tuple(sorted([cand, member])) in pairs for member in clique):
        clique.append(cand)

n = len(clique)
print(f"Clique size: {n} strains with {n*(n-1)//2} observed pairwise distances.")
virus_to_idx = {v: i for i, v in enumerate(clique)}

D_true = np.zeros((n, n), dtype=np.float64)
sub_df = test2[test2['virus1'].isin(virus_to_idx) & test2['virus2'].isin(virus_to_idx)]
for _, row in sub_df.iterrows():
    i = virus_to_idx[row['virus1']]
    j = virus_to_idx[row['virus2']]
    D_true[i, j] = row['distance']
    D_true[j, i] = row['distance']

# Fit True MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=15, max_iter=1000)
coords_true = mds.fit_transform(D_true)

def parse_year(name):
    m = re.search(r'(\d+)$', str(name).strip().split('/')[-1])
    if m:
        yy = int(m.group(1))
        if yy >= 1000: return yy
        return 2000 + yy if yy < 50 else 1900 + yy
    return 2020

years = np.array([parse_year(v) for v in clique])

def export_single_map_svg(coords, title, subtitle, out_path):
    w, h = 600, 520
    margin = 40
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
    span_x = max(max_x - min_x, 1e-5)
    span_y = max(max_y - min_y, 1e-5)
    
    sx = (coords[:, 0] - min_x) / span_x * (w - 2 * margin) + margin
    sy = (coords[:, 1] - min_y) / span_y * (h - 2 * margin) + margin
    
    color_map = {2019: "#3b82f6", 2020: "#10b981", 2021: "#f59e0b", 2022: "#ef4444"}
    
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<rect x="15" y="15" width="{w-30}" height="{h-30}" fill="#f8fafc" stroke="#e2e8f0" rx="8"/>',
        f'<text x="30" y="45" font-family="-apple-system, sans-serif" font-size="16" font-weight="bold" fill="#1e3a8a">{title}</text>',
        f'<text x="30" y="65" font-family="-apple-system, sans-serif" font-size="12" fill="#64748b">{subtitle}</text>'
    ]
    
    for gx in range(margin, w - margin, 60):
        lines.append(f'<line x1="{gx}" y1="{margin+30}" x2="{gx}" y2="{h-margin}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
    for gy in range(margin+30, h - margin, 60):
        lines.append(f'<line x1="{margin}" y1="{gy}" x2="{w-margin}" y2="{gy}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        
    for i in range(len(coords)):
        col = color_map.get(years[i], "#64748b")
        lines.append(f'<circle cx="{sx[i]:.1f}" cy="{sy[i]:.1f}" r="6" fill="{col}" fill-opacity="0.85" stroke="#ffffff" stroke-width="1.5"/>')
        
    # Legend
    leg_y = h - 20
    lines.append(f'<text x="30" y="{leg_y}" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#334155">Year:</text>')
    lx = 80
    for yr, col in color_map.items():
        lines.append(f'<circle cx="{lx}" cy="{leg_y-4}" r="5" fill="{col}"/>')
        lines.append(f'<text x="{lx+8}" y="{leg_y}" font-family="-apple-system, sans-serif" font-size="11" fill="#334155">{yr}</text>')
        lx += 65
        
    lines.append('</svg>')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Saved: {out_path}")

figures_dir = os.path.join(base_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)
export_single_map_svg(coords_true, "Ground Truth Antigenic Map (Cohort 2)", "MDS on 100% Observed Ferret HI Distances (62 Strains)", os.path.join(figures_dir, "map_ground_truth.svg"))

def train_and_evaluate_model(name, model, train_ds, epochs=7):
    print(f"\n--- Training {name} ---")
    loader = DataLoader(range(len(train_ds)), batch_size=256, shuffle=True)
    crit = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for ep in range(epochs):
        model.train()
        t_loss = 0
        for idxs in loader:
            opt.zero_grad()
            pred = model(train_ds.u[idxs], train_ds.v[idxs])
            loss = crit(pred, train_ds.d[idxs])
            loss.backward()
            opt.step()
            t_loss += loss.item() * len(idxs)
            
    # Evaluate on Test Set (75,581 pairs)
    model.eval()
    test_loader = DataLoader(range(len(ds_test_c2)), batch_size=512, shuffle=False)
    p_all, t_all = [], []
    with torch.no_grad():
        for idxs in test_loader:
            p_all.extend(model(ds_test_c2.u[idxs], ds_test_c2.v[idxs]).numpy())
            t_all.extend(ds_test_c2.d[idxs].numpy())
    p_all = np.array(p_all)
    t_all = np.array(t_all)
    
    rmse = float(np.sqrt(np.mean((t_all - p_all)**2)))
    mae = float(np.mean(np.abs(t_all - p_all)))
    rho = float(spearmanr(t_all, p_all)[0])
    
    # Mathematical Invariant Checks
    with torch.no_grad():
        sample_u = ds_test_c2.u[:100]
        sample_v = ds_test_c2.v[:100]
        d_uv = model(sample_u, sample_v)
        d_vu = model(sample_v, sample_u)
        sym_error = float(torch.mean(torch.abs(d_uv - d_vu)).item())
        
        d_uu = model(sample_u, sample_u)
        self_dist = float(torch.mean(torch.abs(d_uu)).item())
        neg_rate = float((p_all < 0).mean() * 100)
        
    # Predict on 62-strain clique for MDS
    D_pred = np.zeros((n, n), dtype=np.float64)
    with torch.no_grad():
        for i in range(n):
            u_i = embs[clique[i]].unsqueeze(0)
            for j in range(i + 1, n):
                v_j = embs[clique[j]].unsqueeze(0)
                d_val = float(model(u_i, v_j).item())
                D_pred[i, j] = d_val
                D_pred[j, i] = d_val
                
    mds_pred = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=15, max_iter=1000)
    coords_pred = mds_pred.fit_transform(D_pred)
    
    proc_t, proc_p, disparity = procrustes(coords_true, coords_pred)
    r_ax1, _ = pearsonr(proc_t[:, 0], proc_p[:, 0])
    r_ax2, _ = pearsonr(proc_t[:, 1], proc_p[:, 1])
    
    return {
        "model_name": name,
        "test_rmse": round(rmse, 4),
        "test_mae": round(mae, 4),
        "test_spearman_rho": round(rho, 4),
        "symmetry_error": round(sym_error, 4),
        "self_distance_D_uu": round(self_dist, 4),
        "negative_predictions_pct": round(neg_rate, 2),
        "procrustes_disparity": round(float(disparity), 4),
        "axis_1_pearson_r": round(float(r_ax1), 4),
        "axis_2_pearson_r": round(float(r_ax2), 4),
        "coords": coords_pred
    }

models_to_test = [
    ("Model 1 (Original Interacting)", Model1_Original(), ds_train_c2, "map_model1_original.svg"),
    ("Model 2 (Symmetric High-Cap)", Model2_Symmetric(), ds_train_c2, "map_model2_symmetric.svg"),
    ("Model 3 (Metric Projection)", Model3_MetricProjection(), ds_train_c2, "map_model3_metric_projection.svg"),
    ("Model 4 (Combined Cohort 1+2)", Model1_Original(), ds_train_comb, "map_model4_combined_cohort.svg")
]

comparison_results = []
for name, mod, ds, svg_name in models_to_test:
    res = train_and_evaluate_model(name, mod, ds, epochs=6)
    coords = res.pop("coords")
    export_single_map_svg(coords, name, f"Test MAE: {res['test_mae']} | Rho: {res['test_spearman_rho']} | Disparity: {res['procrustes_disparity']}", os.path.join(figures_dir, svg_name))
    comparison_results.append(res)

df_comp = pd.DataFrame(comparison_results)
print("\n================ COMPREHENSIVE MODEL COMPARISON TABLE ================")
print(df_comp.to_string(index=False))

reports_dir = os.path.join(base_dir, 'reports')
df_comp.to_csv(os.path.join(reports_dir, 'all_models_comparison_table.csv'), index=False)
with open(os.path.join(reports_dir, 'all_models_comparison_table.json'), 'w', encoding='utf-8') as f:
    json.dump(comparison_results, f, indent=2, ensure_ascii=False)
print("Saved comparison table to CSV and JSON.")
