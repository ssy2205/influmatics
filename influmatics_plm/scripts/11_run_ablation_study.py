import os
import sys
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.manifold import MDS
from scipy.spatial import procrustes
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from landmark_fusion import AttributedPairDataset, AugmentedDistanceHead, multilaterate_landmarks, evaluate_out_of_sample_landmarks
from losses import CombinedDistanceLoss
from models import DistanceHead

class StandardPairDataset(Dataset):
    def __init__(self, edge_df, embeddings_dict):
        self.df = edge_df.reset_index(drop=True)
        self.embeddings = embeddings_dict
        self.v1 = self.df['node_A'].values
        self.v2 = self.df['node_B'].values
        self.targets = self.df['distance'].values
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        emb1 = self.embeddings[self.v1[idx]]
        emb2 = self.embeddings[self.v2[idx]]
        dist = torch.tensor(self.targets[idx], dtype=torch.float32)
        return emb1, emb2, dist

def parse_year(name):
    parts = str(name).strip().split('/')
    m = re.search(r'(\d+)$', parts[-1])
    if m:
        yy = int(m.group(1))
        if yy >= 1000:
            return yy
        return 2000 + yy if yy < 50 else 1900 + yy
    return 2020

def evaluate_model(model, dataloader, device, is_augmented=False):
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            if is_augmented:
                emb1, emb2, delta_g, dist = batch
                emb1, emb2, delta_g = emb1.to(device), emb2.to(device), delta_g.to(device)
                p1 = model(emb1, emb2, delta_g)
                p2 = model(emb2, emb1, delta_g)
            else:
                emb1, emb2, dist = batch
                emb1, emb2 = emb1.to(device), emb2.to(device)
                p1 = model(emb1, emb2)
                p2 = model(emb2, emb1)
            pred = torch.clamp((p1 + p2) / 2.0, min=0.0)
            val_preds.extend(pred.cpu().numpy())
            val_targets.extend(dist.numpy())
            
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    
    rmse = float(np.sqrt(mean_squared_error(val_targets, val_preds)))
    mae = float(mean_absolute_error(val_targets, val_preds))
    rho_res, _ = spearmanr(val_targets, val_preds)
    r_res, _ = pearsonr(val_targets, val_preds)
    rho = float(rho_res) if not np.isnan(rho_res) else 0.0
    r = float(r_res) if not np.isnan(r_res) else 0.0
    return {'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'Spearman_rho': round(rho, 4), 'Pearson_r': round(r, 4)}

def train_condition(name, model, train_loader, val_loader, test_loader, criterion, epochs, device, is_augmented=False):
    print(f"\n==========================================")
    print(f"   Training Condition: {name}")
    print(f"==========================================")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val_rmse = float('inf')
    best_model_state = None
    best_val_metrics = {}
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for batch in train_loader:
            if is_augmented:
                emb1, emb2, delta_g, dist = batch
                emb1, emb2, delta_g, dist = emb1.to(device), emb2.to(device), delta_g.to(device), dist.to(device)
                optimizer.zero_grad()
                preds = model(emb1, emb2, delta_g)
            else:
                emb1, emb2, dist = batch
                emb1, emb2, dist = emb1.to(device), emb2.to(device), dist.to(device)
                optimizer.zero_grad()
                preds = model(emb1, emb2)
            loss = criterion(preds, dist)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(dist)
            count += len(dist)
            
        val_m = evaluate_model(model, val_loader, device, is_augmented=is_augmented)
        if val_m['RMSE'] < best_val_rmse:
            best_val_rmse = val_m['RMSE']
            best_val_metrics = val_m
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = " [*Best]"
        else:
            mark = ""
        print(f"Epoch {epoch:02d}/{epochs} | Val RMSE: {val_m['RMSE']:.4f}, MAE: {val_m['MAE']:.4f}, Rho: {val_m['Spearman_rho']:.4f}{mark}")
        
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    test_m = evaluate_model(model, test_loader, device, is_augmented=is_augmented)
    print(f"Test Evaluation -> RMSE: {test_m['RMSE']}, MAE: {test_m['MAE']}, Rho: {test_m['Spearman_rho']}, r: {test_m['Pearson_r']}")
    return model, best_val_metrics, test_m

def predict_clique(model, clique, embeddings, attribute_dict, device, is_augmented=False):
    model.eval()
    n = len(clique)
    D_pred = np.zeros((n, n), dtype=np.float64)
    with torch.no_grad():
        for i in range(n):
            u_emb = embeddings[clique[i]].unsqueeze(0).to(device)
            u_g = attribute_dict[clique[i]]
            for j in range(i + 1, n):
                v_emb = embeddings[clique[j]].unsqueeze(0).to(device)
                v_g = attribute_dict[clique[j]]
                if is_augmented:
                    dg = torch.tensor([[abs(u_g - v_g)]], dtype=torch.float32, device=device)
                    p1 = model(u_emb, v_emb, dg).item()
                    p2 = model(v_emb, u_emb, dg).item()
                else:
                    p1 = model(u_emb, v_emb).item()
                    p2 = model(v_emb, u_emb).item()
                d = max(0.0, (p1 + p2) / 2.0)
                D_pred[i, j] = d
                D_pred[j, i] = d
    return D_pred

def generate_comparative_svg(true_coords, coords_baseline, coords_euclidean, coords_augmented, years, landmark_mask, out_svg_path):
    width = 1600
    height = 460
    pw, ph = 360, 360
    panels = [
        ("(A) Ground-Truth MDS Map", 30, true_coords, False),
        ("(B) Cond.1: Baseline (Mean Pooling)", 420, coords_baseline, False),
        ("(C) Cond.3: Epitope-Weighted Pooling", 810, coords_euclidean, False),
        ("(D) Cond.4: Full Fusion (Epitope+ΔGly+Landmark)", 1200, coords_augmented, True)
    ]
    color_map = {2019: "#3b82f6", 2020: "#10b981", 2021: "#f59e0b", 2022: "#ef4444"}
    
    def scale(pts, bx, by, bw, bh):
        min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
        min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
        sx = (pts[:, 0] - min_x) / max(max_x - min_x, 1e-5) * (bw - 60) + bx + 30
        sy = (pts[:, 1] - min_y) / max(max_y - min_y, 1e-5) * (bh - 60) + by + 30
        return sx, sy

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>.title{font-family:-apple-system,sans-serif;font-size:12px;font-weight:bold;fill:#1e3a8a;}'
        '.legend{font-family:-apple-system,sans-serif;font-size:11px;fill:#334155;}</style>'
    ]
    
    for title, bx, pts, is_anchor in panels:
        by = 50
        svg.append(f'<rect x="{bx}" y="{by}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" rx="8"/>')
        svg.append(f'<text x="{bx+10}" y="{by-12}" class="title">{title}</text>')
        sx, sy = scale(pts, bx, by, pw, ph)
        for i in range(len(pts)):
            col = color_map.get(years[i], "#64748b")
            if is_anchor and landmark_mask[i]:
                svg.append(f'<polygon points="{sx[i]},{sy[i]-7} {sx[i]+5},{sy[i]+5} {sx[i]-5},{sy[i]+5}" fill="#334155" stroke="#0f172a" stroke-width="1.2"/>')
            else:
                svg.append(f'<circle cx="{sx[i]:.1f}" cy="{sy[i]:.1f}" r="4.5" fill="{col}" fill-opacity="0.85" stroke="#ffffff" stroke-width="1"/>')
                
    # Bottom Legend
    ly = 435
    svg.append(f'<text x="50" y="{ly}" class="legend" font-weight="bold">Strain Year:</text>')
    lx = 140
    for yr, col in color_map.items():
        svg.append(f'<circle cx="{lx}" cy="{ly-4}" r="5" fill="{col}"/>')
        svg.append(f'<text x="{lx+10}" y="{ly}" class="legend">{yr}</text>')
        lx += 75
    svg.append(f'<polygon points="{lx+10},{ly-8} {lx+14},{ly} {lx+6},{ly}" fill="#334155"/>')
    svg.append(f'<text x="{lx+20}" y="{ly}" class="legend">Landmark Anchor (K=15)</text>')
    svg.append('</svg>')
    
    with open(out_svg_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(svg))
    print(f"Comparative 4-panel SVG saved to {out_svg_path}")

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device for Ablation Study: {device}")
    
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    splits_dir = os.path.join(base_dir, 'data/splits')
    reports_dir = os.path.join(base_dir, 'reports')
    figures_dir = os.path.join(base_dir, 'figures')
    
    emb_mean_path = os.path.join(base_dir, 'data/processed/embeddings_esm2_650m.pt')
    emb_epitope_path = os.path.join(base_dir, 'data/processed/embeddings_esm2_650m_epitope_weighted.pt')
    glyco_path = os.path.join(base_dir, 'data/processed/glyco_counts.json')
    
    mean_embeddings = torch.load(emb_mean_path, map_location='cpu')
    epitope_embeddings = torch.load(emb_epitope_path, map_location='cpu')
    with open(glyco_path, 'r', encoding='utf-8') as f:
        glyco_raw = json.load(f)
    glyco_dict = {k: int(v['count']) for k, v in glyco_raw.items()}
    
    common_nodes = set(mean_embeddings.keys()).intersection(set(epitope_embeddings.keys())).intersection(set(glyco_dict.keys()))
    print(f"Total verified overlapping nodes across mean, epitope, and glyco: {len(common_nodes):,}")
    
    train_df = pd.read_csv(os.path.join(splits_dir, 'cohort2_train.csv.gz'))
    val_df = pd.read_csv(os.path.join(splits_dir, 'cohort2_val.csv.gz'))
    test_df = pd.read_csv(os.path.join(splits_dir, 'cohort2_test.csv.gz'))
    
    def filter_df(df):
        return df[df['node_A'].isin(common_nodes) & df['node_B'].isin(common_nodes)].reset_index(drop=True)
    train_df, val_df, test_df = filter_df(train_df), filter_df(val_df), filter_df(test_df)
    
    # Pre-extract complete clique for Task 5
    pairs = set((sorted([r['node_A'], r['node_B']])[0], sorted([r['node_A'], r['node_B']])[1]) for _, r in test_df.iterrows())
    c = Counter(test_df['node_A']).copy()
    c.update(Counter(test_df['node_B']))
    clique = []
    for cand in [v for v, cnt in c.most_common()]:
        if all(tuple(sorted([cand, m])) in pairs for m in clique):
            clique.append(cand)
    n = len(clique)
    virus_to_idx = {v: i for i, v in enumerate(clique)}
    D_true = np.zeros((n, n), dtype=np.float64)
    sub_df = test_df[test_df['node_A'].isin(virus_to_idx) & test_df['node_B'].isin(virus_to_idx)]
    for _, row in sub_df.iterrows():
        i, j = virus_to_idx[row['node_A']], virus_to_idx[row['node_B']]
        D_true[i, j] = D_true[j, i] = row['distance']
        
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=20, max_iter=1000)
    true_coords = mds.fit_transform(D_true)
    
    num_anchors = 15
    landmark_indices = np.linspace(0, n - 1, num_anchors, dtype=int)
    landmark_mask = np.zeros(n, dtype=bool)
    landmark_mask[landmark_indices] = True
    held_out_indices = np.where(~landmark_mask)[0]
    
    results = {}
    epochs = 8
    
    # -------------------------------------------------------------
    # 1. Condition 1: Baseline PLM (Mean Pooling, 5120-dim, SmoothL1)
    # -------------------------------------------------------------
    train_ds1 = StandardPairDataset(train_df, mean_embeddings)
    val_ds1 = StandardPairDataset(val_df, mean_embeddings)
    test_ds1 = StandardPairDataset(test_df, mean_embeddings)
    loader_train1 = DataLoader(train_ds1, batch_size=256, shuffle=True)
    loader_val1 = DataLoader(val_ds1, batch_size=256, shuffle=False)
    loader_test1 = DataLoader(test_ds1, batch_size=256, shuffle=False)
    
    model1 = DistanceHead(emb_dim=1280, hidden_dim=512).to(device)
    m1, val_m1, test_m1 = train_condition(
        "Condition 1: Baseline PLM (Mean Pooling, 5120-dim, SmoothL1)",
        model1, loader_train1, loader_val1, loader_test1, nn.SmoothL1Loss(), epochs, device, is_augmented=False
    )
    D_pred1 = predict_clique(m1, clique, mean_embeddings, glyco_dict, device, is_augmented=False)
    oos1 = evaluate_out_of_sample_landmarks(true_coords, D_pred1, landmark_indices, return_details=False)
    
    results["Condition 1 (Baseline PLM)"] = {
        "description": "ESM-2 (650M) Mean Pooling + 5,120-dim Head + Smooth L1 Loss",
        "prospective_test_75k": test_m1,
        "clique_mae": round(float(np.mean(np.abs(D_true[np.triu_indices(n, 1)] - D_pred1[np.triu_indices(n, 1)]))), 4),
        "clique_spearman_rho": round(float(spearmanr(D_true[np.triu_indices(n, 1)], D_pred1[np.triu_indices(n, 1)]).statistic), 4),
        "out_of_sample_procrustes_disparity": round(oos1['disparity'], 4),
        "out_of_sample_axis_1_pearson_r": round(oos1['pearson_r_axis1'], 4)
    }
    
    # -------------------------------------------------------------
    # 2. Condition 2: + N-Glycosylation Shielding Only (Mean Pooling, 5121-dim, Combined Loss)
    # -------------------------------------------------------------
    train_ds2 = AttributedPairDataset(train_df, mean_embeddings, glyco_dict)
    val_ds2 = AttributedPairDataset(val_df, mean_embeddings, glyco_dict)
    test_ds2 = AttributedPairDataset(test_df, mean_embeddings, glyco_dict)
    loader_train2 = DataLoader(train_ds2, batch_size=256, shuffle=True)
    loader_val2 = DataLoader(val_ds2, batch_size=256, shuffle=False)
    loader_test2 = DataLoader(test_ds2, batch_size=256, shuffle=False)
    
    model2 = AugmentedDistanceHead().to(device)
    m2, val_m2, test_m2 = train_condition(
        "Condition 2: + N-Glycosylation Shielding Only (5121-dim, Combined Loss)",
        model2, loader_train2, loader_val2, loader_test2, CombinedDistanceLoss(alpha=0.3, margin=0.1), epochs, device, is_augmented=True
    )
    D_pred2 = predict_clique(m2, clique, mean_embeddings, glyco_dict, device, is_augmented=True)
    oos2 = evaluate_out_of_sample_landmarks(true_coords, D_pred2, landmark_indices, return_details=False)
    
    results["Condition 2 (+ N-Glycosylation Only)"] = {
        "description": "ESM-2 Mean Pooling + 1D N-Glycosylation Difference (5,121-dim) + Combined Loss",
        "prospective_test_75k": test_m2,
        "clique_mae": round(float(np.mean(np.abs(D_true[np.triu_indices(n, 1)] - D_pred2[np.triu_indices(n, 1)]))), 4),
        "clique_spearman_rho": round(float(spearmanr(D_true[np.triu_indices(n, 1)], D_pred2[np.triu_indices(n, 1)]).statistic), 4),
        "out_of_sample_procrustes_disparity": round(oos2['disparity'], 4),
        "out_of_sample_axis_1_pearson_r": round(oos2['pearson_r_axis1'], 4)
    }

    # -------------------------------------------------------------
    # 3. Condition 3: + Epitope-Weighted Pooling Only (Epitope Pooling, 5120-dim, Combined Loss)
    # -------------------------------------------------------------
    train_ds3 = StandardPairDataset(train_df, epitope_embeddings)
    val_ds3 = StandardPairDataset(val_df, epitope_embeddings)
    test_ds3 = StandardPairDataset(test_df, epitope_embeddings)
    loader_train3 = DataLoader(train_ds3, batch_size=256, shuffle=True)
    loader_val3 = DataLoader(val_ds3, batch_size=256, shuffle=False)
    loader_test3 = DataLoader(test_ds3, batch_size=256, shuffle=False)
    
    model3 = DistanceHead(emb_dim=1280, hidden_dim=512).to(device)
    m3, val_m3, test_m3 = train_condition(
        "Condition 3: + Epitope-Weighted Pooling Only (5120-dim, Combined Loss)",
        model3, loader_train3, loader_val3, loader_test3, CombinedDistanceLoss(alpha=0.3, margin=0.1), epochs, device, is_augmented=False
    )
    D_pred3 = predict_clique(m3, clique, epitope_embeddings, glyco_dict, device, is_augmented=False)
    oos3 = evaluate_out_of_sample_landmarks(true_coords, D_pred3, landmark_indices, return_details=False)
    
    results["Condition 3 (+ Epitope Pooling Only)"] = {
        "description": "ESM-2 Epitope-Weighted Pooling (Koel 3.5x, Sites A-E 2.0x) + 5,120-dim Head + Combined Loss",
        "prospective_test_75k": test_m3,
        "clique_mae": round(float(np.mean(np.abs(D_true[np.triu_indices(n, 1)] - D_pred3[np.triu_indices(n, 1)]))), 4),
        "clique_spearman_rho": round(float(spearmanr(D_true[np.triu_indices(n, 1)], D_pred3[np.triu_indices(n, 1)]).statistic), 4),
        "out_of_sample_procrustes_disparity": round(oos3['disparity'], 4),
        "out_of_sample_axis_1_pearson_r": round(oos3['pearson_r_axis1'], 4)
    }

    # -------------------------------------------------------------
    # 4. Condition 4: + Full Fusion (Epitope Pooling + Glycosylation + Combined Loss)
    # -------------------------------------------------------------
    train_ds4 = AttributedPairDataset(train_df, epitope_embeddings, glyco_dict)
    val_ds4 = AttributedPairDataset(val_df, epitope_embeddings, glyco_dict)
    test_ds4 = AttributedPairDataset(test_df, epitope_embeddings, glyco_dict)
    loader_train4 = DataLoader(train_ds4, batch_size=256, shuffle=True)
    loader_val4 = DataLoader(val_ds4, batch_size=256, shuffle=False)
    loader_test4 = DataLoader(test_ds4, batch_size=256, shuffle=False)
    
    model4 = AugmentedDistanceHead().to(device)
    m4, val_m4, test_m4 = train_condition(
        "Condition 4: Full Fusion (Epitope Pooling + N-Glycosylation + Combined Loss)",
        model4, loader_train4, loader_val4, loader_test4, CombinedDistanceLoss(alpha=0.3, margin=0.1), epochs, device, is_augmented=True
    )
    D_pred4 = predict_clique(m4, clique, epitope_embeddings, glyco_dict, device, is_augmented=True)
    oos4, details4 = evaluate_out_of_sample_landmarks(true_coords, D_pred4, landmark_indices, return_details=True)
    
    results["Condition 4 (Full Fusion)"] = {
        "description": "ESM-2 Epitope Pooling + N-Glycosylation Difference (5,121-dim) + Combined Loss",
        "prospective_test_75k": test_m4,
        "clique_mae": round(float(np.mean(np.abs(D_true[np.triu_indices(n, 1)] - D_pred4[np.triu_indices(n, 1)]))), 4),
        "clique_spearman_rho": round(float(spearmanr(D_true[np.triu_indices(n, 1)], D_pred4[np.triu_indices(n, 1)]).statistic), 4),
        "out_of_sample_procrustes_disparity": round(oos4['disparity'], 4),
        "out_of_sample_axis_1_pearson_r": round(oos4['pearson_r_axis1'], 4)
    }
    
    # Save results JSON
    ablation_json_path = os.path.join(reports_dir, 'ablation_study_results.json')
    with open(ablation_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full ablation results to {ablation_json_path}")
    print(json.dumps(results, indent=2))
    
    # Coordinates for comparative plot:
    # (A) Ground-Truth MDS
    # already computed as true_coords above

    # (B) Condition 1 Baseline: free MDS on predicted distance matrix
    coords_baseline = mds.fit_transform(D_pred1)

    # (C) Condition 3: Epitope-Weighted Pooling – free MDS on D_pred3
    coords_epitope = mds.fit_transform(D_pred3)

    # (D) Condition 4 Full Fusion: landmark multilateration
    _, details4 = evaluate_out_of_sample_landmarks(
        true_coords, D_pred4, landmark_indices, return_details=True
    )
    pred_held_out = details4["pred_coords_held_out"]
    coords_full = np.zeros_like(true_coords)
    coords_full[landmark_indices] = true_coords[landmark_indices]
    coords_full[held_out_indices] = pred_held_out

    years = np.array([parse_year(v) for v in clique])
    comp_svg_path = os.path.join(figures_dir, 'comparative_antigenic_maps.svg')
    generate_comparative_svg(true_coords, coords_baseline, coords_epitope, coords_full, years, landmark_mask, comp_svg_path)

if __name__ == '__main__':
    main()
