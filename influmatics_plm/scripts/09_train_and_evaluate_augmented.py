import os
import sys
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.manifold import MDS
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from landmark_fusion import AttributedPairDataset, AugmentedDistanceHead, multilaterate_landmarks, evaluate_out_of_sample_landmarks
from losses import CombinedDistanceLoss

def parse_year(name):
    parts = str(name).strip().split('/')
    m = re.search(r'(\d+)$', parts[-1])
    if m:
        yy = int(m.group(1))
        if yy >= 1000:
            return yy
        return 2000 + yy if yy < 50 else 1900 + yy
    return 2020

def run_evaluation(model, dataloader, device):
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for emb1, emb2, delta_g, dist in dataloader:
            emb1, emb2 = emb1.to(device), emb2.to(device)
            delta_g = delta_g.to(device)
            # Symmetrize prediction across both orientations
            p1 = model(emb1, emb2, delta_g)
            p2 = model(emb2, emb1, delta_g)
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

def generate_svg_plot(true_coords, pred_coords, proc_true, proc_pred, years, landmark_mask, disparity, out_svg_path):
    width = 1500
    height = 540
    panel_w = 460
    panel_h = 420
    
    color_map = {
        2019: "#3b82f6", # Blue
        2020: "#10b981", # Green
        2021: "#f59e0b", # Amber
        2022: "#ef4444"  # Red
    }
    
    def scale_coords(coords, target_box):
        min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
        min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
        span_x = max(max_x - min_x, 1e-5)
        span_y = max(max_y - min_y, 1e-5)
        bx, by, bw, bh = target_box
        margin = 40
        sx = (coords[:, 0] - min_x) / span_x * (bw - 2 * margin) + bx + margin
        sy = (coords[:, 1] - min_y) / span_y * (bh - 2 * margin) + by + margin
        return sx, sy

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<style>',
        '  .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 14px; font-weight: bold; fill: #1e3a8a; }',
        '  .subtitle { font-family: -apple-system, sans-serif; font-size: 11px; fill: #64748b; }',
        '  .legend { font-family: -apple-system, sans-serif; font-size: 11px; fill: #334155; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>'
    ]

    panels = [
        ("(A) Ground-Truth Map (15 Landmarks ★ + 47 Test ●)", (30, 60, panel_w, panel_h), true_coords, False),
        ("(B) Multilaterated Map (Fixed Anchors + Predicted)", (520, 60, panel_w, panel_h), pred_coords, False),
        (f"(C) Out-of-Sample Overlay (Disparity: {disparity:.4f})", (1010, 60, panel_w, panel_h), None, True)
    ]

    for p_title, (bx, by, bw, bh), coords, is_overlay in panels:
        svg_parts.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="#f8fafc" stroke="#cbd5e1" rx="8"/>')
        svg_parts.append(f'<text x="{bx+15}" y="{by-15}" class="title">{p_title}</text>')
        
        for gx in range(bx + 40, bx + bw - 20, 80):
            svg_parts.append(f'<line x1="{gx}" y1="{by+20}" x2="{gx}" y2="{by+bh-20}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        for gy in range(by + 40, by + bh - 20, 80):
            svg_parts.append(f'<line x1="{bx+20}" y1="{gy}" x2="{bx+bw-20}" y2="{gy}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')

        if not is_overlay:
            sx, sy = scale_coords(coords, (bx, by, bw, bh))
            for i in range(len(coords)):
                col = color_map.get(years[i], "#64748b")
                if landmark_mask[i]:
                    # Landmark anchor: draw star/diamond
                    svg_parts.append(f'<polygon points="{sx[i]},{sy[i]-8} {sx[i]+6},{sy[i]+6} {sx[i]-6},{sy[i]+6}" fill="#475569" stroke="#0f172a" stroke-width="1.5"/>')
                else:
                    # Test strain: circle
                    svg_parts.append(f'<circle cx="{sx[i]:.1f}" cy="{sy[i]:.1f}" r="5.5" fill="{col}" fill-opacity="0.9" stroke="#ffffff" stroke-width="1"/>')
        else:
            # Only out-of-sample points in proc_true and proc_pred
            all_pts = np.vstack([proc_true, proc_pred])
            all_sx, all_sy = scale_coords(all_pts, (bx, by, bw, bh))
            h = len(proc_true)
            t_sx, t_sy = all_sx[:h], all_sy[:h]
            p_sx, p_sy = all_sx[h:], all_sy[h:]
            test_years = years[~landmark_mask]

            for i in range(h):
                svg_parts.append(f'<line x1="{t_sx[i]:.1f}" y1="{t_sy[i]:.1f}" x2="{p_sx[i]:.1f}" y2="{p_sy[i]:.1f}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="2,2"/>')
            for i in range(h):
                svg_parts.append(f'<circle cx="{t_sx[i]:.1f}" cy="{t_sy[i]:.1f}" r="4" fill="#94a3b8" fill-opacity="0.6"/>')
            for i in range(h):
                col = color_map.get(test_years[i], "#64748b")
                svg_parts.append(f'<circle cx="{p_sx[i]:.1f}" cy="{p_sy[i]:.1f}" r="5.5" fill="{col}" fill-opacity="0.9" stroke="#ffffff" stroke-width="1"/>')

    # Legend
    legend_y = 515
    svg_parts.append(f'<text x="50" y="{legend_y}" class="legend" font-weight="bold">Test Strain Year:</text>')
    lx = 170
    for yr, col in color_map.items():
        svg_parts.append(f'<circle cx="{lx}" cy="{legend_y-4}" r="5" fill="{col}"/>')
        svg_parts.append(f'<text x="{lx+10}" y="{legend_y}" class="legend">{yr}</text>')
        lx += 75
        
    svg_parts.append(f'<polygon points="{lx+10},{legend_y-10} {lx+15},{legend_y} {lx+5},{legend_y}" fill="#475569"/>')
    svg_parts.append(f'<text x="{lx+22}" y="{legend_y}" class="legend">Landmark Anchor (K=15)</text>')
    lx += 180
    svg_parts.append(f'<circle cx="{lx+10}" cy="{legend_y-4}" r="4" fill="#94a3b8"/>')
    svg_parts.append(f'<text x="{lx+20}" y="{legend_y}" class="legend">True Held-out Pos</text>')
    svg_parts.append(f'<line x1="{lx+140}" y1="{legend_y-4}" x2="{lx+165}" y2="{legend_y-4}" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="2,2"/>')
    svg_parts.append(f'<text x="{lx+175}" y="{legend_y}" class="legend">Residual Vector (H=47)</text>')

    svg_parts.append('</svg>')
    
    with open(out_svg_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(svg_parts))
    print(f"Saved unconfounded Vector SVG to {out_svg_path}")

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Executing Augmented Training & Evaluation on: {device}")
    
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    splits_dir = os.path.join(base_dir, 'data/splits')
    models_dir = os.path.join(base_dir, 'models/checkpoints')
    reports_dir = os.path.join(base_dir, 'reports')
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Load Embeddings and Glycosylation Counts
    emb_path = os.path.join(base_dir, 'data/processed/embeddings_esm2_650m.pt')
    glyco_path = os.path.join(base_dir, 'data/processed/glyco_counts.json')
    
    raw_embeddings = torch.load(emb_path, map_location='cpu')
    with open(glyco_path, 'r', encoding='utf-8') as f:
        glyco_raw = json.load(f)
    glyco_dict = {k: int(v['count']) for k, v in glyco_raw.items()}
    
    # Restrict to nodes present in both
    common_nodes = set(raw_embeddings.keys()).intersection(set(glyco_dict.keys()))
    embeddings = {k: raw_embeddings[k] for k in common_nodes}
    attribute_dict = {k: glyco_dict[k] for k in common_nodes}
    print(f"Total valid attributed nodes: {len(embeddings):,}")
    
    # 2. Load Splits
    train_df = pd.read_csv(os.path.join(splits_dir, 'cohort2_train.csv.gz'))
    val_df = pd.read_csv(os.path.join(splits_dir, 'cohort2_val.csv.gz'))
    test_df = pd.read_csv(os.path.join(splits_dir, 'cohort2_test.csv.gz'))
    
    # Filter valid edges
    def filter_edges(df):
        return df[df['node_A'].isin(common_nodes) & df['node_B'].isin(common_nodes)].reset_index(drop=True)
        
    train_df = filter_edges(train_df)
    val_df = filter_edges(val_df)
    test_df = filter_edges(test_df)
    
    print(f"Filtered Cohort 2 Pairs -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    # Node Disjointness Assertion
    train_nodes = set(train_df['node_A']).union(set(train_df['node_B']))
    test_nodes = set(test_df['node_A']).union(set(test_df['node_B']))
    assert train_nodes.isdisjoint(test_nodes), "Data Leakage detected!"
    print(f"Strict Node Disjointness Verified: 0 overlapping nodes.")
    
    # 3. Create Datasets using Codex's AttributedPairDataset
    train_dataset = AttributedPairDataset(train_df, embeddings, attribute_dict)
    val_dataset = AttributedPairDataset(val_df, embeddings, attribute_dict)
    test_dataset = AttributedPairDataset(test_df, embeddings, attribute_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 4. Model: AugmentedDistanceHead (5,121 dimensions)
    model = AugmentedDistanceHead().to(device)
    criterion = CombinedDistanceLoss(alpha=0.3, margin=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_rmse = float('inf')
    best_val_metrics = {}
    best_ckpt_path = os.path.join(models_dir, 'cohort2_augmented_best.pt')
    
    epochs = 10
    print("\n--- Training 5,121-dim Augmented Interaction Model ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for emb1, emb2, delta_g, dist in train_loader:
            emb1, emb2 = emb1.to(device), emb2.to(device)
            delta_g, dist = delta_g.to(device), dist.to(device)
            optimizer.zero_grad()
            preds = model(emb1, emb2, delta_g)
            loss = criterion(preds, dist)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(dist)
            
        train_loss = total_loss / len(train_dataset)
        val_metrics = run_evaluation(model, val_loader, device)
        
        if val_metrics['RMSE'] < best_val_rmse:
            best_val_rmse = val_metrics['RMSE']
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), best_ckpt_path)
            mark = " [*Best Model Saved]"
        else:
            mark = ""
            
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val RMSE: {val_metrics['RMSE']:.4f}, MAE: {val_metrics['MAE']:.4f}, Rho: {val_metrics['Spearman_rho']:.4f}{mark}")
        
    # 5. Evaluate on Prospective Future Test Set
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_metrics = run_evaluation(model, test_loader, device)
    print(f"\n>> Final Future Test Set Metrics (75,581 unseen pairs, 2019-2022):")
    print(json.dumps(test_metrics, indent=2))
    
    # 6. Task 5 Out-of-Sample Landmark MDS on 100% Observed Clique
    print("\n--- Task 5: Pure Out-of-Sample Landmark MDS Evaluation ---")
    pairs = set()
    for _, r in test_df.iterrows():
        u, v = sorted([r['node_A'], r['node_B']])
        pairs.add((u, v))

    c = Counter(test_df['node_A']).copy()
    c.update(Counter(test_df['node_B']))
    candidates = [v for v, cnt in c.most_common()]

    clique = []
    for cand in candidates:
        if all(tuple(sorted([cand, member])) in pairs for member in clique):
            clique.append(cand)

    n = len(clique)
    print(f"Identified 100% Complete Test Clique: {n} strains.")
    virus_to_idx = {v: i for i, v in enumerate(clique)}
    
    D_true = np.zeros((n, n), dtype=np.float64)
    sub_df = test_df[test_df['node_A'].isin(virus_to_idx) & test_df['node_B'].isin(virus_to_idx)]
    for _, row in sub_df.iterrows():
        i = virus_to_idx[row['node_A']]
        j = virus_to_idx[row['node_B']]
        D_true[i, j] = row['distance']
        D_true[j, i] = row['distance']

    # Predict Pairwise Matrix using Augmented Model
    D_pred = np.zeros((n, n), dtype=np.float64)
    with torch.no_grad():
        for i in range(n):
            u_emb = embeddings[clique[i]].unsqueeze(0).to(device)
            u_g = attribute_dict[clique[i]]
            for j in range(i + 1, n):
                v_emb = embeddings[clique[j]].unsqueeze(0).to(device)
                v_g = attribute_dict[clique[j]]
                dg = torch.tensor([[abs(u_g - v_g)]], dtype=torch.float32, device=device)
                p1 = model(u_emb, v_emb, dg).item()
                p2 = model(v_emb, u_emb, dg).item()
                dist = max(0.0, (p1 + p2) / 2.0)
                D_pred[i, j] = dist
                D_pred[j, i] = dist

    # Ground truth reference MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=20, max_iter=1000)
    true_coords = mds.fit_transform(D_true)

    # Landmark Selection (K=15 evenly spaced)
    num_anchors = 15
    landmark_indices = np.linspace(0, n - 1, num_anchors, dtype=int)
    landmark_mask = np.zeros(n, dtype=bool)
    landmark_mask[landmark_indices] = True
    
    landmark_coords = true_coords[landmark_indices]
    held_out_indices = np.where(~landmark_mask)[0]
    distances_to_landmarks = D_pred[held_out_indices][:, landmark_indices]

    # Multilateration
    print(f"Multilaterating {len(held_out_indices)} held-out nodes against {num_anchors} landmarks...")
    pred_held_out_positions, diagnostics = multilaterate_landmarks(landmark_coords, distances_to_landmarks)

    # Full reconstructed array for plotting
    pred_coords_full = np.zeros_like(true_coords)
    pred_coords_full[landmark_indices] = landmark_coords
    pred_coords_full[held_out_indices] = pred_held_out_positions

    # Codex evaluate_out_of_sample_landmarks
    true_held_out_coords = true_coords[held_out_indices]
    oos_metrics, details = evaluate_out_of_sample_landmarks(
        true_coords, D_pred, landmark_indices, return_details=True
    )
    aligned_true_oos = details["aligned_true"]
    aligned_pred_oos = details["aligned_pred"]
    
    print("\n>> Pure Out-of-Sample Procrustes Evaluation (Zero Landmark Contamination):")
    print(json.dumps(oos_metrics, indent=2))

    # Save metrics JSON
    summary_report = {
        "model_architecture": "ESM-2 (650M) + N-Glycosylation Difference (5,121-dim Interaction Head)",
        "training_loss": "SmoothL1Loss + 0.3 * PairwiseRankingLoss (margin=0.1)",
        "test_metrics_prospective_75k_pairs": test_metrics,
        "clique_evaluation_62_strains": {
            "num_strains": n,
            "num_landmarks": num_anchors,
            "num_held_out_test_strains": len(held_out_indices),
            "clique_pairwise_mae": round(float(np.mean(np.abs(D_true[np.triu_indices(n, 1)] - D_pred[np.triu_indices(n, 1)]))), 4),
            "clique_pairwise_rmse": round(float(np.sqrt(np.mean((D_true[np.triu_indices(n, 1)] - D_pred[np.triu_indices(n, 1)])**2))), 4),
            "clique_pairwise_spearman_rho": round(float(spearmanr(D_true[np.triu_indices(n, 1)], D_pred[np.triu_indices(n, 1)]).statistic), 4),
            "out_of_sample_procrustes_disparity": round(oos_metrics['disparity'], 4),
            "out_of_sample_axis_1_pearson_r": round(oos_metrics['pearson_r_axis1'], 4),
            "out_of_sample_axis_1_spearman_rho": round(oos_metrics['spearman_rho_axis1'], 4),
            "out_of_sample_axis_2_pearson_r": round(oos_metrics['pearson_r_axis2'], 4),
            "out_of_sample_axis_2_spearman_rho": round(oos_metrics['spearman_rho_axis2'], 4)
        }
    }
    
    report_file = os.path.join(reports_dir, 'augmented_model_evaluation.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved summary report to {report_file}")

    # Generate 3-panel SVG plot
    years = np.array([parse_year(v) for v in clique])
    svg_path = os.path.join(figures_dir, 'reconstructed_antigenic_map.svg')
    generate_svg_plot(
        true_coords, pred_coords_full, 
        aligned_true_oos, aligned_pred_oos, 
        years, landmark_mask, oos_metrics['disparity'], svg_path
    )

if __name__ == '__main__':
    main()
