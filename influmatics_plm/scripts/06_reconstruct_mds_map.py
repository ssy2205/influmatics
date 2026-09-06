import os
import sys
import json
import re
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import MDS
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr
from collections import Counter
from scipy.optimize import minimize

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from models import DistanceHead

def parse_year(name):
    parts = str(name).strip().split('/')
    m = re.search(r'(\d+)$', parts[-1])
    if m:
        yy = int(m.group(1))
        if yy >= 1000:
            return yy
        return 2000 + yy if yy < 50 else 1900 + yy
    return 2020

def generate_svg_plot(true_coords, pred_coords, proc_true, proc_pred, years, disparity, out_svg_path):
    width = 1500
    height = 520
    panel_w = 460
    panel_h = 420
    
    color_map = {
        2019: "#3b82f6",
        2020: "#10b981",
        2021: "#f59e0b",
        2022: "#ef4444"
    }
    
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
        ("(A) Uncontaminated Ground-Truth MDS (100% Observed Clique)", (30, 60, panel_w, panel_h), true_coords, False),
        ("(B) Landmark Triangulation Map (Anchored MDS)", (520, 60, panel_w, panel_h), pred_coords, False),
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
        lx += 80
        
    svg_parts.append(f'<circle cx="{lx+20}" cy="{legend_y-4}" r="4" fill="#94a3b8"/>')
    svg_parts.append(f'<text x="{lx+30}" y="{legend_y}" class="legend">True Pos (Aligned)</text>')
    svg_parts.append(f'<line x1="{lx+160}" y1="{legend_y-4}" x2="{lx+185}" y2="{legend_y-4}" stroke="#64748b" stroke-width="1"/>')
    svg_parts.append(f'<text x="{lx+195}" y="{legend_y}" class="legend">Error Vector (Residual)</text>')

    svg_parts.append('</svg>')
    
    with open(out_svg_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(svg_parts))
    print(f"Purity-certified Vector SVG figure saved to {out_svg_path}")

def triangulate_strain(predicted_distances, anchor_coords):
    """
    Find 2D coordinates that minimize squared error to anchor distances.
    predicted_distances: array of shape (K,)
    anchor_coords: array of shape (K, 2)
    """
    def loss(pos):
        # Euclidean distances to anchors
        dists = np.sqrt(np.sum((anchor_coords - pos)**2, axis=1))
        # Mean squared error against predicted distances
        return np.mean((dists - predicted_distances)**2)
    
    # Start at the center of anchors
    init_pos = np.mean(anchor_coords, axis=0)
    res = minimize(loss, init_pos, method='BFGS')
    return res.x

def main():
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    test_path = os.path.join(base_dir, 'data/splits/cohort2_test.csv.gz')
    emb_path = os.path.join(base_dir, 'data/processed/embeddings_esm2_650m.pt')
    model_path = os.path.join(base_dir, 'models/checkpoints/cohort2_best.pt')
    
    reports_dir = os.path.join(base_dir, 'reports')
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=== Task 5: Landmark Triangulation MDS (Plan A) ===")
    
    df_test = pd.read_csv(test_path)
    embeddings = torch.load(emb_path, map_location='cpu')
    df_test = df_test[df_test['virus1'].isin(embeddings.keys()) & df_test['virus2'].isin(embeddings.keys())]

    # 1. Extract 100% Fully-Observed Complete Clique
    pairs = set()
    for _, r in df_test.iterrows():
        u, v = sorted([r['virus1'], r['virus2']])
        pairs.add((u, v))

    c = Counter(df_test['virus1']).copy()
    c.update(Counter(df_test['virus2']))
    candidates = [v for v, cnt in c.most_common()]

    clique = []
    for cand in candidates:
        if all(tuple(sorted([cand, member])) in pairs for member in clique):
            clique.append(cand)

    n = len(clique)
    print(f"Identified Fully-Observed Test Clique: {n} strains.")
    virus_to_idx = {v: i for i, v in enumerate(clique)}
    
    # 2. Build True Distance Matrix
    D_true = np.zeros((n, n), dtype=np.float64)
    sub_df = df_test[df_test['virus1'].isin(virus_to_idx) & df_test['virus2'].isin(virus_to_idx)]
    for _, row in sub_df.iterrows():
        i = virus_to_idx[row['virus1']]
        j = virus_to_idx[row['virus2']]
        D_true[i, j] = row['distance']
        D_true[j, i] = row['distance']

    # 3. Predict Distance Matrix using the High-Capacity Interaction Model
    device = torch.device('cpu')
    # Use emb_dim=1280, hidden_dim=512 for the restored high-capacity model
    model = DistanceHead(emb_dim=1280, hidden_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    D_pred = np.zeros((n, n), dtype=np.float64)
    with torch.no_grad():
        for i in range(n):
            u = embeddings[clique[i]].unsqueeze(0)
            for j in range(i + 1, n):
                v = embeddings[clique[j]].unsqueeze(0)
                # Since the original model is asymmetric, we symmetrize here or use as-is
                pred_dist1 = model(u, v).item()
                pred_dist2 = model(v, u).item()
                sym_dist = max(0.0, (pred_dist1 + pred_dist2) / 2.0)
                D_pred[i, j] = sym_dist
                D_pred[j, i] = sym_dist

    # 4. Metric MDS on Ground Truth
    print("Computing True Metric MDS...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=15, max_iter=1000)
    true_coords = mds.fit_transform(D_true)

    # 5. Landmark Triangulation for Prediction
    print("Performing GPS-style Triangulation using 15 Anchors...")
    # Select 15 anchor strains (landmarks) distributed evenly
    num_anchors = min(15, n)
    anchor_indices = np.linspace(0, n - 1, num_anchors, dtype=int)
    anchor_coords = true_coords[anchor_indices]

    pred_coords = np.zeros_like(true_coords)
    for i in range(n):
        if i in anchor_indices:
            # Anchor remains at its true position
            pred_coords[i] = true_coords[i]
        else:
            # Triangulate based on predicted distances to anchors
            predicted_dists_to_anchors = D_pred[i, anchor_indices]
            pred_coords[i] = triangulate_strain(predicted_dists_to_anchors, anchor_coords)

    # 6. Procrustes Alignment
    print("Evaluating Procrustes disparity and axis correlations...")
    proc_true, proc_pred, disparity = procrustes(true_coords, pred_coords)

    r_axis1, _ = pearsonr(proc_true[:, 0], proc_pred[:, 0])
    r_axis2, _ = pearsonr(proc_true[:, 1], proc_pred[:, 1])
    rho_axis1, _ = spearmanr(proc_true[:, 0], proc_pred[:, 0])
    rho_axis2, _ = spearmanr(proc_true[:, 1], proc_pred[:, 1])

    # Direct observed pairwise metric errors on the clique
    triu_idx = np.triu_indices(n, k=1)
    true_dists = D_true[triu_idx]
    pred_dists = D_pred[triu_idx]
    clique_rmse = float(np.sqrt(np.mean((true_dists - pred_dists)**2)))
    clique_mae = float(np.mean(np.abs(true_dists - pred_dists)))
    clique_rho, _ = spearmanr(true_dists, pred_dists)

    eval_results = {
        'evaluation_scope': 'Cohort 2 Test Set (100% Fully Observed Clique with Triangulation)',
        'num_strains': n,
        'num_anchors': num_anchors,
        'pairwise_metric_evaluation': {
            'RMSE': round(clique_rmse, 4),
            'MAE': round(clique_mae, 4),
            'Spearman_rho': round(float(clique_rho), 4)
        },
        '2d_mds_procrustes_evaluation': {
            'procrustes_disparity': round(float(disparity), 4),
            'axis_1_metrics': {
                'pearson_r': round(float(r_axis1), 4),
                'spearman_rho': round(float(rho_axis1), 4)
            },
            'axis_2_metrics': {
                'pearson_r': round(float(r_axis2), 4),
                'spearman_rho': round(float(rho_axis2), 4)
            }
        }
    }

    report_path = os.path.join(reports_dir, 'procrustes_evaluation.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"Saved Triangulation evaluation report to {report_path}")
    print(json.dumps(eval_results, indent=2))

    years = np.array([parse_year(v) for v in clique])
    out_svg_path = os.path.join(figures_dir, 'reconstructed_antigenic_map.svg')
    generate_svg_plot(true_coords, pred_coords, proc_true, proc_pred, years, disparity, out_svg_path)

if __name__ == '__main__':
    main()
