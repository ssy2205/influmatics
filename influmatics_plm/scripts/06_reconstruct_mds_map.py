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
    
    # Year colors: 2019, 2020, 2021, 2022
    color_map = {
        2019: "#3b82f6", # blue
        2020: "#10b981", # green
        2021: "#f59e0b", # amber
        2022: "#ef4444"  # red
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
        '  .title { font-family: -apple-system, sans-serif; font-size: 15px; font-weight: bold; fill: #1e3a8a; }',
        '  .subtitle { font-family: -apple-system, sans-serif; font-size: 11px; fill: #64748b; }',
        '  .axis-label { font-family: -apple-system, sans-serif; font-size: 10px; fill: #94a3b8; }',
        '  .legend { font-family: -apple-system, sans-serif; font-size: 11px; fill: #334155; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>'
    ]

    panels = [
        ("(A) True Antigenic Map (MDS on HI Distances)", (30, 60, panel_w, panel_h), true_coords, False),
        ("(B) Model-Reconstructed Map (MDS on Predictions)", (520, 60, panel_w, panel_h), pred_coords, False),
        (f"(C) Procrustes Superposition (Disparity: {disparity:.4f})", (1010, 60, panel_w, panel_h), None, True)
    ]

    for p_title, (bx, by, bw, bh), coords, is_overlay in panels:
        # draw panel background & border
        svg_parts.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="#f8fafc" stroke="#e2e8f0" rx="8"/>')
        svg_parts.append(f'<text x="{bx+15}" y="{by-15}" class="title">{p_title}</text>')
        
        # grid lines
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
            # combine proc_true and proc_pred for uniform scaling
            all_pts = np.vstack([proc_true, proc_pred])
            all_sx, all_sy = scale_coords(all_pts, (bx, by, bw, bh))
            n = len(proc_true)
            t_sx, t_sy = all_sx[:n], all_sy[:n]
            p_sx, p_sy = all_sx[n:], all_sy[n:]

            # draw connecting error lines
            for i in range(n):
                svg_parts.append(f'<line x1="{t_sx[i]:.1f}" y1="{t_sy[i]:.1f}" x2="{p_sx[i]:.1f}" y2="{p_sy[i]:.1f}" stroke="#64748b" stroke-width="0.8" stroke-opacity="0.4"/>')
            
            # draw true points (gray)
            for i in range(n):
                svg_parts.append(f'<circle cx="{t_sx[i]:.1f}" cy="{t_sy[i]:.1f}" r="4" fill="#94a3b8" fill-opacity="0.6"/>')
            
            # draw predicted points (colored by year)
            for i in range(n):
                col = color_map.get(years[i], "#64748b")
                svg_parts.append(f'<circle cx="{p_sx[i]:.1f}" cy="{p_sy[i]:.1f}" r="5" fill="{col}" fill-opacity="0.9" stroke="#ffffff" stroke-width="1"/>')

    # Legend at bottom
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
    print(f"Vector SVG figure saved to {out_svg_path}")

def main():
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    test_path = os.path.join(base_dir, 'data/splits/cohort2_test.csv.gz')
    emb_path = os.path.join(base_dir, 'data/processed/embeddings_esm2_650m.pt')
    model_path = os.path.join(base_dir, 'models/checkpoints/cohort2_best.pt')
    
    reports_dir = os.path.join(base_dir, 'reports')
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=== Task 5: 2D MDS Map Reconstruction & Procrustes Evaluation ===")
    
    df_test = pd.read_csv(test_path)
    embeddings = torch.load(emb_path, map_location='cpu')
    df_test = df_test[df_test['virus1'].isin(embeddings.keys()) & df_test['virus2'].isin(embeddings.keys())]
    print(f"Total valid test pairs: {len(df_test):,}")

    c = Counter(df_test['virus1']).copy()
    c.update(Counter(df_test['virus2']))
    top_viruses = [v for v, count in c.most_common(80)]
    virus_to_idx = {v: i for i, v in enumerate(top_viruses)}
    n = len(top_viruses)
    print(f"Selected {n} representative test viruses across 2019-2022.")

    device = torch.device('cpu')
    model = DistanceHead(emb_dim=1280).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    D_true = np.full((n, n), np.nan)
    np.fill_diagonal(D_true, 0.0)

    sub_df = df_test[df_test['virus1'].isin(virus_to_idx) & df_test['virus2'].isin(virus_to_idx)]
    for _, row in sub_df.iterrows():
        i = virus_to_idx[row['virus1']]
        j = virus_to_idx[row['virus2']]
        D_true[i, j] = row['distance']
        D_true[j, i] = row['distance']

    D_pred = np.zeros((n, n))
    with torch.no_grad():
        for i in range(n):
            u = embeddings[top_viruses[i]].unsqueeze(0)
            for j in range(i + 1, n):
                v = embeddings[top_viruses[j]].unsqueeze(0)
                pred_dist1 = model(u, v).item()
                pred_dist2 = model(v, u).item()
                sym_dist = max(0.0, (pred_dist1 + pred_dist2) / 2.0)
                D_pred[i, j] = sym_dist
                D_pred[j, i] = sym_dist

    mask_missing = np.isnan(D_true)
    D_true_imputed = np.where(mask_missing, D_pred, D_true)

    print("Fitting Metric MDS...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=10, max_iter=500)
    true_coords = mds.fit_transform(D_true_imputed)
    pred_coords = mds.fit_transform(D_pred)

    print("Running Procrustes superimposition...")
    proc_true, proc_pred, disparity = procrustes(true_coords, pred_coords)

    r_axis1, _ = pearsonr(proc_true[:, 0], proc_pred[:, 0])
    r_axis2, _ = pearsonr(proc_true[:, 1], proc_pred[:, 1])
    rho_axis1, _ = spearmanr(proc_true[:, 0], proc_pred[:, 0])
    rho_axis2, _ = spearmanr(proc_true[:, 1], proc_pred[:, 1])

    eval_results = {
        'evaluation_scope': 'Cohort 2 Test Set (2019-2022 Strains)',
        'num_viruses': n,
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

    report_path = os.path.join(reports_dir, 'procrustes_evaluation.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation report to {report_path}")
    print(json.dumps(eval_results, indent=2))

    years = np.array([parse_year(v) for v in top_viruses])
    out_svg_path = os.path.join(figures_dir, 'reconstructed_antigenic_map.svg')
    generate_svg_plot(true_coords, pred_coords, proc_true, proc_pred, years, disparity, out_svg_path)

if __name__ == '__main__':
    main()
