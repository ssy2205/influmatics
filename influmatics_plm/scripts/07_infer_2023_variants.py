import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from models import DistanceHead

def parse_fasta_2023(fasta_path):
    """
    Parse Data_Sheet_3.FASTA and extract only 2023 records.
    Header format: >[Accession] [Virus Name] [Date YYYY/MM/DD] [Segment]
    Example: >WHO49069 A/Maryland/02/2023 2023/01/01 HA
    """
    records = []
    with open(fasta_path, 'r', encoding='utf-8') as f:
        current_header = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    records.append((current_header, "".join(current_seq)))
                current_header = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)
        if current_header:
            records.append((current_header, "".join(current_seq)))

    strains_2023 = []
    for header, seq in records:
        tokens = header.split()
        if len(tokens) >= 4:
            accession = tokens[0]
            segment = tokens[-1]
            date_str = tokens[-2]
            virus_name = " ".join(tokens[1:-2])
        else:
            virus_name = header
            date_str = "2023"
            
        if "2023" in date_str or "/2023" in virus_name:
            strains_2023.append({
                'virus_name': virus_name,
                'year': 2023,
                'date': date_str,
                'sequence': seq
            })
            
    return strains_2023

def extract_esm2_embeddings(sequences, device, batch_size=1):
    """
    Extract 1280-dimensional mean-pooled ESM-2 embeddings (excluding CLS and EOS).
    """
    print("Loading ESM-2 model (facebook/esm2_t33_650M_UR50D)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    model.eval()
    
    embeddings = []
    print(f"Extracting embeddings for {len(sequences)} strains on {device}...")
    with torch.no_grad():
        for seq in tqdm(sequences, desc="ESM-2 Embedding"):
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            hidden = outputs.last_hidden_state
            mask = inputs['attention_mask']
            seq_len = mask.sum().item()
            if seq_len > 2:
                valid_hidden = hidden[0, 1:seq_len-1, :]
            else:
                valid_hidden = hidden[0, :, :]
                
            mean_pooled = valid_hidden.mean(dim=0).cpu()
            embeddings.append(mean_pooled)
            
    return torch.stack(embeddings)

def main():
    parser = argparse.ArgumentParser(description="Infer antigenic distance of 2023 variants against vaccine strain.")
    parser.add_argument("--drift-threshold", type=float, default=2.0, help="Antigenic drift threshold in a.u. (default: 2.0)")
    parser.add_argument("--vaccine-strain", type=str, default="A/Darwin/9/2021", help="Target vaccine strain name")
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Execution Device: {device}")

    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    project_root = os.path.abspath(os.path.join(base_dir, '../..'))

    fasta_path = os.path.join(project_root, 'PMC10834737_SupplementaryFiles/Data_Sheet_3.FASTA')
    if not os.path.exists(fasta_path):
        fasta_path = os.path.join(base_dir, 'PMC10834737_SupplementaryFiles/Data_Sheet_3.FASTA')
    
    emb_path = os.path.join(base_dir, 'data/processed/embeddings_esm2_650m.pt')
    model_path = os.path.join(base_dir, 'models/checkpoints/cohort2_best.pt')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    out_csv_path = os.path.join(reports_dir, '2023_vaccine_escape_predictions.csv')

    print(f"1. Reading 2023 variant sequences from: {fasta_path}")
    strains_2023 = parse_fasta_2023(fasta_path)
    print(f"   Found {len(strains_2023)} strains from 2023.")
    if len(strains_2023) == 0:
        print("Error: No 2023 strains found in FASTA!")
        sys.exit(1)

    print(f"2. Loading reference vaccine strain embedding: '{args.vaccine_strain}'")
    if not os.path.exists(emb_path):
        print(f"Error: Processed embeddings not found at {emb_path}")
        sys.exit(1)
    
    processed_embs = torch.load(emb_path, map_location='cpu')
    if args.vaccine_strain not in processed_embs:
        print(f"Error: Vaccine strain '{args.vaccine_strain}' not found in {emb_path}")
        sys.exit(1)
    vaccine_emb = processed_embs[args.vaccine_strain]

    print("3. Extracting ESM-2 embeddings for 2023 strains...")
    sequences = [s['sequence'] for s in strains_2023]
    var_embs = extract_esm2_embeddings(sequences, device)

    print(f"4. Loading trained DistanceHead model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)
    model = DistanceHead(emb_dim=1280).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"5. Predicting antigenic distance from '{args.vaccine_strain}'...")
    num_strains = len(strains_2023)
    vaccine_batch = vaccine_emb.unsqueeze(0).expand(num_strains, -1).to(device)
    var_embs_device = var_embs.to(device)

    with torch.no_grad():
        pred_distances = model(var_embs_device, vaccine_batch).cpu().numpy()

    # Build results DataFrame
    results = []
    for i, s in enumerate(strains_2023):
        dist = float(pred_distances[i])
        is_drift = bool(dist >= args.drift_threshold)
        results.append({
            'virus_name': s['virus_name'],
            'year': s['year'],
            'predicted_distance': round(dist, 4),
            'is_drift': is_drift
        })

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(by='predicted_distance', ascending=False).reset_index(drop=True)

    df_out.to_csv(out_csv_path, index=False)
    print(f"\nSaved predictions to: {out_csv_path}")

    drift_candidates = df_out[df_out['is_drift']]
    print("\n" + "=" * 60)
    print("        2023 Variant Vaccine Escape Prediction Summary       ")
    print("=" * 60)
    print(f"Total 2023 variants evaluated : {len(df_out)}")
    print(f"Reference vaccine strain     : {args.vaccine_strain}")
    print(f"Antigenic drift threshold    : {args.drift_threshold:.1f} a.u.")
    print(f"Predicted escape variants    : {len(drift_candidates)} ({len(drift_candidates)/len(df_out)*100:.1f}%)")
    print("-" * 60)

    if len(drift_candidates) > 0:
        print("Top 10 High-Risk Escape Variants:")
        print(f"{'Rank':<5} {'Virus Name':<40} {'Pred Dist (a.u.)':<18} {'Drift?'}")
        print("-" * 68)
        for idx, row in drift_candidates.head(10).iterrows():
            print(f"{idx+1:<5} {row['virus_name']:<40} {row['predicted_distance']:<18.4f} {row['is_drift']}")
    else:
        print("No variants met or exceeded the 2.0 a.u. drift threshold.")
    print("=" * 60)

if __name__ == '__main__':
    main()
