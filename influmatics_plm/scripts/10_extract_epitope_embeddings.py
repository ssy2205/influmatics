import os
import torch
from transformers import AutoTokenizer, EsmModel
import pandas as pd
from tqdm import tqdm

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from epitope_pooling import EpitopeWeightedPooling

def load_fasta(filepath):
    seqs = {}
    with open(filepath, 'r') as file:
        header = None
        seq = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    seqs[header] = "".join(seq)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header:
            seqs[header] = "".join(seq)
    return seqs

def get_virus_to_seq_mapping():
    base_dir = os.path.abspath(os.path.join(script_dir, '../../..'))
    supp_dir = os.path.join(base_dir, 'PMC10834737_SupplementaryFiles')
    
    seqs1 = load_fasta(os.path.join(supp_dir, 'Data_Sheet_1.FASTA'))
    seqs2 = load_fasta(os.path.join(supp_dir, 'Data_Sheet_2.FASTA'))
    
    virus_to_seq = {}
    
    # 1. Cohort 1
    df4_s1 = pd.read_excel(os.path.join(supp_dir, 'Data_Sheet_4.XLSX'), sheet_name=0, header=None)
    df4_s2 = pd.read_excel(os.path.join(supp_dir, 'Data_Sheet_4.XLSX'), sheet_name=1, header=None)
    for i in range(2, 272):
        short_name = str(df4_s2.iloc[i, 1]).strip()
        acc = str(df4_s1.iloc[i+1, 1]).strip().replace('>', '')
        full_name = str(df4_s1.iloc[i+1, 2]).strip()
        for h, s in seqs1.items():
            if h.startswith(acc) or full_name.lower() in h.lower() or full_name.split('/')[-2] in h:
                virus_to_seq[short_name] = s
                break
                
    # 2. Cohort 2
    df5 = pd.read_excel(os.path.join(supp_dir, 'Data_Sheet_5.XLSX'), sheet_name='Sheet', header=None)
    for idx, row in df5.iloc[3:].iterrows():
        val = str(row[0]).strip()
        if val and val != 'nan':
            for h, s in seqs2.items():
                if val in h:
                    virus_to_seq[val] = s
                    break
    return virus_to_seq

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    out_path = os.path.join(base_dir, 'data/processed/embeddings_esm2_650m_epitope_weighted.pt')
    
    virus_to_seq = get_virus_to_seq_mapping()
    print(f"Total viruses to extract with epitope weighting: {len(virus_to_seq):,}")
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    model.eval()
    
    pooler = EpitopeWeightedPooling(seq_len=329, w_koel=3.5, w_epitope=2.0, w_framework=1.0).to(device)
    
    embeddings = {}
    items = list(virus_to_seq.items())
    batch_size = 16
    
    print("Extracting batched epitope-weighted embeddings...")
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            batch_names = [name for name, seq in batch_items]
            batch_seqs = [seq for name, seq in batch_items]
            
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            hidden = outputs.last_hidden_state  # (B, L, 1280)
            mask = inputs['attention_mask']     # (B, L)
            
            for b, name in enumerate(batch_names):
                seq_len = mask[b].sum().item()
                # Exclude <cls> and <eos>
                if seq_len > 2:
                    valid_tokens = hidden[b, 1:seq_len-1, :] # (L_valid, 1280)
                else:
                    valid_tokens = hidden[b, :, :]
                    
                pooled = pooler(valid_tokens).cpu()
                embeddings[name] = pooled
                
            if (i // batch_size) % 10 == 0 or (i + batch_size) >= len(items):
                print(f"Processed {min(i + batch_size, len(items))}/{len(items)} strains...")
                
    torch.save(embeddings, out_path)
    print(f"Successfully saved {len(embeddings)} epitope-weighted embeddings to {out_path}")

if __name__ == '__main__':
    main()
