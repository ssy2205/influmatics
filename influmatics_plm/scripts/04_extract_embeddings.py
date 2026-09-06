import os
import torch
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm
import pandas as pd

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
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    supp_dir = os.path.join(base_dir, 'PMC10834737_SupplementaryFiles')
    
    # 1. Load FASTA 1 & 2
    seqs1 = load_fasta(os.path.join(supp_dir, 'Data_Sheet_1.FASTA'))
    seqs2 = load_fasta(os.path.join(supp_dir, 'Data_Sheet_2.FASTA'))
    
    virus_to_seq = {}
    
    # 2. Map Cohort 1 (Short name in Sheet 2 -> Accession/Full name in Sheet 1 -> FASTA 1)
    df4_s1 = pd.read_excel(os.path.join(supp_dir, 'Data_Sheet_4.XLSX'), sheet_name=0, header=None)
    df4_s2 = pd.read_excel(os.path.join(supp_dir, 'Data_Sheet_4.XLSX'), sheet_name=1, header=None)
    
    for i in range(2, 272):
        short_name = str(df4_s2.iloc[i, 1]).strip()
        acc = str(df4_s1.iloc[i+1, 1]).strip().replace('>', '')
        full_name = str(df4_s1.iloc[i+1, 2]).strip()
        
        found_seq = None
        for h, s in seqs1.items():
            if h.startswith(acc) or full_name.lower() in h.lower() or full_name.split('/')[-2] in h:
                found_seq = s
                break
        if found_seq is not None:
            virus_to_seq[short_name] = found_seq
            
    print(f"Cohort 1 mapped: {len([k for k in virus_to_seq if k.startswith('BI/') or '/' in k and len(k.split('/')[0]) <= 3])} strains")
    
    # 3. Map Cohort 2 (Full names in Sheet 1 -> FASTA 2)
    df5 = pd.read_excel(os.path.join(supp_dir, 'Data_Sheet_5.XLSX'), sheet_name='Sheet', header=None)
    c2_count = 0
    for idx, row in df5.iloc[3:].iterrows():
        val = str(row[0]).strip()
        if val and val != 'nan':
            for h, s in seqs2.items():
                if val in h:
                    virus_to_seq[val] = s
                    c2_count += 1
                    break
    print(f"Cohort 2 mapped: {c2_count} strains")
    return virus_to_seq

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'embeddings_esm2_650m.pt')
    
    embeddings = {}
    if os.path.exists(out_path):
        print(f"Loading existing embeddings from {out_path}...")
        embeddings = torch.load(out_path, map_location='cpu')
        print(f"Found {len(embeddings)} existing embeddings.")
        
    virus_to_seq = get_virus_to_seq_mapping()
    missing_viruses = {k: v for k, v in virus_to_seq.items() if k not in embeddings}
    
    print(f"Total target viruses: {len(virus_to_seq)}, Already extracted: {len(embeddings)}, Need extraction: {len(missing_viruses)}")
    
    if len(missing_viruses) > 0:
        print("Loading ESM-2 model...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
        model.eval()
        
        print(f"Extracting embeddings for {len(missing_viruses)} viruses on {device}...")
        with torch.no_grad():
            for v, seq in tqdm(missing_viruses.items()):
                inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: val.to(device) for k, val in inputs.items()}
                outputs = model(**inputs)
                
                hidden = outputs.last_hidden_state
                mask = inputs['attention_mask']
                seq_len = mask.sum().item()
                if seq_len > 2:
                    valid_hidden = hidden[0, 1:seq_len-1, :]
                else:
                    valid_hidden = hidden[0, :, :]
                    
                mean_pooled = valid_hidden.mean(dim=0).cpu()
                embeddings[v] = mean_pooled
                
        torch.save(embeddings, out_path)
        print(f"Updated and saved total {len(embeddings)} embeddings to {out_path}")
    else:
        print("All target virus embeddings are already present!")

if __name__ == '__main__':
    main()
