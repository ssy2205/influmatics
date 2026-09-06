import os
import re
import json
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

def find_pngs(seq):
    # Asn-X-Ser/Thr where X is not Proline
    pattern = r'N[^P][ST]'
    return [m.start() + 1 for m in re.finditer(pattern, seq)]

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    supp_dir = os.path.abspath(os.path.join(base_dir, '../../PMC10834737_SupplementaryFiles'))
    out_path = os.path.join(base_dir, 'data/processed/glyco_counts.json')
    
    seqs1 = load_fasta(os.path.join(supp_dir, 'Data_Sheet_1.FASTA'))
    seqs2 = load_fasta(os.path.join(supp_dir, 'Data_Sheet_2.FASTA'))
    
    # Map all virus names
    virus_to_glyco = {}
    
    # 1. Cohort 1
    df4_s1 = pd.read_excel(os.path.join(supp_dir, 'Data_Sheet_4.XLSX'), sheet_name=0, header=None)
    df4_s2 = pd.read_excel(os.path.join(supp_dir, 'Data_Sheet_4.XLSX'), sheet_name=1, header=None)
    for i in range(2, 272):
        short_name = str(df4_s2.iloc[i, 1]).strip()
        acc = str(df4_s1.iloc[i+1, 1]).strip().replace('>', '')
        full_name = str(df4_s1.iloc[i+1, 2]).strip()
        for h, s in seqs1.items():
            if h.startswith(acc) or full_name.lower() in h.lower() or full_name.split('/')[-2] in h:
                sites = find_pngs(s)
                virus_to_glyco[short_name] = {'count': len(sites), 'sites': sites}
                break
                
    # 2. Cohort 2
    df5 = pd.read_excel(os.path.join(supp_dir, 'Data_Sheet_5.XLSX'), sheet_name='Sheet', header=None)
    for idx, row in df5.iloc[3:].iterrows():
        val = str(row[0]).strip()
        if val and val != 'nan':
            for h, s in seqs2.items():
                if val in h:
                    sites = find_pngs(s)
                    virus_to_glyco[val] = {'count': len(sites), 'sites': sites}
                    break
                    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(virus_to_glyco, f, indent=2)
    print(f"Successfully extracted N-glycosylation profiles for {len(virus_to_glyco)} strains to {out_path}")

if __name__ == '__main__':
    main()
