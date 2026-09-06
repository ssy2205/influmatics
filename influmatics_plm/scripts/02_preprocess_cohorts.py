import pandas as pd
import numpy as np
import os
import sys

# Add scripts directory to path to import pairwise_nan_mae
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from pairwise_nan_mae import pairwise_nan_mae

def parse_titer(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str == '*' or val_str == '':
        return np.nan
    if val_str.startswith('<'):
        try:
            return float(val_str[1:]) / 2.0
        except ValueError:
            return np.nan
    elif val_str.startswith('>'):
        try:
            return float(val_str[1:]) * 2.0
        except ValueError:
            return np.nan
    else:
        try:
            return float(val_str)
        except ValueError:
            return np.nan

def process_cohort(df, virus_col_idx, data_row_start, data_col_start, out_path):
    # Extract virus names and matrix
    viruses = df.iloc[data_row_start:, virus_col_idx].astype(str).values
    matrix_raw = df.iloc[data_row_start:, data_col_start:].values
    
    # Apply parse_titer
    parse_vec = np.vectorize(parse_titer, otypes=[float])
    matrix_cleaned = parse_vec(matrix_raw)
    
    # Log2 transform
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix_log2 = np.log2(matrix_cleaned)
        
    # Compute pairwise MAE
    pairwise_df = pairwise_nan_mae(matrix_log2, min_shared_features=3)
    
    # Map row indices back to virus names
    pairwise_df['virus1'] = viruses[pairwise_df['row_i_idx']]
    pairwise_df['virus2'] = viruses[pairwise_df['row_j_idx']]
    
    # Select desired columns
    out_df = pairwise_df[['virus1', 'virus2', 'distance', 'shared_count']]
    
    # Save as compressed CSV
    out_df.to_csv(out_path, index=False, compression='gzip')
    print(f"Saved {len(out_df)} pairs to {out_path}")


def main():
    os.makedirs('influmatics_code/influmatics_plm/data/processed', exist_ok=True)
    
    print("Processing Cohort 1...")
    # Sheet2 of Data_Sheet_4 is '抗原数据'
    df4 = pd.read_excel('PMC10834737_SupplementaryFiles/Data_Sheet_4.XLSX', sheet_name='抗原数据', header=None)
    process_cohort(
        df=df4, 
        virus_col_idx=1, 
        data_row_start=2, 
        data_col_start=2, 
        out_path='influmatics_code/influmatics_plm/data/processed/cohort1_pairwise.csv.gz'
    )
    
    print("Processing Cohort 2...")
    # Sheet1 of Data_Sheet_5 is 'Sheet'
    df5 = pd.read_excel('PMC10834737_SupplementaryFiles/Data_Sheet_5.XLSX', sheet_name='Sheet', header=None)
    process_cohort(
        df=df5, 
        virus_col_idx=0, 
        data_row_start=3, 
        data_col_start=1, 
        out_path='influmatics_code/influmatics_plm/data/processed/cohort2_pairwise.csv.gz'
    )

if __name__ == '__main__':
    main()
