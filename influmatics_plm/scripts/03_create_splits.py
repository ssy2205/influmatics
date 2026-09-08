import os
import sys
import json
import pandas as pd
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from rolling_temporal_split import rolling_temporal_split

def parse_year_cohort2(name):
    # e.g. A/Panama/2007/1999 -> 1999, A/Firenze/24/08 -> 2008
    parts = str(name).strip().split('/')
    m = re.search(r'(\d+)$', parts[-1])
    if m:
        yy = int(m.group(1))
        if yy >= 1000:
            return yy
        return 2000 + yy if yy < 50 else 1900 + yy
    return None

def process_splits():
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(base_dir, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    splits_dir = os.path.join(data_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    print("\n=== Processing Cohort 2 Splits with Expanding Window (2003 - 2018) ===")
    c2_path = os.path.join(processed_dir, 'cohort2_pairwise.csv.gz')
    df2 = pd.read_csv(c2_path)
    
    # We rename columns to fit rolling_temporal_split format
    df2['item_id'] = df2['virus1']
    df2['reference_id'] = df2['virus2']
    df2['assay_time'] = df2['virus1'].map(parse_year_cohort2) # Assume test virus defines assay time
    
    # Let's filter out NaN assay_time
    df2 = df2.dropna(subset=['assay_time'])
    
    # The EXPERIMENT_PLAN.md specifies: Fold 1: <=2012 -> 2013-14, Fold 2: <=2014 -> 2015-16, Fold 3: <=2016 -> 2017-18
    # Which corresponds to splits = [2012, 2014, 2016, 2018]
    splits = [2012, 2014, 2016, 2018]
    folds = rolling_temporal_split(df2, time_col='assay_time', splits=splits, strict_cold_start=False)
    
    summary2 = []
    
    for idx, (train_df, val_df) in enumerate(folds):
        fold_num = idx + 1
        train_path = os.path.join(splits_dir, f'cohort2_fold{fold_num}_train.csv.gz')
        val_path = os.path.join(splits_dir, f'cohort2_fold{fold_num}_val.csv.gz')
        
        train_df.to_csv(train_path, index=False, compression='gzip')
        val_df.to_csv(val_path, index=False, compression='gzip')
        
        summary2.append({
            "fold": fold_num,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "train_items": len(set(train_df['item_id'])),
            "val_items": len(set(val_df['item_id']))
        })
        print(f"Fold {fold_num}: Train={len(train_df)} pairs, Val={len(val_df)} pairs")

    with open(os.path.join(splits_dir, 'cohort2_rolling_splits.json'), 'w', encoding='utf-8') as f:
        json.dump(summary2, f, indent=2, ensure_ascii=False)
    print("Cohort 2 rolling split summary saved to cohort2_rolling_splits.json")

if __name__ == '__main__':
    process_splits()
