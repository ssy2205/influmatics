import os
import sys
import json
import re
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from temporal_group_split import temporal_group_split

def parse_year_cohort1(name):
    # e.g. BI/15793/68 -> 1968
    parts = str(name).strip().split('/')
    m = re.search(r'(\d+)$', parts[-1])
    if m:
        yy = int(m.group(1))
        if yy >= 1000:
            return yy
        return 1900 + yy if yy >= 60 else 2000 + yy
    return None

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

    print("=== Processing Cohort 1 Splits (1968 - 2003) ===")
    c1_path = os.path.join(processed_dir, 'cohort1_pairwise.csv.gz')
    df1 = pd.read_csv(c1_path)
    df1['node_A'] = df1['virus1']
    df1['node_B'] = df1['virus2']
    df1['year_A'] = df1['node_A'].map(parse_year_cohort1)
    df1['year_B'] = df1['node_B'].map(parse_year_cohort1)

    train1, val1, test1, summary1 = temporal_group_split(
        df1, 
        train_cutoff_year=1995, 
        val_cutoff_year=1999
    )

    train1.to_csv(os.path.join(splits_dir, 'cohort1_train.csv.gz'), index=False, compression='gzip')
    val1.to_csv(os.path.join(splits_dir, 'cohort1_val.csv.gz'), index=False, compression='gzip')
    test1.to_csv(os.path.join(splits_dir, 'cohort1_test.csv.gz'), index=False, compression='gzip')

    with open(os.path.join(splits_dir, 'cohort1_splits.json'), 'w', encoding='utf-8') as f:
        json.dump(summary1, f, indent=2, ensure_ascii=False)
    print("Cohort 1 split summary saved to cohort1_splits.json")
    print(json.dumps(summary1, indent=2))

    print("\n=== Processing Cohort 2 Splits (2003 - 2022) ===")
    c2_path = os.path.join(processed_dir, 'cohort2_pairwise.csv.gz')
    df2 = pd.read_csv(c2_path)
    df2['node_A'] = df2['virus1']
    df2['node_B'] = df2['virus2']
    df2['year_A'] = df2['node_A'].map(parse_year_cohort2)
    df2['year_B'] = df2['node_B'].map(parse_year_cohort2)

    train2, val2, test2, summary2 = temporal_group_split(
        df2, 
        train_cutoff_year=2015, 
        val_cutoff_year=2018
    )

    train2.to_csv(os.path.join(splits_dir, 'cohort2_train.csv.gz'), index=False, compression='gzip')
    val2.to_csv(os.path.join(splits_dir, 'cohort2_val.csv.gz'), index=False, compression='gzip')
    test2.to_csv(os.path.join(splits_dir, 'cohort2_test.csv.gz'), index=False, compression='gzip')

    with open(os.path.join(splits_dir, 'cohort2_splits.json'), 'w', encoding='utf-8') as f:
        json.dump(summary2, f, indent=2, ensure_ascii=False)
    print("Cohort 2 split summary saved to cohort2_splits.json")
    print(json.dumps(summary2, indent=2))

if __name__ == '__main__':
    process_splits()
