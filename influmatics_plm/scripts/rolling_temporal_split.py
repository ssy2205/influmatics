import pandas as pd
import numpy as np

def rolling_temporal_split(df, time_col='assay_time', splits=[2012, 2014, 2016, 2018], strict_cold_start=False):
    """
    Generate expanding-window temporal cross-validation folds.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'item_id', 'reference_id', and time_col.
        time_col (str): Name of the time column. Assumed to be year integers.
        splits (list): List of year boundaries for generating the sequential rolling folds.
                       For N splits, N-1 folds will be generated.
        strict_cold_start (bool): If True, both item_id and reference_id in val must not be in train.
                                  If False, only item_id in val must not be in train's item_id.
                                  
    Returns:
        list of tuples: [(train_df, val_df), ...] for each fold.
    """
    folds = []
    
    for i in range(len(splits) - 1):
        train_end = splits[i]
        val_end = splits[i+1]
        
        train_mask = df[time_col] <= train_end
        val_mask = (df[time_col] > train_end) & (df[time_col] <= val_end)
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        
        # Condition 1: item_id in val must not appear in train's item_id (Query-node disjoint)
        train_items = set(train_df['item_id'])
        val_df = val_df[~val_df['item_id'].isin(train_items)]
        
        # Condition 2: strict_cold_start
        if strict_cold_start:
            train_refs = set(train_df['reference_id'])
            # In strict mode, reference_id of val must also not be present in train
            val_df = val_df[~val_df['reference_id'].isin(train_refs)]
        
        folds.append((train_df, val_df))
        
    return folds

if __name__ == '__main__':
    # 1. Generate Synthetic Data
    np.random.seed(42)
    
    n_edges = 100
    years = np.random.choice([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018], size=n_edges)
    item_ids = [f"item_{np.random.randint(1, 30)}" for _ in range(n_edges)]
    ref_ids = [f"ref_{np.random.randint(1, 20)}" for _ in range(n_edges)]
    distances = np.random.rand(n_edges)
    
    df = pd.DataFrame({
        'item_id': item_ids,
        'reference_id': ref_ids,
        'distance': distances,
        'assay_time': years
    })
    
    print(f"Generated DataFrame with {len(df)} rows.")
    
    # 2. Test default (Query-node disjoint)
    print("\n--- Testing standard Expanding-window Temporal CV ---")
    splits = [2012, 2014, 2016, 2018]
    folds = rolling_temporal_split(df, time_col='assay_time', splits=splits, strict_cold_start=False)
    
    for idx, (train, val) in enumerate(folds):
        train_items = set(train['item_id'])
        val_items = set(val['item_id'])
        
        # Validate query-node disjoint
        overlap = train_items.intersection(val_items)
        assert len(overlap) == 0, f"Leakage found in Fold {idx+1}: Overlapping item_ids {overlap}"
        
        # Validate time splitting
        if len(train) > 0:
            assert train['assay_time'].max() <= splits[idx], "Train time condition violated"
        if len(val) > 0:
            assert val['assay_time'].min() > splits[idx], "Val time condition violated (lower bound)"
            assert val['assay_time'].max() <= splits[idx+1], "Val time condition violated (upper bound)"
            
        print(f"Fold {idx+1} | Train Size: {len(train):>3} | Val Size: {len(val):>3} | Train Items: {len(train_items):>2} | Val Items: {len(val_items):>2}")
            
    print(">> Standard expanding-window temporal CV checks passed!")
    
    # 3. Test strict_cold_start=True
    print("\n--- Testing Strict Cold-start Expanding-window Temporal CV ---")
    strict_folds = rolling_temporal_split(df, time_col='assay_time', splits=splits, strict_cold_start=True)
    
    for idx, (train, val) in enumerate(strict_folds):
        train_items = set(train['item_id'])
        val_items = set(val['item_id'])
        train_refs = set(train['reference_id'])
        val_refs = set(val['reference_id'])
        
        # Validate query-node and reference-node disjoint
        item_overlap = train_items.intersection(val_items)
        ref_overlap = train_refs.intersection(val_refs)
        
        assert len(item_overlap) == 0, f"Leakage found in Fold {idx+1}: Overlapping item_ids {item_overlap}"
        assert len(ref_overlap) == 0, f"Leakage found in Fold {idx+1}: Overlapping reference_ids {ref_overlap}"
        
        print(f"Fold {idx+1} | Train Size: {len(train):>3} | Val Size: {len(val):>3} | Val Items (Strict): {len(val_items):>2} | Val Refs (Strict): {len(val_refs):>2}")
        
    print(">> Strict cold-start CV checks passed!")
