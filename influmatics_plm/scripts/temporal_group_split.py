"""Temporal edge splitting with node-disjoint test holdout."""

from numbers import Integral

import numpy as np
import pandas as pd


def temporal_group_split(df, train_cutoff_year, val_cutoff_year):
    """Return (train_df, val_df, test_df, summary).

    Temporal candidates:
      train: max(year_A, year_B) <= train_cutoff_year
      val:   train_cutoff_year < max(year_A, year_B) <= val_cutoff_year
      test:  min(year_A, year_B) > val_cutoff_year

    Test has priority: remove edges touching any test node from both train
    and validation. Train and validation may share nodes. Edges crossing the
    validation cutoff are dropped. Node IDs are assumed to be hashable and
    consistently represented in both endpoint columns. Missing node IDs or
    non-finite/non-integral years are rejected. Distance is carried unchanged.

    Outputs preserve input columns, row order, and index (including duplicates).
    Input is not modified. Counts describe edge rows, not unique node pairs.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    required = ["node_A", "node_B", "distance", "year_A", "year_B"]
    if not df.columns.is_unique:
        raise ValueError("df must have unique column names")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for name, value in (("train_cutoff_year", train_cutoff_year),
                        ("val_cutoff_year", val_cutoff_year)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if train_cutoff_year >= val_cutoff_year:
        raise ValueError("train_cutoff_year must be < val_cutoff_year")
    if df[["node_A", "node_B"]].isna().to_numpy().any():
        raise ValueError("Node IDs must not be missing")

    # Convert years only, never node IDs, to avoid changing node identity.
    years = []
    for column in ("year_A", "year_B"):
        values = df[column]
        if len(values) and (
            not pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)
            or pd.api.types.is_complex_dtype(values.dtype)
        ):
            raise ValueError(f"{column} must contain numeric integer years")
        array = values.to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(array).all() or (array != np.floor(array)).any():
            raise ValueError(f"{column} must contain finite integer years")
        years.append(array)
    pair_year = np.maximum(*years)
    train_candidate = pair_year <= train_cutoff_year
    val_candidate = (pair_year > train_cutoff_year) & (pair_year <= val_cutoff_year)
    test_mask = np.minimum(*years) > val_cutoff_year

    def node_ids(frame):
        return pd.unique(pd.concat(
            [frame["node_A"], frame["node_B"]], ignore_index=True
        ))

    test_df = df.loc[test_mask].copy()
    test_nodes = node_ids(test_df)
    touches_test = (
        df["node_A"].isin(test_nodes) | df["node_B"].isin(test_nodes)
    ).to_numpy()
    train_mask = train_candidate & ~touches_test
    val_mask = val_candidate & ~touches_test
    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()

    def stats(frame):
        return {"n_samples": len(frame), "n_nodes": len(node_ids(frame))}

    temporal_unassigned = ~(train_candidate | val_candidate | test_mask)
    summary = {
        "train": stats(train_df),
        "val": stats(val_df),
        "test": stats(test_df),
        "input_n_samples": len(df),
        "dropped": {
            "n_samples": int((~(train_mask | val_mask | test_mask)).sum()),
            "temporal_gap": int(temporal_unassigned.sum()),
            "train_test_node_overlap": int((train_candidate & touches_test).sum()),
            "val_test_node_overlap": int((val_candidate & touches_test).sum()),
        },
    }
    return train_df, val_df, test_df, summary
