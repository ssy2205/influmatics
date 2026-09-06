"""Pairwise mean absolute differences over shared non-NaN features."""

from numbers import Integral

import numpy as np
import pandas as pd


def pairwise_nan_mae(matrix, min_shared_features=3, *, block_size=256):
    """Return row_i_idx, row_j_idx, distance, shared_count for i < j.

    Only NaN denotes missingness; infinities retain NumPy arithmetic semantics.
    Computation uses float64. Worst-case runtime is O(N**2 * M), with
    O(block_size * M) arithmetic workspace in addition to input validity and
    O(N**2) output storage. Rows with too few valid features are pruned first.
    """
    for name, value in (("min_shared_features", min_shared_features),
                        ("block_size", block_size)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be >= 1")

    x = np.asarray(matrix)
    if x.ndim != 2 or x.dtype.kind not in "fiu":
        raise ValueError("matrix must be a 2D real numeric array")
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    rows = np.flatnonzero(valid.sum(axis=1) >= min_shared_features)
    n = len(rows)
    # Preserve original indices after pruning.
    x, valid = x[rows], valid[rows]
    capacity = n * (n - 1) // 2
    row_i = np.empty(capacity, dtype=np.int64)
    row_j = np.empty(capacity, dtype=np.int64)
    distances = np.empty(capacity, dtype=np.float64)
    counts = np.empty(capacity, dtype=np.int64)
    scratch = np.empty((min(block_size, n), x.shape[1]), dtype=np.float64)
    used = 0

    for i in range(n - 1):
        for start in range(i + 1, n, block_size):
            stop = min(start + block_size, n)
            shared = valid[start:stop] & valid[i]
            shared_count = shared.sum(axis=1)
            keep = np.flatnonzero(shared_count >= min_shared_features)
            if not keep.size:
                continue
            diff = scratch[:stop - start]
            diff.fill(0.0)
            with np.errstate(invalid="ignore", over="ignore"):
                np.subtract(x[start:stop], x[i], out=diff, where=shared)
                np.abs(diff, out=diff)
                totals = diff.sum(axis=1)
            end = used + keep.size
            row_i[used:end] = rows[i]
            row_j[used:end] = rows[start + keep]
            counts[used:end] = shared_count[keep]
            distances[used:end] = totals[keep] / shared_count[keep]
            used = end

    return pd.DataFrame({
        "row_i_idx": row_i[:used],
        "row_j_idx": row_j[:used],
        "distance": distances[:used],
        "shared_count": counts[:used],
    })
