"""Train/validation/test splitting helpers."""

from __future__ import annotations

import hashlib

import pandas as pd


def temporal_split(
    frame: pd.DataFrame,
    train_end_year: int,
    valid_years: list[int] | tuple[int, ...],
    test_start_year: int,
    year_column: str = "year",
) -> pd.DataFrame:
    """Add a split column using collection year boundaries."""

    if year_column not in frame.columns:
        raise ValueError(f"Missing year column: {year_column}")
    output = frame.copy()
    years = pd.to_numeric(output[year_column], errors="raise").astype(int)
    valid_set = set(valid_years)
    output["split"] = "unused"
    output.loc[years <= train_end_year, "split"] = "train"
    output.loc[years.isin(valid_set), "split"] = "valid"
    output.loc[years >= test_start_year, "split"] = "test"
    _require_nonempty_splits(output, required=("train", "valid", "test"))
    return output


def deterministic_random_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.7,
    valid_fraction: float = 0.15,
    id_column: str = "seq_id",
    salt: str = "influenza_protein_dl",
) -> pd.DataFrame:
    """Add a stable pseudo-random split column based on sequence IDs."""

    if id_column not in frame.columns:
        raise ValueError(f"Missing id column: {id_column}")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1")
    if train_fraction + valid_fraction >= 1:
        raise ValueError("train_fraction + valid_fraction must be < 1")

    output = frame.copy()
    split_values: list[str] = []
    for seq_id in output[id_column].astype(str):
        value = _stable_fraction(f"{salt}:{seq_id}")
        if value < train_fraction:
            split_values.append("train")
        elif value < train_fraction + valid_fraction:
            split_values.append("valid")
        else:
            split_values.append("test")
    output["split"] = split_values
    _require_nonempty_splits(output, required=("train", "valid", "test"))
    return output


def split_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return split counts."""

    if "split" not in frame.columns:
        return {}
    return {
        str(split): int(count)
        for split, count in frame["split"].value_counts().sort_index().items()
    }


def _stable_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    integer = int(digest[:16], 16)
    return integer / float(0xFFFFFFFFFFFFFFFF)


def _require_nonempty_splits(frame: pd.DataFrame, required: tuple[str, ...]) -> None:
    counts = split_counts(frame)
    missing = [name for name in required if counts.get(name, 0) == 0]
    if missing:
        raise ValueError(f"Split produced empty partitions: {','.join(missing)}")
