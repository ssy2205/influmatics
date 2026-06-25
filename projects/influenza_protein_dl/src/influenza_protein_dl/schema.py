"""Schema validation for influenza protein modeling tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MANIFEST_REQUIRED_COLUMNS = {
    "seq_id",
    "length",
    "valid_aa_count",
    "invalid_aa_count",
    "ambiguous_aa_count",
    "stop_count",
    "gap_count",
}

METADATA_REQUIRED_COLUMNS = {
    "seq_id",
    "subtype",
    "gene",
    "protein_region",
    "collection_date",
    "year",
    "host",
    "country",
    "source",
}

COORDINATE_LABEL_REQUIRED_COLUMNS = {
    "seq_id",
    "antigenic_x",
    "antigenic_y",
}


@dataclass(frozen=True)
class ValidationResult:
    """Validation result for tabular inputs."""

    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a TSV file as strings by default."""

    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def validate_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> list[str]:
    """Return missing-column errors for a table."""

    missing = sorted(required_columns.difference(frame.columns))
    if not missing:
        return []
    return [f"{table_name} is missing required columns: {','.join(missing)}"]


def validate_unique_ids(frame: pd.DataFrame, table_name: str) -> list[str]:
    """Return duplicate seq_id errors."""

    if "seq_id" not in frame.columns:
        return []
    duplicates = sorted(frame.loc[frame["seq_id"].duplicated(), "seq_id"].unique())
    if not duplicates:
        return []
    preview = ",".join(duplicates[:10])
    suffix = "" if len(duplicates) <= 10 else f" and {len(duplicates) - 10} more"
    return [f"{table_name} has duplicate seq_id values: {preview}{suffix}"]


def validate_manifest(frame: pd.DataFrame) -> ValidationResult:
    """Validate a protein manifest table."""

    errors = validate_columns(frame, MANIFEST_REQUIRED_COLUMNS, "manifest")
    errors.extend(validate_unique_ids(frame, "manifest"))
    errors.extend(_validate_nonnegative_int(frame, "length", "manifest"))
    errors.extend(_validate_nonnegative_int(frame, "invalid_aa_count", "manifest"))
    warnings: list[str] = []
    if "invalid_aa_count" in frame.columns:
        invalid_rows = _numeric_series(frame["invalid_aa_count"]) > 0
        if invalid_rows.any():
            warnings.append(
                f"manifest contains {int(invalid_rows.sum())} rows with invalid amino-acid characters"
            )
    return ValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def validate_metadata(frame: pd.DataFrame) -> ValidationResult:
    """Validate sequence metadata."""

    errors = validate_columns(frame, METADATA_REQUIRED_COLUMNS, "metadata")
    errors.extend(validate_unique_ids(frame, "metadata"))
    errors.extend(_validate_year(frame, "year", "metadata"))
    return ValidationResult(ok=not errors, errors=tuple(errors))


def validate_coordinate_labels(frame: pd.DataFrame) -> ValidationResult:
    """Validate antigenic coordinate labels."""

    errors = validate_columns(frame, COORDINATE_LABEL_REQUIRED_COLUMNS, "labels")
    errors.extend(validate_unique_ids(frame, "labels"))
    errors.extend(_validate_float(frame, "antigenic_x", "labels"))
    errors.extend(_validate_float(frame, "antigenic_y", "labels"))
    return ValidationResult(ok=not errors, errors=tuple(errors))


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _validate_nonnegative_int(
    frame: pd.DataFrame,
    column: str,
    table_name: str,
) -> list[str]:
    if column not in frame.columns:
        return []
    numeric = _numeric_series(frame[column])
    invalid = numeric.isna() | (numeric < 0) | (numeric % 1 != 0)
    if invalid.any():
        return [f"{table_name}.{column} must contain nonnegative integers"]
    return []


def _validate_year(frame: pd.DataFrame, column: str, table_name: str) -> list[str]:
    if column not in frame.columns:
        return []
    numeric = _numeric_series(frame[column])
    invalid = numeric.isna() | (numeric < 1900) | (numeric > 2100) | (numeric % 1 != 0)
    if invalid.any():
        return [f"{table_name}.{column} must contain plausible integer years"]
    return []


def _validate_float(frame: pd.DataFrame, column: str, table_name: str) -> list[str]:
    if column not in frame.columns:
        return []
    numeric = _numeric_series(frame[column])
    invalid = numeric.isna()
    if invalid.any():
        return [f"{table_name}.{column} must contain numeric values"]
    return []
