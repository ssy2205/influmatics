"""Build supervised modeling datasets from manifest, metadata, and labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schema import (
    ValidationResult,
    read_tsv,
    validate_coordinate_labels,
    validate_manifest,
    validate_metadata,
)


@dataclass(frozen=True)
class DatasetBuildResult:
    """Dataset table plus validation warnings."""

    frame: pd.DataFrame
    warnings: tuple[str, ...]


def build_coordinate_dataset(
    manifest: pd.DataFrame,
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    mutation_features: pd.DataFrame | None = None,
) -> DatasetBuildResult:
    """Join input tables into one antigenic-coordinate modeling dataset."""

    validation_errors: list[str] = []
    warnings: list[str] = []
    for result in (
        validate_manifest(manifest),
        validate_metadata(metadata),
        validate_coordinate_labels(labels),
    ):
        validation_errors.extend(result.errors)
        warnings.extend(result.warnings)
    if validation_errors:
        raise ValueError("\n".join(validation_errors))

    manifest_ids = set(manifest["seq_id"])
    metadata_ids = set(metadata["seq_id"])
    label_ids = set(labels["seq_id"])

    missing_metadata = sorted((manifest_ids & label_ids) - metadata_ids)
    missing_labels = sorted((manifest_ids & metadata_ids) - label_ids)
    if missing_metadata:
        warnings.append(
            f"{len(missing_metadata)} labeled manifest rows are missing metadata"
        )
    if missing_labels:
        warnings.append(
            f"{len(missing_labels)} manifest+metadata rows are missing labels and will be dropped"
        )

    dataset = manifest.merge(metadata, on="seq_id", how="inner", validate="one_to_one")
    dataset = dataset.merge(labels, on="seq_id", how="inner", validate="one_to_one")
    if mutation_features is not None:
        if "seq_id" not in mutation_features.columns:
            raise ValueError("mutation_features is missing required column: seq_id")
        dataset = dataset.merge(
            mutation_features,
            on="seq_id",
            how="left",
            validate="one_to_one",
        )
        feature_columns = [
            column
            for column in mutation_features.columns
            if column != "seq_id"
        ]
        dataset[feature_columns] = dataset[feature_columns].fillna(0)
    dataset = _coerce_numeric_columns(dataset)
    return DatasetBuildResult(frame=dataset, warnings=tuple(warnings))


def build_coordinate_dataset_from_paths(
    manifest_path: str | Path,
    metadata_path: str | Path,
    labels_path: str | Path,
    mutation_features_path: str | Path | None = None,
) -> DatasetBuildResult:
    """Load TSV inputs and build a coordinate dataset."""

    mutation_features = read_tsv(mutation_features_path) if mutation_features_path else None
    return build_coordinate_dataset(
        manifest=read_tsv(manifest_path),
        metadata=read_tsv(metadata_path),
        labels=read_tsv(labels_path),
        mutation_features=mutation_features,
    )


def write_dataset(frame: pd.DataFrame, path: str | Path) -> None:
    """Write a dataset TSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, sep="\t", index=False)


def validation_summary(results: list[ValidationResult]) -> tuple[bool, list[str], list[str]]:
    """Collapse several validation results for CLIs."""

    errors: list[str] = []
    warnings: list[str] = []
    for result in results:
        errors.extend(result.errors)
        warnings.extend(result.warnings)
    return not errors, errors, warnings


def _coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    numeric_columns = [
        "length",
        "valid_aa_count",
        "invalid_aa_count",
        "ambiguous_aa_count",
        "stop_count",
        "gap_count",
        "year",
        "antigenic_x",
        "antigenic_y",
    ]
    numeric_columns.extend([column for column in output.columns if column.startswith("aa_")])
    numeric_columns.extend(
        [
            column
            for column in output.columns
            if column.startswith("antigenic_site_")
        ]
    )
    for column in numeric_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="raise")
    return output
