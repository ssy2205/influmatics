import math

import pandas as pd

from influenza_protein_dl.baseline import (
    FeatureSpec,
    evaluate_predictions,
    fit_ridge,
    infer_feature_columns,
)
from influenza_protein_dl.dataset import build_coordinate_dataset
from influenza_protein_dl.mutation_features import build_mutation_feature_table
from influenza_protein_dl.schema import validate_coordinate_labels, validate_metadata
from influenza_protein_dl.splits import deterministic_random_split, temporal_split


def _manifest():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s2", "s3", "s4"],
            "description": ["", "", "", ""],
            "length": [329, 329, 329, 329],
            "valid_aa_count": [329, 329, 328, 329],
            "invalid_aa_count": [0, 0, 0, 0],
            "ambiguous_aa_count": [0, 0, 1, 0],
            "stop_count": [0, 0, 0, 0],
            "gap_count": [0, 0, 0, 0],
            "aa_A": [0.1, 0.2, 0.3, 0.4],
            "aa_C": [0.2, 0.2, 0.1, 0.1],
        }
    )


def _metadata():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s2", "s3", "s4"],
            "subtype": ["H3N2", "H3N2", "H3N2", "H3N2"],
            "gene": ["HA", "HA", "HA", "HA"],
            "protein_region": ["HA1", "HA1", "HA1", "HA1"],
            "collection_date": ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
            "year": ["2020", "2021", "2022", "2023"],
            "host": ["human", "human", "human", "human"],
            "country": ["KR", "KR", "KR", "KR"],
            "source": ["fixture", "fixture", "fixture", "fixture"],
        }
    )


def _labels():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s2", "s3", "s4"],
            "antigenic_x": ["0.0", "1.0", "2.0", "3.0"],
            "antigenic_y": ["0.0", "2.0", "4.0", "6.0"],
        }
    )


def test_metadata_validation_rejects_unparseable_year():
    metadata = _metadata()
    metadata.loc[0, "year"] = "twenty"

    result = validate_metadata(metadata)

    assert not result.ok
    assert "metadata.year" in result.errors[0]


def test_label_validation_rejects_missing_coordinate():
    labels = _labels().drop(columns=["antigenic_y"])

    result = validate_coordinate_labels(labels)

    assert not result.ok
    assert "antigenic_y" in result.errors[0]


def test_build_coordinate_dataset_coerces_numeric_columns():
    result = build_coordinate_dataset(_manifest(), _metadata(), _labels())

    assert len(result.frame) == 4
    assert result.frame["year"].dtype.kind in {"i", "u"}
    assert result.frame["antigenic_x"].dtype.kind == "f"


def test_build_coordinate_dataset_merges_optional_mutation_features():
    mutation_features = pd.DataFrame(
        {
            "seq_id": ["s2"],
            "aa_mutation_count": ["2"],
            "antigenic_site_mutation_count": ["1"],
        }
    )

    result = build_coordinate_dataset(
        _manifest(),
        _metadata(),
        _labels(),
        mutation_features=mutation_features,
    )

    by_id = result.frame.set_index("seq_id")
    assert by_id.loc["s2", "aa_mutation_count"] == 2
    assert by_id.loc["s1", "aa_mutation_count"] == 0
    assert by_id.loc["s2", "antigenic_site_mutation_count"] == 1


def test_build_mutation_feature_table_counts_antigenic_sites():
    mutations = pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2"],
            "position": ["10", "20", "30"],
            "mutation_type": ["substitution", "synonymous", "aa_deletion"],
            "coordinate_space": ["aa", "aa", "aa"],
        }
    )

    features = build_mutation_feature_table(
        mutations,
        antigenic_sites={"A": {10, 20}, "B": {30}},
    ).set_index("seq_id")

    assert features.loc["s1", "aa_mutation_count"] == 2
    assert features.loc["s1", "aa_substitution_count"] == 1
    assert features.loc["s1", "antigenic_site_A_mutation_count"] == 2
    assert features.loc["s2", "aa_deletion_count"] == 1
    assert features.loc["s2", "antigenic_site_B_mutation_count"] == 1


def test_temporal_split_assigns_expected_partitions():
    dataset = build_coordinate_dataset(_manifest(), _metadata(), _labels()).frame

    split = temporal_split(
        dataset,
        train_end_year=2021,
        valid_years=[2022],
        test_start_year=2023,
    )

    assert dict(zip(split["seq_id"], split["split"])) == {
        "s1": "train",
        "s2": "train",
        "s3": "valid",
        "s4": "test",
    }


def test_random_split_is_deterministic():
    dataset = pd.DataFrame({"seq_id": [f"s{i}" for i in range(200)]})

    first = deterministic_random_split(dataset)
    second = deterministic_random_split(dataset)

    assert first["split"].tolist() == second["split"].tolist()


def test_fit_ridge_predicts_training_rows():
    dataset = build_coordinate_dataset(_manifest(), _metadata(), _labels()).frame
    feature_columns = infer_feature_columns(dataset)
    model = fit_ridge(dataset, FeatureSpec(feature_columns=feature_columns), alpha=0.1)

    predictions = model.predict(dataset)
    metrics = evaluate_predictions(
        dataset[["antigenic_x", "antigenic_y"]].to_numpy(),
        predictions,
    )

    assert predictions.shape == (4, 2)
    assert math.isfinite(metrics["mae_mean"])
