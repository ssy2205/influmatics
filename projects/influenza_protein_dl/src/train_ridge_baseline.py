"""Train a ridge baseline for antigenic coordinate prediction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from influenza_protein_dl.baseline import (
    DEFAULT_TARGET_COLUMNS,
    FeatureSpec,
    evaluate_predictions,
    fit_ridge,
    infer_feature_columns,
    prediction_frame,
    target_matrix,
)
from influenza_protein_dl.splits import deterministic_random_split, split_counts, temporal_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Labeled dataset TSV")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--split-strategy",
        choices=["existing", "temporal", "random"],
        default="existing",
    )
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--valid-years", default="2022", help="Comma-separated validation years")
    parser.add_argument("--test-start-year", type=int, default=2023)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = pd.read_csv(args.dataset, sep="\t")
    frame = apply_split(frame, args)

    train_frame = frame.loc[frame["split"] == "train"].copy()
    if train_frame.empty:
        raise ValueError("Training split is empty")

    feature_columns = infer_feature_columns(frame)
    feature_spec = FeatureSpec(feature_columns=feature_columns)
    model = fit_ridge(train_frame, feature_spec=feature_spec, alpha=args.alpha)

    predictions = model.predict(frame)
    prediction_rows = prediction_frame(frame, predictions, DEFAULT_TARGET_COLUMNS)

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows.to_csv(output_dir / "predictions.tsv", sep="\t", index=False)

    metrics = {
        "model": "ridge_baseline",
        "alpha": args.alpha,
        "features": list(feature_columns),
        "split_counts": split_counts(frame),
        "by_split": {},
    }
    for split_name in ("train", "valid", "test"):
        split_frame = frame.loc[frame["split"] == split_name]
        if split_frame.empty:
            continue
        split_pred = predictions[split_frame.index.to_numpy()]
        split_true = target_matrix(split_frame, DEFAULT_TARGET_COLUMNS)
        metrics["by_split"][split_name] = evaluate_predictions(split_true, split_pred)

    (output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True, allow_nan=False)
    )
    print(f"Wrote predictions and metrics to {output_dir}")
    return 0


def apply_split(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Apply the requested split strategy."""

    if args.split_strategy == "existing":
        if "split" not in frame.columns:
            raise ValueError("Dataset has no split column; choose --split-strategy temporal or random")
        return frame
    if args.split_strategy == "temporal":
        valid_years = [
            int(year.strip())
            for year in args.valid_years.split(",")
            if year.strip()
        ]
        return temporal_split(
            frame,
            train_end_year=args.train_end_year,
            valid_years=valid_years,
            test_start_year=args.test_start_year,
        )
    return deterministic_random_split(frame)


def _json_safe(value):
    """Replace non-finite floats with None for standards-compliant JSON."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
