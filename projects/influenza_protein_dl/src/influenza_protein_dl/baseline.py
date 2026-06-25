"""Small baseline models for antigenic coordinate prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLUMNS = (
    "length",
    "valid_aa_count",
    "invalid_aa_count",
    "ambiguous_aa_count",
    "stop_count",
    "gap_count",
)
DEFAULT_TARGET_COLUMNS = ("antigenic_x", "antigenic_y")


@dataclass(frozen=True)
class FeatureSpec:
    """Feature and target columns used by a model."""

    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS


@dataclass(frozen=True)
class Standardizer:
    """Feature standardization parameters."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


@dataclass(frozen=True)
class RidgeModel:
    """Closed-form multi-output ridge regression model."""

    weights: np.ndarray
    standardizer: Standardizer
    feature_spec: FeatureSpec
    alpha: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = design_matrix(frame, self.feature_spec.feature_columns)
        x_scaled = self.standardizer.transform(x)
        x_with_bias = add_bias_column(x_scaled)
        return x_with_bias @ self.weights


def infer_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Infer numeric baseline feature columns from a dataset."""

    aa_columns = tuple(sorted(column for column in frame.columns if column.startswith("aa_")))
    mutation_columns = tuple(
        sorted(
            column
            for column in frame.columns
            if column.startswith("antigenic_site_")
        )
    )
    base_columns = tuple(column for column in DEFAULT_FEATURE_COLUMNS if column in frame.columns)
    return base_columns + aa_columns + mutation_columns


def design_matrix(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    """Return numeric feature matrix."""

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing feature columns: {','.join(missing)}")
    return frame.loc[:, list(feature_columns)].astype(float).to_numpy()


def target_matrix(frame: pd.DataFrame, target_columns: tuple[str, ...]) -> np.ndarray:
    """Return numeric target matrix."""

    missing = [column for column in target_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing target columns: {','.join(missing)}")
    return frame.loc[:, list(target_columns)].astype(float).to_numpy()


def fit_standardizer(values: np.ndarray) -> Standardizer:
    """Fit feature standardization parameters."""

    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return Standardizer(mean=mean, scale=scale)


def fit_ridge(
    train_frame: pd.DataFrame,
    feature_spec: FeatureSpec,
    alpha: float = 1.0,
) -> RidgeModel:
    """Fit multi-output ridge regression."""

    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    x = design_matrix(train_frame, feature_spec.feature_columns)
    y = target_matrix(train_frame, feature_spec.target_columns)
    standardizer = fit_standardizer(x)
    x_scaled = standardizer.transform(x)
    x_with_bias = add_bias_column(x_scaled)
    penalty = np.eye(x_with_bias.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x_with_bias.T @ x_with_bias + penalty, x_with_bias.T @ y)
    return RidgeModel(
        weights=weights,
        standardizer=standardizer,
        feature_spec=feature_spec,
        alpha=alpha,
    )


def add_bias_column(values: np.ndarray) -> np.ndarray:
    """Prepend a bias column."""

    return np.concatenate([np.ones((values.shape[0], 1)), values], axis=1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
) -> dict[str, float]:
    """Calculate regression metrics."""

    metrics: dict[str, float] = {}
    residual = y_pred - y_true
    metrics["mae_mean"] = float(np.mean(np.abs(residual)))
    metrics["rmse_mean"] = float(np.sqrt(np.mean(residual**2)))
    for idx, target in enumerate(target_columns):
        target_residual = residual[:, idx]
        metrics[f"{target}_mae"] = float(np.mean(np.abs(target_residual)))
        metrics[f"{target}_rmse"] = float(np.sqrt(np.mean(target_residual**2)))
        metrics[f"{target}_pearson"] = pearsonr(y_true[:, idx], y_pred[:, idx])
    return metrics


def pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation, or nan when undefined."""

    if len(a) < 2:
        return float("nan")
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    if denominator == 0:
        return float("nan")
    return float(np.sum(a_centered * b_centered) / denominator)


def prediction_frame(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
    model_name: str = "ridge_baseline",
) -> pd.DataFrame:
    """Create a prediction TSV-ready frame."""

    output_columns = ["seq_id"]
    if "split" in frame.columns:
        output_columns.append("split")
    output = frame.loc[:, output_columns].copy()
    for idx, target in enumerate(target_columns):
        output[f"target_{target}"] = frame[target].astype(float).to_numpy()
        output[f"pred_{target}"] = predictions[:, idx]
    output["prediction_model"] = model_name
    return output
