"""Nested cross-validation harness.

Two design choices carry most of the weight here.

**Thresholds are chosen inside the training fold, never on the test fold.** F1 and
sensitivity both depend on where the probability is cut. Picking that cut-off on the same
rows the metric is reported on inflates every headline number, and on a 402-row cohort the
inflation is large. Each outer training fold therefore runs its own inner CV, selects the
cut-off there, and applies it unchanged to the held-out rows.

**Hyperparameters are tuned inside the training fold too**, for the same reason.

The outer loop is a repeated stratified 5-fold: 5 splits as requested, repeated 10 times
because a single 5-fold on 402 rows is noisy enough to reorder the leaderboard by chance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from data import Dataset
from models import ModelSpec

N_SPLITS = 5
N_REPEATS = 10
INNER_SPLITS = 5
RANDOM_STATE = 0

# Screening priority: a missed case costs far more than a false alarm, so alongside the
# F1-optimal cut-off every model also reports the cut-off meeting this sensitivity floor.
TARGET_SENSITIVITY = 0.90

ECE_BINS = 10


@dataclass(frozen=True)
class OperatingPoint:
    """Counts and rates at one probability cut-off."""

    threshold: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float


def _confusion_rates(y: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float, float]:
    """Return `(sensitivity, specificity, precision, f1)` for hard predictions."""
    tp = float(np.sum((predicted == 1) & (y == 1)))
    fp = float(np.sum((predicted == 1) & (y == 0)))
    fn = float(np.sum((predicted == 0) & (y == 1)))
    tn = float(np.sum((predicted == 0) & (y == 0)))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    denom = precision + sensitivity
    f1 = 2.0 * precision * sensitivity / denom if denom else 0.0
    return sensitivity, specificity, precision, f1


def operating_point(y: np.ndarray, proba: np.ndarray, threshold: float) -> OperatingPoint:
    """Evaluate one cut-off.

    Args:
        y: Binary labels.
        proba: Predicted probability of the positive class.
        threshold: Rows with `proba >= threshold` are called positive.

    Returns:
        The rates at that cut-off.
    """
    sensitivity, specificity, precision, f1 = _confusion_rates(y, (proba >= threshold).astype(int))
    return OperatingPoint(threshold, sensitivity, specificity, precision, f1)


def _candidate_thresholds(proba: np.ndarray) -> np.ndarray:
    """Every cut-off that can produce a distinct split of `proba`, plus the trivial ones."""
    return np.unique(np.concatenate([[0.0], proba, [1.0 + 1e-9]]))


def best_f1_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    """Cut-off maximising F1.

    Args:
        y: Binary labels.
        proba: Predicted probability of the positive class.

    Returns:
        The maximising threshold; ties break towards the higher cut-off, which is the more
        conservative referral rate.
    """
    candidates = _candidate_thresholds(proba)
    scores = np.array([operating_point(y, proba, t).f1 for t in candidates])
    return float(candidates[np.flatnonzero(scores == scores.max())[-1]])


def sensitivity_floor_threshold(
    y: np.ndarray, proba: np.ndarray, floor: float = TARGET_SENSITIVITY
) -> float:
    """Highest cut-off still achieving at least `floor` sensitivity.

    Args:
        y: Binary labels.
        proba: Predicted probability of the positive class.
        floor: Minimum acceptable sensitivity.

    Returns:
        The threshold. Falls back to 0.0 when no cut-off reaches the floor, i.e. flag
        everybody rather than silently miss the target.
    """
    candidates = _candidate_thresholds(proba)
    feasible = [t for t in candidates if operating_point(y, proba, t).sensitivity >= floor]
    return float(max(feasible)) if feasible else 0.0


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, bins: int = ECE_BINS) -> float:
    """Weighted mean gap between predicted probability and observed frequency.

    Args:
        y: Binary labels.
        proba: Predicted probability of the positive class.
        bins: Number of equal-width bins over [0, 1].

    Returns:
        The error in [0, 1]; 0 is perfect calibration.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(proba, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        total += mask.sum() * abs(proba[mask].mean() - y[mask].mean())
    return float(total / len(y))


def _tune(spec: ModelSpec, x: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Grid-search hyperparameters on the training fold only.

    Args:
        spec: The model being evaluated.
        x: Training features.
        y: Training labels.
        seed: Seed for the inner split.

    Returns:
        The winning parameter dict, empty when the model has no grid.
    """
    if not spec.param_grid:
        return {}
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    search = GridSearchCV(
        clone(spec.estimator),
        spec.param_grid,
        scoring="average_precision",
        cv=inner,
        n_jobs=-1,
    )
    return search.fit(x, y).best_params_


def _fit(spec: ModelSpec, params: dict, x: np.ndarray, y: np.ndarray):
    """Clone the spec's estimator with `params` applied and fit it."""
    return clone(spec.estimator).set_params(**params).fit(x, y)


def _inner_thresholds(
    spec: ModelSpec, params: dict, x: np.ndarray, y: np.ndarray, seed: int
) -> tuple[float, float]:
    """Choose both cut-offs using only the training fold.

    Hyperparameters are held fixed at `params` here. They were already selected on this
    same training fold, so re-searching them per inner split would multiply the fit count
    by the grid size for no gain in honesty — the test fold stays unseen either way.

    Args:
        spec: The model being evaluated.
        params: Hyperparameters chosen by `_tune`.
        x: Training features.
        y: Training labels.
        seed: Seed for the inner split.

    Returns:
        `(f1_threshold, sensitivity_floor_threshold)`.
    """
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    held_out = np.zeros(len(y))
    for train_i, test_i in inner.split(x, y):
        model = _fit(spec, params, x[train_i], y[train_i])
        held_out[test_i] = model.predict_proba(x[test_i])[:, 1]
    return best_f1_threshold(y, held_out), sensitivity_floor_threshold(y, held_out)


def cross_validate_model(spec: ModelSpec, dataset: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full nested CV for one model.

    Args:
        spec: The model to evaluate.
        dataset: The feature set to evaluate it on.

    Returns:
        `(fold_metrics, pooled_predictions)`. `fold_metrics` has one row per outer fold.
        `pooled_predictions` holds the out-of-fold probability for every row of the first
        repeat, which is what the ROC, PR and reliability curves are drawn from.
    """
    outer = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    rows: list[dict] = []
    pooled = np.full(len(dataset.y), np.nan)

    for fold, (train_i, test_i) in enumerate(outer.split(dataset.x, dataset.y)):
        repeat = fold // N_SPLITS
        seed = RANDOM_STATE + repeat
        x_train, y_train = dataset.x[train_i], dataset.y[train_i]
        x_test, y_test = dataset.x[test_i], dataset.y[test_i]

        params = _tune(spec, x_train, y_train, seed)
        f1_threshold, floor_threshold = _inner_thresholds(spec, params, x_train, y_train, seed)
        proba = _fit(spec, params, x_train, y_train).predict_proba(x_test)[:, 1]
        if repeat == 0:
            pooled[test_i] = proba

        at_f1 = operating_point(y_test, proba, f1_threshold)
        at_floor = operating_point(y_test, proba, floor_threshold)
        rows.append(
            {
                "model": spec.name,
                "family": spec.family,
                "feature_set": dataset.name,
                "fold": fold,
                "repeat": repeat,
                "roc_auc": roc_auc_score(y_test, proba),
                "pr_auc": average_precision_score(y_test, proba),
                "brier": brier_score_loss(y_test, proba),
                "ece": expected_calibration_error(y_test, proba),
                "log_loss": log_loss(y_test, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
                "f1": at_f1.f1,
                "sensitivity": at_f1.sensitivity,
                "specificity": at_f1.specificity,
                "precision": at_f1.precision,
                "threshold_f1": f1_threshold,
                "sens_at_floor": at_floor.sensitivity,
                "spec_at_floor": at_floor.specificity,
                "f1_at_floor": at_floor.f1,
                "threshold_floor": floor_threshold,
                "accuracy": float(np.mean((proba >= f1_threshold).astype(int) == y_test)),
            }
        )

    predictions = pd.DataFrame(
        {
            "model": spec.name,
            "family": spec.family,
            "feature_set": dataset.name,
            "y": dataset.y,
            "proba": pooled,
        }
    )
    return pd.DataFrame(rows), predictions


def head_to_head(fold_metrics: pd.DataFrame, challenger: str, reference: str) -> pd.DataFrame:
    """Compare two models fold by fold, which is what decides whether a gap is real.

    Comparing two leaderboard means says nothing about whether the gap survives the noise
    of a 402-row cohort. Both models saw exactly the same 50 splits, so the per-fold
    differences are paired and a signed-rank test applies.

    The p-values are optimistic and are reported as a rough guide only: repeated k-fold
    reuses the same rows across repeats, so the 50 differences are not independent. The
    effect sizes are the part to read.

    Args:
        fold_metrics: Per-fold metrics for a single feature set.
        challenger: Model whose advantage is being measured.
        reference: Model it is measured against.

    Returns:
        One row per metric with both means, the paired difference and a signed-rank
        p-value.

    Raises:
        ValueError: If either model is missing, or the two did not see the same folds.
    """
    from scipy.stats import wilcoxon

    left = fold_metrics[fold_metrics["model"] == challenger].sort_values("fold")
    right = fold_metrics[fold_metrics["model"] == reference].sort_values("fold")
    if left.empty or right.empty:
        raise ValueError(f"missing fold metrics for {challenger!r} or {reference!r}")
    if not left["fold"].to_numpy().tolist() == right["fold"].to_numpy().tolist():
        raise ValueError("the two models were not evaluated on the same folds")

    rows = []
    for metric in ("f1", "sensitivity", "specificity", "roc_auc", "pr_auc", "brier", "ece"):
        difference = left[metric].to_numpy() - right[metric].to_numpy()
        rows.append(
            {
                "metric": metric,
                "challenger": challenger,
                "reference": reference,
                "challenger_mean": left[metric].mean(),
                "reference_mean": right[metric].mean(),
                "difference": difference.mean(),
                "wilcoxon_p": wilcoxon(difference).pvalue if difference.any() else 1.0,
            }
        )
    return pd.DataFrame(rows)


def summarise(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold metrics into one ranked row per model.

    Args:
        fold_metrics: Concatenated output of `cross_validate_model`.

    Returns:
        Mean and standard deviation of every metric, sorted by mean F1 descending — the
        stated first priority.
    """
    metrics = [
        "f1",
        "sensitivity",
        "specificity",
        "precision",
        "roc_auc",
        "pr_auc",
        "brier",
        "ece",
        "log_loss",
        "accuracy",
        "sens_at_floor",
        "spec_at_floor",
        "threshold_f1",
    ]
    grouped = fold_metrics.groupby(["feature_set", "model", "family"], sort=False)[metrics]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values("f1_mean", ascending=False)
