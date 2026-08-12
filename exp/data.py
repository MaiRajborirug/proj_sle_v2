"""Loading and feature-set construction for the SLE screening bakeoff.

The seven screening criteria are read from `criteria_d9.json` via `core.load_criteria()` so
this module never becomes a second source of truth for which criteria exist or in what
order. The CSV's column names happen to match the criteria `key` values exactly, spaces
included (`SCL or DL`, `Oral Ulcer`, `Joint involvement`).
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import core  # noqa: E402  (path shim above must run first)

CSV_FILE = _HERE / "SLE_NotSLE.csv"
LABEL_COLUMN = "Diagnosis"

# Age bands mirror app.py's AGE_BANDS so the feature set cannot encode resolution the app is
# unable to collect. Upper bounds are exclusive; the final band is open-ended.
AGE_BAND_EDGES = (20, 30, 40, 50, 60)
AGE_BAND_LABELS = ("<20", "20-29", "30-39", "40-49", "50-59", "60+")

D9_ONLY = "d9"
D9_DEMO = "d9+demo"
FEATURE_SETS = (D9_ONLY, D9_DEMO)


@dataclass(frozen=True)
class Dataset:
    """A feature matrix and its labels, plus the metadata the harness needs downstream.

    Attributes:
        name: Feature-set name, one of `FEATURE_SETS`.
        x: Feature matrix, shape `(n_samples, n_features)`, float64.
        y: Binary labels, shape `(n_samples,)`, 1 = SLE.
        feature_names: Column names of `x`, in order.
        criterion_index: Positions in `feature_names` of the seven screening criteria.
    """

    name: str
    x: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    criterion_index: tuple[int, ...]

    @property
    def prevalence(self) -> float:
        """Fraction of rows labelled SLE."""
        return float(self.y.mean())


def criterion_keys() -> tuple[str, ...]:
    """Return the seven criterion keys in the order `criteria_d9.json` declares them."""
    return tuple(c["key"] for c in core.load_criteria())


def load_raw() -> pd.DataFrame:
    """Read the case-control CSV.

    Returns:
        The full 402-row frame with every original column, unmodified.

    Raises:
        FileNotFoundError: If `SLE_NotSLE.csv` is missing from this directory.
    """
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"expected the cohort CSV at {CSV_FILE}")
    return pd.read_csv(CSV_FILE)


def age_band(age: pd.Series) -> pd.Series:
    """Bin ages into the six bands app.py offers.

    Args:
        age: Ages in years.

    Returns:
        Integer band index, 0 = `<20` through 5 = `60+`.
    """
    return pd.Series(np.digitize(age.to_numpy(), AGE_BAND_EDGES), index=age.index)


def build_dataset(name: str) -> Dataset:
    """Assemble one feature set from the cohort CSV.

    Args:
        name: `"d9"` for the seven criteria alone, `"d9+demo"` to append `Sex` and the
            banded age. Both are the only inputs a booth kiosk can actually collect.

    Returns:
        The assembled `Dataset`.

    Raises:
        ValueError: If `name` is not one of `FEATURE_SETS`, or the CSV lacks a criterion
            column.
    """
    if name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {name!r}; expected one of {FEATURE_SETS}")

    frame = load_raw()
    keys = criterion_keys()
    missing = [k for k in keys if k not in frame.columns]
    if missing:
        raise ValueError(f"cohort CSV is missing criterion columns: {missing}")

    features = frame.loc[:, list(keys)].astype(float)
    if name == D9_DEMO:
        features["Sex"] = frame["Sex"].astype(float)
        features["AgeBand"] = age_band(frame["Age"]).astype(float)

    return Dataset(
        name=name,
        x=features.to_numpy(dtype=float),
        y=frame[LABEL_COLUMN].to_numpy(dtype=int),
        feature_names=tuple(features.columns),
        criterion_index=tuple(range(len(keys))),
    )


def enumerate_patterns(n_criteria: int = 7) -> np.ndarray:
    """Enumerate every possible tick-box state of the screening form.

    Args:
        n_criteria: Number of binary criteria.

    Returns:
        Array of shape `(2 ** n_criteria, n_criteria)` holding each combination once, in
        ascending binary order so row 0 is "nothing ticked".
    """
    return np.array(list(itertools.product([0.0, 1.0], repeat=n_criteria)))


def bayes_ceiling(dataset: Dataset) -> float:
    """Best accuracy any deterministic classifier could reach on this feature set.

    Rows sharing an identical feature vector but disagreeing on the label are
    irreducible; the ceiling counts the majority label within each distinct vector.

    Args:
        dataset: The dataset to measure.

    Returns:
        Accuracy in [0, 1].
    """
    frame = pd.DataFrame(dataset.x, columns=list(dataset.feature_names))
    frame["_y"] = dataset.y
    grouped = frame.groupby(list(dataset.feature_names))["_y"]
    correct = grouped.agg(lambda s: max(int(s.sum()), len(s) - int(s.sum()))).sum()
    return float(correct) / len(dataset.y)
