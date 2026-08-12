"""Checks that decide whether a leaderboard winner is safe to put in front of the public.

Cross-validated F1 answers "does this model fit the cohort". It cannot answer "does this
model behave sanely on a booth visitor", because the cohort's control group is 202 other
sick hospital patients rather than healthy walk-ins. These five checks answer the second
question:

1. `monotonicity_audit` — does ticking one more symptom ever *lower* the risk?
2. `all_zero_probe` — what does the model say when nothing at all is ticked?
3. `effect_directions` — which criteria did the model learn a negative association for?
4. `ppv_by_prevalence` — what does the output mean once the base rate is 0.1%, not 50%?
5. `separation_report` — which criteria are separable enough to blow up an unpenalised fit?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from data import Dataset, enumerate_patterns
from models import ModelSpec

# A booth screening the general public sits at the far left of this range; the cohort sits
# at the far right by construction. Spanning both makes the gap explicit.
PREVALENCE_GRID = (0.001, 0.005, 0.01, 0.05, 0.5)

# Predictions differing by less than this are treated as equal, so floating-point noise in
# a tree ensemble is not reported as a monotonicity violation.
MONOTONE_TOLERANCE = 1e-9


def fit_full(spec: ModelSpec, dataset: Dataset):
    """Fit a model on the entire cohort, for the structural checks below.

    Cross-validation measures generalisation; these checks interrogate the shape of the
    decision surface, which is best read off the model that saw all the data.

    Args:
        spec: The model to fit.
        dataset: The feature set to fit on.

    Returns:
        The fitted estimator.
    """
    return clone(spec.estimator).fit(dataset.x, dataset.y)


def _pattern_matrix(dataset: Dataset) -> np.ndarray:
    """All 2^7 criterion patterns, padded with median demographics where present."""
    patterns = enumerate_patterns(len(dataset.criterion_index))
    if dataset.x.shape[1] == patterns.shape[1]:
        return patterns
    padding = np.median(dataset.x[:, patterns.shape[1] :], axis=0)
    return np.column_stack([patterns, np.tile(padding, (len(patterns), 1))])


def monotonicity_audit(spec: ModelSpec, dataset: Dataset) -> dict:
    """Count edges of the 2^7 input lattice where adding a symptom lowers predicted risk.

    Every pair of patterns differing in exactly one criterion is one edge. A model fit for
    a screening instrument should never fall along one: reporting an extra symptom must not
    make the tool more reassuring.

    Args:
        spec: The model to audit.
        dataset: The feature set it was built for.

    Returns:
        Keys `model`, `feature_set`, `violations`, `edges`, `worst_drop`, `p_all_zero`,
        `p_all_one`, `violation_example`.
    """
    model = fit_full(spec, dataset)
    patterns = _pattern_matrix(dataset)
    proba = model.predict_proba(patterns)[:, 1]
    n_criteria = len(dataset.criterion_index)

    index = {tuple(row[:n_criteria]): i for i, row in enumerate(patterns)}
    violations = 0
    edges = 0
    worst_drop = 0.0
    example = ""
    for row, i in index.items():
        for bit in range(n_criteria):
            if row[bit] == 1.0:
                continue
            higher = list(row)
            higher[bit] = 1.0
            j = index[tuple(higher)]
            edges += 1
            drop = proba[i] - proba[j]
            if drop > MONOTONE_TOLERANCE:
                violations += 1
                if drop > worst_drop:
                    worst_drop = drop
                    added = dataset.feature_names[bit]
                    example = f"adding {added!r}: {proba[i]:.3f} -> {proba[j]:.3f}"

    return {
        "model": spec.name,
        "family": spec.family,
        "feature_set": dataset.name,
        "violations": violations,
        "edges": edges,
        "violation_rate": violations / edges if edges else 0.0,
        "worst_drop": worst_drop,
        "p_all_zero": float(proba[0]),
        "p_all_one": float(proba[-1]),
        "violation_example": example,
    }


def all_zero_probe(spec: ModelSpec, dataset: Dataset) -> float:
    """Predicted probability of SLE when no criterion is ticked.

    v1's shipped model returned 0.921 here, which is why it was withdrawn (`core.py:3-10`).
    On this cohort the empty pattern really is 8/10 positive — those are patients diagnosed
    on serology alone — so a model that reproduces the number is fitting the data correctly
    and is still unusable in a booth.

    Args:
        spec: The model to probe.
        dataset: The feature set it was built for.

    Returns:
        The probability in [0, 1].
    """
    return float(fit_full(spec, dataset).predict_proba(_pattern_matrix(dataset)[:1])[0, 1])


def effect_directions(dataset: Dataset, seed: int = 0) -> pd.DataFrame:
    """Report each feature's learned direction against its clinical direction.

    Uses a plain L2 logistic fit for signed coefficients and a random forest for
    permutation importance, so both a linear and a non-linear view are covered.

    Args:
        dataset: The feature set to analyse.
        seed: Seed for the permutation test.

    Returns:
        One row per feature: `logistic_coef`, `odds_ratio`, `permutation_importance`,
        `p_given_sle`, `p_given_not_sle`, and `inverted` marking features whose learned
        direction contradicts the EULAR/ACR premise that a criterion raises risk.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    linear = LogisticRegression(l1_ratio=0.0, C=1.0, max_iter=1000).fit(dataset.x, dataset.y)
    forest = RandomForestClassifier(n_estimators=300, random_state=seed).fit(dataset.x, dataset.y)
    importance = permutation_importance(
        forest, dataset.x, dataset.y, n_repeats=20, random_state=seed, scoring="roc_auc"
    )

    positive = dataset.x[dataset.y == 1]
    negative = dataset.x[dataset.y == 0]
    coefficients = linear.coef_.ravel()
    return pd.DataFrame(
        {
            "feature": dataset.feature_names,
            "feature_set": dataset.name,
            "logistic_coef": coefficients,
            "odds_ratio": np.exp(coefficients),
            "permutation_importance": importance.importances_mean,
            "p_given_sle": positive.mean(axis=0),
            "p_given_not_sle": negative.mean(axis=0),
            "inverted": [
                bool(c < 0 and i in dataset.criterion_index) for i, c in enumerate(coefficients)
            ],
        }
    )


def shift_prevalence(proba: np.ndarray, source: float, target: float) -> np.ndarray:
    """Re-anchor probabilities from one base rate to another.

    Only the intercept moves: the log-odds are shifted by
    `logit(target) - logit(source)`. This is the standard correction for a case-control
    sample, and it assumes the class-conditional feature distributions themselves transfer,
    which is exactly the assumption this cohort violates for joint involvement.

    Args:
        proba: Probabilities calibrated at the `source` base rate.
        source: Base rate the probabilities were fitted at.
        target: Base rate to re-express them at.

    Returns:
        The shifted probabilities.

    Raises:
        ValueError: If either base rate is outside the open interval (0, 1).
    """
    if not 0.0 < source < 1.0 or not 0.0 < target < 1.0:
        raise ValueError(f"base rates must lie in (0, 1); got source={source}, target={target}")
    shifted = logit(np.clip(proba, 1e-9, 1 - 1e-9)) + logit(target) - logit(source)
    return expit(shifted)


def ppv_by_prevalence(
    sensitivity: float, specificity: float, grid: tuple[float, ...] = PREVALENCE_GRID
) -> pd.DataFrame:
    """Translate a fixed operating point into what it means at each base rate.

    Sensitivity and specificity are properties of the test; PPV is not. A tool holding
    90% sensitivity and 80% specificity looks excellent on a 50/50 cohort and refers 200
    well people per true case at booth prevalence. This table is where that shows up.

    Args:
        sensitivity: True-positive rate at the chosen cut-off.
        specificity: True-negative rate at the chosen cut-off.
        grid: Base rates to evaluate.

    Returns:
        One row per base rate with `ppv`, `npv`, `referrals_per_1000` screened, and
        `screened_per_case_found`.
    """
    rows = []
    for prevalence in grid:
        tp = sensitivity * prevalence
        fp = (1.0 - specificity) * (1.0 - prevalence)
        fn = (1.0 - sensitivity) * prevalence
        tn = specificity * (1.0 - prevalence)
        flagged = tp + fp
        rows.append(
            {
                "prevalence": prevalence,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ppv": tp / flagged if flagged else 0.0,
                "npv": tn / (tn + fn) if tn + fn else 0.0,
                "referrals_per_1000": 1000.0 * flagged,
                "screened_per_case_found": 1.0 / tp if tp else float("inf"),
            }
        )
    return pd.DataFrame(rows)


def separation_report(dataset: Dataset) -> pd.DataFrame:
    """Flag criteria that perfectly or near-perfectly separate the two classes.

    A criterion absent from every control produces an infinite maximum-likelihood
    coefficient. Regularisation keeps the fit finite, but the resulting weight is an
    artefact of how the controls were sampled rather than a measured effect size.

    Args:
        dataset: The feature set to analyse.

    Returns:
        One row per binary feature with its class-conditional rates, empirical odds ratio
        (Haldane-corrected), and a `quasi_separated` flag.
    """
    rows = []
    for i, name in enumerate(dataset.feature_names):
        column = dataset.x[:, i]
        if not np.all(np.isin(column, (0.0, 1.0))):
            continue
        a = float(np.sum((column == 1) & (dataset.y == 1)))
        b = float(np.sum((column == 1) & (dataset.y == 0)))
        c = float(np.sum((column == 0) & (dataset.y == 1)))
        d = float(np.sum((column == 0) & (dataset.y == 0)))
        rows.append(
            {
                "feature": name,
                "feature_set": dataset.name,
                "n_positive_cases": int(a),
                "n_positive_controls": int(b),
                "p_given_sle": a / (a + c) if a + c else 0.0,
                "p_given_not_sle": b / (b + d) if b + d else 0.0,
                # Haldane-Anscombe: add 0.5 to every cell so a zero does not make this
                # undefined. The point of the row is that the true value is unbounded.
                "odds_ratio": ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)),
                "quasi_separated": bool(a == 0 or b == 0),
            }
        )
    return pd.DataFrame(rows)
