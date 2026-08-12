"""Derive the points table and band cut-offs that `criteria_d9.json` ships.

Run it to regenerate and verify:

    exp/.venv/bin/python exp/fit_points_model.py

The fit is an ordinary penalised logistic regression with one twist: the ridge penalty
pulls the weights toward the **published EULAR/ACR 2019 values** rather than toward zero.

    minimise   logloss(Xw + b)  +  λ‖w − κ·w_eular‖²      subject to  w ≥ ε

That single change is what makes the result usable. Fitted freely on this cohort, oral
ulcer and joint involvement both take *negative* weight — true of a case-control sample
whose 202 controls are rheumatology patients saturated with joint complaints, and false of
anyone walking past a booth. Shrinking toward the literature keeps those two criteria
positive without discarding what the cohort genuinely does show, and states the assumption
as one number, λ, instead of hiding it in a hand-adjustment.

`κ` is free so the prior finds its own scale; `ε` keeps every weight strictly positive,
which is what makes the resulting score monotonic and lets `core.compute_band` be monotonic
in turn. λ = 2 was chosen by cross-validating the whole procedure — fit, round, apply a
fixed cut-off — which is what `cross_validated` below reproduces. See REPORT.md §6 and the
repository README for the reasoning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

_HERE = Path(__file__).parent
for path in (str(_HERE), str(_HERE.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

import core  # noqa: E402
import data  # noqa: E402

LAMBDA = 2.0
EPSILON = 0.05
POINT_CAP = 12
N_SPLITS, N_REPEATS, RANDOM_STATE = 5, 10, 0

# Cut-offs are placed where the likelihood ratio changes, not at even intervals. See
# `band_report` for the measured values.
BANDS = ((core.GREEN, 0, 2), (core.YELLOW, 3, 7), (core.ORANGE, 8, 12), (core.RED, 13, 36))


def fit_weights(x: np.ndarray, y: np.ndarray, prior: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """Fit log-odds weights shrunk toward `prior`, bounded strictly positive.

    Args:
        x: Feature matrix of the seven binary criteria.
        y: Binary labels, 1 = SLE.
        prior: Published EULAR/ACR weight per criterion, same order as `x`'s columns.
        lam: Shrinkage strength. 0 trusts the cohort alone; large values reproduce the
            prior's shape.

    Returns:
        One non-negative weight per criterion, in log-odds units.
    """
    y = y.astype(float)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, b, kappa = theta[:-2], theta[-2], theta[-1]
        z = x @ w + b
        residual = expit(z) - y
        gap = w - kappa * prior
        gradient = np.empty_like(theta)
        gradient[:-2] = x.T @ residual + 2.0 * lam * gap
        gradient[-2] = residual.sum()
        gradient[-1] = -2.0 * lam * float(gap @ prior)
        loss = float(np.sum(np.logaddexp(0.0, z) - y * z)) + lam * float(gap @ gap)
        return loss, gradient

    bounds = [(EPSILON, None)] * x.shape[1] + [(None, None), (0.0, None)]
    start = np.r_[prior * 0.3, 0.0, 0.3]
    result = minimize(
        objective, start, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000}
    )
    return result.x[:-2]


def to_points(weights: np.ndarray, cap: int = POINT_CAP) -> np.ndarray:
    """Rescale log-odds weights to small positive integers.

    The smallest weight becomes 1 and the largest is capped, so the table fits on a paper
    form. Rounding is nearly free — it preserves the ranking almost exactly, which
    `main` verifies by comparing AUC before and after.

    Args:
        weights: Positive log-odds weights.
        cap: Largest allowed point value.

    Returns:
        Integer points, every one at least 1.
    """
    points = np.round(weights / weights.min()).astype(int)
    if points.max() > cap:
        points = np.round(points * cap / points.max()).astype(int)
    return np.maximum(1, points)


def band_report(scores: np.ndarray, y: np.ndarray) -> list[dict]:
    """Measure each band's likelihood ratio, which is what the cut-offs were chosen on.

    A likelihood ratio near 1 means the band leaves the visitor's risk unchanged — the
    reason YELLOW exists as a separate, non-referring band rather than being folded into
    GREEN or ORANGE.

    Args:
        scores: Points total per patient.
        y: Binary labels, 1 = SLE.

    Returns:
        One dict per band with counts and its likelihood ratio. Bands containing no
        controls use the rule-of-three bound, so the ratio is a floor rather than infinity.
    """
    rows = []
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    for name, low, high in BANDS:
        mask = (scores >= low) & (scores <= high)
        p_given_sle = float(y[mask].sum()) / n_pos
        p_given_not = float((1 - y[mask]).sum()) / n_neg
        rows.append(
            {
                "band": name,
                "range": f"{low}-{high}",
                "n": int(mask.sum()),
                "sle": int(y[mask].sum()),
                "controls": int((1 - y[mask]).sum()),
                "lr": p_given_sle / max(p_given_not, 3.0 / n_neg),
                "lr_is_floor": p_given_not == 0.0,
            }
        )
    return rows


def cross_validated(x: np.ndarray, y: np.ndarray, prior: np.ndarray, threshold: int) -> dict:
    """Cross-validate the whole procedure: refit, re-round, then apply a fixed cut-off.

    Args:
        x: Feature matrix.
        y: Binary labels.
        prior: EULAR/ACR weights.
        threshold: Referral cut-off applied to the held-out fold.

    Returns:
        Mean `auc`, `sensitivity` and `specificity` over the outer folds.
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    auc, sens, spec = [], [], []
    for train, test in splitter.split(x, y):
        points = to_points(fit_weights(x[train], y[train], prior))
        scores, labels = x[test] @ points, y[test]
        flagged = scores >= threshold
        tp = float(np.sum(flagged & (labels == 1)))
        fp = float(np.sum(flagged & (labels == 0)))
        fn = float(np.sum(~flagged & (labels == 1)))
        tn = float(np.sum(~flagged & (labels == 0)))
        auc.append(roc_auc_score(labels, scores))
        sens.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))
    return {
        "auc": float(np.mean(auc)),
        "sensitivity": float(np.mean(sens)),
        "specificity": float(np.mean(spec)),
    }


def main() -> None:
    """Refit, print the table and bands, and check them against `criteria_d9.json`."""
    criteria = core.load_criteria()
    dataset = data.build_dataset(data.D9_ONLY)
    prior = np.array([c["eular_score"] for c in criteria], dtype=float)

    weights = fit_weights(dataset.x, dataset.y, prior)
    points = to_points(weights)
    scores = dataset.x @ points

    print(f"fitted with lambda={LAMBDA}, epsilon={EPSILON}\n")
    print(f"{'criterion':<20}{'points':>8}{'log-odds':>10}{'EULAR':>8}")
    for c, w, p in zip(criteria, weights, points, strict=True):
        print(f"{c['key']:<20}{p:>8}{w:>10.2f}{int(c['eular_score']):>8}")
    print(f"{'total':<20}{points.sum():>8}")

    continuous = dataset.x @ weights
    print(
        f"\nAUC continuous {roc_auc_score(dataset.y, continuous):.4f}"
        f"  ->  rounded to points {roc_auc_score(dataset.y, scores):.4f}"
    )

    print(f"\n{'band':<8}{'range':>8}{'n':>6}{'SLE':>6}{'ctrl':>6}{'LR':>9}")
    for row in band_report(scores, dataset.y):
        floor = ">=" if row["lr_is_floor"] else "  "
        print(
            f"{row['band']:<8}{row['range']:>8}{row['n']:>6}{row['sle']:>6}"
            f"{row['controls']:>6}{floor}{row['lr']:>7.2f}"
        )

    measured = cross_validated(dataset.x, dataset.y, prior, core.BAND_ORANGE)
    print(
        f"\ncross-validated at the ORANGE cut-off ({core.BAND_ORANGE}+, i.e. referral): "
        f"AUC={measured['auc']:.3f} "
        f"sens={measured['sensitivity']:.3f} spec={measured['specificity']:.3f}"
    )

    shipped = np.array([c["score"] for c in criteria])
    if np.array_equal(points, shipped):
        print("\ncriteria_d9.json matches this fit.")
    else:
        print(
            f"\nMISMATCH — criteria_d9.json has {list(shipped)}, this fit gives "
            f"{list(points)}. Update the file or the fit, not both silently."
        )


if __name__ == "__main__":
    main()
