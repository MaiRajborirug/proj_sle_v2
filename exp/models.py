"""The model zoo: unconstrained learners, monotone-constrained learners, rule baselines.

Three families are compared.

`unconstrained`
    Off-the-shelf sklearn classifiers, free to learn any relationship in the cohort —
    including that joint involvement predicts *against* SLE, which is true of this
    case-control sample and false of a screening booth.

`monotone`
    Learners constrained so that ticking one more criterion can never lower the predicted
    risk. This is the shape the EULAR/ACR instrument has by construction, and the property
    `tests/test_core.py` already requires of the shipped scorer.

`baseline`
    The incumbent rule from `core.compute_score`, plus a trivial reference rule. These are
    what a new model has to beat to be worth adopting.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import core  # noqa: E402  (path shim above must run first)

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - exercised only on hosts without libomp
    HAS_XGBOOST = False

RANDOM_STATE = 0

UNCONSTRAINED = "unconstrained"
MONOTONE = "monotone"
BASELINE = "baseline"

# Highest score the seven-criterion restricted EULAR/ACR instrument can reach: the maximum
# weight within each of the four domains it covers (6 + 6 + 4 + 2).
EULAR_MAX_SCORE = 18


class NonNegativeLogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic regression with per-feature lower bounds of zero on selected coefficients.

    This is the monotone headline candidate: constraining the criterion coefficients to be
    non-negative makes the fitted model a data-driven points table directly comparable to
    the published EULAR/ACR weights, while keeping the smooth probability output that the
    tree models only approximate.

    sklearn's `LogisticRegression` has no coefficient bounds, so the penalised log-loss is
    minimised directly with L-BFGS-B.

    Args:
        c: Inverse L2 regularisation strength, matching sklearn's convention.
        constrained_index: Positions of the features to bound at `>= 0`. Features outside
            this set stay unbounded, which is what demographics want — sex and age carry no
            prior direction. Empty means an ordinary L2 logistic fit.
        max_iter: Iteration cap handed to L-BFGS-B.
    """

    def __init__(
        self,
        c: float = 1.0,
        constrained_index: tuple[int, ...] = (),
        max_iter: int = 500,
    ) -> None:
        self.c = c
        self.constrained_index = constrained_index
        self.max_iter = max_iter

    def fit(self, x: np.ndarray, y: np.ndarray) -> NonNegativeLogisticRegression:
        """Fit the bounded model.

        Args:
            x: Feature matrix, shape `(n_samples, n_features)`.
            y: Binary labels, shape `(n_samples,)`.

        Returns:
            self
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = x.shape[1]
        self.classes_ = np.array([0, 1])

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            weights, intercept = theta[:-1], theta[-1]
            logits = x @ weights + intercept
            # log(1 + exp(z)) evaluated stably, so large logits do not overflow.
            loss = float(np.sum(np.logaddexp(0.0, logits) - y * logits))
            penalty = float(np.sum(weights**2)) / (2.0 * self.c)
            residual = expit(logits) - y
            grad = np.empty_like(theta)
            grad[:-1] = x.T @ residual + weights / self.c
            grad[-1] = float(np.sum(residual))
            return loss + penalty, grad

        bounds: list[tuple[float | None, float | None]] = [(None, None)] * (n_features + 1)
        for i in self.constrained_index:
            bounds[i] = (0.0, None)

        result = minimize(
            objective,
            x0=np.zeros(n_features + 1),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter},
        )
        self.coef_ = result.x[:-1].reshape(1, -1)
        self.intercept_ = result.x[-1:]
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Return the log-odds for each row of `x`."""
        return np.asarray(x, dtype=float) @ self.coef_.ravel() + self.intercept_[0]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape `(n_samples, 2)`."""
        p = expit(self.decision_function(x))
        return np.column_stack([1.0 - p, p])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return hard labels at the 0.5 cut-off."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class EularRuleScorer(BaseEstimator, ClassifierMixin):
    """The shipped EULAR/ACR restricted score, wrapped as an estimator.

    Scoring is delegated to `core.compute_score` rather than reimplemented, so the baseline
    in this bakeoff is exactly the instrument the app uses today.

    Args:
        criterion_index: Positions of the seven criteria within the feature matrix.
        platt: When True, `fit` learns a one-dimensional logistic calibration of the raw
            score so the model's probabilities are comparable to the learners on Brier and
            ECE. When False the model is fixed: `fit` learns nothing and the reported
            probability is `score / 18`, which ranks correctly but is not calibrated.
    """

    def __init__(self, criterion_index: tuple[int, ...] = (), platt: bool = False) -> None:
        self.criterion_index = criterion_index
        self.platt = platt

    def _raw_score(self, x: np.ndarray) -> np.ndarray:
        criteria = core.load_criteria()
        keys = [c["key"] for c in criteria]
        columns = np.asarray(x, dtype=float)[:, list(self.criterion_index)]
        scores = np.empty(len(columns))
        for row_i, row in enumerate(columns):
            values = {key: bool(row[i]) for i, key in enumerate(keys)}
            scores[row_i], _ = core.compute_score(values, criteria)
        return scores

    def fit(self, x: np.ndarray, y: np.ndarray) -> EularRuleScorer:
        """Fit the Platt calibration if enabled; otherwise record the classes only.

        Args:
            x: Feature matrix.
            y: Binary labels.

        Returns:
            self
        """
        self.classes_ = np.array([0, 1])
        if self.platt:
            self._platt = LogisticRegression().fit(self._raw_score(x).reshape(-1, 1), y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape `(n_samples, 2)`."""
        scores = self._raw_score(x)
        p = (
            self._platt.predict_proba(scores.reshape(-1, 1))[:, 1]
            if self.platt
            else scores / EULAR_MAX_SCORE
        )
        return np.column_stack([1.0 - p, p])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return hard labels at the 0.5 cut-off."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class AnyNonJointRule(BaseEstimator, ClassifierMixin):
    """Flag anyone reporting at least one criterion other than joint involvement.

    Included because on this cohort it is a genuinely strong rule, which is itself the
    finding: joint involvement is the one criterion the control group is saturated with.

    Args:
        criterion_index: Positions of the seven criteria within the feature matrix.
        joint_position: Index of joint involvement within those seven.
    """

    def __init__(self, criterion_index: tuple[int, ...] = (), joint_position: int = 5) -> None:
        self.criterion_index = criterion_index
        self.joint_position = joint_position

    def fit(self, x: np.ndarray, y: np.ndarray) -> AnyNonJointRule:
        """Record the classes. The rule has no parameters to learn."""
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape `(n_samples, 2)`."""
        keep = [j for j in range(len(self.criterion_index)) if j != self.joint_position]
        columns = np.asarray(x, dtype=float)[:, list(self.criterion_index)][:, keep]
        # Graded rather than binary so ROC/PR curves have more than one operating point.
        p = np.clip(columns.sum(axis=1) / columns.shape[1], 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return hard labels: positive if any non-joint criterion is present."""
        return (self.predict_proba(x)[:, 1] > 0.0).astype(int)


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the bakeoff.

    Attributes:
        name: Display name, unique within the zoo.
        family: One of `UNCONSTRAINED`, `MONOTONE`, `BASELINE`.
        estimator: An unfitted estimator, cloned before every fit.
        param_grid: Grid searched inside each outer training fold. Empty means no tuning.
        note: Short annotation carried into the results tables.
    """

    name: str
    family: str
    estimator: object
    param_grid: dict = field(default_factory=dict)
    note: str = ""


def _monotone_constraints(n_features: int, criterion_index: tuple[int, ...]) -> list[int]:
    """Build a per-feature constraint vector: 1 for criteria, 0 for demographics."""
    constraints = [0] * n_features
    for i in criterion_index:
        constraints[i] = 1
    return constraints


def build_zoo(n_features: int, criterion_index: tuple[int, ...]) -> list[ModelSpec]:
    """Assemble every model to be compared for a given feature set.

    Args:
        n_features: Number of columns in the feature matrix.
        criterion_index: Positions of the seven screening criteria within those columns.

    Returns:
        The model specs, in a stable order.
    """
    constraints = _monotone_constraints(n_features, criterion_index)
    specs: list[ModelSpec] = [
        ModelSpec(
            name="Logistic (L2)",
            family=UNCONSTRAINED,
            estimator=LogisticRegression(l1_ratio=0.0, max_iter=1000),
            param_grid={"C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        ),
        ModelSpec(
            name="Logistic (L1)",
            family=UNCONSTRAINED,
            estimator=LogisticRegression(l1_ratio=1.0, solver="saga", max_iter=5000),
            param_grid={"C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        ),
        ModelSpec(
            name="Bernoulli NB",
            family=UNCONSTRAINED,
            estimator=BernoulliNB(),
            param_grid={"alpha": [0.1, 0.5, 1.0, 2.0]},
        ),
        ModelSpec(
            name="Decision tree",
            family=UNCONSTRAINED,
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            param_grid={"max_depth": [2, 3, 4, 5], "min_samples_leaf": [5, 10, 20]},
        ),
        ModelSpec(
            name="Random forest",
            family=UNCONSTRAINED,
            estimator=RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE),
            param_grid={"max_depth": [3, 5, None], "min_samples_leaf": [1, 5, 10]},
        ),
        ModelSpec(
            name="Extra trees",
            family=UNCONSTRAINED,
            estimator=ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_STATE),
            param_grid={"max_depth": [3, 5, None], "min_samples_leaf": [1, 5, 10]},
        ),
        ModelSpec(
            name="HistGradientBoosting",
            family=UNCONSTRAINED,
            estimator=HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            param_grid={"max_depth": [2, 3, None], "learning_rate": [0.05, 0.1]},
        ),
        ModelSpec(
            name="SVC (RBF)",
            family=UNCONSTRAINED,
            estimator=Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE), cv=3),
                    ),
                ]
            ),
            param_grid={"clf__estimator__C": [0.3, 1.0, 3.0]},
            note="matches the v1 model family",
        ),
        ModelSpec(
            name="k-NN",
            family=UNCONSTRAINED,
            estimator=Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
            param_grid={"clf__n_neighbors": [5, 11, 21, 31]},
        ),
        ModelSpec(
            name="Non-negative logistic",
            family=MONOTONE,
            estimator=NonNegativeLogisticRegression(constrained_index=criterion_index),
            param_grid={"c": [0.1, 0.3, 1.0, 3.0, 10.0]},
            note="data-driven points table",
        ),
        ModelSpec(
            name="Monotone HistGB",
            family=MONOTONE,
            estimator=HistGradientBoostingClassifier(
                monotonic_cst=constraints, random_state=RANDOM_STATE
            ),
            param_grid={"max_depth": [2, 3, None], "learning_rate": [0.05, 0.1]},
        ),
        ModelSpec(
            name="EULAR/ACR rule (shipped)",
            family=BASELINE,
            estimator=EularRuleScorer(criterion_index=criterion_index, platt=False),
            note="core.compute_score, uncalibrated",
        ),
        ModelSpec(
            name="EULAR/ACR rule (Platt)",
            family=BASELINE,
            estimator=EularRuleScorer(criterion_index=criterion_index, platt=True),
            note="same ranking, calibrated on the training fold",
        ),
        ModelSpec(
            name="Any non-joint criterion",
            family=BASELINE,
            estimator=AnyNonJointRule(criterion_index=criterion_index),
            note="trivial reference rule",
        ),
    ]

    if HAS_XGBOOST:
        common = {
            "n_estimators": 300,
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "verbosity": 0,
        }
        grid = {"max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]}
        specs.insert(
            7,
            ModelSpec(
                name="XGBoost",
                family=UNCONSTRAINED,
                estimator=XGBClassifier(**common),
                param_grid=grid,
            ),
        )
        specs.append(
            ModelSpec(
                name="Monotone XGBoost",
                family=MONOTONE,
                estimator=XGBClassifier(monotone_constraints=tuple(constraints), **common),
                param_grid=grid,
            )
        )

    return specs
