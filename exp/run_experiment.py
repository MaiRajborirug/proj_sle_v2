"""Run the whole bakeoff: `exp/.venv/bin/python exp/run_experiment.py`.

Writes CSVs to `exp/results/` and PNGs to `exp/figures/`, then prints the leaderboard and
the safety verdict. Deterministic — every seed is fixed, so reruns reproduce byte-identical
tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
for path in (str(_HERE), str(_HERE.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

import core  # noqa: E402
import data  # noqa: E402
import diagnostics  # noqa: E402
import evaluate  # noqa: E402
import models  # noqa: E402
import plots  # noqa: E402

RESULTS_DIR = _HERE / "results"

# Anchors computed independently from the raw CSV before any of this code existed. If a
# refactor breaks the data path these are what catch it.
EXPECTED_PREVALENCE = 0.4975
EXPECTED_BAYES_CEILING = 0.9303
EXPECTED_RULE_AUC = 0.6474
ANCHOR_TOLERANCE = 5e-4


def check_anchors(dataset: data.Dataset, summary: pd.DataFrame) -> None:
    """Verify the pipeline reproduces the hand-computed properties of the cohort.

    Args:
        dataset: The `d9` feature set.
        summary: The aggregated `d9` leaderboard.

    Raises:
        AssertionError: If any anchor drifts beyond `ANCHOR_TOLERANCE`, or a model beats
            the Bayes ceiling — which would mean the test folds are leaking into training.
    """
    assert abs(dataset.prevalence - EXPECTED_PREVALENCE) < ANCHOR_TOLERANCE, dataset.prevalence
    ceiling = data.bayes_ceiling(dataset)
    assert abs(ceiling - EXPECTED_BAYES_CEILING) < ANCHOR_TOLERANCE, ceiling

    rule = summary[summary["model"] == "EULAR/ACR rule (shipped)"]["roc_auc_mean"].iloc[0]
    assert abs(rule - EXPECTED_RULE_AUC) < 5e-3, rule

    over = summary[summary["accuracy_mean"] > ceiling + 0.02]
    assert over.empty, f"models beat the Bayes ceiling, suspect leakage:\n{over['model'].tolist()}"


def operating_table(summary: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Sensitivity and specificity at the sensitivity-floor cut-off, for named models.

    Args:
        summary: The aggregated leaderboard.
        names: Models to include, in display order.

    Returns:
        Frame indexed by model name with `sensitivity` and `specificity` columns.
    """
    rows = summary.set_index("model").loc[names]
    return pd.DataFrame(
        {"sensitivity": rows["sens_at_floor_mean"], "specificity": rows["spec_at_floor_mean"]}
    )


def run_feature_set(name: str) -> dict:
    """Cross-validate every model on one feature set and audit the results.

    Args:
        name: One of `data.FEATURE_SETS`.

    Returns:
        Keys `dataset`, `folds`, `summary`, `predictions`, `audit`, `zoo`.
    """
    dataset = data.build_dataset(name)
    zoo = models.build_zoo(dataset.x.shape[1], dataset.criterion_index)
    print(f"\n=== feature set {name!r}: {len(zoo)} models, {dataset.x.shape[0]} rows ===")

    folds, predictions, audits = [], [], []
    for spec in zoo:
        fold_metrics, pooled = evaluate.cross_validate_model(spec, dataset)
        folds.append(fold_metrics)
        predictions.append(pooled)
        audits.append(diagnostics.monotonicity_audit(spec, dataset))
        print(
            f"  {spec.name:<26} F1={fold_metrics['f1'].mean():.3f}"
            f"  sens={fold_metrics['sensitivity'].mean():.3f}"
            f"  AUC={fold_metrics['roc_auc'].mean():.3f}"
            f"  Brier={fold_metrics['brier'].mean():.3f}"
            f"  viol={audits[-1]['violations']}"
        )

    all_folds = pd.concat(folds, ignore_index=True)
    return {
        "dataset": dataset,
        "folds": all_folds,
        "summary": evaluate.summarise(all_folds),
        "predictions": pd.concat(predictions, ignore_index=True),
        "audit": pd.DataFrame(audits),
        "zoo": zoo,
    }


def main() -> None:
    """Run both feature sets, write every artefact, and print the verdict."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {name: run_feature_set(name) for name in data.FEATURE_SETS}

    summary = pd.concat([r["summary"] for r in results.values()], ignore_index=True)
    folds = pd.concat([r["folds"] for r in results.values()], ignore_index=True)
    audit = pd.concat([r["audit"] for r in results.values()], ignore_index=True)
    separation = pd.concat(
        [diagnostics.separation_report(r["dataset"]) for r in results.values()], ignore_index=True
    )
    directions = pd.concat(
        [diagnostics.effect_directions(r["dataset"]) for r in results.values()], ignore_index=True
    )

    primary = results[data.D9_ONLY]
    check_anchors(primary["dataset"], primary["summary"])

    curve_models = plots.curve_selection(primary["summary"], primary["predictions"])
    operating = operating_table(primary["summary"], curve_models)
    ppv = pd.concat(
        [
            diagnostics.ppv_by_prevalence(row["sensitivity"], row["specificity"]).assign(model=name)
            for name, row in operating.iterrows()
        ],
        ignore_index=True,
    )

    best = {
        family: primary["summary"][primary["summary"]["family"] == family]
        .sort_values("f1_mean", ascending=False)["model"]
        .iloc[0]
        for family in (models.UNCONSTRAINED, models.MONOTONE)
    }
    duels = pd.concat(
        [
            evaluate.head_to_head(
                primary["folds"], best[models.UNCONSTRAINED], best[models.MONOTONE]
            ),
            evaluate.head_to_head(
                primary["folds"], best[models.MONOTONE], "EULAR/ACR rule (shipped)"
            ),
        ],
        ignore_index=True,
    )
    duels.to_csv(RESULTS_DIR / "head_to_head.csv", index=False)

    folds.to_csv(RESULTS_DIR / "cv_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    audit.to_csv(RESULTS_DIR / "monotonicity.csv", index=False)
    ppv.to_csv(RESULTS_DIR / "ppv_by_prevalence.csv", index=False)
    separation.to_csv(RESULTS_DIR / "separation.csv", index=False)
    directions.to_csv(RESULTS_DIR / "effect_directions.csv", index=False)

    points_spec = next(s for s in primary["zoo"] if s.name == "Non-negative logistic")
    written = plots.plot_all(
        dataset=primary["dataset"],
        fold_metrics=primary["folds"],
        summary=primary["summary"],
        predictions=primary["predictions"],
        audit=primary["audit"],
        operating=operating,
        points_spec=points_spec,
        criteria=core.load_criteria(),
    )

    print(f"\nwrote {len(written)} figures to {plots.FIGURE_DIR}")
    print(f"wrote {len(list(RESULTS_DIR.glob('*.csv')))} tables to {RESULTS_DIR}")
    print("\nbest unconstrained vs best monotone, paired over the 50 folds:")
    print(duels.head(7).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_monotone = (
        primary["summary"][primary["summary"]["family"] == models.MONOTONE]
        .sort_values("f1_mean", ascending=False)
        .iloc[0]
    )
    shipped = primary["summary"][primary["summary"]["model"] == "EULAR/ACR rule (shipped)"].iloc[0]
    print(
        f"\nbest monotone model: {best_monotone['model']} "
        f"F1={best_monotone['f1_mean']:.3f} AUC={best_monotone['roc_auc_mean']:.3f}\n"
        f"shipped rule:        F1={shipped['f1_mean']:.3f} AUC={shipped['roc_auc_mean']:.3f}\n"
        f"Bayes ceiling on these 7 fields: {data.bayes_ceiling(primary['dataset']):.4f} accuracy"
    )


if __name__ == "__main__":
    main()
