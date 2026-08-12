# `exp/` — model bakeoff for the seven-criterion screen

Self-contained experiment. Nothing here is imported by the app; `app.py`, `core.py` and
`storage.py` are untouched by this directory. The findings live in **[REPORT.md](REPORT.md)**.

## Question

The app currently scores its seven observable criteria with the published EULAR/ACR 2019
weights (`core.compute_score`). Would a simple learned model — logistic regression, random
forest, gradient boosting — do better as a cheap screen before a patient commits to the
full lab panel?

Judged on **F1 and missed cases first, calibration second**, with monotone models preferred
because that is the shape the EULAR/ACR instrument already has.

## Run it

```bash
uv venv exp/.venv --python 3.13
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python exp/.venv/bin/python -r exp/requirements.txt
exp/.venv/bin/python exp/run_experiment.py
```

Takes a few minutes. Deterministic — every seed is fixed, so a rerun reproduces the tables
byte for byte. Outputs land in `exp/results/` (CSV) and `exp/figures/` (PNG); both are
regenerated from scratch on each run.

## Layout

| file | what it does |
|---|---|
| `data.py` | Loads the cohort, builds the `d9` and `d9+demo` feature sets, computes the Bayes ceiling |
| `models.py` | The 16-model zoo: unconstrained, monotone-constrained, and the rule baselines |
| `evaluate.py` | Nested repeated stratified 5-fold CV with in-fold threshold selection |
| `diagnostics.py` | Monotonicity audit, empty-form probe, effect directions, prevalence shift |
| `plots.py` | The nine figures |
| `run_experiment.py` | Orchestrator — run this |

## Design notes

**Thresholds and hyperparameters are both chosen inside the training fold.** F1 and
sensitivity depend on where the probability is cut; choosing that cut on the rows the metric
is reported from inflates every headline number, badly so on 402 rows. Each outer training
fold runs its own inner 5-fold to pick both, then applies them unchanged to held-out data.

**The outer loop is 5-fold repeated 10 times** — 5 splits as asked for, repeated because a
single 5-fold on this cohort is noisy enough to reorder the leaderboard by chance.

**The baseline is the real thing.** `EularRuleScorer` calls `core.compute_score` rather than
reimplementing the weights, so what the bakeoff has to beat is exactly what the app ships.

**Three anchors are asserted on every run** (`run_experiment.check_anchors`): cohort
prevalence 0.4975, Bayes-optimal accuracy 0.9303 on the seven fields, and shipped-rule ROC
AUC 0.6474. They were computed by hand from the raw CSV before any of this code existed. A
model exceeding the Bayes ceiling means the folds are leaking, and the run aborts.

## Data

`SLE_NotSLE.csv` — 402 rows, 200 SLE and 202 not, 24 features of which the app collects
seven. Carried over from v1. Read `REPORT.md` before drawing conclusions from it: the
control group is 202 other sick hospital patients, not healthy walk-ins, and that fact
drives the entire result.
