# Which model should screen the seven criteria?

Bakeoff of 16 models over `SLE_NotSLE.csv` (402 patients, 200 SLE / 202 not), restricted to the
seven observable criteria the app collects. Nested repeated stratified 5-fold cross-validation,
50 outer folds, thresholds and hyperparameters both chosen inside the training fold.

Reproduce with `exp/.venv/bin/python exp/run_experiment.py` — see [README.md](README.md).

---

## Answer

**Recommended: the non-negative logistic points model.** F1 **0.884**, ROC AUC **0.925**,
Brier **0.089**, ECE **0.071**, and **zero** monotonicity violations across all 128 possible
inputs.

All three monotone models are statistically tied (§3.1) — `Monotone XGBoost` tops the sort at
F1 0.890 but leads `Monotone HistGB` by 0.0026 at *p = 0.94*. The points model is picked on
structure, not score: see §3.2.

**Ceiling, for reference:** the best unconstrained model (`Extra trees`, or equally
`Random forest` / `SVC`) reaches F1 **0.910** and AUC **0.945**. Monotonicity costs about
**2 points of F1** and **5 points of sensitivity** — and *buys* better calibration
(ECE 0.059 vs 0.094).

**Both are far better than what the app ships.** The current EULAR/ACR restricted rule scores
F1 **0.639** and AUC **0.647** on this cohort — worse than the trivial rule "flag anyone with any
criterion other than joint involvement" (F1 0.834, AUC 0.871).

**But — and this decides the matter — none of these numbers transfer to a public booth.** The
control group is 202 other sick hospital patients, not healthy walk-ins. Section 4 sets out what
that breaks and what it does not.

---

## 1. The cohort, before any modelling

![Data profile](figures/01_data_profile.png)

| criterion | P(present \| SLE) | P(present \| not SLE) | empirical OR |
|---|---:|---:|---:|
| Proteinuria | 0.590 | 0.005 | 193 |
| SCL or DL | 0.240 | **0.000** | ∞ (quasi-separated) |
| ACL | 0.260 | 0.045 | 7.2 |
| Alopecia | 0.220 | 0.015 | 16.2 |
| Fever | 0.145 | 0.064 | 2.4 |
| Oral Ulcer | 0.110 | 0.124 | **0.88** |
| Joint involvement | 0.515 | 0.861 | **0.17** |

Two criteria run backwards. Joint involvement — worth **+6** in the published EULAR/ACR
instrument, tied for the heaviest weight — is *commoner in the controls than in the cases*, because
170 of the 402 patients present as joint-involvement-only and 154 of those are controls. The
control arm is a rheumatology comparison group, and this is what it looks like.

Two more facts bound everything downstream:

- **Bayes-optimal accuracy on these seven fields is 0.9303.** 252 of 402 patients fall into 9
  tick-box patterns that contain both classes. No model, however complex, can exceed this; a run
  that does is leaking and `run_experiment.check_anchors` aborts.
- **The empty pattern is 8/10 SLE.** Ten patients tick nothing at all and eight of them have lupus,
  diagnosed on serology the booth cannot see. This is the source of v1's notorious 0.921.

---

## 2. Leaderboard

Mean over 50 outer folds; `sd` is across folds. Sorted by F1, the stated first priority.

| model | family | F1 | sens | spec | ROC AUC | PR AUC | Brier | ECE | viol. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Extra trees | unconstrained | **0.910** ±.034 | 0.897 | 0.928 | 0.945 | 0.947 | 0.077 | 0.094 | 41 |
| Random forest | unconstrained | 0.910 ±.033 | 0.895 | 0.930 | 0.944 | 0.946 | 0.078 | 0.094 | 113 |
| SVC (RBF) | unconstrained | 0.910 ±.034 | 0.898 | 0.925 | 0.939 | 0.934 | 0.077 | 0.074 | 223 |
| XGBoost | unconstrained | 0.907 ±.035 | 0.885 | 0.935 | 0.944 | 0.943 | 0.075 | 0.064 | 104 |
| HistGradientBoosting | unconstrained | 0.897 ±.038 | 0.878 | 0.921 | 0.941 | 0.942 | 0.081 | 0.070 | 149 |
| Logistic (L2) | unconstrained | 0.894 ±.038 | 0.887 | 0.903 | 0.937 | 0.939 | 0.090 | 0.092 | 128 |
| Bernoulli NB | unconstrained | 0.891 ±.038 | 0.875 | 0.912 | 0.936 | 0.938 | 0.086 | 0.069 | 128 |
| **Monotone XGBoost** | **monotone** | **0.890** ±.038 | 0.846 | 0.947 | 0.926 | 0.930 | 0.085 | **0.059** | **0** |
| **Monotone HistGB** | **monotone** | 0.888 ±.039 | 0.843 | 0.947 | 0.925 | 0.929 | 0.086 | 0.062 | **0** |
| **Non-negative logistic** | **monotone** | 0.884 ±.037 | 0.843 | 0.938 | 0.925 | 0.928 | 0.089 | 0.071 | **0** |
| Decision tree | unconstrained | 0.883 ±.037 | 0.830 | 0.952 | 0.927 | 0.923 | 0.084 | 0.055 | 17 |
| Logistic (L1) | unconstrained | 0.881 ±.036 | 0.858 | 0.912 | 0.935 | 0.939 | 0.089 | 0.070 | 128 |
| *Any non-joint criterion* | *baseline* | 0.834 ±.035 | 0.880 | 0.772 | 0.871 | 0.841 | 0.290 | 0.347 | 0 |
| k-NN | unconstrained | 0.815 ±.101 | 0.890 | 0.661 | 0.908 | 0.915 | 0.142 | 0.205 | 34 |
| *EULAR/ACR rule (shipped)* | *baseline* | **0.639** ±.041 | 0.862 | **0.205** | **0.647** | 0.741 | 0.231 | 0.269 | 0 |
| *EULAR/ACR rule (Platt)* | *baseline* | 0.638 ±.041 | 0.852 | 0.224 | 0.647 | 0.741 | 0.219 | 0.290 | 0 |

![Leaderboard](figures/05_metric_comparison.png)

![ROC](figures/02_roc_curves.png)
![Precision-recall](figures/03_pr_curves.png)

**Adding sex and age changes nothing.** The `d9+demo` feature set moves the best F1 from 0.910 to
0.906 and the best AUC from 0.945 to 0.950 — inside fold noise. Age looks informative (SLE mean 38
vs control 50) but that gap is the age profile of a rheumatology control arm, not a property of
lupus. Keep the demographics optional, as the app already has them.

### Why the shipped rule scores so badly here

Its AUC of 0.647 is not a coding error. The rule gives joint involvement +6, which lands 160 of the
202 controls at exactly 6 points — the yellow band. At its ≥6 cut-off it flags 86% of cases and 80%
of controls: specificity **0.205**. On this cohort it is close to a coin toss, entirely because it
weights the one criterion that runs backwards here.

That is an indictment of the cohort as much as of the rule. The published weights were derived
against a mixed comparison group; this control arm is not that.

---

## 3. Calibration

![Calibration](figures/04_calibration.png)

Calibration was the second priority and it does not force a trade-off against monotonicity — the
constrained models are the better calibrated ones. All three beat the top unconstrained models on
ECE: `Monotone XGBoost` **0.059**, `Monotone HistGB` 0.062, the points model 0.071, against 0.094
for both `Extra trees` and `Random forest`. Brier separates them by less than 0.02 either way.

Both rule baselines are badly calibrated (ECE 0.269 / 0.347). Platt-scaling the rule fixes the
Brier a little (0.231 → 0.219) and cannot fix the ranking at all — AUC is unchanged at 0.647,
because a monotone rescaling cannot reorder anyone.

### 3.1 The three monotone models are tied

Paired over the same 50 folds. Read the effect sizes; the p-values are optimistic because
repeated k-fold reuses rows across repeats.

| comparison | ΔF1 | p | Δsensitivity | p | ΔECE | p |
|---|---:|---:|---:|---:|---:|---:|
| Mono XGBoost − Mono HistGB | +0.0026 | **0.94** | +0.0035 | 0.57 | −0.0029 | 0.010 |
| Mono XGBoost − Non-neg logistic | +0.0065 | 0.027 | +0.0035 | 0.47 | −0.0119 | 0.009 |
| Mono HistGB − Non-neg logistic | +0.0039 | 0.005 | 0.0000 | 0.80 | −0.0090 | 0.36 |

`Monotone XGBoost` heads the leaderboard by 0.0026 F1 over `Monotone HistGB` at *p* = 0.94 —
that is sort order, not evidence. Its lead over the points model is 0.0065 F1 and 0.012 ECE:
detectable, and too small to choose on. Nothing separates any of the three on **sensitivity**,
the metric that matters most for a screen.

So the choice inside the monotone family has to be made on grounds other than the score.

### 3.2 Why the points model, and not the boosted trees

There are only **128 possible patients** — seven binary criteria. Of those:

| | count |
|---|---:|
| patterns never observed | **68 of 128** |
| patterns with fewer than 5 patients | 45 (82 of 402 patients) |
| patterns with fewer than 10 patients | 53 (127 of 402 patients) |

The task is estimating a 128-cell lookup table with more than half the cells empty. The fitted
`Monotone XGBoost` uses **300 trees** to do it; the points model uses **8 numbers** (7 weights
and an intercept). Once monotonicity is imposed there is no smooth structure left for the extra
flexibility to find. Both models return *something* on the 68 unobserved patterns, but on
different terms: the ensemble returns whichever leaf the pattern happens to fall into, an
arbitrary step inherited from splits fitted elsewhere, while the points model returns the sum of
its weights — an extrapolation on a stated, checkable assumption. When the training cohort is
known to be the wrong population, the assumption you can state is the safer one.

Three practical points follow:

- **It is inspectable and publishable.** Seven weights (§4.2) can be read, argued with by a
  clinician, and hand-computed at a booth. A 300-tree ensemble can only be trusted.
- **It is the same shape as the instrument already shipped.** `core.compute_score` already sums
  per-criterion weights; adopting the points model changes those weights and the aggregation rule
  (§5), not the architecture. No serialised model artefact, no new scoring code path.
- **It keeps the deployment lean.** `requirements.txt` stays at three packages; no xgboost, no
  joblib, no model file to version or load.

If none of that matters for a given deployment, `Monotone XGBoost` buys 0.0065 more F1 and is a
perfectly defensible pick. This is a tie broken on engineering grounds, not on accuracy.

### 3.3 Best unconstrained vs best monotone, paired over the same 50 folds

| metric | Extra trees | Monotone XGBoost | difference |
|---|---:|---:|---:|
| F1 | 0.9103 | 0.8903 | −0.0200 |
| sensitivity | 0.8965 | 0.8460 | −0.0505 |
| ROC AUC | 0.9447 | 0.9258 | −0.0189 |
| Brier | 0.0774 | 0.0845 | +0.0072 |
| **ECE** | 0.0938 | **0.0594** | **−0.0344** |

The gaps are consistent across folds rather than noise (see `results/head_to_head.csv`; the
signed-rank p-values there are optimistic, because repeated k-fold reuses rows across repeats, so
read the effect sizes). The trade is real and small: give up 2 points of F1 and 5 of sensitivity,
get better calibration and the structural guarantee below.

---

## 4. Why the leaderboard is not the decision

### 4.1 Every unconstrained model can be argued out of a referral

![Monotonicity audit](figures/06_monotonicity_audit.png)

Enumerating all 128 tick-box states and every one-symptom edge between them:

| model | violating edges | worst case observed |
|---|---:|---|
| SVC (RBF) | 223 / 448 (50%) | adding *joint involvement*: 0.832 → 0.115 |
| HistGradientBoosting | 149 (33%) | adding *oral ulcer*: 0.686 → 0.082 |
| Logistic (L2 / L1), Bernoulli NB | 128 (29%) | adding *joint involvement*: 0.585 → 0.397 |
| Random forest | 113 (25%) | adding *oral ulcer*: **0.802 → 0.000** |
| XGBoost | 104 (23%) | adding *oral ulcer*: 0.772 → 0.022 |
| Extra trees | 41 (9%) | adding *oral ulcer*: 0.800 → 0.000 |
| Decision tree | 17 (4%) | adding *ACL*: 1.000 → 0.000 |
| **all three monotone models** | **0** | — |

Read the Random forest row as a booth interaction: a visitor is told 80% risk, mentions they also
have a mouth ulcer, and is told 0%. That is not a defensible screening tool no matter what its F1
is, and it is the same defect that withdrew v1's model.

The empty-form probe tells the same story. With nothing ticked, `SVC` returns **0.832**, the tree
ensembles **0.77–0.80**, and `k-NN` **1.000** — against v1's withdrawn **0.921**. The monotone
models return **0.12–0.17**, and the shipped rule returns **0**. Every one of those unconstrained
numbers is a *correct* fit to a cohort where 8 of 10 symptom-free patients had lupus, and none of
them is usable on a walk-in.

### 4.2 What the constraint costs, in points

![Points model](figures/09_points_model.png)

The non-negative logistic fit is the clearest view of what the cohort would do if allowed. Rescaled
onto the published 0–18 range:

| criterion | EULAR/ACR 2019 | learned from this cohort |
|---|---:|---:|
| Proteinuria | 4 | **6.0** |
| SCL or DL | 4 | 4.2 |
| Alopecia | 2 | 3.0 |
| ACL | 6 | **2.0** |
| Fever | 2 | 1.1 |
| Oral Ulcer | 2 | **0** (constraint binding) |
| Joint involvement | 6 | **0** (constraint binding) |

The constraint is doing real work on exactly the two criteria section 1 flagged. Unconstrained,
both would take negative weight.

### 4.3 The base rate, which nothing above accounts for

![Prevalence shift](figures/08_prevalence_shift.png)

Sensitivity and specificity are properties of the test; predictive value is not. At the
sensitivity-floor operating point:

| model | sens | spec | PPV @ 50% | PPV @ 5% | PPV @ 1% | PPV @ 0.1% | referred per 1,000 @ 0.1% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Extra trees | 0.90 | 0.92 | 0.92 | 0.38 | 0.106 | **0.012** | 77 |
| Monotone XGBoost | 0.89 | 0.82 | 0.83 | 0.21 | 0.047 | **0.005** | 181 |
| EULAR/ACR rule (shipped) | 0.95 | 0.09 | 0.51 | 0.05 | 0.011 | **0.001** | **906** |

At a public booth (~0.1% prevalence) the best model refers 77 people per 1,000 to find roughly one
case in 86 referrals; the shipped rule refers **906 per 1,000**, which is not screening. That
ranking is the one solid, transferable conclusion in this section.

The PPV values themselves are not. They are computed from sensitivity and specificity measured
against *this* control group. Booth attendees are not rheumatology patients, so the real specificity
is unknown — plausibly higher, since healthy people tick fewer boxes, but unmeasured. Treat the
column as an order of magnitude, not a number.

---

## 5. What to do

**Do not deploy any of these models to the public booth on this data.** Not because they fit
badly — they fit this cohort well — but because they were fitted against the wrong comparison
group. `core.py` already reaches that conclusion for v1's model; this bakeoff confirms it holds for
every model family, and quantifies it.

**Keep the EULAR/ACR rule for now.** Its AUC of 0.647 here is a measurement against a control arm
that does not represent booth attendees. It is monotone, scores 0 on an empty form, and is citable.
Its poor showing is a reason to seek better data, not a reason to swap in a model whose failure mode
is worse.

**If the tool is instead pointed at clinic pre-lab triage** — patients already symptomatic, being
considered for the full panel, i.e. the population this cohort actually samples — then the
non-negative logistic points model is a defensible replacement for the rule and would be a large
improvement (F1 0.884 vs 0.639, AUC 0.925 vs 0.647). Adoption means new `score` values in
`criteria_d9.json` plus two changes in `core.compute_score`: sum the weights instead of taking the
domain maximum, and — only if a calibrated probability is wanted rather than a band — add the
fitted intercept and pass the total through a logistic. The monotonicity property in
`tests/test_core.py:110-143` holds by construction either way.

**What would settle the booth question:** a consecutive sample of booth attendees with the seven
criteria recorded and a real diagnostic follow-up, including the people who tick nothing. Even a few
hundred rows would fix the two things this cohort cannot: the direction of joint involvement, and
the intercept. The app already logs exactly these seven fields per submission
(`storage.build_header`), so the collection mechanism exists.

**Reconsider the yellow band regardless.** `core.BAND_YELLOW = 6` is marked provisional. On any
population, joint involvement alone reaches 6 and triggers a referral. Whether that is right is a
clinical call, but it is the single decision that most determines the referral volume.

---

## 6. What actually shipped

The recommendation in §3.2 was adopted with one refinement, driven by the objection that a
cohort whose controls are rheumatology patients cannot be trusted to say a criterion is
worthless. Rather than letting oral ulcer and joint involvement fall to zero, the fit is
**shrunk toward the published EULAR/ACR weights** instead of toward zero:

```
minimise   logloss(Xw + b)  +  λ‖w − κ·w_eular‖²      subject to  w ≥ ε
```

λ = 2, chosen by cross-validating the whole procedure. Reproduce with
`exp/.venv/bin/python exp/fit_points_model.py`, which refits, rounds, and verifies the
result against `criteria_d9.json`.

| criterion | shipped points | EULAR/ACR |
|---|---:|---:|
| Proteinuria | 12 | 4 |
| SCL or DL | 8 | 4 |
| ACL | 7 | 6 |
| Alopecia | 5 | 2 |
| Fever | 2 | 2 |
| Oral Ulcer | 1 | 2 |
| Joint involvement | 1 | 6 |

Cross-validated at the referral cut-off: **AUC 0.905, sensitivity 0.802, specificity
0.966**, against 0.647 / 0.862 / 0.205 for the published weights.

The cost of the hedge is about **0.019 of AUC** versus letting those two weights go to
zero — and that penalty is itself measured on the cohort whose controls are saturated with
joint involvement, so it is an upper bound.

### Four bands, not three

Cut-offs were placed where the likelihood ratio changes:

| band | score | LR | n | SLE | controls |
|---|---|---:|---:|---:|---:|
| GREEN | 0–2 | 0.14 | 208 | 25 | 183 |
| YELLOW | 3–7 | **0.93** | 25 | 12 | 13 |
| ORANGE | 8–12 | 10.1 | 66 | 60 | 6 |
| RED | 13+ | ≥35 | 103 | 103 | 0 |

Scores 3–7 carry a likelihood ratio indistinguishable from 1 — that range leaves risk
exactly where it started. Folding it into GREEN would falsely reassure 12 SLE patients;
folding it into ORANGE would refer 13 controls to find them. It therefore gets its own
band, which reports the finding and asks for nothing. **Referral starts at ORANGE.**

This also corrects an earlier conclusion in this report. §3 selected a cut-off targeting
sensitivity above the shipped rule's 0.862, reaching 0.875 — but the sensitivity gained
came entirely from the 3–7 range, which is noise. Chasing sensitivity alone was the wrong
objective at booth prevalence.

### The dipstick dependency

Proteinuria carries 12 of the 36 points and 59% of the cohort's SLE patients have it. With
proteinuria unavailable the table collapses to **AUC 0.651, sensitivity 0.545** — no better
than the published weights. A urine dipstick is a hard requirement for this form, not an
enhancement.

## Files

| output | contents |
|---|---|
| `results/summary.csv` | Full leaderboard, both feature sets, mean and sd for every metric |
| `results/cv_metrics.csv` | Raw per-fold metrics — 50 folds × 16 models × 2 feature sets |
| `results/head_to_head.csv` | Paired fold-by-fold comparisons |
| `results/monotonicity.csv` | Violation counts, worst drop, empty-form probability |
| `results/ppv_by_prevalence.csv` | Operating points re-expressed at five base rates |
| `results/separation.csv` | Class-conditional rates and quasi-separation flags |
| `results/effect_directions.csv` | Coefficients, odds ratios, permutation importance |
| `figures/` | The nine figures above |
