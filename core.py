"""Scoring and triage banding for the v2 screening app.

**Why there is no machine-learning model here.** v1 ships `model_d9.joblib`, a
`CalibratedClassifierCV` over an SVC trained on `SLE_NotSLE.csv` — a 202/200 case-control
cohort of hospital patients. Evaluated across all 128 possible d9 inputs, it returns
**0.921 with no symptoms ticked**, places 117/128 inputs in the highest risk band, and is
non-monotonic: reporting an isolated fever *lowers* its output to 0.185. That behaviour is
correct for its training cohort, where the comparison group was other sick people, and
invalid for a public booth, where the comparison group is healthy people. It is therefore
not used for scoring. See README.md.

v2 instead uses the published **EULAR/ACR 2019** weights restricted to the seven
observable d9 criteria, with the standard domain-maximum rule. That score is monotonic by
construction, scores 0 when nothing is present, and is a citable instrument rather than a
black box.

This module does not import Streamlit, so it can be unit-tested and reused from analysis
scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

APP_VERSION = "2.0.0"

_HERE = Path(__file__).parent
CRITERIA_FILE = _HERE / "criteria_d9.json"
ARTICLES_FILE = _HERE / "articles.json"

# Triage cut-offs on the restricted EULAR/ACR score, which ranges 0–18 for these seven
# criteria.
#
# RED is set at 10 to match the published EULAR/ACR 2019 classification threshold, reached
# here on observable findings alone. YELLOW is set at 6, the weight of a single major
# finding (acute cutaneous lupus, or joint involvement), so one major sign prompts
# follow-up while an isolated minor sign (fever, oral ulcer or alopecia, each 2) does not.
#
# PROVISIONAL — these need clinical sign-off, and should be revisited after the first
# event against the real distribution of booth scores and the referral clinic's capacity.
BAND_YELLOW = 6
BAND_RED = 10

GREEN, YELLOW, RED = "green", "yellow", "red"


def load_criteria() -> list[dict]:
    """Load the seven d9 criteria.

    Returns:
        The criteria as stored in `criteria_d9.json`, in display order.
    """
    with open(CRITERIA_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_articles() -> list[dict]:
    """Load the reading-material posters shown on the บทความ page.

    Returns:
        The posters as stored in `articles.json`, in display order.
    """
    with open(ARTICLES_FILE, encoding="utf-8") as f:
        return json.load(f)


def compute_score(values: dict[str, bool], criteria: list[dict]) -> tuple[int, dict[str, int]]:
    """Score the ticked criteria using EULAR/ACR domain-maximum weighting.

    Within each domain only the highest-weighted present criterion counts, which is the
    standard EULAR/ACR 2019 rule.

    Args:
        values: Maps criterion `key` to whether it was ticked. Missing keys read as absent.
        criteria: The criteria list from `load_criteria()`.

    Returns:
        `(total, breakdown)` where `breakdown` maps each domain present in `criteria` to
        the points it contributed. Domains contributing nothing map to 0.
    """
    breakdown: dict[str, int] = {}
    for c in criteria:
        present = bool(values.get(c["key"], False))
        points = c["score"] if present else 0
        breakdown[c["domain"]] = max(breakdown.get(c["domain"], 0), points)
    return sum(breakdown.values()), breakdown


def compute_band(score: int) -> str:
    """Map a restricted EULAR/ACR score onto a triage band.

    Args:
        score: Total from `compute_score`.

    Returns:
        One of `GREEN`, `YELLOW`, `RED`.
    """
    if score >= BAND_RED:
        return RED
    if score >= BAND_YELLOW:
        return YELLOW
    return GREEN
