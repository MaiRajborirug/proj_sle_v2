# Testing strategy
#
# The input domain is only 2^7 = 128 states, so where a property must hold universally the
# test enumerates all of them rather than sampling. Elsewhere the partition is:
#
# compute_score — by how the domain-maximum rule applies:
#     no criteria ticked (boundary: empty)      -> 0
#     one criterion in a domain                 -> that weight
#     two criteria in the SAME domain           -> the higher weight only, not the sum
#     criteria across different domains         -> weights add
#     every criterion ticked (boundary: full)   -> 18
#   Plus the absent-key case: a criterion missing from the dict must read as absent
#   rather than raise, because the form only writes keys the user interacted with.
#
# compute_band — by the two cut-offs, covering each subdomain and both boundaries, which
#   is where an off-by-one would live: 0..5 green, 6 exactly, 7..9 yellow, 10 exactly,
#   11..18 red. Boundary class: 0 and 18, the extremes of the legal score range.
#
# Properties (exhaustive over all 128 inputs): monotonicity, and the score staying within
#   its declared range. Monotonicity is the specific defect that disqualified v1's model,
#   so it is asserted directly rather than assumed.
#
# criteria_d9.json — guards the data the scoring reads: the seven keys, and the EULAR/ACR
#   weights, which must not drift from the published values.

import itertools
from pathlib import Path

import pytest

import core

REPO = Path(__file__).resolve().parents[1]

CRITERIA = core.load_criteria()
KEYS = [c["key"] for c in CRITERIA]


def score_of(*ticked):
    return core.compute_score({k: True for k in ticked}, CRITERIA)[0]


def all_input_states():
    """Every one of the 128 possible tick combinations."""
    for combo in itertools.product([False, True], repeat=len(KEYS)):
        yield dict(zip(KEYS, combo))


def test_criteria_are_the_seven_d9_features():
    assert KEYS == ["Fever", "ACL", "SCL or DL", "Oral Ulcer",
                    "Alopecia", "Joint involvement", "Proteinuria"]


def test_eular_weights_match_published_values():
    weights = {c["key"]: c["score"] for c in CRITERIA}
    assert weights == {
        "Fever": 2, "ACL": 6, "SCL or DL": 4, "Oral Ulcer": 2,
        "Alopecia": 2, "Joint involvement": 6, "Proteinuria": 4,
    }


def test_every_referenced_image_exists():
    """Images churn as consented photographs replace placeholders; a stale path in
    criteria_d9.json would only surface as a broken card at the booth."""
    missing = [src for c in CRITERIA for src in c["images"] if not (REPO / src).is_file()]
    assert missing == []


def test_criteria_declare_an_images_list():
    """Zero images is legal (ไข้ has none); the key itself must always be present."""
    for c in CRITERIA:
        assert isinstance(c["images"], list), c["key"]


def test_nothing_ticked_scores_zero():
    """The defect that disqualified v1's model: it returned 0.921 for this input."""
    assert score_of() == 0


def test_everything_ticked_scores_the_maximum():
    assert score_of(*KEYS) == 18


def test_single_criterion_scores_its_own_weight():
    assert score_of("ACL") == 6
    assert score_of("Fever") == 2


def test_same_domain_takes_the_maximum_not_the_sum():
    """ACL(6) and Alopecia(2) are both Mucocutaneous, so the domain contributes 6."""
    assert score_of("ACL", "Alopecia") == 6
    assert score_of("SCL or DL", "Oral Ulcer", "Alopecia") == 4


def test_different_domains_add():
    """Fever(Constitutional,2) + Joint(Musculoskeletal,6) + Proteinuria(Renal,4)."""
    assert score_of("Fever", "Joint involvement", "Proteinuria") == 12


def test_missing_keys_read_as_absent():
    score, _ = core.compute_score({"ACL": True}, CRITERIA)
    assert score == 6


def test_breakdown_reports_every_domain():
    _, breakdown = core.compute_score({"ACL": True}, CRITERIA)
    assert breakdown == {"Constitutional": 0, "Mucocutaneous": 6,
                         "Musculoskeletal": 0, "Renal": 0}


@pytest.mark.parametrize("score,expected", [
    (0, core.GREEN), (5, core.GREEN),
    (6, core.YELLOW), (7, core.YELLOW), (9, core.YELLOW),
    (10, core.RED), (18, core.RED),
])
def test_compute_band(score, expected):
    assert core.compute_band(score) == expected


def test_score_stays_in_range_for_every_possible_input():
    for values in all_input_states():
        assert 0 <= core.compute_score(values, CRITERIA)[0] <= 18


def test_scoring_is_monotonic_for_every_possible_input():
    """Adding a finding must never lower the score — v1's model failed exactly this."""
    for values in all_input_states():
        base = core.compute_score(values, CRITERIA)[0]
        for key in KEYS:
            if values[key]:
                continue
            with_extra = {**values, key: True}
            assert core.compute_score(with_extra, CRITERIA)[0] >= base, (key, values)


def test_band_is_monotonic_for_every_possible_input():
    rank = {core.GREEN: 0, core.YELLOW: 1, core.RED: 2}
    for values in all_input_states():
        base = rank[core.compute_band(core.compute_score(values, CRITERIA)[0])]
        for key in KEYS:
            if values[key]:
                continue
            with_extra = {**values, key: True}
            assert rank[core.compute_band(core.compute_score(with_extra, CRITERIA)[0])] >= base
