# Testing strategy
#
# The input domain is only 2^7 = 128 states, so where a property must hold universally the
# test enumerates all of them rather than sampling. Elsewhere the partition is:
#
# compute_score — by how many criteria contribute, and from where:
#     no criteria ticked (boundary: empty)      -> 0
#     one criterion                             -> that weight
#     two criteria in the SAME domain           -> they ADD (this is what changed when the
#                                                  domain-maximum rule was dropped)
#     criteria across different domains         -> they add
#     every criterion ticked (boundary: full)   -> 36
#   Plus the absent-key case: a criterion missing from the dict must read as absent
#   rather than raise, because the form only writes keys the user interacted with.
#
# compute_band — by the three cut-offs, covering each subdomain and every boundary, which
#   is where an off-by-one would live: 0..2 green, 3 exactly, 4..7 yellow, 8 exactly,
#   9..12 orange, 13 exactly, 14..36 red. Boundary class: 0 and 36, the extremes of the
#   legal score range.
#
# Properties (exhaustive over all 128 inputs): monotonicity of both the score and the band,
#   and the score staying within its declared range. Monotonicity is the specific defect
#   that disqualified v1's model, so it is asserted directly rather than assumed; here it
#   follows from every weight being positive, which is itself asserted.
#
# criteria_d9.json — guards the data the scoring reads: the seven keys, the fitted weights,
#   and the published EULAR/ACR weights they were shrunk toward. Both must be pinned: the
#   fitted values because the bands are calibrated to them, and `eular_score` because
#   exp/fit_points_model.py consumes it as the prior and would otherwise shrink toward its
#   own output.

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


def test_fitted_weights_are_the_calibrated_ones():
    """The band cut-offs are calibrated to these exact weights; changing one without
    refitting the bands would silently move every triage decision."""
    weights = {c["key"]: c["score"] for c in CRITERIA}
    assert weights == {
        "Fever": 2, "ACL": 7, "SCL or DL": 8, "Oral Ulcer": 1,
        "Alopecia": 5, "Joint involvement": 1, "Proteinuria": 12,
    }


def test_published_eular_weights_are_preserved_as_the_prior():
    """exp/fit_points_model.py shrinks toward these. If `score` overwrote them the fit
    would shrink toward its own previous output and drift on every rerun."""
    prior = {c["key"]: c["eular_score"] for c in CRITERIA}
    assert prior == {
        "Fever": 2, "ACL": 6, "SCL or DL": 4, "Oral Ulcer": 2,
        "Alopecia": 2, "Joint involvement": 6, "Proteinuria": 4,
    }


def test_every_weight_is_positive():
    """Monotonicity follows from this. A zero or negative weight would mean reporting one
    more symptom could leave the score unchanged or lower it."""
    assert [c["key"] for c in CRITERIA if c["score"] <= 0] == []


def test_reported_accuracy_matches_the_measurement():
    """These are shown to staff on the result screen, so a stale value is a claim made to a
    volunteer about how much to trust a green. Measured output of exp/fit_points_model.py
    at core.BAND_ORANGE — regenerate rather than edit."""
    assert core.SENSITIVITY_AT_REFERRAL == 0.802
    assert core.SPECIFICITY_AT_REFERRAL == 0.966
    assert core.SENSITIVITY_WITHOUT_URINE == 0.378


def test_losing_the_urine_result_costs_most_of_the_sensitivity():
    """The reason the app asks whether a urine result exists at all. If these ever converge
    the question has stopped earning its place on the form."""
    assert core.SENSITIVITY_WITHOUT_URINE < core.SENSITIVITY_AT_REFERRAL / 2


def test_criteria_declare_a_thai_test_requirement():
    """Empty is legal for the six criteria a visitor can observe unaided. Proteinuria's
    must not be, since it is the only one needing a lab result and its caption is the only
    place the form says so."""
    for c in CRITERIA:
        assert isinstance(c["requires_test_th"], str), c["key"]
    by_key = {c["key"]: c for c in CRITERIA}
    assert by_key["Proteinuria"]["requires_test_th"].strip()
    assert by_key["Proteinuria"]["requires_test"].strip()


def test_every_referenced_image_exists():
    """Images churn as consented photographs replace placeholders; a stale path in
    criteria_d9.json would only surface as a broken card at the booth."""
    missing = [src for c in CRITERIA for src in c["images"] if not (REPO / src).is_file()]
    assert missing == []


def test_article_has_content_and_cites_its_sources():
    ARTICLE = core.load_article()
    assert len(ARTICLE) > 500
    assert "แหล่งอ้างอิง" in ARTICLE


def test_article_embeds_no_images():
    """docs/adr/0001 removed every third-party poster because none were licensed for reuse.

    A Markdown image would put one back without anyone revisiting that decision, so the
    rule is enforced here rather than left to review. Links to sources are fine — only
    reproduction is not.
    """
    assert "![" not in core.load_article()


def test_no_unlicensed_posters_remain_in_the_repo():
    assert list((REPO / "images").glob("poster_*")) == []


def test_criteria_declare_an_images_list():
    """Zero images is legal (ไข้ has none); the key itself must always be present."""
    for c in CRITERIA:
        assert isinstance(c["images"], list), c["key"]


def test_nothing_ticked_scores_zero():
    """The defect that disqualified v1's model: it returned 0.921 for this input."""
    assert score_of() == 0


def test_everything_ticked_scores_the_maximum():
    assert score_of(*KEYS) == 36
    assert core.max_score(CRITERIA) == 36


def test_single_criterion_scores_its_own_weight():
    assert score_of("ACL") == 7
    assert score_of("Fever") == 2
    assert score_of("Proteinuria") == 12


def test_same_domain_adds_rather_than_taking_the_maximum():
    """ACL(7) and Alopecia(5) are both Mucocutaneous. Under the old domain-maximum rule
    this was 7; summing is the behaviour change that motivated the refit."""
    assert score_of("ACL", "Alopecia") == 12
    assert score_of("SCL or DL", "Oral Ulcer", "Alopecia") == 14


def test_different_domains_add():
    """Fever(Constitutional,2) + Joint(Musculoskeletal,1) + Proteinuria(Renal,12)."""
    assert score_of("Fever", "Joint involvement", "Proteinuria") == 15


def test_missing_keys_read_as_absent():
    score, _ = core.compute_score({"ACL": True}, CRITERIA)
    assert score == 7


def test_breakdown_reports_every_domain():
    _, breakdown = core.compute_score({"ACL": True, "Alopecia": True}, CRITERIA)
    assert breakdown == {"Constitutional": 0, "Mucocutaneous": 12,
                         "Musculoskeletal": 0, "Renal": 0}


@pytest.mark.parametrize("score,expected", [
    (0, core.GREEN), (2, core.GREEN),
    (3, core.YELLOW), (4, core.YELLOW), (7, core.YELLOW),
    (8, core.ORANGE), (9, core.ORANGE), (12, core.ORANGE),
    (13, core.RED), (36, core.RED),
])
def test_compute_band(score, expected):
    assert core.compute_band(score) == expected


def test_isolated_vague_findings_never_leave_green():
    """The 154 controls who report only joint pain, and the 15 who report only an oral
    ulcer, must not be referred — that failure is what made the previous rule refer 90%
    of everyone."""
    assert core.compute_band(score_of("Joint involvement")) == core.GREEN
    assert core.compute_band(score_of("Oral Ulcer")) == core.GREEN
    assert core.compute_band(score_of("Joint involvement", "Oral Ulcer")) == core.GREEN


def test_isolated_proteinuria_reaches_orange_but_not_red():
    """Proteinuria alone warrants a doctor, but on its own it is far more often diabetic,
    hypertensive or benign than lupus, so it must not read as an urgent lupus result."""
    assert core.compute_band(score_of("Proteinuria")) == core.ORANGE


def test_score_stays_in_range_for_every_possible_input():
    for values in all_input_states():
        assert 0 <= core.compute_score(values, CRITERIA)[0] <= core.max_score(CRITERIA)


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
    rank = {band: i for i, band in enumerate(core.BAND_ORDER)}
    for values in all_input_states():
        base = rank[core.compute_band(core.compute_score(values, CRITERIA)[0])]
        for key in KEYS:
            if values[key]:
                continue
            with_extra = {**values, key: True}
            assert rank[core.compute_band(core.compute_score(with_extra, CRITERIA)[0])] >= base
