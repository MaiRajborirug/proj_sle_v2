# Testing strategy
#
# build_header — one representative: the real criteria list. Asserts the column contract
#   the analysis scripts depend on (metadata, then one column per criterion, then result).
#
# SheetRecorder — partition on `enabled`, the flag that implements test mode. The disabled
#   case is the one that matters: it must report success without touching the network, or
#   a demo would either crash or pollute the study dataset. Row construction is tested
#   through _row_from, including the missing-column case.
#
# Throttle — partition by client state: under the limit, at the limit, over the limit,
#   and the empty-IP case (throttling disabled rather than everyone blocked). Window
#   expiry is covered by driving the clock rather than sleeping.

import core
import storage
from throttle import Throttle


def test_build_header_column_contract():
    header = storage.build_header(core.load_criteria())
    assert header[:7] == ["timestamp_utc", "session_uuid", "submission_seq", "mode",
                          "app_version", "sex", "age_band"]
    assert header[7:14] == ["Fever", "ACL", "SCL_or_DL", "Oral_Ulcer", "Alopecia",
                            "Joint_involvement", "Proteinuria"]
    assert header[14:] == ["n_criteria", "eular_score", "band"]
    assert len(header) == 17


def test_disabled_recorder_reports_success_without_network():
    recorder = storage.SheetRecorder(["a", "b"], enabled=False)
    assert recorder.append({"a": 1, "b": 2}) is True
    assert recorder.pending_count == 0


def test_row_from_fills_missing_columns_with_blank():
    recorder = storage.SheetRecorder(["a", "b", "c"], enabled=False)
    assert recorder._row_from({"a": 1, "c": 3}) == [1, "", 3]


def test_enabled_recorder_queues_row_when_credentials_absent(monkeypatch):
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.delenv("SHEET_ID", raising=False)
    recorder = storage.SheetRecorder(["a"], enabled=True)
    assert recorder.append({"a": 1}) is False
    assert recorder.pending_count == 1


def test_throttle_allows_up_to_the_limit():
    t = Throttle(max_events=3, window_seconds=600)
    assert [t.allow("1.2.3.4") for _ in range(3)] == [True, True, True]


def test_throttle_blocks_past_the_limit():
    t = Throttle(max_events=3, window_seconds=600)
    for _ in range(3):
        t.allow("1.2.3.4")
    assert t.allow("1.2.3.4") is False


def test_throttle_is_per_client():
    t = Throttle(max_events=1, window_seconds=600)
    assert t.allow("1.2.3.4") is True
    assert t.allow("1.2.3.4") is False
    assert t.allow("5.6.7.8") is True


def test_throttle_disabled_when_no_client_ip():
    t = Throttle(max_events=1, window_seconds=600)
    assert all(t.allow("") for _ in range(10))


def test_throttle_forgets_hits_outside_the_window(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr("throttle.time.monotonic", lambda: clock["now"])
    t = Throttle(max_events=2, window_seconds=60)
    assert t.allow("1.2.3.4") is True
    assert t.allow("1.2.3.4") is True
    assert t.allow("1.2.3.4") is False
    clock["now"] += 61
    assert t.allow("1.2.3.4") is True
