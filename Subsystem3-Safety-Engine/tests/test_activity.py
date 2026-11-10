from datetime import datetime, timedelta, timezone

from alert_engine.activity import build_periods, load_rows, summarise
from alert_engine.sinks import LocalSqliteSink

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def row(offset_s, posture, tier="safe", person_id="p-1", score=10):
    return {
        "person_id": person_id, "camera_id": "cam-1", "zone": "lounge", "posture": posture, "tier": tier,
        "risk_score": score, "started_at": (T0 + timedelta(seconds=offset_s)).isoformat(),
    }


def test_duration_runs_until_next_row():
    rows = [row(0, "walking"), row(10, "bending", "warning"), row(16, "standing")]
    periods = build_periods(rows, T0 + timedelta(seconds=20))
    assert [p.duration_s for p in periods] == [10, 6, 4]


def test_latest_row_is_ongoing():
    periods = build_periods([row(0, "walking"), row(10, "sitting")], T0 + timedelta(seconds=20))
    assert [p.ongoing for p in periods] == [False, True]


def test_unmonitored_is_never_ongoing():
    periods = build_periods([row(0, "walking"), row(10, "unmonitored", None, score=None)], T0 + timedelta(seconds=20))
    assert periods[-1].ongoing is False


def test_people_are_timed_separately():
    rows = [row(0, "walking", person_id="p-1"), row(5, "sitting", person_id="p-2"), row(10, "sitting", person_id="p-1")]
    periods = build_periods(rows, T0 + timedelta(seconds=20))
    assert [(p.person_id, p.duration_s) for p in periods] == [("p-1", 10), ("p-2", 15), ("p-1", 10)]


def test_out_of_order_timestamps_never_give_negative_duration():
    periods = build_periods([row(10, "walking"), row(0, "sitting")], T0 + timedelta(seconds=20))
    assert periods[0].duration_s == 0


def test_summary_totals_exclude_unmonitored_time():
    rows = [row(0, "sitting"), row(10, "unmonitored", None, score=None), row(30, "sitting"), row(35, "walking")]
    summary = summarise(build_periods(rows, T0 + timedelta(seconds=40)), "p-1")
    assert summary.posture_totals_s == {"sitting": 15, "walking": 5}
    assert summary.unmonitored_s == 20


def test_unsafe_then_alert_counts_as_one_risk_period():
    rows = [
        row(0, "bending", "warning"), row(5, "bending", "unsafe"), row(8, "bending", "alert"),
        row(12, "standing"), row(15, "bending", "unsafe"), row(18, "standing"),
    ]
    summary = summarise(build_periods(rows, T0 + timedelta(seconds=20)), "p-1")
    assert summary.risk_periods == 2
    assert summary.alert_periods == 1


def test_load_rows_filters_by_person_in_logged_order(tmp_path):
    sink = LocalSqliteSink(sqlite_path=tmp_path / "events.db")
    sink.start()
    sink.write_activity("p-1", "cam-1", "lounge", "walking", "safe", 10, T0)
    sink.write_activity("p-2", "cam-1", "lounge", "sitting", "safe", 5, T0)
    sink.write_activity("p-1", "cam-1", "lounge", "sitting", "safe", 5, T0 + timedelta(seconds=1))
    sink.stop()
    assert [r["posture"] for r in load_rows(tmp_path / "events.db", "p-1")] == ["walking", "sitting"]
