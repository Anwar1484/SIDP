import json
import sqlite3
from datetime import datetime, timezone

from alert_engine.schemas import ClassifiedEvent
from alert_engine.sinks import LocalSqliteSink


def make_sink(tmp_path) -> LocalSqliteSink:
    sink = LocalSqliteSink(sqlite_path=tmp_path / "events.db")
    sink.start()
    return sink


def query_all(tmp_path, table: str) -> list[dict]:
    conn = sqlite3.connect(tmp_path / "events.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def make_classified_event(**overrides) -> ClassifiedEvent:
    defaults = dict(
        alert_id="alert-1",
        event_id="event-1",
        person_id="p-1",
        camera_id="cam-1",
        zone="lounge",
        posture="bending",
        tier="unsafe",
        risk_score=70,
        duration_s=6.4,
        reasons=["bending sustained 6.4 s"],
        detected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        dispatched_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return ClassifiedEvent(**defaults)


def test_write_status_stores_row(tmp_path):
    sink = make_sink(tmp_path)
    sink.write_status("p-1", "bending", "unsafe", 70, datetime(2026, 1, 1, tzinfo=timezone.utc))
    sink._queue.join()
    sink.stop()

    rows = query_all(tmp_path, "resident_status")
    assert len(rows) == 1
    assert rows[0]["person_id"] == "p-1"
    assert rows[0]["posture"] == "bending"
    assert rows[0]["tier"] == "unsafe"
    assert rows[0]["risk_score"] == 70


def test_write_status_upserts_by_person(tmp_path):
    sink = make_sink(tmp_path)
    sink.write_status("p-1", "bending", "unsafe", 70, datetime(2026, 1, 1, tzinfo=timezone.utc))
    sink.write_status("p-1", "sitting", "safe", 5, datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc))
    sink._queue.join()
    sink.stop()

    rows = query_all(tmp_path, "resident_status")
    assert len(rows) == 1
    assert rows[0]["posture"] == "sitting"
    assert rows[0]["tier"] == "safe"


def test_write_alert_stores_full_record(tmp_path):
    sink = make_sink(tmp_path)
    event = make_classified_event()
    sink.write_alert(event)
    sink._queue.join()
    sink.stop()

    rows = query_all(tmp_path, "alerts")
    assert len(rows) == 1
    row = rows[0]
    assert row["alert_id"] == "alert-1"
    assert row["person_id"] == "p-1"
    assert row["tier"] == "unsafe"
    assert row["risk_score"] == 70
    assert json.loads(row["reasons"]) == ["bending sustained 6.4 s"]


def test_write_alert_does_not_dedupe(tmp_path):
    sink = make_sink(tmp_path)
    sink.write_alert(make_classified_event(alert_id="alert-1"))
    sink.write_alert(make_classified_event(alert_id="alert-2"))
    sink._queue.join()
    sink.stop()

    rows = query_all(tmp_path, "alerts")
    assert len(rows) == 2


def test_write_camera_status_upserts(tmp_path):
    sink = make_sink(tmp_path)
    sink.write_camera_status("cam-1", "monitored", datetime(2026, 1, 1, tzinfo=timezone.utc))
    sink.write_camera_status("cam-1", "unmonitored", datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc))
    sink._queue.join()
    sink.stop()

    rows = query_all(tmp_path, "camera_status")
    assert len(rows) == 1
    assert rows[0]["status"] == "unmonitored"


def test_writes_survive_stop_and_restart(tmp_path):
    sink = make_sink(tmp_path)
    sink.write_status("p-1", "sitting", "safe", 5, datetime(2026, 1, 1, tzinfo=timezone.utc))
    sink._queue.join()
    sink.stop()

    sink2 = make_sink(tmp_path)
    sink2.write_status("p-2", "standing", "safe", 5, datetime(2026, 1, 1, tzinfo=timezone.utc))
    sink2._queue.join()
    sink2.stop()

    rows = query_all(tmp_path, "resident_status")
    assert {r["person_id"] for r in rows} == {"p-1", "p-2"}


def test_bad_job_does_not_crash_worker(tmp_path):
    sink = make_sink(tmp_path)
    sink._queue.put(("nonsense", ()))
    sink.write_status("p-1", "sitting", "safe", 5, datetime(2026, 1, 1, tzinfo=timezone.utc))
    sink._queue.join()
    sink.stop()

    rows = query_all(tmp_path, "resident_status")
    assert len(rows) == 1  # worker kept running after the bad job
