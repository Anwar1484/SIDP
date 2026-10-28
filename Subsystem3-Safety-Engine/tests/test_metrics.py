import json
from datetime import datetime, timedelta, timezone

from alert_engine.metrics import MetricsLogger

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_log_alert_appends_jsonl_record(tmp_path):
    logger = MetricsLogger(path=tmp_path / "metrics.jsonl")
    logger.log_alert(
        alert_id="alert-1",
        detected_at=BASE_TIME,
        received_at=BASE_TIME + timedelta(seconds=0.05),
        confirmed_at=BASE_TIME + timedelta(seconds=0.3),
        dispatched_at=BASE_TIME + timedelta(seconds=0.4),
    )
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["alert_id"] == "alert-1"
    assert record["latency_s"] == 0.4


def test_log_alert_appends_multiple_records(tmp_path):
    logger = MetricsLogger(path=tmp_path / "metrics.jsonl")
    for i in range(3):
        logger.log_alert(
            alert_id=f"alert-{i}",
            detected_at=BASE_TIME,
            received_at=BASE_TIME,
            confirmed_at=BASE_TIME,
            dispatched_at=BASE_TIME,
        )
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
