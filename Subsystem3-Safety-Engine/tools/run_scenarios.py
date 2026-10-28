"""Runs every scenario from CLAUDE.md section 10 in-process and reports pass/fail
against the requirements in section 12.

Runs entirely without MQTT, a broker, TTS or a speaker: it drives the same
pipeline main.py wires up (validate -> debounce -> score -> dispatch ->
watchdog), but with recording fakes standing in for the sink/audio/publish
ports, and a synthetic clock standing in for wall-clock time so an 8 s
dropout scenario finishes instantly instead of taking 8 real seconds.

Usage:
    python tools/run_scenarios.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from alert_engine.config import load_config  # noqa: E402
from alert_engine.debounce import DebounceTracker  # noqa: E402
from alert_engine.dispatcher import Dispatcher  # noqa: E402
from alert_engine.ml_scoring import HybridScorer  # noqa: E402
from alert_engine.main import CameraOccupancy, RejectionLogger, make_camera_status_handler, make_posture_handler  # noqa: E402
from alert_engine.metrics import MetricsLogger  # noqa: E402
from alert_engine.schemas import ClassifiedEvent  # noqa: E402
from alert_engine.watchdog import CameraWatchdog  # noqa: E402

SCENARIO_LOG_DIR = ROOT / "logs" / "scenario_runs"


def _load_simulate_publisher():
    spec = importlib.util.spec_from_file_location("simulate_publisher", ROOT / "tools" / "simulate_publisher.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["simulate_publisher"] = module
    spec.loader.exec_module(module)
    return module


sim = _load_simulate_publisher()


class _SyntheticClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def iso(self) -> str:
        return self.now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{self.now.microsecond // 1000:03d}Z"


def _generate_timed_frames(scenario_name: str) -> list[tuple[object, datetime]]:
    """Runs a scenario generator against a synthetic clock instead of real time.sleep.

    Preserves the real inter-frame gaps (e.g. the 8 s camera dropout) as
    timestamp deltas, without actually waiting for them.
    """
    clock = _SyntheticClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    original_now_iso = sim.now_iso
    sim.now_iso = clock.iso
    timed_frames: list[tuple[object, datetime]] = []
    try:
        for frame in sim.SCENARIOS[scenario_name]():
            timed_frames.append((frame.payload, clock.now))
            clock.now = clock.now + timedelta(seconds=frame.delay_s)
    finally:
        sim.now_iso = original_now_iso
    return timed_frames


class RecordingSink:
    def __init__(self) -> None:
        self.statuses: list[tuple] = []
        self.alerts: list[ClassifiedEvent] = []
        self.camera_statuses: list[tuple] = []

    def write_status(self, person_id, posture, tier, risk_score, updated_at) -> None:
        self.statuses.append((person_id, posture, tier, risk_score, updated_at))

    def write_alert(self, event: ClassifiedEvent) -> None:
        self.alerts.append(event)

    def write_camera_status(self, camera_id, status, last_seen) -> None:
        self.camera_statuses.append((camera_id, status, last_seen))


class RecordingAudio:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def enqueue(self, tier, zone, repeat) -> None:
        self.calls.append((tier, zone, repeat))


@dataclass
class ScenarioReport:
    name: str
    frame_count: int = 0
    published: list[str] = field(default_factory=list)
    rejected_count: int = 0
    sink: RecordingSink = field(default_factory=RecordingSink)
    audio: RecordingAudio = field(default_factory=RecordingAudio)
    camera_snapshots: list[dict] = field(default_factory=list)

    @property
    def published_events(self) -> list[dict]:
        return [json.loads(p) for p in self.published]


def run_scenario(scenario_name: str, config) -> ScenarioReport:
    report = ScenarioReport(name=scenario_name)
    SCENARIO_LOG_DIR.mkdir(parents=True, exist_ok=True)
    rejection_logger = RejectionLogger(path=SCENARIO_LOG_DIR / f"rejected_{scenario_name}.jsonl")
    metrics = MetricsLogger(path=SCENARIO_LOG_DIR / f"metrics_{scenario_name}.jsonl")
    debounce = DebounceTracker(config.debounce)
    occupancy = CameraOccupancy()
    scorer = HybridScorer.from_config(config)
    dispatcher = Dispatcher(
        config=config.dispatch, sink=report.sink, audio=report.audio, publish=report.published.append
    )

    camera_status_handler = make_camera_status_handler(report.sink, debounce, occupancy, lambda payload: None)

    def on_status_change(snapshot):
        report.camera_snapshots.append(snapshot)
        camera_status_handler(snapshot)

    watchdog = CameraWatchdog(camera_timeout_s=config.watchdog.camera_timeout_s, on_status_change=on_status_change)

    cursor = {"now": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    handler = make_posture_handler(
        config, rejection_logger, debounce, dispatcher, watchdog, metrics, occupancy, scorer,
        clock=lambda: cursor["now"],
    )

    timed_frames = _generate_timed_frames(scenario_name)
    report.frame_count = len(timed_frames)
    for payload, at in timed_frames:
        cursor["now"] = at
        watchdog.check(now=at)  # simulate the periodic poll thread noticing a stale camera
        if isinstance(payload, str):
            raw = payload.encode("utf-8")
        elif "__note__" in payload:
            continue
        else:
            raw = json.dumps(payload).encode("utf-8")
        handler("posture/events", raw)

    report.rejected_count = rejection_logger.count
    return report


@dataclass
class Check:
    requirement: str
    description: str
    passed: bool
    detail: str


def check_r1_alert_within_2s(report: ScenarioReport) -> Check:
    alert_events = [a for a in report.sink.alerts if a.tier == "alert"]
    if not alert_events:
        return Check("R1", "alert within 2s of confirmation", False, "no alert-tier event was produced")
    first = alert_events[0]
    latency = (first.dispatched_at - first.detected_at).total_seconds()
    return Check("R1", "alert within 2s of confirmation", latency <= 2.0, f"latency={latency:.3f}s")


def check_r2_no_alert_on_glitch(report: ScenarioReport) -> Check:
    passed = len(report.sink.alerts) == 0
    return Check("R2", "single-frame glitch triggers no alert", passed, f"alerts recorded={len(report.sink.alerts)}")


def check_r3_false_alerts_below_10pct(report: ScenarioReport) -> Check:
    events = report.published_events
    if not events:
        return Check("R3", "false alerts on normal activity < 10%", False, "no events were classified")
    non_safe = sum(1 for e in events if e["tier"] != "safe")
    fraction = non_safe / len(events)
    return Check("R3", "false alerts on normal activity < 10%", fraction < 0.10, f"non-safe fraction={fraction:.1%}")


def check_r4_valid_events_classified(report: ScenarioReport) -> Check:
    passed = len(report.published) > 0
    return Check(
        "R4", "valid events get classified into a tier", passed, f"classified events={len(report.published)}"
    )


def check_r5_malformed_rejected_without_crash(report: ScenarioReport) -> Check:
    passed = report.rejected_count > 0
    return Check(
        "R5", "malformed events rejected, engine keeps running", passed,
        f"rejected={report.rejected_count}, classified={len(report.published)} (run completed without raising)",
    )


def check_dropout_recovers(report: ScenarioReport) -> Check:
    statuses = [next(iter(snap.values()))["status"] for snap in report.camera_snapshots]
    saw_unmonitored = "unmonitored" in statuses
    ends_monitored = bool(statuses) and statuses[-1] == "monitored"
    passed = saw_unmonitored and ends_monitored
    return Check(
        "section 10", "camera flagged unmonitored, then recovers", passed, f"status transitions={statuses}"
    )


SCENARIO_CHECKS = {
    "fall": [check_r1_alert_within_2s],
    "glitch": [check_r2_no_alert_on_glitch],
    "normal": [check_r3_false_alerts_below_10pct],
    "bending": [check_r4_valid_events_classified],
    "dropout": [check_dropout_recovers],
    "malformed": [check_r5_malformed_rejected_without_crash],
}

ALL_SCENARIOS = ["normal", "glitch", "bending", "fall", "dropout", "malformed", "mixed"]


def main() -> int:
    config = load_config(str(ROOT / "config" / "config.yaml"))
    all_checks: list[Check] = []

    print(f"{'SCENARIO':<10} {'FRAMES':>7} {'PUBLISHED':>10} {'REJECTED':>9} {'ALERTS':>7} {'AUDIO':>6}")
    for name in ALL_SCENARIOS:
        report = run_scenario(name, config)
        print(
            f"{name:<10} {report.frame_count:>7} {len(report.published):>10} {report.rejected_count:>9} "
            f"{len(report.sink.alerts):>7} {len(report.audio.calls):>6}"
        )
        for check_fn in SCENARIO_CHECKS.get(name, []):
            all_checks.append(check_fn(report))

    print()
    print(f"{'REQ':<10} {'RESULT':<6} {'DESCRIPTION':<40} DETAIL")
    all_passed = True
    for check in all_checks:
        result = "PASS" if check.passed else "FAIL"
        all_passed = all_passed and check.passed
        print(f"{check.requirement:<10} {result:<6} {check.description:<40} {check.detail}")

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
