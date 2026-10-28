# S3 Alert Engine

Subsystem 3 of the RBB4013 SIDP elderly posture detection system. Validates,
debounces, scores and classifies posture events from the vision model (S2),
triggers local audio alerts, and publishes classified events for the
dashboard (S4). See `../CLAUDE.md` for the full build brief.

This phase runs fully offline: MQTT is local, storage is SQLite, audio is
local TTS + speakers. No cloud services, no Firebase (that's a later phase).

## Setup

Requires Python 3.11 and a local Mosquitto broker on `localhost:1883`.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .                # makes `python -m alert_engine.main` resolve the package
```

Install Mosquitto if you don't have it:

```powershell
winget install EclipseFoundation.Mosquitto
```

Copy `.env.example` to `.env` if/when the Firebase phase is enabled. Nothing
is required in `.env` for this phase.

## Running

Start the engine (subscribes to `posture/events`, publishes to
`alert/events`, `alert/status/cameras`, `alert/audio/devices`):

```bash
python -m alert_engine.main
```

In another terminal, feed it fake posture events:

```bash
python tools/simulate_publisher.py --scenario fall
```

Watch live alerts and camera status:

```bash
python tools/monitor.py
```

List speakers and switch the output device at runtime:

```bash
python tools/select_audio.py --list
python tools/select_audio.py --device-id 3
```

Run every scenario and check pass/fail against the requirements in
CLAUDE.md section 12, without needing a broker, TTS or a speaker (runs the
same pipeline in-process against recording fakes and a synthetic clock, so
an 8 s dropout scenario finishes instantly):

```bash
python tools/run_scenarios.py
```

## Tests

```bash
pytest
```

Unit tests never need MQTT, a speaker, or real TTS — every stage takes
injected ports (sink, audio, publish callback, clock) so it can be tested
with plain Python objects. See `tests/`.

## Architecture

```
S2 vision  ->  MQTT posture/events  ->  Subscriber  ->  Validator (pydantic)
   ->  Debounce tracker (per person)  ->  Risk scorer  ->  Tier classifier
   ->  Dispatcher  ->  Event sink (local SQLite now, Firebase later)
                   ->  Audio module (unsafe and alert only)
                   ->  MQTT alert/events (for anyone else listening)
Camera watchdog runs alongside and flags cameras with no events for 5 s.
```

Each stage is its own module with an injectable interface and no direct
MQTT/Firebase/hardware dependency, so it's testable with plain Python
objects:

| Module | Responsibility |
|---|---|
| `config.py` | Loads and validates `config/config.yaml` |
| `schemas.py` | Pydantic input/output message models |
| `mqtt_client.py` | paho-mqtt v2 wrapper with reconnect |
| `debounce.py` | Per-person state machine (section 8.2) |
| `scoring.py` | Risk score + tier classification (section 8.3) |
| `dispatcher.py` | Routes classified events by tier (section 8.4) |
| `audio.py` | TTS-to-WAV, device selection, playback queue (section 8.5) |
| `sinks.py` | `EventSink` interface + `LocalSqliteSink` (section 8.6) |
| `watchdog.py` | Camera dropout detection (section 8.7) |
| `metrics.py` | Alert latency logging for R1 (section 8.8) |
| `main.py` | Wires everything together |

Audio playback and SQLite writes each run on their own worker thread behind
a queue, so the dispatcher never blocks on I/O.

### A note on camera dropout and debounce

When `watchdog.py` flags a camera `unmonitored`, `main.py` resets debounce
state for every resident last seen on that camera
(`DebounceTracker.reset_person`). Without this, a person's `duration_s`
would silently span the outage once the camera recovers — e.g. a resident
walking normally for 3 s, then a real 8 s camera dropout, would otherwise
report `duration_s ≈ 11 s` on the first event after recovery, enough to
spuriously cross into the "warning" tier for a completely safe posture. The
reset forces the person to re-earn confirmation from scratch after any
outage.

## Config

All tunables live in `config/config.yaml`. Notable additions beyond
CLAUDE.md's starting values:

- `debounce.safe_postures`: which postures count as "safe" for the
  `resolved` debounce state (defaults to `sitting`, `standing`, `walking`).
  Not in the original sample config; needed so the debounce module doesn't
  have to hard-code posture semantics.

## MQTT interfaces

See CLAUDE.md section 7. `alert/status/cameras` publishes a single retained
JSON object keyed by `camera_id` (not per-camera subtopics), e.g.:

```json
{"cam-01": {"status": "monitored", "last_seen": "2026-10-05T10:42:01.123Z"}}
```

## Requirements status (this phase)

| ID | Requirement | Status |
|---|---|---|
| R1 | Alert within 2 s of confirmation | Met — dispatch is synchronous once confirmed; verified via `run_scenarios.py` |
| R2 | Single-frame misreads trigger 0 alerts | Met — debounce window absorbs single glitch frames |
| R3 | False alerts on normal activity < 10% | Met (0% on the `normal` scenario) |
| R4 | Every valid event classified into a tier | Met for confirmed postures; raw frames within the debounce window are intentionally not yet classified |
| R5 | Malformed events rejected, 100%, no crash | Met — see `logs/rejected.jsonl` |
| R6 | No alerts lost when internet drops | Deferred to the Firebase phase (out of scope now) |
| R7 | No video/images leave the laptop | Met by construction — only structured posture events are handled |

## Known limitations / still to confirm

- The final posture list, exact S2 event fields, Firebase DB structure, demo
  laptop OS, and final R1–R7 targets are still being confirmed with the team
  (CLAUDE.md section 16).
- This dev environment didn't have Mosquitto installed; MQTT wiring is
  implemented per section 7.1 and exercised by `simulate_publisher.py` +
  `monitor.py`, but hasn't been run live end-to-end against a broker here.
  `run_scenarios.py` verifies the pipeline logic without needing one.

## Learned risk scoring (supervisor feedback)

Decisions are driven by a trained model, with the original rules kept alongside.

- `src/alert_engine/ml_scoring.py`: softmax-regression model (numpy only) predicting the tier from posture, hold duration, confidence, bend angle and resident risk. `HybridScorer` uses it when `models/risk_model.npz` exists, otherwise falls back to the rule-based `scoring.py`. Falls are always `alert`. Each alert's `reasons` lists the model's top drivers and the rule-based result for comparison.
- Train: `python tools/train_model.py` (synthetic placeholder data) or `python tools/train_model.py --csv labelled.csv` (real labelled events from S2; columns `posture,duration_s,confidence,bend_angle_deg,resident_multiplier,tier`).
- Switch off: set `ml.enabled: false` in `config/config.yaml`.
- The shipped model is trained on synthetic data and must be retrained on real labelled data before any accuracy claim. Output message format (section 7.3) is unchanged.
