# How to Run S3 (Alert Engine) in the Full System

For whoever is putting S1 + S2 + S3 + S4 together on one laptop.

## What S3 does (30 seconds)

S3 listens for posture messages from S2, ignores glitches, scores the risk, picks a tier
(safe / warning / unsafe / alert), speaks a voice warning for unsafe and alert, and sends
the results to S4. It never touches video. **S3 must be running for any alert to happen.**

```
S1 camera -> S2 vision --posture/events--> [ S3 alert engine ] --alert/events--> S4 dashboard
                                                 |--> voice alert on speaker
                                                 |--> local SQLite log (data/events.db)
```

All parts talk through one MQTT broker (Mosquitto). Nobody imports anyone else's code.

## 1. One-time setup (about 5 minutes)

Needs Python 3.11 and Mosquitto.

1. **Install Mosquitto** (Windows): `winget install EclipseFoundation.Mosquitto`
   Make sure the `mosquitto` service is running (Services app, or `Get-Service mosquitto`).
   It must listen on `localhost:1883`.
   - If S2/S4 run on *other* laptops, Mosquitto must also accept remote connections
     (add `listener 1883` and `allow_anonymous true` to `mosquitto.conf`, restart the service),
     and port 1883 must be open in the firewall.
2. **Open a terminal in the `alert-engine` folder** and run:
   ```
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```
   (Do not copy someone else's `.venv` folder. Always create a fresh one.)
3. **Check it works:** `pytest` should report all tests passed. No broker needed for this.
4. **Check the learned model file exists:** `models/risk_model.npz`.
   If missing, create it with `python tools/train_model.py`.
   (Without it S3 still works, using the fixed-rule scoring.)

## 2. Run the system

Always start in this order. Each command in its own terminal, with the venv activated
(`.venv\Scripts\activate`) and the terminal in the `alert-engine` folder.

| Order | What | Command |
|---|---|---|
| 1 | Mosquitto broker | already running as a service |
| 2 | **S3 alert engine** | `python -m alert_engine.main` |
| 3 | Live view of S3 output (optional) | `python tools/monitor.py` |
| 4 | S4 dashboard | start per S4's instructions |
| 5 | S1 camera + S2 vision | start per S1/S2 instructions |

On start, S3 logs `Risk scoring mode: model` (or `rules`) and connects to the broker.
Leave it running. Stop it with Ctrl+C.

## 3. What S2 must send (the contract)

Topic: `posture/events` (QoS 0). One JSON message per detection, about 5-10 per second per person:

```json
{
  "event_id": "uuid string",
  "timestamp": "2026-10-05T10:42:01.123Z",
  "camera_id": "cam-01",
  "person_id": "p-003",
  "posture": "bending",
  "confidence": 0.87,
  "bend_angle_deg": 68.0,
  "zone": "lounge"
}
```

- `posture` must be one of: `sitting`, `standing`, `walking`, `bending`, `lying`, `fall`, `unknown`.
- `confidence` is 0.0 to 1.0. `timestamp` is ISO 8601 in UTC.
- `bend_angle_deg` and `zone` are optional.
- `person_id` must stay the same for the same person; `camera_id` must be stable.
- Send every frame, including normal postures like `sitting`. S3 needs them to clear
  alerts and to know the camera is alive.
- A new posture name must be added to `postures` and `posture_severity` in
  `config/config.yaml` (and the model retrained). Tell Mahdi.

## 4. What S4 gets from S3

| Topic | Contents |
|---|---|
| `alert/events` | Every classified event: tier, risk_score, posture, person, zone, `reasons` (why it fired) |
| `alert/status/cameras` | Each camera `monitored` / `unmonitored` (retained) |
| `alert/audio/devices` | List of available speakers (retained) |
| `alert/audio/select` | S4 -> S3. Publish `{"device_id": 3}` to switch speaker |

Alert fields are documented in the main README (section "MQTT interfaces"). They are stable.
Tell Mahdi before changing any topic or field.

## 5. Test without S1/S2 (for a safe rehearsal)

With S3 running, in another terminal:

```
python tools/simulate_publisher.py --scenario mixed
```

Scenarios: `normal`, `glitch`, `bending`, `fall`, `dropout`, `malformed`, `mixed`.
Watch the result in `python tools/monitor.py`.

Full self-check of requirements R1-R5 (no broker or speaker needed):

```
python tools/run_scenarios.py
```

Expected ending: `ALL CHECKS PASSED`.

## 6. Pick the speaker

```
python tools/select_audio.py --list
python tools/select_audio.py --device-id 3
```

If the chosen device fails, S3 falls back to the OS default speaker.

## 6b. Phone pop-ups (optional, Telegram bot, needs internet)

S3 can send a pop-up to staff phones for unsafe and alert events. It is text only unless camera
photos are turned on (see *Camera photo on alerts* below).

You need the Telegram app and about 3 minutes. Two values are needed: a **bot token** and a **chat ID**.

1. **Make the bot.** In Telegram open **@BotFather** and send `/newbot`. Give it a name, then a
   username ending in `bot`. It replies with a **token** (a long code like `123456789:AAF...`). Copy it.
2. **Choose who gets the alerts.**
   - *Just you:* open a chat with your new bot and send it "hello".
   - *A team:* create a Telegram group (new message icon > New Group), add the staff, then add the
     bot to the group by its username. Send "hello" in the group.
3. **Find the chat ID.** Open this link in a browser, with your token in place of `YOUR_TOKEN`
   (keep the word `bot` before the token, no spaces):
   `https://api.telegram.org/botYOUR_TOKEN/getUpdates`
   Find `"chat":{"id":` in the text and copy the number after it (negative for groups).
   If the result is empty (`"result":[]`), send another message and refresh.
4. **Give both values to S3.** In the `alert-engine` folder copy `.env.example` to `.env` and fill in:
   ```
   TELEGRAM_BOT_TOKEN=your token
   TELEGRAM_CHAT_ID=your chat id
   ```
   No quotes or spaces. Never commit or share `.env`: anyone with the token can use the bot.
5. **Restart S3** (`python -m alert_engine.main`). Test with
   `python tools/simulate_publisher.py --scenario fall`. A pop-up should reach the phone within seconds.
   Without these two values the feature stays off and nothing else changes.

If Mahdi has sent you a ready-made `.env` file, put it in the `alert-engine` folder, skip steps 1-4
and do step 5. Never commit it.

The laptop running S3 needs internet for this. If nothing arrives, the S3 log shows
`Phone notification failed` with the reason. Each person must have pressed **Start** in the bot chat
(or be in the group) to receive messages.

By default only emergencies send a message: tier `alert` AND posture `fall` or `lying`
(set under `notify.tiers` and `notify.postures` in `config/config.yaml`; `postures: []` means any).
The same per-person cooldown applies, so phones are not spammed.

### Camera photo on alerts (off by default)

With `notify.attach_frame: true`, an alert arrives as one photo with a short caption. This sends
resident images to Telegram's servers, so R7 must be revised before turning it on.

Until the camera is hooked up, S3 looks for the latest frame at `data/frames/<camera_id>.jpg`, for
example `data/frames/cam-01.jpg`. Whatever supplies frames should write to a temporary file and then
rename it, so S3 never reads a half-written picture. On Windows that rename fails while S3 is reading
the file, so the feed must catch `PermissionError` and try again on its next frame.

If the frame is missing, more than `notify.frame_max_age_s` seconds from the event, or fails to
upload, the alert is sent as text instead. `data/frames/` is gitignored: never commit resident images.

To test without a camera, run these in separate terminals with S3 already running. The first one
keeps a labelled dummy picture fresh, and the second triggers one alert:
```
python tools/dummy_frames.py --cameras cam-04
python tools/simulate_publisher.py --scenario fall
```
The S3 log then shows `Phone alert sent with photo`, or `sent as text` if it fell back.

## 6c. Activity history

S3 keeps a history of what each resident did in the `activity_log` table of `data/events.db`. A row is
added only when a confirmed posture or tier changes, so one-frame glitches never appear. To see it:
```
python tools/activity_report.py
python tools/activity_report.py --person p-003 --hours 24
```
Each person gets a timeline and totals per posture. An `unmonitored` row means nobody was watching:
either their camera dropped out or S3 was shut down. If S3 crashes rather than being stopped with
Ctrl+C, no `unmonitored` row is written, so the last activity before the crash will look too long.

## 7. Settings you may need to change (`config/config.yaml`)

- `mqtt.host`: set to the broker laptop's IP if S3 is not on the same machine as Mosquitto.
- `ml.enabled`: `false` = fixed rules only, `true` = trained model (rules are the fallback).
- `debounce`, `tiers`, `resident_profiles`: sensitivity and risk levels.
- `audio.messages`: the spoken warning text.

## 8. If something is wrong

| Symptom | Check |
|---|---|
| S3 says it could not connect to the broker | Mosquitto not running, or wrong `mqtt.host`/port |
| S2 is sending but nothing appears | Open `logs/rejected.jsonl`: it shows each bad message and the reason (wrong posture name, missing field, bad timestamp) |
| Camera shows `unmonitored` | No valid messages from that `camera_id` for 5 s |
| No voice alert | Wrong speaker: run `select_audio.py --list`. Voice files are rebuilt on first start in `assets/audio/` |
| Alert seems late | Check `logs/metrics.jsonl` (detected / received / confirmed / dispatched times) |
| No alert for a brief posture | By design: a posture needs 3 good detections (confidence >= 0.6) within 1.5 s |

## 9. Notes for the review

- The risk model is trained on **synthetic placeholder data**; it is not yet validated on
  real residents. Retrain with real labelled data: `python tools/train_model.py --csv labelled.csv`.
- Only event data leaves the vision side by default. S3 handles images only if
  `notify.attach_frame` is turned on (see section 6b and R7).
- Firebase is not built yet. S3 stores locally in `data/events.db` and publishes over MQTT.

Questions about S3: Mahdi.
