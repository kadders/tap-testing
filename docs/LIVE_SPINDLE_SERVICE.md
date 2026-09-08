# Live spindle service (RRF SBC → MQTT)

Headless daemon that streams ADXL345 data using the same live-spindle / `record_stream` path as `homing_gui`, publishes MQTT when SBC mode is active, and can start/stop with RRF print jobs.

## Architecture

```text
systemd tap-spindle.service
        │
        ▼
live_spindle_service ──record_stream──► CSV under data/live_spindle/service/
        │                      └── MQTT batches (SBC-gated)
        │
        ├── optional ustreamer → ffmpeg → session.mp4 (+ YouTube RTMP tee)
        ├── rr_model job sync (optional)
        ├── rr_model ArborCTL globals → spindle MQTT + JSONL
        └── /run/tap-spindle/status.json  (tray / monitoring)
```

Optional session video (`TAP_VIDEO_ENABLED=1`): ffmpeg ingests ustreamer MJPEG, burns in live telemetry aligned to `recording_t0_mono`, and optionally tees to YouTube when preflight passes. See **[VIDEO_RECORDING.md](VIDEO_RECORDING.md)**.

RRF/DSF should **subscribe only** to summary topics (`analysis`, `status`, `alert`), not `accel/batch`. See [MQTT_TELEMETRY.md](MQTT_TELEMETRY.md) and the wire-format guide [MQTT_PAYLOAD_REFERENCE.md](MQTT_PAYLOAD_REFERENCE.md).

## Idle timeout

If ADXL samples stop arriving for **`MQTT_SESSION_IDLE_TIMEOUT_S`** (default **25**), the `record_stream` idle watchdog stops the worker, sets **`stop_reason: idle_timeout`**, publishes `status=error`, and runs MQTT `end_session` in `finally`. Jarvis collector idle finalize (default 30s) is the backstop if that stop is lost.

Diagnose the last stop:

```bash
bash scripts/diagnose_recording_stop.sh
bash scripts/diagnose_recording_stop.sh 20260819_161616
```

## Job-sync stop grace

While recording, stop when RRF reports **`completed`**, **`idle`**, no loaded file, or any non-active status (including **`busy`** after `stop.g` — file name may still be in the object model). Grace: **`TAP_JOB_SYNC_STOP_GRACE_S`** (default **3** seconds). Set to `0` for immediate stop.

If RRF HTTP poll fails while recording (e.g. e-stop power cycle, board unreachable), stop after **`TAP_RRF_DISCONNECT_STOP_S`** (default **3** seconds) with `stop_reason: rrf_disconnect`. Successful poll clears the timer.

## Stop reasons

`/run/tap-spindle/status.json` includes **`stop_reason`** on the last stop:

| Value | Meaning |
|-------|---------|
| `idle_timeout` | No ADXL samples for `MQTT_SESSION_IDLE_TIMEOUT_S` |
| `job_completed` | RRF `state.status` is `completed` |
| `job_no_file` | No file in RRF job object model |
| `job_sync_inactive` | Other non-active job state (e.g. `idle`, `busy` after job done) |
| `rrf_halted` | RRF `state.status` is `halted` (e-stop / fault) |
| `rrf_shutdown` | RRF `state.status` is `off` or `shutdown` |
| `rrf_disconnect` | RRF HTTP unreachable for `TAP_RRF_DISCONNECT_STOP_S` while recording |
| `worker_exception` | ADXL worker crashed |
| `service_shutdown` | systemd stop / service exit |
| `manual` | Other `stop_recording()` call |

MQTT `end_session` and error status payloads may include the same `stop_reason` field.

## Install on DuetPi (this repo at `/mnt/repos/tap-testing`)

The shared NFS checkout’s `.venv` is often unusable on the Pi (no `+x`, wrong arch). The install script creates a **Pi-local** venv at `~/.venvs/tap-testing`.

```bash
sudo bash /mnt/repos/tap-testing/scripts/install_live_spindle_service.sh
sudoedit /etc/default/tap-spindle   # verify TAP_MQTT_HOST=mqtt.jarvis.lan
sudo systemctl restart tap-spindle
journalctl -u tap-spindle -f
cat /run/tap-spindle/status.json
```

## Modes

| Flag / env | Behavior |
|------------|----------|
| `--job-sync` / `TAP_LIVE_JOB_SYNC=1` (default) | Record while RRF job is active |
| `--always-on` / `TAP_LIVE_ALWAYS_ON=1` | Continuous recording from start |
| `--idle` | No auto recording (useful with `--tray` only) |
| `--tray` / `TAP_LIVE_TRAY=1` | System tray when `DISPLAY`/`WAYLAND_DISPLAY` set |
| `--video-test` / `--video-test-s 30` | One-shot video with live RRF/ADXL overlay (optional YouTube) then exit |

## Manual run

```bash
cd /mnt/repos/tap-testing
source .venv/bin/activate
export TAP_RRF_BASE=http://127.0.0.1
export TAP_MQTT_HOST=mqtt.jarvis.lan
export TAP_MQTT_PORT=1883
python -m tap_testing.live_spindle_service --job-sync
```

## Optional tray (desktop session)

```bash
# user unit (after logging into a GUI session)
mkdir -p ~/.config/systemd/user
cp deploy/systemd/tap-spindle-tray.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now tap-spindle-tray.service
```

Prefer the **system** `tap-spindle` unit for capture; use the tray unit only as a status indicator, or run one process with `--job-sync --tray` under a graphical session.

## RRF MQTT (publish only — do not subscribe to tap/#)

RRF MQTT subscribe only shows messages in DWC and does **not** run G-code, so we **do not** subscribe Duet to `tap/#`. Enable the MQTT client for **publishing** machine events to Jarvis.

**SBC (Milo):** put MQTT in **`/opt/dsf/sd/sys/dsf-config.g`** on the Pi — **not** in SD `config.g` or `nxt-user-overrides.g` (NeXT skips `M586` in SBC mode; overrides run on the board before WiFi is reliable). After boot, if sim jobs report `M118: MQTT client is not connected`, run `bash scripts/ensure_rrf_mqtt.sh` or regenerate sim gcode (4th-combinator embeds an `M586` reconnect preamble).

```gcode
; dsf-config.g (SBC Pi) — publish only
G4 S20
M586.4 C"milo"
M586 P4 H"mqtt.jarvis.lan" R1883 S1
G4 S3
```

Diagnostics: `bash scripts/diagnose_rrf_mqtt.sh [sim.gcode]`

Optional: publish tool/job events from macros (`tpost.g`, etc.) to `duet/{machine}/…` for the Jarvis collector (see jarvis `docs/services/tap-collector.md`). Live ADXL + tool timeline from this service still go to `tap/{device}/…`.

## Status JSON

Example `/run/tap-spindle/status.json`:

```json
{
  "state": "recording",
  "recording": true,
  "mqtt_enabled": true,
  "rrf_connected": true,
  "rrf_job_active": true,
  "rrf_status": "processing",
  "current_tool": 3,
  "job_file": "0:/gcodes/part.gcode",
  "sample_rate_hz": 800.0,
  "output_path": ".../homing.csv",
  "spindle_rpm": 24000.0,
  "spindle_load_percent": 23.3,
  "video_recording": true,
  "video_path": ".../session.mp4",
  "youtube_live": false,
  "video_stream_remote": false
}
```

When `TAP_VIDEO_MODE=remote`, `video_stream_remote` is true during active jobs and
YouTube RTMP is handled by the cluster `tap-stream` service
([VIDEO_STREAMING.md](VIDEO_STREAMING.md)).

While recording, each run directory also gets:

- `tool-events.jsonl` — RRF `state.currentTool` timeline (`t_s` aligned with ADXL)
- `spindle-telemetry.jsonl` — ArborCTL Hz / RPM / optional load (`t_s` aligned with ADXL)
- `tools-snapshot.json` — RRF tool table at start
- `run-meta.json` — session / job metadata
- `session.mp4`, `video-meta.json` — optional job video when `TAP_VIDEO_ENABLED=1`
  (or under `TAP_VIDEO_DIR/{session_id}/` if set; `run-meta.json` records `video_path`)
  ([VIDEO_RECORDING.md](VIDEO_RECORDING.md))

Tool and spindle MQTT: see [MQTT_TELEMETRY.md](MQTT_TELEMETRY.md). While recording, poll interval defaults to **`TAP_RRF_TOOL_POLL_S=0.25`**. Spindle values come from RRF object-model globals (ArborCTL daemon owns RS-485).
