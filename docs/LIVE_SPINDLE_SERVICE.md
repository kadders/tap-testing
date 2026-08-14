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
        ├── rr_model job sync (optional)
        ├── rr_model ArborCTL globals → spindle MQTT + JSONL
        └── /run/tap-spindle/status.json  (tray / monitoring)
```

RRF/DSF should **subscribe only** to summary topics (`analysis`, `status`, `alert`), not `accel/batch`. See [MQTT_TELEMETRY.md](MQTT_TELEMETRY.md) and the wire-format guide [MQTT_PAYLOAD_REFERENCE.md](MQTT_PAYLOAD_REFERENCE.md).

## Idle timeout

If ADXL samples stop arriving for **`MQTT_SESSION_IDLE_TIMEOUT_S`** (default **25**), the service publishes `status=error` with `last_error: idle_timeout` and stops recording so MQTT `end_session` still runs. Jarvis collector idle finalize (default 30s) is the backstop if that stop is lost.

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

RRF MQTT subscribe only shows messages in DWC and does **not** run G-code, so we **do not** subscribe Duet to `tap/#`. Enable the MQTT client for **publishing** machine events to Jarvis:

```gcode
; config.g / dsf-config.g (DSF/RRF 3.6+) — publish only
M586.4 C"milo"
M586 P4 H"mqtt.jarvis.lan" R1883 S1
```

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
  "spindle_load_percent": 23.3
}
```

While recording, each run directory also gets:

- `tool-events.jsonl` — RRF `state.currentTool` timeline (`t_s` aligned with ADXL)
- `spindle-telemetry.jsonl` — ArborCTL Hz / RPM / optional load (`t_s` aligned with ADXL)
- `tools-snapshot.json` — RRF tool table at start
- `run-meta.json` — session / job metadata

Tool and spindle MQTT: see [MQTT_TELEMETRY.md](MQTT_TELEMETRY.md). While recording, poll interval defaults to **`TAP_RRF_TOOL_POLL_S=0.25`**. Spindle values come from RRF object-model globals (ArborCTL daemon owns RS-485).
