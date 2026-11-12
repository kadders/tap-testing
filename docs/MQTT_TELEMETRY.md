# MQTT telemetry (optional Jarvis integration)

Optional publish path for live tap-test data into a Mosquitto broker consumed by the Jarvis **`tap_collector`** service. Enabled when **`TAP_MQTT_HOST`** is set **and** (by default) RRF reports **SBC / DSF mode**. Local CSV recording is unchanged.

## RRF SBC mode (recommended setup)

On a Duet **SBC** layout the Raspberry Pi runs Duet Software Framework (DSF) beside this package. That is the intended home for MQTT telemetry: the Pi already has the ADXL, network, and can reach Jarvis/Mosquitto.

**Detection:** `GET /rr_model?key=sbc&flags=v` — same rule DWC uses (`model.sbc !== null`). Standalone Duet WiFi/Ethernet (no SBC) returns null and MQTT stays off.

```text
[ADXL on Pi] → tap-testing (SBC) → MQTT → Mosquitto → jarvis tap_collector
                 ↑
          RRF/DSF on same Pi (or LAN)
          rr_model.sbc confirms SBC mode
```

### Typical env on the SBC Pi

```bash
# Reach the board’s HTTP API (often localhost when DSF proxies on the Pi)
export TAP_RRF_BASE=http://127.0.0.1
# or http://milo.local

export TAP_MQTT_HOST=mqtt.jarvis.lan
export TAP_MQTT_PORT=1883           # standard MQTT port
export TAP_MQTT_DEVICE_ID=milo
# TAP_MQTT_REQUIRE_SBC=1 is the default — MQTT only if rr_model.sbc is set

pip install paho-mqtt
python -m tap_testing.run_cycle --no-plot
```

Jarvis host:

```bash
docker compose --profile telemetry up -d mosquitto tap_collector
```

### Lab / no-Duet override

```bash
export TAP_MQTT_FORCE=1          # skip SBC probe
# or
export TAP_MQTT_REQUIRE_SBC=0
```

## Topics

All topics use prefix `tap/{device_id}/…`. `device_id` comes from **`TAP_MQTT_DEVICE_ID`** (default: hostname).

| Topic | QoS | When |
|-------|-----|------|
| `tap/{device_id}/session` | 1 | Session start / stop / tap_detected |
| `tap/{device_id}/accel/batch` | 0 | Batched accelerometer windows (~100 ms) |
| `tap/{device_id}/tool` | 1 | Active RRF tool start / change / stop (live spindle) |
| `tap/{device_id}/spindle` | 0 | ArborCTL live Hz / RPM / optional load (job recording) |
| `tap/{device_id}/analysis` | 1 | Analysis summary (`source`: `pi` or `jarvis`) |
| `tap/{device_id}/modbus` | 0 | VFD Modbus rows (when Modbus recording is on) |
| `tap/{device_id}/status` | 1 | Publisher connect / SBC gate / errors |

On connect in SBC mode, a `status` message with `"state": "sbc"` may include `distribution` / `dsf_version` from the object model.

## Payloads (JSON)

**Full field-by-field reference:** [MQTT_PAYLOAD_REFERENCE.md](MQTT_PAYLOAD_REFERENCE.md) (topics, QoS, units, correlation rules, examples).

### Session

One parent `session_id` covers the whole recording (all accel batches, tool events, Modbus, and analysis). Lifecycle keeps the existing `event` field and adds an explicit discriminator:

| `event` | `event_type` | Meaning |
|---------|--------------|---------|
| `start` | `start_session` | Recording began |
| `stop` | `end_session` | Recording ended |
| `tap_detected` | (optional) | Tap/impact marker |

```json
{
  "event": "start",
  "event_type": "start_session",
  "session_id": "uuid",
  "device_id": "milo",
  "mode": "tap|cycle|stream|homing|live_spindle",
  "sample_rate_hz": 800.0,
  "ts": 1710000000.0,
  "job_file": "0:/gcodes/part.gcode",
  "tool_number": 3,
  "tool_name": "1/4 EM",
  "gcode_sha256": "…",
  "gcode_summary": { "tools_used": [3], "move_stats": { "n_g0": 10, "n_g1": 100 } }
}
```

`job_file` is included on **both** `start_session` and `end_session` when a print job is active (from `rr_model` `job.file.fileName` when an RRF client is available). Live-spindle may also attach top-level `gcode_sha256` plus a compact `gcode_summary` (tools, move stats, etc. — without repeating `gcode_sha256` / `job_file`) from a read-only `/rr_download` of that file (bytes are discarded after hashing — Jarvis resolves the CAM NFS share by basename + hash). Stop payloads may include final `n_batches` / `n_samples` / `tool_number`. Status publishes (`tap/.../status`) carry the active `session_id` while recording (`state: recording|idle|error`).

### Accel batch

```json
{
  "session_id": "uuid",
  "seq": 0,
  "t0_s": 0.0,
  "dt_s": 0.00125,
  "ax": [0.01, 0.02],
  "ay": [0.0, 0.01],
  "az": [1.0, 0.99]
}
```

### Tool (active cutter timeline)

Published by `live_spindle_service` on selection changes **and** when the RRF tool table reports offset/state diffs (poll interval `TAP_RRF_TOOL_TABLE_POLL_S`, default **1.0** s; also refreshed immediately on selection change).

`t_s` shares the **same monotonic recording origin** as ADXL batches (`time_basis: recording_monotonic`). Root compatibility fields remain (`event`, `tool_number`, `previous_tool`, `tool_name`, `t_s`, `ts`, job/file position). Schema **v2** adds:

| Field | Role |
|-------|------|
| `schema_version` | `2` |
| `event_id` | Stable id for collector idempotency |
| `event_type` | `tool_selected` / `tool_deselected` / `tool_offset_changed` / `tool_table_snapshot` |
| `tool_snapshot` | Normalized RRF tool (named axis offsets from `move.axes`, probed flags, heaters/fans, compact row) |
| `offsets` | Named-axis mm offsets (+ probed) for the selected tool |

`event` stays `start` | `change` | `stop`. Machine invariant for this shop: **RRF `Tn` maps 1:1 to Fusion `post-process.number`**. Jarvis may auto-resolve GUID, but every segment retains both the labeled RRF slot and the resolved Fusion fields.

```json
{
  "schema_version": 2,
  "event_id": "te-…",
  "event": "change",
  "event_type": "tool_selected",
  "session_id": "20260717_210000",
  "seq": 2,
  "t_s": 83.241,
  "ts": 1710000000.0,
  "time_basis": "recording_monotonic",
  "tool_number": 5,
  "previous_tool": 2,
  "tool_name": "1/4 EM",
  "job_file": "part.gcode",
  "file_position": 18492,
  "offsets": {
    "kind": "rrf_tool_axis",
    "unit": "mm",
    "axes": {"X": 0.0, "Y": 0.0, "Z": -40.12},
    "offsets_probed": true
  },
  "tool_snapshot": {
    "number": 5,
    "name": "1/4 EM",
    "offsets_mm": {"X": 0.0, "Y": 0.0, "Z": -40.12}
  }
}
```

On disk (beside `homing.csv`):

| File | Role |
|------|------|
| `tool-events.jsonl` | Same enriched events as MQTT (authoritative for offline import) |
| `tools-snapshot.json` | Versioned RRF tool table at start/end |
| `run-meta.json` | Session id, job file, sample rate, initial tool |

### Analysis

```json
{
  "session_id": "uuid",
  "source": "pi",
  "fn_hz": 920.5,
  "fn_hz_uncertainty": 2.1,
  "avoid_rpm": [13800, 6900],
  "suggested_rpm_min": 15000,
  "suggested_rpm_max": 18000,
  "n_teeth": 4,
  "sample_rate_hz": 800.0,
  "ts": 1710000000.0
}
```

### Modbus

`t_s` is **recording-relative** (same monotonic origin as ADXL batches and tool events).
`ts` is wall-clock Unix time for the poll.

```json
{
  "session_id": "uuid",
  "t_s": 1.234,
  "ts": 1710000000.0,
  "registers": {"hr_513": 2000, "ir_0": 200}
}
```

H100 register words are always **deci-Hz**. Convert with `RPM = 120 × (raw/10) / poles`.

### Spindle (ArborCTL)

Published by `live_spindle_service` while recording. Reads RRF globals (`arborVFDStatus`, `arborVFDPower`) — does **not** poll Modbus. `t_s` shares the ADXL origin.

```json
{
  "session_id": "uuid",
  "t_s": 12.5,
  "source": "arborctl",
  "hz": 400.0,
  "rpm": 24000.0,
  "poles": 2,
  "commanded_rpm": 24000.0,
  "power_available": true,
  "watts": 350.0,
  "load_percent": 23.3,
  "h100_fc4_count": 13
}
```

H100 load comes from ArborCTL’s FC4 monitor (13 words). Short clones (`h100Fc4Count = 2`) omit `watts` / `load_percent`. See [MQTT_PAYLOAD_REFERENCE.md](MQTT_PAYLOAD_REFERENCE.md#spindle).

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `TAP_MQTT_HOST` | (unset) | Broker hostname; use `mqtt.jarvis.lan` on the Jarvis LAN (`mqtt://` must not be included); unset = MQTT disabled |
| `TAP_MQTT_PORT` | `1883` | Broker port |
| `TAP_MQTT_DEVICE_ID` | hostname | Topic device segment |
| `TAP_MQTT_BATCH_MS` | `100` | Accel batch window (ms) |
| `TAP_MQTT_USERNAME` | | Optional username |
| `TAP_MQTT_PASSWORD` | | Optional password |
| `TAP_MQTT_CLIENT_ID` | `tap-{device_id}` | MQTT client id |
| `TAP_MQTT_REQUIRE_SBC` | `1` | Require `rr_model.sbc` before connecting |
| `TAP_MQTT_FORCE` | `0` | Skip SBC probe (lab / no Duet) |
| `TAP_RRF_BASE` | `http://milo.local` | RRF HTTP base used for SBC probe / tool poll |
| `TAP_RRF_PASSWORD` | | Optional RRF session password |
| `TAP_RRF_POLL_S` | `1.0` | Job-sync poll interval when idle |
| `TAP_RRF_TOOL_POLL_S` | `0.25` | Selection / job poll interval while recording |
| `TAP_RRF_TOOL_TABLE_POLL_S` | `1.0` | Full tool-table refresh while recording (offset diffs) |
| `MQTT_SESSION_IDLE_TIMEOUT_S` | `25` | Edge watchdog: stop recording / emit `end_session` if no ADXL samples for this many seconds (slightly under collector idle so the publisher usually wins) |
| `TAP_JOB_SYNC_STOP_GRACE_S` | `3` | Seconds RRF job must stay inactive before job-sync stop (0 = immediate) |
| `TAP_RRF_DISCONNECT_STOP_S` | `3` | Seconds RRF HTTP may fail while recording before `stop_reason=rrf_disconnect` (0 = immediate) |

## Session idle / fault close

- **Publisher (this package):** `record_stream` idle watchdog (single path). After `MQTT_SESSION_IDLE_TIMEOUT_S` with no samples it sets `stop_reason=idle_timeout`, publishes `status=error`, and stops so `end_session` still runs in `finally`.
- **Collector (Jarvis):** if `end_session` is lost, idle sweep finalizes after `TAP_COLLECTOR_IDLE_TIMEOUT_S` (default 30s), including empty sessions that never received batches.

## Dependency

```bash
pip install paho-mqtt
```

If `paho-mqtt` is missing and `TAP_MQTT_HOST` is set, recording continues; MQTT is skipped with a warning. If SBC mode is required and the board is standalone / unreachable, MQTT is skipped with a warning and CSV recording continues.

## Jarvis collector

See sibling repo `jarvis` → `docs/services/tap-collector.md` (Compose profile `telemetry`). The collector subscribes to **`tap/#`** (this package) and **`duet/#`** (optional RRF macro publishes).

## Duet / RRF MQTT (publish only)

RRF **3.6+** can connect to Mosquitto and publish with **`M118 P6`** (`P6` = MQTT message type; do not use `L6` — `L` is log level 0–3). MQTT **subscribe** only shows text in DWC and does **not** execute G-code. **Do not** subscribe Duet to `tap/#`.

On **SBC**, configure **`M586 P4`** in **`dsf-config.g`** on the Pi (not SD `config.g` / `nxt-user-overrides.g`). See [LIVE_SPINDLE_SERVICE.md](LIVE_SPINDLE_SERVICE.md) and `scripts/diagnose_rrf_mqtt.sh`.

To send machine tool/job events into Jarvis, publish to `duet/{machine}/…` (see jarvis tap-collector docs). Sim jobs from 4th-combinator use `cam/{device}/…`. Live ADXL, the tool timeline, and ArborCTL spindle samples from `live_spindle_service` remain on `tap/{device}/…`.
