# MQTT payload reference (tap-testing → Jarvis)

Canonical wire-format for topics under `tap/{device_id}/…`. Setup and environment variables live in [MQTT_TELEMETRY.md](MQTT_TELEMETRY.md); this page documents **what each message contains**.

Publisher: `tap_testing.mqtt_telemetry.MqttTelemetryPublisher` (and `tool_telemetry` / `spindle_telemetry` for live spindle).  
Collector: Jarvis `tap_collector` (subscribe `tap/#`).

## Shared rules

| Concept | Rule |
|---------|------|
| Topic prefix | `tap/{device_id}/…` — `device_id` from `TAP_MQTT_DEVICE_ID` (default hostname) |
| Encoding | UTF-8 JSON, compact (`separators=(",", ":")`) |
| `session_id` | One id per recording session; all accel batches, tool events, Modbus rows, and analysis for that run share it |
| `t_s` | Seconds since recording start (`time.monotonic()` origin). Same basis for ADXL CSV, tool events, Modbus, and spindle samples |
| `ts` | Unix wall-clock seconds (`time.time()`) when the message was built |
| `seq` | Per-session counter for accel batches and tool events (starts at 0) |
| Fail-open | Missing `paho-mqtt`, SBC gate failure, or publish errors never block CSV capture; publishes may be dropped |
| QoS | Session / tool / analysis / status → **1**; accel batch / modbus / spindle → **0** |

```text
session (start) ──► accel/batch* ──► tool* ──► spindle* ──► modbus* ──► session (stop)
                         │
                         └── analysis (after cycle / analyze CLI)
```

## Topics overview

| Topic leaf | QoS | Publisher | When |
|------------|-----|-----------|------|
| `session` | 1 | `MqttTelemetryPublisher` | Start / stop / tap detected |
| `accel/batch` | 0 | `MqttTelemetryPublisher.emit_sample` | ~every `TAP_MQTT_BATCH_MS` (default 100 ms) while recording |
| `tool` | 1 | `ToolEventRecorder` via MQTT | Live spindle: selection, offsets, snapshots |
| `analysis` | 1 | `analyze` / `run_cycle` | After FFT / RPM guidance |
| `modbus` | 0 | Homing GUI Modbus poll | One row per successful poll |
| `spindle` | 0 | `live_spindle_service` via MQTT | ArborCTL Hz / RPM / optional load while a job is recording |
| `status` | 1 | Publisher lifecycle | Connect, SBC gate, recording, idle, error |

---

## `session`

**Topic:** `tap/{device_id}/session`

### `start_session` (`event: start`)

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `event` | string | yes | `"start"` |
| `event_type` | string | yes | `"start_session"` |
| `session_id` | string | yes | UUID or timestamp id (live spindle uses `YYYYmmdd_HHMMSS`) |
| `device_id` | string | yes | Topic device segment |
| `mode` | string | yes | `tap` \| `cycle` \| `stream` \| `homing` \| `live_spindle` |
| `sample_rate_hz` | float | yes | Configured ADXL rate |
| `ts` | float | yes | Wall clock |
| `job_file` | string | if known | RRF `job.file.fileName` |
| `tool_number` | int | if known | RRF `state.currentTool` (−1 → often omitted or null) |
| `tool_name` | string | if known | From RRF tools table |
| `gcode_sha256` | string | live spindle | Hex digest of `/rr_download` bytes |
| `gcode_summary` | object | live spindle | Compact summary **without** repeating `gcode_sha256` / `job_file` |

```json
{
  "event": "start",
  "event_type": "start_session",
  "session_id": "20260718_213000",
  "device_id": "milo-sbc",
  "mode": "live_spindle",
  "sample_rate_hz": 800.0,
  "ts": 1710000000.0,
  "job_file": "0:/gcodes/part.gcode",
  "tool_number": 3,
  "tool_name": "1/4 EM",
  "gcode_sha256": "a1b2c3…",
  "gcode_summary": {
    "source": "rrf_download",
    "gcode_resolution_status": "rrf_download",
    "n_lines": 12040,
    "n_bytes": 482193,
    "tools_used": [3, 5],
    "tool_change_lines": [{"line": 120, "byte_offset": 4102, "tool_number": 3}],
    "spindle_commands": [{"cmd": "M3", "s": 18000.0, "count": 2}],
    "feed_commands": [{"f": 1200.0, "count": 40}],
    "move_stats": {"n_g0": 210, "n_g1": 9800},
    "operations": [";OPERATION: Contour1"]
  }
}
```

### `end_session` (`event: stop`)

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `event` | string | yes | `"stop"` |
| `event_type` | string | yes | `"end_session"` |
| `session_id` | string | yes | Same as start |
| `device_id` | string | yes | |
| `mode` | string | yes | |
| `sample_rate_hz` | float | yes | |
| `ts` | float | yes | |
| `n_batches` | int | yes | Accel batch count (or override) |
| `n_samples` | int | optional | If caller provides |
| `tool_number` | int | optional | Final tool |
| `job_file` | string | if known | From start cache or explicit arg |

### `tap_detected`

| Field | Type | Notes |
|-------|------|-------|
| `event` | string | `"tap_detected"` |
| `session_id` | string | |
| `mode` | string | |
| `sample_rate_hz` | float | |
| `t_s` | float | Recording-relative impact time |
| `ts` | float | Wall clock |

---

## `accel/batch`

**Topic:** `tap/{device_id}/accel/batch` · QoS 0

Batched windows of accelerometer samples (g). One batch is flushed when the time span ≥ `TAP_MQTT_BATCH_MS` or the buffer reaches `round(batch_ms/1000 * sample_rate_hz)` samples.

| Field | Type | Notes |
|-------|------|-------|
| `session_id` | string | Active session |
| `seq` | int | 0-based batch index within the session |
| `t0_s` | float | Recording-relative time of first sample in batch |
| `dt_s` | float | Mean sample interval (s); ≈ `1/sample_rate_hz` |
| `ax`, `ay`, `az` | float[] | Acceleration in **g**; same length |

```json
{
  "session_id": "20260718_213000",
  "seq": 0,
  "t0_s": 0.0,
  "dt_s": 0.00125,
  "ax": [0.01, 0.02],
  "ay": [0.0, 0.01],
  "az": [1.0, 0.99]
}
```

Correlate sample *i* to recording time: `t_s ≈ t0_s + i * dt_s`.

---

## `tool`

**Topic:** `tap/{device_id}/tool` · QoS 1 · Schema **v2**

Published by `live_spindle_service` / `ToolEventRecorder`. `t_s` uses the **same monotonic origin** as ADXL.

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | int | `2` |
| `event_id` | string | Stable id for collector idempotency |
| `event` | string | `start` \| `change` \| `stop` (compat) |
| `event_type` | string | `tool_selected` \| `tool_deselected` \| `tool_offset_changed` \| `tool_table_snapshot` |
| `session_id` | string | |
| `seq` | int | |
| `t_s` | float | Recording-relative |
| `ts` | float | Wall clock |
| `time_basis` | string | `"recording_monotonic"` |
| `tool_number` | int \| null | RRF slot (−1 / null = none) |
| `previous_tool` | int \| null | |
| `tool_name` | string \| null | |
| `job_file` | string \| null | |
| `file_position` | int \| null | RRF byte offset into G-code |
| `rrf_slot` | int \| null | Same as tool_number (explicit for Fusion mapping) |
| `offsets` | object | Named-axis mm offsets (see below) |
| `tool_snapshot` | object | Normalized RRF tool row |
| `previous_tool_snapshot` | object | On offset / selection changes |
| `tools` | array | Full table on snapshot events |

**Machine invariant:** RRF `Tn` maps 1:1 to Fusion `post-process.number`.

```json
{
  "schema_version": 2,
  "event_id": "tevt-abc123def456",
  "event": "change",
  "event_type": "tool_selected",
  "session_id": "20260718_213000",
  "seq": 2,
  "t_s": 83.241,
  "ts": 1710000000.0,
  "time_basis": "recording_monotonic",
  "tool_number": 5,
  "previous_tool": 2,
  "tool_name": "1/4 EM",
  "job_file": "part.gcode",
  "file_position": 18492,
  "rrf_slot": 5,
  "offsets": {
    "kind": "rrf_tool_axis",
    "unit": "mm",
    "axis_letters": ["X", "Y", "Z"],
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

On-disk companions (beside `homing.csv`): `tool-events.jsonl`, `tools-snapshot.json`, `run-meta.json`.

---

## `analysis`

**Topic:** `tap/{device_id}/analysis` · QoS 1

Published after `run_cycle` or the `analyze` CLI when MQTT is enabled.

| Field | Type | Notes |
|-------|------|-------|
| `session_id` | string | Cycle/run id or CSV stem |
| `source` | string | `"pi"` (on-device) or `"jarvis"` |
| `fn_hz` | float | Natural frequency |
| `fn_hz_uncertainty` | float \| null | Measurement uncertainty |
| `avoid_rpm` | float[] | Tooth-pass resonance RPMs |
| `suggested_rpm_min` | float | Stable pocket low |
| `suggested_rpm_max` | float | Stable pocket high |
| `n_teeth` | int | Flutes used |
| `sample_rate_hz` | float | |
| `ts` | float | Wall clock |

```json
{
  "session_id": "20260718_120000",
  "source": "pi",
  "fn_hz": 920.5,
  "fn_hz_uncertainty": 2.1,
  "avoid_rpm": [13800.0, 6900.0],
  "suggested_rpm_min": 15000.0,
  "suggested_rpm_max": 18000.0,
  "n_teeth": 4,
  "sample_rate_hz": 800.0,
  "ts": 1710000000.0
}
```

---

## `modbus`

**Topic:** `tap/{device_id}/modbus` · QoS 0

| Field | Type | Notes |
|-------|------|-------|
| `session_id` | string | |
| `t_s` | float | Recording-relative (aligned with ADXL) |
| `ts` | float | Wall-clock poll time |
| `registers` | object | Flat map (`hr_*`, `ir_*`, `di_*`, `co_*`); no `t_s`/`ts` keys |

```json
{
  "session_id": "20260718_213000",
  "t_s": 1.234,
  "ts": 1710000000.0,
  "registers": {"hr_513": 2000, "ir_0": 200}
}
```

H100 `hr_513` / `ir_0` / `ir_1` are **always deci-Hz** (value × 10). Convert with `RPM = 120 × (raw/10) / poles`. Do not treat raw `< 1000` as Hz.

---

## `spindle`

**Topic:** `tap/{device_id}/spindle` · QoS 0

Decoded ArborCTL object-model sample (not raw Modbus). `t_s` uses the same recording-monotonic origin as ADXL.

Identity: **`RPM = 120 × Hz / poles`**. Live H100 Hz/RPM are taken from `arborVFDStatus` as-is; `poles` comes from `arborMotorSpec`. Do not re-scale with FluidNC `rpm*60/10`.

| Field | Type | Notes |
|-------|------|-------|
| `session_id` | string | |
| `t_s` | float | Recording-relative |
| `ts` | float | Wall clock |
| `source` | string | `arborctl` (observed) or `rrf_spindle` (reference fallback) |
| `spindle_index` | int | RRF spindle slot |
| `hz` | float | Electrical output frequency |
| `rpm` | float | Mechanical RPM from ArborCTL |
| `poles` | float | Nameplate poles |
| `commanded_rpm` | float | RRF `spindles[S].active` when present |
| `commanded_hz` | float | `commanded_rpm × poles / 120` |
| `running` / `dir` / `stable` / `comm_ready` | bool/int | Status vector |
| `power_available` | bool | H100: `h100Fc4Count > 2`; other drivers: power vector present |
| `watts` / `load_percent` | float | Only when `power_available` (idle `0` is real) |
| `h100_fc4_count` | int | Diagnostic; 13 = long monitor, 2 = short clone |
| `fault` | bool | `arborState[S][4]` when present |

```json
{
  "schema_version": 1,
  "session_id": "20260718_213000",
  "t_s": 12.5,
  "ts": 1710000000.0,
  "source": "arborctl",
  "spindle_index": 0,
  "hz": 400.0,
  "rpm": 24000.0,
  "poles": 2,
  "commanded_rpm": 24000.0,
  "commanded_hz": 400.0,
  "running": true,
  "dir": 1,
  "stable": true,
  "comm_ready": true,
  "power_available": true,
  "watts": 350.0,
  "load_percent": 23.3,
  "h100_fc4_count": 13,
  "time_basis": "recording_monotonic"
}
```

On disk (beside `homing.csv`): `spindle-telemetry.jsonl` — same objects as MQTT.

---

## `status`

**Topic:** `tap/{device_id}/status` · QoS 1

Lifecycle / health frames. Common `state` values:

| `state` | Meaning |
|---------|---------|
| `connected` | MQTT client connected |
| `disconnected` | Client closing |
| `sbc` | SBC gate passed (`distribution`, `dsf_version` may be present) |
| `recording` | Session active |
| `idle` | Session ended |
| `error` | Capture or service error (`last_error`) |

Always includes `device_id`, `ts`; may include `session_id`, `mode`, `job_file`, `tool_number`, `sample_rate_hz`, `n_batches`.

```json
{
  "state": "recording",
  "device_id": "milo-sbc",
  "ts": 1710000000.0,
  "session_id": "20260718_213000",
  "mode": "live_spindle",
  "sample_rate_hz": 800.0,
  "job_file": "0:/gcodes/part.gcode",
  "tool_number": 3
}
```

---

## Correlation cheat-sheet

1. Join all streams on **`session_id`**.
2. Align ADXL, tools, spindle, and Modbus on **`t_s`** (recording monotonic).
3. Use **`ts`** only for wall-clock / cross-host ordering.
4. Resolve CAM files with **`gcode_sha256`** + basename of **`job_file`**.
5. Segment vibration by tool with **`tool`** events (`t_s` cut points).
6. Prefer on-disk `tool-events.jsonl` / `spindle-telemetry.jsonl` if MQTT batches were dropped (QoS 0).
7. Prefer ArborCTL `spindle` RPM/load over Fusion / RRF `spindleRpm` static values.
