# Changelog

Notable changes to this project are documented here.

## Unreleased — 2026-07-18

### Added

- Optional MQTT telemetry for Jarvis, including batched accelerometer samples,
  session lifecycle events, analysis summaries, Modbus rows, ArborCTL spindle
  speed/load, service status, and RRF tool events.
- SBC/DSF detection through the RRF object model so MQTT is enabled on the
  intended DuetPi deployment, with explicit environment overrides for lab use.
- A headless `live_spindle_service` that records continuously or follows RRF
  print-job state, publishes MQTT telemetry, and writes a machine-readable
  status file.
- RRF tool telemetry with a recording-aligned timeline, tool selection and
  offset-change events, tool-table snapshots, and per-run metadata.
- G-code SHA-256 + compact summary (`gcode_summary`) via read-only
  `/rr_download` on live-spindle session start for Jarvis CAM resolution
  (bytes discarded after hashing).
- Session `job_file` enrichment on MQTT start/stop across record, cycle,
  homing, and live-spindle workflows.
- systemd service, environment, and optional tray-unit templates under
  `deploy/systemd/`.
- Documentation for MQTT configuration, topic and payload schemas, SBC
  deployment, service modes, status output, and RRF integration.
- Unit tests for MQTT publishing, the live spindle service, tool telemetry,
  G-code summary, and the expanded RRF HTTP helpers.
- `.editorconfig` / `.gitattributes` to enforce LF line endings (avoids
  `bash\r` shebang failures on the SBC).
- Canonical MQTT wire-format guide: `docs/MQTT_PAYLOAD_REFERENCE.md`
  (session, accel batch, tool v2, analysis, modbus, spindle, status).
- Live spindle service publishes ArborCTL Hz/RPM/load from RRF globals
  (`tap/{device}/spindle`, `spindle-telemetry.jsonl`) while a job is recording.
  H100 load uses the FC4 monitor (`h100Fc4Count`); short clones omit load.
  RPM identity is `120 × Hz / poles` (never FluidNC `rpm*60/10`).

### Changed

- ADXL345 default interface is **SPI** (Mode 3, CE0 / GPIO 8); I2C remains
  available via `TAP_ADXL345_INTERFACE=i2c`. README, wiring docs, and the
  systemd environment template are aligned with that default.
- Tap, streaming, cycle, homing, and standalone analysis workflows can publish
  telemetry when MQTT is configured; local CSV capture remains the primary
  output and continues if MQTT is unavailable.
- Homing recordings can publish synchronized Modbus rows.
- Modbus `t_s` uses the same `recording_t0_mono` origin as ADXL / tool events
  (wall-clock retained as `ts`); CSV notes `# time_basis, recording_monotonic`.
- Cycle analysis publishes its final natural-frequency and RPM guidance.
- RRF HTTP support now includes SBC probing, job/tool context, file download,
  axis and tool table parsing, normalized tool snapshots, and offset comparison
  helpers.
- The package command help and README now list the live spindle service and
  optional MQTT integration.
- Default cycle spacing is 5 s (tests updated to match).
- `rpm_from_sfm` is the exact inverse of `sfm_from_rpm` (no `0.262` approximation).
