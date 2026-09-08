# Changelog

Notable changes to this project are documented here.

## Unreleased — 2026-07-18

### Added

- **Remote YouTube streaming** (`TAP_VIDEO_MODE=remote`): Pi publishes
  `tap/{device}/video_overlay` over MQTT; cluster `tap-stream` (Jarvis k8s)
  pulls Pi MJPEG and encodes to YouTube RTMP. See `docs/VIDEO_STREAMING.md`.
- ustreamer launcher template without `--slowdown` (`deploy/ustreamer/`).
- Source validation script: `scripts/check_ustreamer_source.sh`.
- Cluster manual test script: `scripts/cluster_youtube_stream_test.sh`.
- Status JSON field: `video_stream_remote`.
- MQTT topic `video_overlay` documented in `docs/MQTT_PAYLOAD_REFERENCE.md`.
- Recording **stop_reason** in status JSON, MQTT `end_session`, and
  `scripts/diagnose_recording_stop.sh`.
- Job-sync stop grace (`TAP_JOB_SYNC_STOP_GRACE_S`, default 3s) and stop on
  `completed` / cleared file / non-active status (no longer treats `busy`+file
  as still recording after job end).
- Single ADXL idle watchdog in `record_stream` (removed duplicate RRF-poll path).
- Local job video uses wall-clock PTS + CFR at `TAP_VIDEO_FPS` (default 15) and
  optional `TAP_VIDEO_MAX_WIDTH` (default 640) so session duration matches the job
  instead of collapsing into a short sped-up clip when the Pi falls behind.
- Gentler ffmpeg shutdown: `q` then wait before SIGINT; clearer SIGBUS log hinting
  at `TAP_VIDEO_ENCODER=libx264`.
- RRF sim MQTT diagnostics: `scripts/diagnose_rrf_mqtt.sh`, `scripts/ensure_rrf_mqtt.sh`;
  docs clarify SBC `M586 P4` belongs in `dsf-config.g` (not `nxt-user-overrides.g`) and
  `M118 P6` (not `L6`) for MQTT publish.

### Added (earlier)
- Optional session video for `live_spindle_service`: ustreamer MJPEG → ffmpeg →
  `session.mp4` per job with live telemetry overlay aligned to
  `recording_t0_mono`.
- Optional YouTube Live RTMP tee (gated preflight: requires `TAP_VIDEO_ENABLED`,
  `TAP_YOUTUBE_ENABLED`, and `TAP_YOUTUBE_STREAM_KEY`; skips with logged reason
  when any setting is missing). Default ingest: `rtmp://a.rtmp.youtube.com/live2`.
- `tap_testing/video_recording.py`, unit tests, and `docs/VIDEO_RECORDING.md`
  (YouTube setup steps, daemon env, preflight skip reasons).
- Status JSON fields: `video_recording`, `video_path`, `youtube_live`.
- Session video orientation via `TAP_VIDEO_ROTATE` (0/90/180/270 clockwise) and
  `TAP_VIDEO_FLIP` (`h`/`v`/`hv`) in `/etc/default/tap-spindle`.
- Session video save path via `TAP_VIDEO_DIR` (`{dir}/{session_id}/session.mp4`;
  unset keeps files beside `homing.csv`).
- `--video-test` (default 30s) smoke-tests ustreamer → ffmpeg with the same live
  RRF/ADXL overlay as a job, plus the YouTube preflight
  (`scripts/video_test.sh`).
- Video overlay HUD: spindle RPM/load, installed tool (name refreshed on change),
  live feed mm/min, XYZA work positions from `move.axes`, RRF job
  elapsed/remaining when present, and ADXL X/Y/Z level-meter bars
  (`TAP_VIDEO_ACCEL_SCALE_G`).
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
