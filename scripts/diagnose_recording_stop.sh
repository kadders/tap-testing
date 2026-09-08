#!/usr/bin/env bash
# Diagnose why tap-spindle stopped or finalized a recording session.
# Usage: bash scripts/diagnose_recording_stop.sh [session_id]
set -euo pipefail

SESSION_ID="${1:-}"
STATUS_PATH="${TAP_LIVE_STATUS_PATH:-/run/tap-spindle/status.json}"
OUTPUT_ROOT="${TAP_LIVE_OUTPUT_DIR:-/mnt/repos/tap-testing/data/live_spindle/service}"
SINCE="${DIAG_SINCE:-2 hours ago}"

echo "=== tap-spindle recording stop diagnostics ==="
echo "status:   $STATUS_PATH"
echo "sessions: $OUTPUT_ROOT"
echo "since:    $SINCE"
echo

if [[ -f "$STATUS_PATH" ]]; then
  echo "--- status.json ---"
  python3 - "$STATUS_PATH" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
keys = (
    "state", "recording", "session_id", "stop_reason", "last_error",
    "rrf_status", "rrf_job_active", "rrf_connected", "output_path",
    "video_recording", "video_stream_remote",
)
for k in keys:
    if k in d:
        print(f"  {k}: {d[k]}")
PY
else
  echo "status.json not found"
fi

echo
echo "--- recent journal (stop / timeout / ffmpeg) ---"
if command -v journalctl >/dev/null 2>&1; then
  journalctl -u tap-spindle --since "$SINCE" --no-pager 2>/dev/null \
    | grep -iE "recording stopped|recording started|idle timeout|stop_reason|job inactive|ffmpeg did not|recording failed|worker still running" \
    | tail -30 || echo "(no matching lines)"
else
  echo "journalctl not available"
fi

pick_dir() {
  if [[ -n "$SESSION_ID" ]]; then
    echo "$OUTPUT_ROOT/$SESSION_ID"
    return
  fi
  if [[ -f "$STATUS_PATH" ]]; then
    python3 - "$STATUS_PATH" "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
status = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
sid = status.get("session_id") or ""
if sid and (root / sid).is_dir():
    print(root / sid)
elif status.get("output_path"):
    print(Path(status["output_path"]).parent)
PY
  fi
}

RUN_DIR="$(pick_dir 2>/dev/null || true)"
echo
if [[ -n "${RUN_DIR:-}" && -d "$RUN_DIR" ]]; then
  echo "--- run dir: $RUN_DIR ---"
  ls -la "$RUN_DIR"
  if [[ -f "$RUN_DIR/homing.csv" ]]; then
    python3 - "$RUN_DIR/homing.csv" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
lines = p.read_text().strip().splitlines()
data = [ln for ln in lines if ln and not ln.startswith("#")]
print(f"  homing.csv data rows: {len(data)}")
if data:
    last = data[-1].split(",")
    if last:
        print(f"  last t_s: {last[0]}")
PY
  fi
else
  echo "--- run dir: not found ---"
fi

echo
echo "Stop reason guide:"
echo "  idle_timeout       — no ADXL samples for MQTT_SESSION_IDLE_TIMEOUT_S (default 25s)"
echo "  job_completed      — RRF state.status completed"
echo "  job_no_file        — job.file.fileName cleared / no file loaded"
echo "  job_sync_inactive  — idle, busy, etc. with no active print/simulate"
echo "  worker_exception   — ADXL worker crashed"
echo "  service_shutdown   — tap-spindle service stopping"
echo "  manual             — stop_recording() without a specific reason"
