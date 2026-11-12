#!/usr/bin/env bash
# Phase 0: validate ustreamer MJPEG source freshness (no ffmpeg / YouTube).
# Usage: bash scripts/check_ustreamer_source.sh [ustreamer_base_url] [duration_s]
set -euo pipefail

BASE="${1:-http://127.0.0.1:8081}"
DURATION="${2:-10}"
export USTREAMER_BASE="$BASE"
export USTREAMER_CHECK_DURATION="$DURATION"

echo "=== ustreamer source check ==="
echo "base:     $BASE"
echo "duration: ${DURATION}s"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 required" >&2
  exit 1
fi

python3 - <<PY
import json
import os
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

base = os.environ.get("USTREAMER_BASE", "http://127.0.0.1:8081")
duration = float(os.environ.get("USTREAMER_CHECK_DURATION", "10"))
state_url = base.rstrip("/") + "/state"
snap_url = base.rstrip("/") + "/snapshot"

def fetch_state():
    with urlopen(state_url, timeout=5) as r:
        return json.load(r)

def fetch_snapshot_headers():
    req = Request(snap_url, headers={"User-Agent": "tap-testing/check_ustreamer_source"})
    with urlopen(req, timeout=5) as r:
        hdrs = dict(r.headers.items())
        _ = r.read(4096)
        return hdrs

try:
    state = fetch_state()
except URLError as e:
    print(f"FAIL: cannot reach {state_url}: {e}", file=sys.stderr)
    sys.exit(1)

print("state:", json.dumps(state, indent=2))
result = state.get("result") or {}
src = result.get("source") or {}
stream = result.get("stream") or {}
print()
print(f"capture: {src.get('resolution')} @ {src.get('captured_fps')} fps (desired {src.get('desired_fps')})")
print(f"clients: {stream.get('clients')} queued_fps={stream.get('queued_fps')}")
if int(stream.get("clients") or 0) > 1:
    print("WARN: multiple MJPEG clients — use a single consumer during tests")

print()
print(f"Sampling snapshot headers for {duration:.0f}s...")
t0 = time.monotonic()
last_grab = None
max_gap = 0.0
n = 0
while time.monotonic() - t0 < duration:
    hdrs = fetch_snapshot_headers()
    grab = hdrs.get("X-UStreamer-Grab-Timestamp") or hdrs.get("x-ustreamer-grab-timestamp")
    if grab is not None:
        grab_f = float(grab)
        if last_grab is not None:
            max_gap = max(max_gap, grab_f - last_grab)
        last_grab = grab_f
    n += 1
    time.sleep(0.5)

print(f"samples: {n}")
if last_grab is not None:
    print(f"max inter-grab gap (ustreamer monotonic): {max_gap:.3f}s")
print("OK: source reachable — compare live preview latency visually during motion")
PY
