#!/usr/bin/env bash
# Timed video smoke test with live RRF/ADXL overlay (same env as tap-spindle).
# Usage: bash scripts/video_test.sh [duration_s]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${TAP_SPINDLE_ENV:-/etc/default/tap-spindle}"
SVC_USER="${SUDO_USER:-${USER:-kad}}"
HOME_DIR="$(getent passwd "$SVC_USER" 2>/dev/null | cut -d: -f6 || echo "$HOME")"
VENV="${TAP_SPINDLE_VENV:-$HOME_DIR/.venvs/tap-testing}"
DURATION="${1:-30}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
else
  echo "Warning: $ENV_FILE not found — using current environment" >&2
fi

if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet tap-spindle 2>/dev/null; then
  echo "Note: tap-spindle is running. Stop it if YouTube ingest would conflict:" >&2
  echo "  sudo systemctl stop tap-spindle" >&2
fi

PYTHON="$VENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m tap_testing.live_spindle_service --video-test --video-test-s "$DURATION"
