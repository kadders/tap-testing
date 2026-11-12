#!/usr/bin/env bash
# Verify NeXT post (duet/#) and tap-spindle (tap/#) MQTT side-by-side during a job.
# Metadata-only collector sessions usually mean duet/ tool events arrived without
# tap/ accel batches (tap-spindle not recording / not publishing).
#
# Usage (on a host that can reach the broker, ideally Milo):
#   bash scripts/verify_duet_tap_mqtt.sh [seconds]
# Env: TAP_MQTT_HOST (default mqtt.jarvis.lan), TAP_MQTT_PORT, TAP_MQTT_DEVICE_ID
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${TAP_SPINDLE_ENV:-/etc/default/tap-spindle}"
SECS="${1:-30}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

HOST="${TAP_MQTT_HOST:-mqtt.jarvis.lan}"
PORT="${TAP_MQTT_PORT:-1883}"
DEVICE="${TAP_MQTT_DEVICE_ID:-milo}"
DEVICE="${DEVICE%-sbc}"

echo "=== verify duet/# vs tap/# for ${SECS}s ==="
echo "broker: ${HOST}:${PORT}  device: ${DEVICE}"
echo "Start or continue a job that includes NeXT post M118 tool publishes."
echo

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

if ! command -v mosquitto_sub >/dev/null 2>&1; then
  echo "ERROR: mosquitto_sub required (apt install mosquitto-clients)"
  exit 1
fi

mosquitto_sub -h "$HOST" -p "$PORT" -t "duet/${DEVICE}/#" -v >"$OUT/duet.log" 2>"$OUT/duet.err" &
DPID=$!
mosquitto_sub -h "$HOST" -p "$PORT" -t "tap/${DEVICE}/#" -v >"$OUT/tap.log" 2>"$OUT/tap.err" &
TPID=$!

sleep "$SECS"
kill "$DPID" "$TPID" 2>/dev/null || true
wait "$DPID" "$TPID" 2>/dev/null || true

duet_n=$(wc -l <"$OUT/duet.log" | tr -d ' ')
tap_n=$(wc -l <"$OUT/tap.log" | tr -d ' ')
tap_batch=$(grep -c '/accel/batch' "$OUT/tap.log" || true)
tap_sess=$(grep -c '/session' "$OUT/tap.log" || true)
duet_tool=$(grep -c '/tool' "$OUT/duet.log" || true)

echo "--- counts ---"
echo "  duet/# messages:     ${duet_n}  (tool lines: ${duet_tool})"
echo "  tap/# messages:      ${tap_n}  (session: ${tap_sess}, accel/batch: ${tap_batch})"
echo

if [[ "$duet_tool" -gt 0 && "$tap_batch" -eq 0 ]]; then
  echo "RESULT: metadata-only pattern — duet tool events without tap accel batches."
  echo "  Fix: ensure tap-spindle is running and recording (job-sync / TAP_MQTT_HOST)."
  echo "  Status: cat /run/tap-spindle/status.json"
  echo "  RRF MQTT (M118): bash ${ROOT}/scripts/diagnose_rrf_mqtt.sh"
  exit 2
fi

if [[ "$duet_tool" -eq 0 && "$tap_batch" -eq 0 ]]; then
  echo "RESULT: no traffic on either path in ${SECS}s — is a job running? broker reachable?"
  echo "  RRF: bash ${ROOT}/scripts/diagnose_rrf_mqtt.sh"
  echo "  ensure: bash ${ROOT}/scripts/ensure_rrf_mqtt.sh"
  exit 3
fi

if [[ "$duet_tool" -eq 0 && "$tap_batch" -gt 0 ]]; then
  echo "RESULT: tap accel OK; no duet tool events — check posted gcode has M118 P6 and RRF MQTT connected."
  echo "  bash ${ROOT}/scripts/diagnose_rrf_mqtt.sh /path/to/job.gcode"
  exit 4
fi

echo "RESULT: both duet tool and tap accel traffic seen — post + tap-spindle look live."
echo "Sample duet:"
head -n 3 "$OUT/duet.log" || true
echo "Sample tap session/batch:"
grep -E '/(session|accel/batch)' "$OUT/tap.log" | head -n 3 || true
exit 0
