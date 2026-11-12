#!/usr/bin/env bash
# Re-apply RRF MQTT client (M586 P4) on SBC — boot dsf-config may run before WiFi is ready.
# Usage: bash scripts/ensure_rrf_mqtt.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${TAP_SPINDLE_ENV:-/etc/default/tap-spindle}"
RRF_BASE="${TAP_RRF_BASE:-http://127.0.0.1}"
DEVICE_ID="${TAP_MQTT_DEVICE_ID:-milo}"
MQTT_HOST="${TAP_MQTT_HOST:-mqtt.jarvis.lan}"
MQTT_PORT="${TAP_MQTT_PORT:-1883}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
  RRF_BASE="${TAP_RRF_BASE:-$RRF_BASE}"
  DEVICE_ID="${TAP_MQTT_DEVICE_ID:-$DEVICE_ID}"
  MQTT_HOST="${TAP_MQTT_HOST:-$MQTT_HOST}"
  MQTT_PORT="${TAP_MQTT_PORT:-$MQTT_PORT}"
fi

# Sim gcode uses cam/{device}/… — strip -sbc suffix if set for tap-spindle.
CAM_DEVICE="${TAP_CAM_DEVICE_ID:-${DEVICE_ID%%-sbc}}"

send_gcode() {
  local line="$1"
  curl -sfG "${RRF_BASE%/}/rr_gcode" --data-urlencode "gcode=${line}" >/dev/null
}

echo "RRF MQTT: client=${CAM_DEVICE} broker=${MQTT_HOST}:${MQTT_PORT}"
send_gcode "M586.4 C\"${CAM_DEVICE}\""
sleep 1
send_gcode "M586 P4 H\"${MQTT_HOST}\" R${MQTT_PORT} S1"
sleep 2
echo "Done. Test with:"
echo "  curl -sG '${RRF_BASE%/}/rr_gcode' --data-urlencode 'gcode=M118 P6 S\"{\"\"state\"\":\"\"connected\"\",\"\"mode\"\":\"\"test\"\"}\" T\"cam/${CAM_DEVICE}/status\"'"
