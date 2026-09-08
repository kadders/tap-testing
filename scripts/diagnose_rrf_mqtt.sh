#!/usr/bin/env bash
# Diagnose RRF/DSF MQTT for sim M118 P6 (separate from tap-spindle paho MQTT).
# Usage: bash scripts/diagnose_rrf_mqtt.sh [sim.gcode]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${TAP_SPINDLE_ENV:-/etc/default/tap-spindle}"
RRF_BASE="${TAP_RRF_BASE:-http://127.0.0.1}"
SIM_FILE="${1:-}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
  RRF_BASE="${TAP_RRF_BASE:-$RRF_BASE}"
fi

echo "=== RRF MQTT diagnostics (sim M118 P6) ==="
echo "RRF base: ${RRF_BASE}"

DSF_SYS="${DSF_SYS:-/opt/dsf/sd/sys}"
for f in "$DSF_SYS/dsf-config.g" "$DSF_SYS/nxt-user-overrides.g"; do
  echo "--- $(basename "$f") ---"
  if [[ -f "$f" ]]; then
    grep -in M586 "$f" || echo "(no M586 — expected in dsf-config.g on SBC, not user-overrides)"
  else
    echo "(missing)"
  fi
done

echo "--- rr_model network ---"
curl -sf "${RRF_BASE%/}/rr_model?key=network" | python3 -c "
import json, sys
d = json.load(sys.stdin).get('result', {})
for iface in d.get('interfaces', []):
    print(f\"  {iface.get('type')}: state={iface.get('state')} activeProtocols={iface.get('activeProtocols')}\")
" 2>/dev/null || echo "  (rr_model unavailable)"

echo "--- Pi broker reachability ---"
HOST="${TAP_MQTT_HOST:-mqtt.jarvis.lan}"
PORT="${TAP_MQTT_PORT:-1883}"
if command -v mosquitto_pub >/dev/null 2>&1; then
  mosquitto_pub -h "$HOST" -p "$PORT" -t test/milo/diag -m ok && echo "  mosquitto_pub: OK"
else
  PYTHON="${TAP_SPINDLE_VENV:-$HOME/.venvs/tap-testing}/bin/python"
  if [[ -x "$PYTHON" ]]; then
    "$PYTHON" -c "
import os, paho.mqtt.client as mqtt
h=os.environ.get('TAP_MQTT_HOST','mqtt.jarvis.lan')
p=int(os.environ.get('TAP_MQTT_PORT','1883'))
c=mqtt.Client(client_id='rrf-mqtt-diag')
c.connect(h,p,60)
c.publish('test/milo/diag','ok')
c.disconnect()
print('  paho publish: OK')
" && true
  else
    echo "  (install mosquitto-clients or paho-mqtt to test broker from Pi)"
  fi
fi

if [[ -n "$SIM_FILE" && -f "$SIM_FILE" ]]; then
  echo "--- sim gcode M118 scan: $SIM_FILE ---"
  grep -n '^M118' "$SIM_FILE" | head -5 || echo "  (no M118 lines)"
  bad=0
  while IFS= read -r ln; do
    if [[ ${#ln} -gt 240 ]]; then
      echo "  WARN: line >240 chars (${#ln})"
      bad=1
    fi
    if [[ "$ln" == *"M118 L6"* ]]; then
      echo "  WARN: M118 L6 (use P6 for MQTT)"
      bad=1
    fi
    if [[ "$ln" == M118\ P6* && "$ln" != *'S"'* ]]; then
      echo "  WARN: M118 P6 missing quoted S\"…\""
      bad=1
    fi
  done < <(grep '^M118' "$SIM_FILE" || true)
  if grep -q 'M586 P4' "$SIM_FILE"; then
    echo "  OK: sim includes M586 reconnect preamble"
  else
    echo "  NOTE: no M586 in sim — regenerate with current 4th-combinator or run ensure_rrf_mqtt.sh before job"
  fi
  [[ "$bad" -eq 0 ]] || echo "  (see warnings above)"
fi

echo "--- tap-spindle MQTT (separate client) ---"
if [[ -f /run/tap-spindle/status.json ]]; then
  python3 -c "import json; d=json.load(open('/run/tap-spindle/status.json')); print('  mqtt_enabled:', d.get('mqtt_enabled'), 'detail:', d.get('mqtt_detail'))"
else
  echo "  (status.json not found)"
fi

echo "--- fix ---"
echo "  SBC: put M586 P4 in ${DSF_SYS}/dsf-config.g (not nxt-user-overrides.g on SD)"
echo "  Then: bash ${ROOT}/scripts/ensure_rrf_mqtt.sh"
echo "  Regenerate sim: python3 -m fourth_combinator JOB_DIR --sim --mqtt-device-id milo"
