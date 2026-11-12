#!/usr/bin/env bash
# Install live-spindle systemd service on DuetPi / SBC.
# Run on the Pi: sudo bash /mnt/repos/tap-testing/scripts/install_live_spindle_service.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_SRC="$ROOT/deploy/systemd/tap-spindle.service"
DEFAULT_SRC="$ROOT/deploy/systemd/tap-spindle.default"
RUN_DIR=/run/tap-spindle
SVC_USER="${SUDO_USER:-kad}"
SVC_GROUP="$(id -gn "$SVC_USER" 2>/dev/null || echo kad)"
HOME_DIR="$(getent passwd "$SVC_USER" | cut -d: -f6)"
VENV="${TAP_SPINDLE_VENV:-$HOME_DIR/.venvs/tap-testing}"

echo "Installing from $ROOT (user=$SVC_USER venv=$VENV)"

SKIP_DEPS="${SKIP_DEPS:-0}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating Pi-local venv at $VENV (do not use NFS .venv — often no +x / wrong arch)..."
  sudo -u "$SVC_USER" mkdir -p "$(dirname "$VENV")"
  sudo -u "$SVC_USER" python3 -m venv "$VENV"
  sudo -u "$SVC_USER" "$VENV/bin/pip" install -U pip
fi

if [[ "$SKIP_DEPS" == "1" ]]; then
  echo "SKIP_DEPS=1 — leaving Python deps in $VENV unchanged"
else
  echo "Installing Python deps into $VENV..."
  sudo -u "$SVC_USER" "$VENV/bin/pip" install -r "$ROOT/requirements.txt"
  sudo -u "$SVC_USER" "$VENV/bin/pip" install 'paho-mqtt>=1.6.0'
  sudo -u "$SVC_USER" "$VENV/bin/pip" install 'pystray>=0.19' 'Pillow>=9.0' || true
fi

install -d -m 0755 "$RUN_DIR"
chown "$SVC_USER:$SVC_GROUP" "$RUN_DIR"
cat >/etc/tmpfiles.d/tap-spindle.conf <<EOF
d /run/tap-spindle 0755 $SVC_USER $SVC_GROUP -
EOF
systemd-tmpfiles --create /etc/tmpfiles.d/tap-spindle.conf || true

tmp_unit="$(mktemp)"
sed \
  -e "s/^User=.*/User=$SVC_USER/" \
  -e "s/^Group=.*/Group=$SVC_GROUP/" \
  -e "s|^ExecStart=.*|ExecStart=$VENV/bin/python -m tap_testing.live_spindle_service|" \
  -e "s|^WorkingDirectory=.*|WorkingDirectory=$ROOT|" \
  "$UNIT_SRC" >"$tmp_unit"
# Ensure PYTHONPATH points at repo root
if ! grep -q '^Environment=PYTHONPATH=' "$tmp_unit"; then
  sed -i "/EnvironmentFile=/a Environment=PYTHONPATH=$ROOT" "$tmp_unit"
else
  sed -i "s|^Environment=PYTHONPATH=.*|Environment=PYTHONPATH=$ROOT|" "$tmp_unit"
fi

if [[ ! -f /etc/default/tap-spindle ]]; then
  install -m 0644 "$DEFAULT_SRC" /etc/default/tap-spindle
  echo "Wrote /etc/default/tap-spindle — MQTT defaults to mqtt.jarvis.lan:1883"
else
  echo "Keeping existing /etc/default/tap-spindle (set TAP_MQTT_HOST=mqtt.jarvis.lan if needed)"
fi

install -m 0644 "$tmp_unit" /etc/systemd/system/tap-spindle.service
rm -f "$tmp_unit"
systemctl daemon-reload
systemctl enable tap-spindle.service
systemctl restart tap-spindle.service
systemctl --no-pager --full status tap-spindle.service || true

echo
echo "Done. Logs: journalctl -u tap-spindle -f"
echo "Status:    cat /run/tap-spindle/status.json"
echo "RRF MQTT subscribe (analysis/status only) — see docs/LIVE_SPINDLE_SERVICE.md"
echo "Venv:      $VENV"
