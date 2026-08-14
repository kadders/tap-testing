#!/usr/bin/env bash
# Update tap-testing on the live-spindle host and restart the systemd service.
#
# The repo lives on a shared drive, so code changes made on the dev machine are
# already present on the host — no git pull is needed (and would risk clobbering
# local changes). This just refreshes the systemd unit and restarts the service.
#
# Usage:
#   scripts/update_live_spindle_service.sh              # auto: local on SBC, else SSH
#   scripts/update_live_spindle_service.sh local        # force local (no SSH)
#   scripts/update_live_spindle_service.sh [user@]host  # force SSH to host
#
# Env overrides:
#   TAP_SPINDLE_HOST   SSH target when remote (default: kad@192.168.86.65)
#   TAP_SPINDLE_REPO   Repo path (default: /mnt/repos/tap-testing, or this checkout)
#   INSTALL_DEPS=1     Also reinstall Python deps in the venv (slow; default: skip)
set -euo pipefail

SCRIPT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_HOST="${TAP_SPINDLE_HOST:-kad@192.168.86.65}"
DEFAULT_REPO="${TAP_SPINDLE_REPO:-}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

skip_deps=1
[[ "$INSTALL_DEPS" == "1" ]] && skip_deps=0

resolve_repo() {
  if [[ -n "$DEFAULT_REPO" ]]; then
    printf '%s\n' "$DEFAULT_REPO"
    return
  fi
  if [[ -f /mnt/repos/tap-testing/scripts/install_live_spindle_service.sh ]]; then
    printf '%s\n' /mnt/repos/tap-testing
    return
  fi
  if [[ -f "$SCRIPT_REPO/scripts/install_live_spindle_service.sh" ]]; then
    printf '%s\n' "$SCRIPT_REPO"
    return
  fi
  printf '%s\n' /mnt/repos/tap-testing
}

is_spindle_host() {
  local repo=$1
  [[ -f "$repo/scripts/install_live_spindle_service.sh" ]] || return 1
  [[ -f /etc/systemd/system/tap-spindle.service ]] && return 0
  [[ -f /lib/systemd/system/tap-spindle.service ]] && return 0
  # Unit may only be known to systemd (already installed)
  systemctl cat tap-spindle.service &>/dev/null
}

run_update() {
  local repo=$1
  local install="$repo/scripts/install_live_spindle_service.sh"

  if [[ ! -f "$install" ]]; then
    echo "ERROR: missing install script: $install" >&2
    exit 1
  fi

  echo "==> Reinstalling unit + restarting (SKIP_DEPS=$skip_deps)"
  sudo SKIP_DEPS="$skip_deps" bash "$install"

  echo "==> Recent logs:"
  sudo journalctl -u tap-spindle -n 20 --no-pager || true
}

run_remote() {
  local host=$1
  local repo=$2
  local SSH_OPTS=(
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=3
  )

  echo "==> Connecting to $host (repo=$repo)…"
  if ! ssh "${SSH_OPTS[@]}" "$host" "test -f '$repo/scripts/install_live_spindle_service.sh'"; then
    echo "ERROR: cannot reach $host or missing $repo/scripts/install_live_spindle_service.sh" >&2
    exit 1
  fi

  local ssh_sudo=(ssh "${SSH_OPTS[@]}")
  if ssh "${SSH_OPTS[@]}" "$host" "sudo -n true" &>/dev/null; then
    # Passwordless sudo available — no TTY needed.
    :
  elif [[ -t 0 ]]; then
    # Interactive terminal: allocate a TTY so sudo can prompt.
    ssh_sudo=(ssh -tt "${SSH_OPTS[@]}")
  else
    echo "ERROR: $host has no passwordless sudo and stdin is not a TTY." >&2
    echo "       Run this script from an interactive shell, or configure NOPASSWD for:" >&2
    echo "         $repo/scripts/install_live_spindle_service.sh" >&2
    echo "         journalctl" >&2
    exit 1
  fi

  echo "==> Reinstalling unit + restarting (SKIP_DEPS=$skip_deps)"
  "${ssh_sudo[@]}" "$host" \
    "sudo SKIP_DEPS='$skip_deps' bash '$repo/scripts/install_live_spindle_service.sh'"

  echo "==> Recent logs:"
  "${ssh_sudo[@]}" "$host" \
    "sudo journalctl -u tap-spindle -n 20 --no-pager" || true

  echo "==> Done. Follow logs with: ssh $host 'journalctl -u tap-spindle -f'"
}

REPO="$(resolve_repo)"
ARG="${1:-}"

case "$ARG" in
  local|--local)
    echo "==> Forcing local update (repo=$REPO)"
    if ! is_spindle_host "$REPO"; then
      echo "ERROR: this machine does not look like the spindle host (no tap-spindle unit / repo)." >&2
      echo "       Use: $0 ${DEFAULT_HOST}" >&2
      exit 1
    fi
    run_update "$REPO"
    echo "==> Done. Follow logs with: journalctl -u tap-spindle -f"
    ;;
  "")
    if is_spindle_host "$REPO"; then
      echo "==> Detected local spindle host (repo=$REPO) — skipping SSH"
      run_update "$REPO"
      echo "==> Done. Follow logs with: journalctl -u tap-spindle -f"
    else
      run_remote "$DEFAULT_HOST" "${DEFAULT_REPO:-/mnt/repos/tap-testing}"
    fi
    ;;
  *)
    run_remote "$ARG" "${DEFAULT_REPO:-/mnt/repos/tap-testing}"
    ;;
esac
