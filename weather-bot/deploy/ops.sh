#!/usr/bin/env bash
set -euo pipefail

VM_NAME="${VM_NAME:-weather-bot}"
ZONE="${ZONE:-us-east1-b}"
PROJECT="${PROJECT:-weather-488111}"
REMOTE_USER="${REMOTE_USER:-tombaxter}"
WORKDIR="/home/${REMOTE_USER}/weather-bot"
SERVICE="weather-bot"

LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

usage() {
  cat <<'EOF'
Usage: ./deploy/ops.sh <command>

Commands:
  status      Show systemd status
  heartbeat   Show latest heartbeat lines
  trades      Show latest paper trade lines
  logs        Show recent bot + error logs
  follow      Stream live journald logs
  restart     Restart service and show status
  cron        Show installed cron jobs
  sync        Pull live data from VM to local (logs + data)
EOF
}

ssh_cmd() {
  gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --project "${PROJECT}" --command "$1"
}

cmd="${1:-}"
case "${cmd}" in
  status)
    ssh_cmd "sudo systemctl status ${SERVICE} --no-pager"
    ;;
  heartbeat)
    ssh_cmd "tail -n 300 ${WORKDIR}/logs/bot.log | grep -E 'HEARTBEAT|DISCOVERY|SCAN_MODE' | tail -n 40 || true"
    ;;
  trades)
    ssh_cmd "tail -n 300 ${WORKDIR}/logs/bot.log | grep -E 'PAPER TRADE|SIGNAL' | tail -n 60 || true"
    ;;
  logs)
    ssh_cmd "echo '--- bot.log ---'; tail -n 120 ${WORKDIR}/logs/bot.log || true; echo '--- bot_error.log ---'; tail -n 120 ${WORKDIR}/logs/bot_error.log || true"
    ;;
  follow)
    ssh_cmd "sudo journalctl -u ${SERVICE} -f"
    ;;
  restart)
    ssh_cmd "sudo systemctl restart ${SERVICE} && sudo systemctl status ${SERVICE} --no-pager"
    ;;
  cron)
    ssh_cmd "crontab -l || true"
    ;;
  sync)
    echo "Pulling data from VM -> local..."
    mkdir -p "${LOCAL_DIR}/logs" "${LOCAL_DIR}/data"
    gcloud compute scp \
      "${VM_NAME}:${WORKDIR}/logs/trades.csv" \
      "${LOCAL_DIR}/logs/trades.csv" \
      --zone "${ZONE}" --project "${PROJECT}" 2>/dev/null || echo "  trades.csv: not found on VM yet"
    gcloud compute scp \
      "${VM_NAME}:${WORKDIR}/logs/signals.csv" \
      "${LOCAL_DIR}/logs/signals.csv" \
      --zone "${ZONE}" --project "${PROJECT}" 2>/dev/null || echo "  signals.csv: not found on VM yet"
    gcloud compute scp \
      "${VM_NAME}:${WORKDIR}/data/positions.json" \
      "${LOCAL_DIR}/data/positions.json" \
      --zone "${ZONE}" --project "${PROJECT}" 2>/dev/null || echo "  positions.json: not found on VM yet"
    gcloud compute scp \
      "${VM_NAME}:${WORKDIR}/logs/calibration.json" \
      "${LOCAL_DIR}/logs/calibration.json" \
      --zone "${ZONE}" --project "${PROJECT}" 2>/dev/null || echo "  calibration.json: not found on VM yet"
    echo "Sync complete. Refresh the Streamlit dashboard."
    ;;
  *)
    usage
    exit 1
    ;;
esac
