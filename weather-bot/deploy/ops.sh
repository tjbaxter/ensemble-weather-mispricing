#!/usr/bin/env bash
set -euo pipefail

VM_NAME="${VM_NAME:-weather-bot}"
ZONE="${ZONE:-us-east1-b}"
PROJECT="${PROJECT:-weather-488111}"
REMOTE_USER="${REMOTE_USER:-tombaxter}"
WORKDIR="/home/${REMOTE_USER}/weather-bot"
SERVICE="weather-bot"

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
  *)
    usage
    exit 1
    ;;
esac
