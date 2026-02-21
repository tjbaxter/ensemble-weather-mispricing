#!/usr/bin/env bash
set -euo pipefail

# Run from local machine:
#   ./deploy/redeploy.sh
#
# Optional overrides:
#   VM_NAME=weather-bot ZONE=us-east1-b REMOTE_USER=myuser ./deploy/redeploy.sh

VM_NAME="${VM_NAME:-weather-bot}"
ZONE="${ZONE:-us-east1-b}"
REMOTE_USER="${REMOTE_USER:-$USER}"
BOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_WORKDIR="/home/${REMOTE_USER}/weather-bot"

echo "==> Syncing repo to VM (excluding local venv/logs/cache)"
TMP_ARCHIVE="/tmp/weather-bot-redeploy-$$.tgz"
trap 'rm -f "${TMP_ARCHIVE}"' EXIT
tar -C "${BOT_DIR}" \
  --exclude=".venv" \
  --exclude="venv" \
  --exclude="logs" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  -czf "${TMP_ARCHIVE}" .
gcloud compute scp --zone "${ZONE}" "${TMP_ARCHIVE}" "${VM_NAME}:~/weather-bot.tgz"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  rm -rf '${REMOTE_WORKDIR}' && \
  mkdir -p '${REMOTE_WORKDIR}' && \
  tar -xzf ~/weather-bot.tgz -C '${REMOTE_WORKDIR}' && \
  rm -f ~/weather-bot.tgz"

echo "==> Installing/updating dependencies and restarting service"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  set -euo pipefail && \
  cd '${REMOTE_WORKDIR}' && \
  mkdir -p '${REMOTE_WORKDIR}/logs' && \
  python3 -m venv venv && \
  source venv/bin/activate && \
  pip install --upgrade pip && \
  pip install -r requirements.txt && \
  sudo systemctl daemon-reload && \
  sudo systemctl restart weather-bot && \
  sudo systemctl status weather-bot --no-pager"

echo "==> Recent heartbeat lines"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  grep HEARTBEAT '${REMOTE_WORKDIR}/logs/bot.log' | tail -10 || true"
