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
ARCHIVE_EXCLUDES=(
  --exclude="./.env"
  --exclude="./.venv"
  --exclude="./venv"
  --exclude="./logs"
  --exclude="./__pycache__"
  --exclude="*.pyc"
  --exclude="./data/*.json"
  --exclude="./data/*.db"
  --exclude="./data/*.db-*"
  --exclude="./data/*.sqlite"
  --exclude="./data/*.sqlite3"
  --exclude="./data/*.csv"
  --exclude="./data/*.jsonl"
  --exclude="./data/*.parquet"
  --exclude="./data/*.tmp"
  --exclude="./data/*.lock"
)

echo "==> Syncing code bundle to VM (preserving remote runtime data)"
TMP_ARCHIVE="/tmp/weather-bot-redeploy-$$.tgz"
trap 'rm -f "${TMP_ARCHIVE}"' EXIT
tar -C "${BOT_DIR}" \
  "${ARCHIVE_EXCLUDES[@]}" \
  -czf "${TMP_ARCHIVE}" .
gcloud compute scp --zone "${ZONE}" "${TMP_ARCHIVE}" "${VM_NAME}:~/weather-bot.tgz"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  set -euo pipefail && \
  REMOTE_STAGE=\$(mktemp -d /tmp/weather-bot-stage-XXXXXX) && \
  mkdir -p '${REMOTE_WORKDIR}' && \
  tar -xzf ~/weather-bot.tgz -C \"\${REMOTE_STAGE}\" && \
  python3 \"\${REMOTE_STAGE}/deploy/safe_remote_sync.py\" \"\${REMOTE_STAGE}\" '${REMOTE_WORKDIR}' && \
  rm -rf \"\${REMOTE_STAGE}\" ~/weather-bot.tgz"

echo "==> Installing/updating dependencies and restarting service"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  set -euo pipefail && \
  cd '${REMOTE_WORKDIR}' && \
  mkdir -p '${REMOTE_WORKDIR}/logs' && \
  if [ ! -f /etc/weather-bot.env ]; then \
    printf '%s\n' \
      'PAPER_TRADING=true' \
      'LIVE_TRADING=false' \
      'INITIAL_BANKROLL=300' \
      'REQUIRE_VPN=true' \
      'STATION_PRIORITY_FILTER=HIGH,MEDIUM,LOW' \
      'CLOB_PREFILTER_PRIORITY=HIGH,MEDIUM,LOW' \
      'SETTLEMENT_WATCHER_POLL_SECONDS=10' \
      'SETTLEMENT_WATCHER_OFFICIAL_REFRESH_SECONDS=60' \
      'DASHBOARD_API_HOST=127.0.0.1' \
      'DASHBOARD_API_PORT=8510' \
      \"DASHBOARD_API_TOKEN=\$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')\" \
      'MET_OFFICE_API_KEY=' \
      'ACCUWEATHER_API_KEY=' | sudo tee /etc/weather-bot.env >/dev/null; \
  fi && \
  if ! sudo grep -q '^DASHBOARD_API_HOST=' /etc/weather-bot.env; then \
    echo 'DASHBOARD_API_HOST=127.0.0.1' | sudo tee -a /etc/weather-bot.env >/dev/null; \
  fi && \
  if ! sudo grep -q '^DASHBOARD_API_PORT=' /etc/weather-bot.env; then \
    echo 'DASHBOARD_API_PORT=8510' | sudo tee -a /etc/weather-bot.env >/dev/null; \
  fi && \
  if ! sudo grep -q '^DASHBOARD_API_TOKEN=' /etc/weather-bot.env; then \
    printf 'DASHBOARD_API_TOKEN=%s\n' \"\$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')\" | sudo tee -a /etc/weather-bot.env >/dev/null; \
  fi && \
  sudo chmod 600 /etc/weather-bot.env && \
  sudo chown root:root /etc/weather-bot.env && \
  python3 -m venv venv && \
  source venv/bin/activate && \
  pip install --upgrade pip && \
  pip install -r requirements.txt && \
  sed -e 's#__USER__#${REMOTE_USER}#g' -e 's#__WORKDIR__#${REMOTE_WORKDIR}#g' '${REMOTE_WORKDIR}/deploy/weather-dashboard.service.template' | sudo tee /etc/systemd/system/weather-dashboard.service >/dev/null && \
  sed -e 's#__USER__#${REMOTE_USER}#g' -e 's#__WORKDIR__#${REMOTE_WORKDIR}#g' '${REMOTE_WORKDIR}/deploy/weather-dashboard-api.service.template' | sudo tee /etc/systemd/system/weather-dashboard-api.service >/dev/null && \
  sed -e 's#__USER__#${REMOTE_USER}#g' -e 's#__WORKDIR__#${REMOTE_WORKDIR}#g' '${REMOTE_WORKDIR}/deploy/weather-settlement-watcher.service.template' | sudo tee /etc/systemd/system/weather-settlement-watcher.service >/dev/null && \
  sudo systemctl daemon-reload && \
  sudo systemctl restart weather-bot && \
  sudo systemctl restart weather-dashboard && \
  sudo systemctl restart weather-dashboard-api && \
  sudo systemctl restart weather-settlement-watcher && \
  sudo systemctl status weather-bot --no-pager"

echo "==> Recent heartbeat lines"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  (sudo journalctl -u weather-bot --no-pager -n 300 | grep HEARTBEAT | tail -10) || \
  (grep HEARTBEAT '${REMOTE_WORKDIR}/logs/bot.log' | tail -10) || true"

echo "==> Settlement watcher status"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  sudo systemctl status weather-settlement-watcher --no-pager && \
  sudo systemctl status weather-dashboard-api --no-pager && \
  tail -n 20 '${REMOTE_WORKDIR}/logs/settlement_watcher.log' || true"
