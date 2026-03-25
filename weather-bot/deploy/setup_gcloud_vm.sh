#!/usr/bin/env bash
set -euo pipefail

VM_NAME="${VM_NAME:-weather-bot}"
ZONE="${ZONE:-us-east1-b}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-small}"
DISK_SIZE="${DISK_SIZE:-20GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2404-lts-amd64}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
TAG="${TAG:-weather-bot}"
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

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI is required. Install it first."
  exit 1
fi

echo "==> Ensuring VM exists (${VM_NAME} in ${ZONE})"
if ! gcloud compute instances describe "${VM_NAME}" --zone "${ZONE}" >/dev/null 2>&1; then
  gcloud compute instances create "${VM_NAME}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --boot-disk-size="${DISK_SIZE}" \
    --tags="${TAG}"
else
  echo "VM already exists; skipping create."
fi

echo "==> Installing base packages"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  sudo apt-get update && \
  sudo apt-get install -y python3 python3-pip python3-venv git tmux logrotate"

echo "==> Copying code bundle (preserving remote runtime data)"
TMP_ARCHIVE="/tmp/weather-bot-deploy-$$.tgz"
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

echo "==> Setting up Python environment"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  mkdir -p '${REMOTE_WORKDIR}/logs' && \
  cd '${REMOTE_WORKDIR}' && \
  python3 -m venv venv && \
  source venv/bin/activate && \
  pip install --upgrade pip && \
  pip install -r requirements.txt"

echo "==> Installing systemd services + healthcheck + logrotate"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  set -euo pipefail && \
  mkdir -p '${REMOTE_WORKDIR}/logs' && \
  chmod +x '${REMOTE_WORKDIR}/deploy/healthcheck.sh' '${REMOTE_WORKDIR}/deploy/redeploy.sh' '${REMOTE_WORKDIR}/deploy/setup_gcloud_vm.sh' '${REMOTE_WORKDIR}/deploy/install_cron_jobs.sh' '${REMOTE_WORKDIR}/deploy/install_dashboard_service.sh' '${REMOTE_WORKDIR}/deploy/install_dashboard_api_service.sh' || true && \
  sed -e 's#__USER__#${REMOTE_USER}#g' -e 's#__WORKDIR__#${REMOTE_WORKDIR}#g' '${REMOTE_WORKDIR}/deploy/weather-bot.service.template' | sudo tee /etc/systemd/system/weather-bot.service >/dev/null && \
  sed -e 's#__USER__#${REMOTE_USER}#g' -e 's#__WORKDIR__#${REMOTE_WORKDIR}#g' -e 's#__PORT__#8501#g' '${REMOTE_WORKDIR}/deploy/weather-dashboard.service.template' | sudo tee /etc/systemd/system/weather-dashboard.service >/dev/null && \
  sed -e 's#__USER__#${REMOTE_USER}#g' -e 's#__WORKDIR__#${REMOTE_WORKDIR}#g' '${REMOTE_WORKDIR}/deploy/weather-dashboard-api.service.template' | sudo tee /etc/systemd/system/weather-dashboard-api.service >/dev/null && \
  sed -e 's#__USER__#${REMOTE_USER}#g' -e 's#__WORKDIR__#${REMOTE_WORKDIR}#g' '${REMOTE_WORKDIR}/deploy/weather-settlement-watcher.service.template' | sudo tee /etc/systemd/system/weather-settlement-watcher.service >/dev/null && \
  sed -e 's#__WORKDIR__#${REMOTE_WORKDIR}#g' '${REMOTE_WORKDIR}/deploy/weather-bot-logrotate' | sudo tee /etc/logrotate.d/weather-bot >/dev/null"

gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  set -euo pipefail && \
  DASHBOARD_API_TOKEN=\$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))') && \
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
      \"DASHBOARD_API_TOKEN=\${DASHBOARD_API_TOKEN}\" \
      'MET_OFFICE_API_KEY=' \
      'ACCUWEATHER_API_KEY=' | sudo tee /etc/weather-bot.env >/dev/null; \
  fi && \
  sudo chmod 600 /etc/weather-bot.env && \
  sudo chown root:root /etc/weather-bot.env && \
  sudo systemctl daemon-reload && \
  sudo systemctl enable weather-bot && \
  sudo systemctl enable weather-dashboard && \
  sudo systemctl enable weather-dashboard-api && \
  sudo systemctl enable weather-settlement-watcher && \
  sudo systemctl restart weather-bot && \
  sudo systemctl restart weather-dashboard && \
  sudo systemctl restart weather-dashboard-api && \
  sudo systemctl restart weather-settlement-watcher"

echo "==> Installing cron suite"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  bash '${REMOTE_WORKDIR}/deploy/install_cron_jobs.sh'"

echo "==> Installing commercial forecast logger cron (daily 19:05 UTC)"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  (crontab -l 2>/dev/null | grep -v 'log_commercial_forecasts.py' || true; \
   echo '5 19 * * * ${REMOTE_WORKDIR}/venv/bin/python3 ${REMOTE_WORKDIR}/scripts/log_commercial_forecasts.py >> ${REMOTE_WORKDIR}/logs/commercial_forecast.log 2>&1') | crontab -"

echo "==> Verifying service status"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  sudo systemctl status weather-bot --no-pager && \
  sudo systemctl status weather-settlement-watcher --no-pager && \
  sudo systemctl status weather-dashboard --no-pager && \
  sudo systemctl status weather-dashboard-api --no-pager && \
  if ! sudo journalctl -u weather-bot --no-pager -n 300 | grep HEARTBEAT | tail -5; then \
    grep HEARTBEAT '${REMOTE_WORKDIR}/logs/bot.log' | tail -5 || true; \
  fi"

cat <<EOF

Setup complete.
Next:
1) Set secrets on VM (not in repo):
   gcloud compute ssh ${VM_NAME} --zone ${ZONE} --command 'sudo nano /etc/weather-bot.env'
2) Restart after editing:
   gcloud compute ssh ${VM_NAME} --zone ${ZONE} --command 'sudo systemctl restart weather-bot weather-settlement-watcher weather-dashboard weather-dashboard-api'
3) Open the private dashboard tunnel:
   gcloud compute ssh ${VM_NAME} --zone ${ZONE} --project weather-488111 -- -N -L 8501:127.0.0.1:8501
4) To power Streamlit Cloud from the VM directly, expose weather-dashboard-api (default 127.0.0.1:8510)
   through your preferred reverse proxy/tunnel and set Streamlit secrets:
   DASHBOARD_DATA_SOURCE=api
   DASHBOARD_API_BASE_URL=<public api base url>
   DASHBOARD_API_TOKEN=<value from /etc/weather-bot.env>
EOF
