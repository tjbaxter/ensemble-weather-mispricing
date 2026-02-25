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

echo "==> Copying weather-bot directory (excluding local venv/logs/cache)"
TMP_ARCHIVE="/tmp/weather-bot-deploy-$$.tgz"
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

echo "==> Setting up Python environment"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  mkdir -p '${REMOTE_WORKDIR}/logs' && \
  cd '${REMOTE_WORKDIR}' && \
  python3 -m venv venv && \
  source venv/bin/activate && \
  pip install --upgrade pip && \
  pip install -r requirements.txt"

echo "==> Installing systemd service + healthcheck + logrotate"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  set -euo pipefail && \
  mkdir -p '${REMOTE_WORKDIR}/logs' && \
  chmod +x '${REMOTE_WORKDIR}/deploy/healthcheck.sh' '${REMOTE_WORKDIR}/deploy/redeploy.sh' '${REMOTE_WORKDIR}/deploy/setup_gcloud_vm.sh' || true && \
  sed -e 's#__USER__#${REMOTE_USER}#g' -e 's#__WORKDIR__#${REMOTE_WORKDIR}#g' '${REMOTE_WORKDIR}/deploy/weather-bot.service.template' | sudo tee /etc/systemd/system/weather-bot.service >/dev/null && \
  sed -e 's#__WORKDIR__#${REMOTE_WORKDIR}#g' '${REMOTE_WORKDIR}/deploy/weather-bot-logrotate' | sudo tee /etc/logrotate.d/weather-bot >/dev/null"

gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  set -euo pipefail && \
  if [ ! -f /etc/weather-bot.env ]; then \
    printf '%s\n' \
      'PAPER_TRADING=true' \
      'LIVE_TRADING=false' \
      'INITIAL_BANKROLL=300' \
      'REQUIRE_VPN=true' \
      'STATION_PRIORITY_FILTER=HIGH,MEDIUM,LOW' \
      'CLOB_PREFILTER_PRIORITY=HIGH,MEDIUM,LOW' \
      'MET_OFFICE_API_KEY=' \
      'ACCUWEATHER_API_KEY=' | sudo tee /etc/weather-bot.env >/dev/null; \
  fi && \
  sudo chmod 600 /etc/weather-bot.env && \
  sudo chown root:root /etc/weather-bot.env && \
  sudo systemctl daemon-reload && \
  sudo systemctl enable weather-bot && \
  sudo systemctl restart weather-bot"

echo "==> Installing healthcheck cron (every 5 minutes)"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  (crontab -l 2>/dev/null | grep -v 'deploy/healthcheck.sh' || true; echo '*/5 * * * * ${REMOTE_WORKDIR}/deploy/healthcheck.sh') | crontab -"

echo "==> Installing commercial forecast logger cron (daily 19:05 UTC)"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  (crontab -l 2>/dev/null | grep -v 'log_commercial_forecasts.py' || true; \
   echo '5 19 * * * ${REMOTE_WORKDIR}/venv/bin/python3 ${REMOTE_WORKDIR}/scripts/log_commercial_forecasts.py >> ${REMOTE_WORKDIR}/logs/commercial_forecast.log 2>&1') | crontab -"

echo "==> Verifying service status"
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "\
  sudo systemctl status weather-bot --no-pager && \
  grep HEARTBEAT '${REMOTE_WORKDIR}/logs/bot.log' | tail -5 || true"

cat <<EOF

Setup complete.
Next:
1) Set secrets on VM (not in repo):
   gcloud compute ssh ${VM_NAME} --zone ${ZONE} --command 'sudo nano /etc/weather-bot.env'
2) Restart after editing:
   gcloud compute ssh ${VM_NAME} --zone ${ZONE} --command 'sudo systemctl restart weather-bot'
EOF
