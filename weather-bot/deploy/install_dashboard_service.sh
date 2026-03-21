#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="weather-dashboard"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SERVICE_TEMPLATE="${SCRIPT_DIR}/weather-dashboard.service.template"
INSTALL_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
PORT="${PORT:-8501}"
USER_NAME="${USER_NAME:-$(id -un)}"
STREAMLIT_BIN="${WORKDIR}/venv/bin/streamlit"

echo "=== Weather Dashboard Service Installer ==="
echo ""

if [[ ! -f "${WORKDIR}/dashboard.py" ]]; then
  echo "ERROR: dashboard.py not found in ${WORKDIR}"
  exit 1
fi

if [[ ! -x "${STREAMLIT_BIN}" ]]; then
  echo "ERROR: streamlit binary not found at ${STREAMLIT_BIN}"
  echo "Install requirements first: ${WORKDIR}/venv/bin/pip install -r ${WORKDIR}/requirements.txt"
  exit 1
fi

mkdir -p "${WORKDIR}/logs"

echo "--- Installing systemd service ---"
sed \
  -e "s#__USER__#${USER_NAME}#g" \
  -e "s#__WORKDIR__#${WORKDIR}#g" \
  -e "s#__PORT__#${PORT}#g" \
  "${SERVICE_TEMPLATE}" | sudo tee "${INSTALL_PATH}" >/dev/null
sudo chmod 644 "${INSTALL_PATH}"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

echo ""
echo "--- Status ---"
sleep 2
sudo systemctl status "${SERVICE_NAME}" --no-pager -l

echo ""
echo "=== Done ==="
echo "Dashboard is bound to 127.0.0.1:${PORT} on the VM."
echo ""
echo "Open a private tunnel from your Mac:"
echo "  gcloud compute ssh weather-bot --zone us-east1-b --project weather-488111 -- -N -L ${PORT}:127.0.0.1:${PORT}"
echo ""
echo "Then visit:"
echo "  http://localhost:${PORT}"
