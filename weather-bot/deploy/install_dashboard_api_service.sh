#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="weather-dashboard-api"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SERVICE_TEMPLATE="${SCRIPT_DIR}/weather-dashboard-api.service.template"
INSTALL_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
USER_NAME="${USER_NAME:-$(id -un)}"
PYTHON_BIN="${WORKDIR}/venv/bin/python3"

echo "=== Weather Dashboard API Service Installer ==="
echo ""

if [[ ! -f "${WORKDIR}/scripts/dashboard_api.py" ]]; then
  echo "ERROR: scripts/dashboard_api.py not found in ${WORKDIR}"
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python binary not found at ${PYTHON_BIN}"
  echo "Install requirements first: ${WORKDIR}/venv/bin/pip install -r ${WORKDIR}/requirements.txt"
  exit 1
fi

mkdir -p "${WORKDIR}/logs"

echo "--- Installing systemd service ---"
sed \
  -e "s#__USER__#${USER_NAME}#g" \
  -e "s#__WORKDIR__#${WORKDIR}#g" \
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
echo "Dashboard API is bound to ${DASHBOARD_API_HOST:-127.0.0.1}:${DASHBOARD_API_PORT:-8510} on the VM."
echo "Expose it via your preferred reverse proxy/tunnel before using DASHBOARD_DATA_SOURCE=api on Streamlit Cloud."
