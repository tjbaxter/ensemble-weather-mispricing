#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/weather-bot}"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/venv/bin/python3}"
PAST_DAYS="${PAST_DAYS:-45}"
CALIBRATION_TIME_UTC="${CALIBRATION_TIME_UTC:-35 2 * * *}" # 02:35 UTC daily
HEALTHCHECK_TIME="${HEALTHCHECK_TIME:-*/5 * * * *}" # every 5 min

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python binary: ${PYTHON_BIN}"
  exit 1
fi

mkdir -p "${WORKDIR}/logs"

HEALTHCHECK_LINE="${HEALTHCHECK_TIME} ${WORKDIR}/deploy/healthcheck.sh"
CALIBRATION_LINE="${CALIBRATION_TIME_UTC} cd ${WORKDIR} && ${PYTHON_BIN} scripts/backtest_calibration.py --past-days ${PAST_DAYS} --output logs/calibration.json --rankings-output logs/model_rankings.json >> logs/calibration_cron.log 2>&1"

(crontab -l 2>/dev/null | grep -Ev "deploy/healthcheck.sh|backtest_calibration.py" || true; echo "${HEALTHCHECK_LINE}"; echo "${CALIBRATION_LINE}") | crontab -

echo "Installed cron jobs:"
crontab -l | grep -E "deploy/healthcheck.sh|backtest_calibration.py"
