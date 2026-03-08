#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/weather-bot}"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/venv/bin/python3}"
PAST_DAYS="${PAST_DAYS:-45}"
CALIBRATION_TIME_UTC="${CALIBRATION_TIME_UTC:-35 2 * * *}" # 02:35 UTC daily
HEALTHCHECK_TIME="${HEALTHCHECK_TIME:-*/5 * * * *}" # every 5 min
RESOLVER_TIME_UTC="${RESOLVER_TIME_UTC:-0 10 * * *}" # 10:00 UTC daily
DATASYNC_TIME_UTC="${DATASYNC_TIME_UTC:-15 10 * * *}" # 10:15 UTC daily (after resolver)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python binary: ${PYTHON_BIN}"
  exit 1
fi

mkdir -p "${WORKDIR}/logs"

HEALTHCHECK_LINE="${HEALTHCHECK_TIME} ${WORKDIR}/deploy/healthcheck.sh"
CALIBRATION_LINE="${CALIBRATION_TIME_UTC} cd ${WORKDIR} && ${PYTHON_BIN} scripts/backtest_calibration.py --past-days ${PAST_DAYS} --output logs/calibration.json --rankings-output logs/model_rankings.json >> logs/calibration_cron.log 2>&1"
RESOLVER_LINE="${RESOLVER_TIME_UTC} cd ${WORKDIR} && ${PYTHON_BIN} scripts/daily_resolver.py >> logs/resolver.log 2>&1"
DATASYNC_LINE="${DATASYNC_TIME_UTC} cd ${WORKDIR} && bash scripts/git_push_data.sh >> logs/datasync.log 2>&1"

FILTER_PATTERN="deploy/healthcheck.sh|backtest_calibration.py|daily_resolver.py|git_push_data.sh"
(crontab -l 2>/dev/null | grep -Ev "${FILTER_PATTERN}" || true; \
  echo "${HEALTHCHECK_LINE}"; \
  echo "${CALIBRATION_LINE}"; \
  echo "${RESOLVER_LINE}"; \
  echo "${DATASYNC_LINE}") | crontab -

echo "Installed cron jobs:"
crontab -l
