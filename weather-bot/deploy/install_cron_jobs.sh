#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/weather-bot}"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/venv/bin/python3}"
PAST_DAYS="${PAST_DAYS:-45}"
CALIBRATION_TIME_UTC="${CALIBRATION_TIME_UTC:-35 2 * * *}" # 02:35 UTC daily
HEALTHCHECK_TIME="${HEALTHCHECK_TIME:-*/5 * * * *}" # every 5 min
RESOLVER_TIME_UTC="${RESOLVER_TIME_UTC:-0 10 * * *}" # 10:00 UTC daily
FAST_SYNC_TIME="${FAST_SYNC_TIME:-*/5 * * * *}"
ARCHIVE_SYNC_TIME="${ARCHIVE_SYNC_TIME:-*/30 * * * *}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python binary: ${PYTHON_BIN}"
  exit 1
fi

mkdir -p "${WORKDIR}/logs"

HEALTHCHECK_LINE="${HEALTHCHECK_TIME} ${WORKDIR}/deploy/healthcheck.sh"
CALIBRATION_LINE="${CALIBRATION_TIME_UTC} cd ${WORKDIR} && ${PYTHON_BIN} scripts/backtest_calibration.py --past-days ${PAST_DAYS} --output logs/calibration.json --rankings-output logs/model_rankings.json >> logs/calibration_cron.log 2>&1"
RESOLVER_LINE="${RESOLVER_TIME_UTC} cd ${WORKDIR} && ${PYTHON_BIN} scripts/daily_resolver.py >> logs/resolver.log 2>&1 && bash deploy/sync_dashboard_data.sh >> logs/git_push.log 2>&1 && bash deploy/sync_archive_data.sh >> logs/git_push.log 2>&1"
FAST_SYNC_LINE="${FAST_SYNC_TIME} ${WORKDIR}/deploy/sync_dashboard_data.sh"
ARCHIVE_SYNC_LINE="${ARCHIVE_SYNC_TIME} ${WORKDIR}/deploy/sync_archive_data.sh"

FILTER_PATTERN="deploy/healthcheck.sh|backtest_calibration.py|daily_resolver.py|git_push_data.sh|deploy/sync_dashboard_data.sh|deploy/sync_archive_data.sh|deploy/sync_data.sh"
(crontab -l 2>/dev/null | grep -Ev "${FILTER_PATTERN}" || true; \
  echo "${HEALTHCHECK_LINE}"; \
  echo "${CALIBRATION_LINE}"; \
  echo "${RESOLVER_LINE}"; \
  echo "${FAST_SYNC_LINE}"; \
  echo "${ARCHIVE_SYNC_LINE}") | crontab -

echo "Installed cron jobs:"
crontab -l
