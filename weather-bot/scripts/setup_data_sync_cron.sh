#!/usr/bin/env bash
# setup_data_sync_cron.sh — install dashboard mirror sync cron jobs on the VM.
#
# The bot keeps writing data locally in /home/tombaxter/weather-bot.
# Fast syncs keep the public dashboard fresh; slower archive syncs handle deep
# logs and archive files without bloating every 5-minute push.

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/weather-bot}"
VENV_PYTHON="${VENV_PYTHON:-$WORKDIR/venv/bin/python3}"
LOG_DIR="$WORKDIR/logs"
FAST_SYNC_TIME="${FAST_SYNC_TIME:-*/5 * * * *}"
ARCHIVE_SYNC_TIME="${ARCHIVE_SYNC_TIME:-*/30 * * * *}"
RESOLVER_TIME_UTC="${RESOLVER_TIME_UTC:-0 10 * * *}"

mkdir -p "$LOG_DIR"

NEW_CRONS=$(cat <<EOF
# Weather bot — fast dashboard mirror sync every 5 minutes
${FAST_SYNC_TIME} ${WORKDIR}/deploy/sync_dashboard_data.sh
# Weather bot — archive/deep-log sync every 30 minutes
${ARCHIVE_SYNC_TIME} ${WORKDIR}/deploy/sync_archive_data.sh
# Weather bot — resolve yesterday's trades at 10:00 UTC, then push current + archive files
${RESOLVER_TIME_UTC} cd ${WORKDIR} && ${VENV_PYTHON} scripts/daily_resolver.py >> ${LOG_DIR}/resolver.log 2>&1 && bash deploy/sync_dashboard_data.sh >> ${LOG_DIR}/git_push.log 2>&1 && bash deploy/sync_archive_data.sh >> ${LOG_DIR}/git_push.log 2>&1
EOF
)

echo "Adding cron entries:"
echo "$NEW_CRONS"
echo ""

(
    crontab -l 2>/dev/null | grep -vE "git_push_data|daily_resolver|deploy/sync_dashboard_data.sh|deploy/sync_archive_data.sh|deploy/sync_data.sh" || true
    echo ""
    echo "$NEW_CRONS"
) | crontab -

echo "✓ Cron entries added"
echo ""
echo "Current crontab:"
crontab -l
