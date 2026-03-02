#!/usr/bin/env bash
# setup_data_sync_cron.sh — add a cron to push data files to git every 30 min
#
# The always-on bot writes positions.json, signals.csv, resolved.csv, etc.
# continuously. This cron pushes those to GitHub so Streamlit Cloud sees them.
# Run ONCE on the VM after installing the systemd service.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PUSH_SCRIPT="$REPO_DIR/weather-bot/scripts/git_push_data.sh"
RESOLVER_SCRIPT="$REPO_DIR/weather-bot/scripts/daily_resolver.py"
VENV_PYTHON="$REPO_DIR/weather-bot/venv/bin/python3"
LOG_DIR="$REPO_DIR/weather-bot/logs"

# Ensure log dir exists
mkdir -p "$LOG_DIR"

# Build the new cron entries
NEW_CRONS=$(cat <<EOF
# Weather bot — push data to GitHub every 30 min so Streamlit Cloud stays fresh
*/30 * * * * cd $REPO_DIR && bash $PUSH_SCRIPT >> $LOG_DIR/git_push.log 2>&1
# Weather bot — resolve yesterday's trades at 10:30 UTC (WU data usually ready by then)
30 10 * * * cd $REPO_DIR && $VENV_PYTHON $RESOLVER_SCRIPT >> $LOG_DIR/resolver.log 2>&1 && bash $PUSH_SCRIPT >> $LOG_DIR/git_push.log 2>&1
EOF
)

echo "Adding cron entries:"
echo "$NEW_CRONS"
echo ""

# Merge with existing crontab (remove old resolver/push entries first to avoid dupes)
(
    crontab -l 2>/dev/null | grep -vE "git_push_data|daily_resolver" || true
    echo ""
    echo "$NEW_CRONS"
) | crontab -

echo "✓ Cron entries added"
echo ""
echo "Current crontab:"
crontab -l
