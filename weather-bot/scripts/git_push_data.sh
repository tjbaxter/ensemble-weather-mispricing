#!/usr/bin/env bash
# git_push_data.sh — auto-commit and push all data files to git after cron runs.
# The Streamlit Cloud dashboard reads these files from git, so pushing keeps it in sync.
#
# Usage: called at the end of each cron job, or run standalone.
#
# SSH key must already be configured for the repo on this machine.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# Pull first to avoid conflicts (cron jobs on VM are the only writer, but be safe)
git pull --rebase --autostash origin main 2>&1 | tail -3

# Stage all data files that change during bot/cron runs
git add \
  weather-bot/data/positions.json \
  weather-bot/data/polymarket_cache.json \
  weather-bot/data/commercial_forecast_log.json \
  weather-bot/data/model_snapshot_log.json \
  weather-bot/data/accuracy_rows_cache.json \
  weather-bot/data/morning_obs_cache.json \
  weather-bot/backtest/data/resolved_markets.json \
  weather-bot/backtest/data/resolved_markets.csv \
  2>/dev/null || true

# Log files need -f because logs/ is gitignored (exceptions added in .gitignore)
git add -f \
  weather-bot/logs/resolved.csv \
  weather-bot/logs/signals.csv \
  weather-bot/logs/trades.csv \
  2>/dev/null || true

# Only commit if there are actual changes
if git diff --cached --quiet; then
  echo "[git_push_data] No changes to commit."
  exit 0
fi

TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M UTC")
git commit -m "auto: data sync ${TIMESTAMP}"
git push origin main

echo "[git_push_data] Pushed data update at ${TIMESTAMP}"
