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

# Stage files defensively: missing files should not abort staging.
add_if_exists() {
  local flags=()
  if [[ "${1:-}" == "-f" ]]; then
    flags+=("-f")
    shift
  fi
  for rel in "$@"; do
    if [[ -f "$rel" ]]; then
      git add "${flags[@]}" "$rel"
    fi
  done
}

add_if_exists \
  weather-bot/data/positions.json \
  weather-bot/data/positions_live.json \
  weather-bot/data/positions_shadow_2a.json \
  weather-bot/data/positions_shadow_2b.json \
  weather-bot/data/positions_shadow_2c.json \
  weather-bot/data/positions_shadow_purdey.json \
  weather-bot/data/positions_shadow_cavendish.json \
  weather-bot/data/positions_shadow_purdey2.json \
  weather-bot/data/positions_shadow_cavendish3.json \
  weather-bot/data/positions_shadow_true_alpha.json \
  weather-bot/data/positions_shadow_props_kelly.json \
  weather-bot/data/model_accuracy_log.json \
  weather-bot/data/polymarket_cache.json \
  weather-bot/data/commercial_forecast_log.json \
  weather-bot/data/model_snapshot_log.json \
  weather-bot/data/accuracy_rows_cache.json \
  weather-bot/data/morning_obs_cache.json \
  weather-bot/backtest/data/resolved_markets.json \
  weather-bot/backtest/data/resolved_markets.csv

# Log files need -f because logs/ is gitignored (exceptions added in .gitignore)
add_if_exists -f \
  weather-bot/logs/resolved.csv \
  weather-bot/logs/signals.csv \
  weather-bot/logs/trades.csv \
  weather-bot/logs/shadow_2a/resolved.csv \
  weather-bot/logs/shadow_2b/resolved.csv \
  weather-bot/logs/shadow_2c/resolved.csv \
  weather-bot/logs/shadow_purdey/resolved.csv \
  weather-bot/logs/shadow_cavendish/resolved.csv \
  weather-bot/logs/shadow_purdey2/resolved.csv \
  weather-bot/logs/shadow_cavendish3/resolved.csv \
  weather-bot/logs/shadow_true_alpha/resolved.csv \
  weather-bot/logs/shadow_props_kelly/resolved.csv

# Only commit if there are actual changes
if git diff --cached --quiet; then
  echo "[git_push_data] No changes to commit."
  exit 0
fi

TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M UTC")
git commit -m "auto: data sync ${TIMESTAMP}"
git push origin main

echo "[git_push_data] Pushed data update at ${TIMESTAMP}"
