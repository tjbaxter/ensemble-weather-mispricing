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

for rel in weather-bot/data/positions*.json; do
  if [[ -f "$rel" ]]; then
    git add "$rel"
  fi
done

add_if_exists \
  weather-bot/data/dashboard_sync_status.json \
  weather-bot/data/model_accuracy_log.json \
  weather-bot/data/polymarket_cache.json \
  weather-bot/data/commercial_forecast_log.json \
  weather-bot/data/model_snapshot_log.json \
  weather-bot/data/settlement_snapshot.json \
  weather-bot/data/settlement_status.json \
  weather-bot/data/settlement_summary.json \
  weather-bot/data/trade_observability.jsonl \
  weather-bot/data/accuracy_rows_cache.json \
  weather-bot/data/morning_obs_cache.json \
  weather-bot/backtest/data/resolved_markets.json \
  weather-bot/backtest/data/resolved_markets.csv

# Log files need -f because logs/ is gitignored (exceptions added in .gitignore)
add_if_exists -f \
  weather-bot/logs/signals.csv \
  weather-bot/logs/trades.csv

for rel in weather-bot/logs/resolved*.csv weather-bot/logs/shadow_*/resolved*.csv; do
  if [[ -f "$rel" ]]; then
    git add -f "$rel"
  fi
done

# Deep observability logs (append-only JSONL, gitignored logs/ path needs -f)
for rel in weather-bot/logs/deep/*.jsonl; do
  if [[ -f "$rel" ]]; then
    git add -f "$rel"
  fi
done

created_commit=0
if ! git diff --cached --quiet; then
  TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M UTC")
  git commit -m "auto: data sync ${TIMESTAMP}"
  created_commit=1
fi

ahead_count="$(git rev-list --count @{u}..HEAD 2>/dev/null || echo 0)"
if [[ "${ahead_count}" -gt 0 ]]; then
  if [[ "${created_commit}" -eq 0 ]]; then
    echo "[git_push_data] Branch already ahead by ${ahead_count} commit(s); retrying push."
  fi
  git push origin main
  echo "[git_push_data] Pushed data update to origin/main."
  exit 0
fi

if [[ "${created_commit}" -eq 1 ]]; then
  echo "[git_push_data] Commit created but branch is not ahead of origin/main."
else
  echo "[git_push_data] No changes to commit or push."
fi
