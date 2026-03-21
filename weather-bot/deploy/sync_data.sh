#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Backward-compatible wrapper: keep the old entrypoint working while the VM
# migrates to separate fast and archive sync cron jobs.
bash "${SCRIPT_DIR}/sync_dashboard_data.sh"
bash "${SCRIPT_DIR}/sync_archive_data.sh"
