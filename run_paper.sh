#!/bin/bash
set -euo pipefail

export PAPER_TRADING=true
export LIVE_TRADING=false

exec "$(dirname "$0")/run.sh" "$@"
