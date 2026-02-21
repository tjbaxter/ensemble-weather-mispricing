#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/weather-bot"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python3 -m pip install -q -r requirements.txt
python3 main.py "$@"
