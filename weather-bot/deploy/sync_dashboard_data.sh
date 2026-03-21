#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

src_root = Path(os.environ.get("WEATHER_BOT_SRC_ROOT", "/home/tombaxter/weather-bot"))
dst_root = Path(os.environ.get("WEATHER_BOT_REPO_ROOT", "/home/tombaxter/repo/weather-bot"))

data_files = [
    "polymarket_cache.json",
    "commercial_forecast_log.json",
    "model_snapshot_log.json",
    "model_accuracy_log.json",
    "accuracy_rows_cache.json",
    "morning_obs_cache.json",
]
log_files = ["resolved.csv", "signals.csv", "trades.csv"]

copied: list[str] = []

for src in sorted((src_root / "data").glob("positions*.json")):
    dst = dst_root / "data" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(f"data/{src.name}")

for name in data_files:
    src = src_root / "data" / name
    if not src.exists():
        continue
    dst = dst_root / "data" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(f"data/{name}")

for name in log_files:
    src = src_root / "logs" / name
    if not src.exists():
        continue
    dst = dst_root / "logs" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(f"logs/{name}")

for src in sorted((src_root / "logs").glob("shadow_*/resolved.csv")):
    rel = src.relative_to(src_root)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(rel))

status_path = dst_root / "data" / "dashboard_sync_status.json"
status_path.parent.mkdir(parents=True, exist_ok=True)
status: dict[str, object]
try:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(status, dict):
        status = {}
except Exception:
    status = {}

status.update(
    {
        "last_fast_sync_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "fast_sync_file_count": len(copied),
        "fast_sync_files": copied,
    }
)
status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

REPO_ROOT="${WEATHER_BOT_REPO_ROOT:-/home/tombaxter/repo/weather-bot}"
cd "$(dirname "${REPO_ROOT}")"
bash "${REPO_ROOT}/scripts/git_push_data.sh" >> /tmp/git_sync.log 2>&1
