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

copied: list[str] = []

for src in sorted((src_root / "data").glob("positions_archive_*.json")):
    dst = dst_root / "data" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(f"data/{src.name}")

for src in sorted((src_root / "logs").glob("resolved_archive_*.csv")):
    dst = dst_root / "logs" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(f"logs/{src.name}")

for src in sorted((src_root / "logs").glob("shadow_*/resolved*.csv")):
    rel = src.relative_to(src_root)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(rel))

deep_dir = src_root / "logs" / "deep"
if deep_dir.exists():
    for src in sorted(deep_dir.glob("*.jsonl")):
        dst = dst_root / "logs" / "deep" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(f"logs/deep/{src.name}")

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
        "last_archive_sync_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "archive_sync_file_count": len(copied),
    }
)
status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

REPO_ROOT="${WEATHER_BOT_REPO_ROOT:-/home/tombaxter/repo/weather-bot}"
SRC_ROOT="${WEATHER_BOT_SRC_ROOT:-/home/tombaxter/weather-bot}"
LOG_PATH="${WEATHER_BOT_SYNC_LOG:-${SRC_ROOT}/logs/git_push.log}"
mkdir -p "$(dirname "${LOG_PATH}")"
cd "$(dirname "${REPO_ROOT}")"
printf '\n[%s] sync_archive_data start\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "${LOG_PATH}"
bash "${REPO_ROOT}/scripts/git_push_data.sh" >> "${LOG_PATH}" 2>&1
