#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="weather-bot"
WORKDIR="${HOME}/weather-bot"
LOG_FILE="${WORKDIR}/logs/bot.log"
HEALTH_LOG="${WORKDIR}/logs/healthcheck.log"
MAX_AGE_SECONDS=600
STARTUP_GRACE_SECONDS=1200

mkdir -p "${WORKDIR}/logs"

if [[ ! -f "${LOG_FILE}" ]]; then
  echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') missing_log_file=${LOG_FILE}" >> "${HEALTH_LOG}"
  sudo systemctl restart "${SERVICE_NAME}"
  exit 0
fi

MAIN_PID="$(systemctl show -p MainPID --value "${SERVICE_NAME}" 2>/dev/null || echo 0)"
UPTIME_SECONDS=""
if [[ -n "${MAIN_PID}" && "${MAIN_PID}" != "0" ]]; then
  UPTIME_SECONDS="$(ps -o etimes= -p "${MAIN_PID}" 2>/dev/null | tr -d ' ' || true)"
fi
if [[ -n "${UPTIME_SECONDS}" ]] && [[ "${UPTIME_SECONDS}" -lt "${STARTUP_GRACE_SECONDS}" ]]; then
  echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') startup_grace pid=${MAIN_PID} uptime_s=${UPTIME_SECONDS}" >> "${HEALTH_LOG}"
  exit 0
fi

if python3 - "${LOG_FILE}" "${MAX_AGE_SECONDS}" <<'PY'
import datetime as dt
import os
import sys

log_file = sys.argv[1]
max_age = int(sys.argv[2])
last = None

with open(log_file, "rb") as f:
    text = f.read().replace(b"\x00", b"").decode("utf-8", "ignore")
for line in text.splitlines():
    if "HEARTBEAT" in line:
        last = line

if last is None:
    # Fallback: if no explicit heartbeat exists, accept recent log activity.
    mtime = dt.datetime.fromtimestamp(os.path.getmtime(log_file), tz=dt.timezone.utc)
    age = (dt.datetime.now(dt.timezone.utc) - mtime).total_seconds()
    if age <= max_age:
        raise SystemExit(0)
    raise SystemExit(2)

prefix = last.split(" | ", 1)[0]
ts = dt.datetime.strptime(prefix, "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=dt.timezone.utc)
age = (dt.datetime.now(dt.timezone.utc) - ts).total_seconds()

if age > max_age:
    raise SystemExit(3)
raise SystemExit(0)
PY
then
  exit 0
fi

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') stale_or_missing_heartbeat restarting_service=${SERVICE_NAME}" >> "${HEALTH_LOG}"
sudo systemctl restart "${SERVICE_NAME}"
