#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="weather-bot"
JOURNAL_IDENTIFIER="${JOURNAL_IDENTIFIER:-${SERVICE_NAME}}"
WORKDIR="${HOME}/weather-bot"
LOG_FILE="${WORKDIR}/logs/bot.log"
HEALTH_LOG="${WORKDIR}/logs/healthcheck.log"
MAX_AGE_SECONDS=600
STARTUP_GRACE_SECONDS=1200

mkdir -p "${WORKDIR}/logs"

log_health() {
  echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') $*" >> "${HEALTH_LOG}"
}

if ! systemctl is-active --quiet "${SERVICE_NAME}"; then
  log_health "service_inactive restarting_service=${SERVICE_NAME}"
  sudo systemctl restart "${SERVICE_NAME}"
  exit 0
fi

MAIN_PID="$(systemctl show -p MainPID --value "${SERVICE_NAME}" 2>/dev/null || echo 0)"
UPTIME_SECONDS=""
if [[ -n "${MAIN_PID}" && "${MAIN_PID}" != "0" ]]; then
  UPTIME_SECONDS="$(ps -o etimes= -p "${MAIN_PID}" 2>/dev/null | tr -d ' ' || true)"
fi
if [[ -n "${UPTIME_SECONDS}" ]] && [[ "${UPTIME_SECONDS}" -lt "${STARTUP_GRACE_SECONDS}" ]]; then
  log_health "startup_grace pid=${MAIN_PID} uptime_s=${UPTIME_SECONDS}"
  exit 0
fi

if HEALTH_REASON="$(
  python3 - "${JOURNAL_IDENTIFIER}" "${LOG_FILE}" "${MAX_AGE_SECONDS}" <<'PY'
import datetime as dt
import os
import subprocess
import sys

journal_identifier = sys.argv[1]
log_file = sys.argv[2]
max_age = int(sys.argv[3])


def recent_journal_activity() -> tuple[bool, str]:
    try:
        output = subprocess.check_output(
            [
                "journalctl",
                "-t",
                journal_identifier,
                "--since",
                f"-{max_age} seconds",
                "--no-pager",
                "-n",
                "200",
                "-o",
                "short-iso",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        return False, f"journal_error={type(exc).__name__}"

    lines = [line for line in output.splitlines() if line.strip()]
    if lines:
        return True, f"journal_lines={len(lines)}"
    return False, "journal_stale"


def recent_file_activity() -> tuple[bool, str]:
    if not os.path.exists(log_file):
        return False, "file_missing"

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
            return True, f"file_mtime_age_s={int(age)}"
        return False, f"file_mtime_age_s={int(age)}"

    prefix = last.split(" | ", 1)[0]
    try:
        ts = dt.datetime.strptime(prefix, "%Y-%m-%d %H:%M:%S,%f").replace(
            tzinfo=dt.timezone.utc
        )
    except ValueError:
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(log_file), tz=dt.timezone.utc)
        age = (dt.datetime.now(dt.timezone.utc) - mtime).total_seconds()
        if age <= max_age:
            return True, f"file_parse_fallback_age_s={int(age)}"
        return False, f"file_parse_fallback_age_s={int(age)}"

    age = (dt.datetime.now(dt.timezone.utc) - ts).total_seconds()
    if age <= max_age:
        return True, f"file_heartbeat_age_s={int(age)}"
    return False, f"file_heartbeat_age_s={int(age)}"


journal_ok, journal_reason = recent_journal_activity()
if journal_ok:
    print(journal_reason)
    raise SystemExit(0)

file_ok, file_reason = recent_file_activity()
if file_ok:
    print(file_reason)
    raise SystemExit(0)

print(f"{journal_reason} {file_reason}")
raise SystemExit(1)
PY
)"; then
  exit 0
fi

log_health "stale_or_missing_activity reason=\"${HEALTH_REASON}\" restarting_service=${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"
exit 0

