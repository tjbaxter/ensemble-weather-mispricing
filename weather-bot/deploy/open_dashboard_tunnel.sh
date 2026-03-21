#!/usr/bin/env bash
set -euo pipefail

VM_NAME="${VM_NAME:-weather-bot}"
ZONE="${ZONE:-us-east1-b}"
PROJECT="${PROJECT:-weather-488111}"
LOCAL_PORT="${LOCAL_PORT:-8501}"
REMOTE_PORT="${REMOTE_PORT:-8501}"

exec gcloud compute ssh "${VM_NAME}" \
  --zone "${ZONE}" \
  --project "${PROJECT}" \
  -- -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"
