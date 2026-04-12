"""Fetch live Weather Underground / Weather.com station observations.

This is the source-of-truth module for the WU observations API.
The same Weather.com endpoint powers the Polymarket resolution source
(wunderground.com/history/daily/...) and the intraday "today's max so far"
reading used for provisional prior-day settlement.

API constants are centralised here; dashboard.py should migrate to importing
from this module rather than maintaining its own copy.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from typing import Any

import requests

_LOG = logging.getLogger("weather-bot.wu_observations")

# ── Weather.com observations API ────────────────────────────────────────────────
WU_OBS_API_URL = (
    "https://api.weather.com/v1/location/{station}/observations/historical.json"
)
WU_OBS_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"

# ICAO → (WU station ID in ICAO:9:COUNTRY format, unit flag)
# unit: "m" = metric (°C), "e" = english (°F)
_ICAO_TO_WU: dict[str, tuple[str, str]] = {
    "RKSI": ("RKSI:9:KR", "m"),
    "EGLC": ("EGLC:9:GB", "m"),
    "KLGA": ("KLGA:9:US", "e"),
    "KATL": ("KATL:9:US", "e"),
    "KORD": ("KORD:9:US", "e"),
    "KMIA": ("KMIA:9:US", "e"),
    "KDFW": ("KDFW:9:US", "e"),
    "SAEZ": ("SAEZ:9:AR", "m"),
    "LFPG": ("LFPG:9:FR", "m"),
    "CYYZ": ("CYYZ:9:CA", "m"),
    "KSEA": ("KSEA:9:US", "e"),
    "LTAC": ("LTAC:9:TR", "m"),
    "NZWN": ("NZWN:9:NZ", "m"),
}

# ── Per-station fetch cache ─────────────────────────────────────────────────────
_CACHE_TTL_SEC = 300  # 5 minutes
_fetch_cache: dict[str, dict[str, Any]] = {}


def fetch_wu_observed_max(
    station_icao: str,
    target_date: str | None = None,
) -> dict[str, Any] | None:
    """Fetch today's (or a specific date's) running max from WU observations.

    Returns dict with keys:
        running_max      float   highest temp observed so far
        latest_temp      float   most recent reading
        n_obs            int     number of observation readings
        fetched_utc      str     ISO timestamp of this fetch
        last_obs_utc     str     ISO timestamp of most recent observation
        unit             str     "C" or "F"
        station_id       str     WU station ID used
    or None if the station is unknown / fetch fails / no observations.
    """
    station_info = _ICAO_TO_WU.get(station_icao)
    if not station_info:
        return None
    station_id, units = station_info

    cache_key = f"{station_icao}/{target_date or 'today'}"
    cached = _fetch_cache.get(cache_key)
    if cached and (time.monotonic() - cached["_mono"]) < _CACHE_TTL_SEC:
        return cached["data"]

    if target_date:
        day_str = target_date.replace("-", "")
    else:
        day_str = date.today().strftime("%Y%m%d")

    try:
        resp = requests.get(
            WU_OBS_API_URL.format(station=station_id),
            params={
                "apiKey": WU_OBS_API_KEY,
                "units": units,
                "startDate": day_str,
                "endDate": day_str,
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        if not obs:
            return None

        temps = [o["temp"] for o in obs if o.get("temp") is not None]
        if not temps:
            return None

        last = obs[-1]
        last_epoch = last.get("valid_time_gmt", 0)
        last_utc = (
            datetime.fromtimestamp(last_epoch, tz=timezone.utc).isoformat()
            if last_epoch
            else ""
        )

        result: dict[str, Any] = {
            "running_max": max(temps),
            "latest_temp": last.get("temp"),
            "n_obs": len(temps),
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "last_obs_utc": last_utc,
            "unit": "C" if units == "m" else "F",
            "station_id": station_id,
        }

        _fetch_cache[cache_key] = {"data": result, "_mono": time.monotonic()}
        return result

    except Exception as exc:
        _LOG.debug("WU observation fetch failed for %s: %s", station_icao, exc)
        return None
