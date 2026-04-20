"""Fetch live Weather Underground / Weather.com station observations.

This is the source-of-truth module for the WU observations API.
The same Weather.com endpoint powers the Polymarket resolution source
(wunderground.com/history/daily/...) and the intraday "today's max so far"
reading used for provisional prior-day settlement.

IMPORTANT: We use the v3/wx/observations/current endpoint which returns
`temperatureMaxSince7Am` - this is the same data source used by the WU
history page and is more reliable than the v1 historical observations API
which suffers from CDN caching issues.

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
# v3 current observations - same source as WU history page, fresher data
WU_CURRENT_OBS_URL = "https://api.weather.com/v3/wx/observations/current"
# v1 historical observations - kept as fallback
WU_HIST_OBS_URL = (
    "https://api.weather.com/v1/location/{station}/observations/historical.json"
)
WU_OBS_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"

# ICAO codes for supported stations
# The v3 API accepts icaoCode directly, v1 needs ICAO:9:COUNTRY format
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
# Short TTL since v3 current obs endpoint has fresher data
_CACHE_TTL_SEC = 60  # 1 minute
_fetch_cache: dict[str, dict[str, Any]] = {}


def fetch_wu_observed_max(
    station_icao: str,
    target_date: str | None = None,
    bypass_cache: bool = False,
) -> dict[str, Any] | None:
    """Fetch today's (or a specific date's) running max from WU observations.

    Uses the v3/wx/observations/current endpoint which provides temperatureMaxSince7Am.
    This is the same data source used by the WU history page and is more reliable
    than the v1 historical observations API which suffers from CDN caching issues.

    Args:
        station_icao: ICAO code (e.g., "LTAC" for Ankara)
        target_date: Optional ISO date string (YYYY-MM-DD), defaults to today.
                     Note: v3 current API only returns today's data - for historical
                     dates, we fall back to the v1 historical API.
        bypass_cache: If True, skip cache and fetch fresh data (use for critical
                      bet decisions where stale data could cause wrong bucket)

    Returns dict with keys:
        running_max      float   highest temp observed so far (temperatureMaxSince7Am)
        latest_temp      float   most recent reading
        n_obs            int     number of observation readings (estimated for v3)
        fetched_utc      str     ISO timestamp of this fetch
        last_obs_utc     str     ISO timestamp of most recent observation
        unit             str     "C" or "F"
        station_id       str     ICAO code
    or None if the station is unknown / fetch fails / no observations.
    """
    station_info = _ICAO_TO_WU.get(station_icao)
    if not station_info:
        return None
    station_id, units = station_info

    # For historical dates, use v1 historical API
    if target_date and target_date != date.today().isoformat():
        return _fetch_historical_obs(station_icao, station_id, units, target_date)

    cache_key = f"{station_icao}/today"
    if not bypass_cache:
        cached = _fetch_cache.get(cache_key)
        if cached and (time.monotonic() - cached["_mono"]) < _CACHE_TTL_SEC:
            return cached["data"]

    try:
        # Use v3 current observations - fresher than v1 historical
        resp = requests.get(
            WU_CURRENT_OBS_URL,
            params={
                "apiKey": WU_OBS_API_KEY,
                "icaoCode": station_icao,
                "units": units,
                "language": "en-US",
                "format": "json",
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()

        # v3 API returns temperatureMaxSince7Am which is what we need
        running_max = data.get("temperatureMaxSince7Am")
        if running_max is None:
            # Fallback to temperatureMax24Hour if maxSince7Am not available
            running_max = data.get("temperatureMax24Hour")
        if running_max is None:
            _LOG.warning("WU v3 %s: no max temperature in response", station_icao)
            return None

        latest_temp = data.get("temperature")
        valid_time_utc = data.get("validTimeUtc", 0)
        last_utc = (
            datetime.fromtimestamp(valid_time_utc, tz=timezone.utc).isoformat()
            if valid_time_utc
            else ""
        )

        result: dict[str, Any] = {
            "running_max": running_max,
            "latest_temp": latest_temp,
            "n_obs": -1,  # v3 doesn't provide observation count
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "last_obs_utc": last_utc,
            "unit": "C" if units == "m" else "F",
            "station_id": station_icao,
        }

        # Log when max temperature updates (for debugging resolution discrepancies)
        old_cached = _fetch_cache.get(cache_key, {}).get("data", {})
        old_max = old_cached.get("running_max")
        if old_max is not None and running_max != old_max:
            _LOG.info(
                "WU %s: max updated %s -> %s°%s (v3 current obs, last_obs=%s)",
                station_icao, old_max, running_max, result["unit"], last_utc
            )

        _fetch_cache[cache_key] = {"data": result, "_mono": time.monotonic()}
        return result

    except Exception as exc:
        _LOG.warning("WU v3 observation fetch failed for %s: %s", station_icao, exc)
        return None


def _fetch_historical_obs(
    station_icao: str,
    station_id: str,
    units: str,
    target_date: str,
) -> dict[str, Any] | None:
    """Fetch historical observations using v1 API (for non-today dates)."""
    day_str = target_date.replace("-", "")
    
    try:
        resp = requests.get(
            WU_HIST_OBS_URL.format(station=station_id),
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

        return {
            "running_max": max(temps),
            "latest_temp": last.get("temp"),
            "n_obs": len(temps),
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "last_obs_utc": last_utc,
            "unit": "C" if units == "m" else "F",
            "station_id": station_id,
        }

    except Exception as exc:
        _LOG.debug("WU v1 historical fetch failed for %s: %s", station_icao, exc)
        return None
