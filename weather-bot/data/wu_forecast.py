"""Weather Underground forecast client.

WU uses IBM GRAF (GFS + ECMWF + proprietary corrections) — the same forecast
the Polymarket retail crowd sees when they check the weather before betting.

By comparing our AI model's prediction against WU's forecast, we get a direct
measure of how wrong the crowd's weather app is — the true model delta.

API: WU's internal api.weather.com endpoint (same key used by wunderground.com).
Rate limits: ~500 calls/day on the embedded key. We cache aggressively.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

# The Weather Company API key embedded in Weather Underground's own website JS.
# This is not a secret — it's loaded in every browser that visits wunderground.com.
_WU_API_KEY = "6532d6454b8aa370768e63d6ba5a832e"
_WU_FORECAST_URL = "https://api.weather.com/v3/wx/forecast/daily/10day"

# Cache: (lat, lon, units) → {fetched_at, data}
_cache: dict[tuple, dict[str, Any]] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour — WU updates forecasts ~hourly


async def get_wu_daily_high(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    target_date: date,
    unit: str = "C",
) -> float | None:
    """Return WU's forecast high temperature for target_date.

    Args:
        client: shared httpx async client
        lat, lon: station coordinates
        target_date: the date we want the high for
        unit: 'C' (metric) or 'F' (imperial/english)

    Returns:
        Forecast high in the requested unit, or None if unavailable.
    """
    wu_units = "e" if unit == "F" else "m"
    cache_key = (round(lat, 4), round(lon, 4), wu_units)
    now = time.monotonic()

    cached = _cache.get(cache_key)
    if cached and (now - cached["fetched_at"]) < _CACHE_TTL_SECONDS:
        payload = cached["data"]
    else:
        try:
            resp = await client.get(
                _WU_FORECAST_URL,
                params={
                    "geocode": f"{lat},{lon}",
                    "format": "json",
                    "units": wu_units,
                    "language": "en-US",
                    "apiKey": _WU_API_KEY,
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            _cache[cache_key] = {"fetched_at": now, "data": payload}
        except (httpx.HTTPError, ValueError, KeyError):
            return None

    valid_times = payload.get("validTimeLocal", [])
    highs = payload.get("calendarDayTemperatureMax", [])

    for ts, high in zip(valid_times, highs):
        if high is None:
            continue
        try:
            day = datetime.fromisoformat(str(ts)).date()
        except (ValueError, TypeError):
            continue
        if day == target_date:
            return float(high)

    return None


async def get_wu_crowd_baseline(
    client: httpx.AsyncClient,
    stations: dict[str, dict],
    target_dates: list[str],
) -> dict[str, dict[str, float | None]]:
    """Fetch WU crowd baseline temps for multiple stations and dates.

    Returns:
        {station_icao: {date_iso: crowd_temp_or_None}}
    """
    results: dict[str, dict[str, float | None]] = {}
    tasks = []

    for icao, station in stations.items():
        lat = float(station["lat"])
        lon = float(station["lon"])
        unit = station.get("resolution_unit", "F")
        results[icao] = {}
        for date_iso in target_dates:
            try:
                target_date = date.fromisoformat(date_iso)
            except ValueError:
                results[icao][date_iso] = None
                continue
            tasks.append((icao, date_iso, lat, lon, unit, target_date))

    async def _fetch(icao: str, date_iso: str, lat: float, lon: float, unit: str, target_date: date) -> None:
        temp = await get_wu_daily_high(client, lat, lon, target_date, unit)
        results[icao][date_iso] = temp

    await asyncio.gather(*[_fetch(*t) for t in tasks])
    return results
