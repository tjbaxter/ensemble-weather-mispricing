"""Optional AccuWeather side-channel logger client.

This client is intentionally read-only for snapshot logging and does not drive
live trading decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
import time
from typing import Any

import httpx


ACCUWEATHER_API_BASE = "https://dataservice.accuweather.com"
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 6 * 60 * 60

# Stable mapping for our station set so we never need repeated geoposition lookups.
KNOWN_LOCATION_KEYS: dict[str, str] = {
    "EGLC": "2532754",
    "KATL": "2140212",
    "KDFW": "336107",
    "KLGA": "2627477",
    "KMIA": "3593859",
    "KORD": "2626577",
    "KSEA": "341357",
    "LFPG": "159190",
    "RKSI": "2331998",
    "SBGR": "36369",
}


@dataclass(frozen=True)
class AccuWeatherSnapshot:
    station_icao: str
    city: str
    forecast_date: str
    forecast_high: float
    unit: str
    model_source: str = "accuweather_daily_1day"


class AccuWeatherClient:
    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 20.0,
        location_cache_path: str = "data/accuweather_location_keys.json",
        forecast_cache_path: str = "data/accuweather_forecast_cache.json",
        meta_cache_path: str = "data/accuweather_meta.json",
        forecast_ttl_seconds: int = 6 * 60 * 60,
        rate_limit_cooldown_seconds: int = DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.http = httpx.AsyncClient(timeout=timeout_seconds, headers={"User-Agent": "WeatherBot/1.0"})
        self.location_cache_path = Path(location_cache_path)
        self.forecast_cache_path = Path(forecast_cache_path)
        self.meta_cache_path = Path(meta_cache_path)
        self.forecast_ttl_seconds = max(60, int(forecast_ttl_seconds))
        self.rate_limit_cooldown_seconds = max(60, int(rate_limit_cooldown_seconds))

        self._location_key_cache: dict[str, str] = dict(KNOWN_LOCATION_KEYS)
        self._location_key_cache.update(self._load_json(self.location_cache_path))
        self._forecast_cache: dict[str, dict[str, Any]] = self._load_json(self.forecast_cache_path)
        self._meta_cache: dict[str, Any] = self._load_json(self.meta_cache_path)
        try:
            self._rate_limited_until_unix = int(self._meta_cache.get("rate_limited_until_unix", 0))
        except (TypeError, ValueError):
            self._rate_limited_until_unix = 0

    async def close(self) -> None:
        await self.http.aclose()

    async def get_daily_high_snapshot(self, station: dict[str, Any]) -> AccuWeatherSnapshot | None:
        if self._is_rate_limited():
            return None

        station_icao = str(station.get("icao", ""))
        unit = "C" if station.get("resolution_unit") == "C" else "F"
        cached_snapshot = self._snapshot_from_cache(station_icao, expected_unit=unit)
        if cached_snapshot is not None:
            return cached_snapshot

        location_key = await self._location_key_for_station(station)
        if not location_key:
            return None

        metric = "true" if unit == "C" else "false"
        response = await self.http.get(
            f"{ACCUWEATHER_API_BASE}/forecasts/v1/daily/1day/{location_key}",
            params={"apikey": self.api_key, "metric": metric},
        )
        if response.status_code in {403, 429}:
            self._mark_rate_limited()
            return None
        response.raise_for_status()
        payload = response.json()
        forecasts = payload.get("DailyForecasts", []) if isinstance(payload, dict) else []
        if not forecasts:
            return None
        first = forecasts[0]
        date_raw = str(first.get("Date", ""))
        try:
            forecast_date = datetime.fromisoformat(date_raw.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            forecast_date = date_raw[:10] if len(date_raw) >= 10 else ""
        temp = first.get("Temperature", {}).get("Maximum", {}).get("Value")
        if temp is None:
            return None
        try:
            forecast_high = float(temp)
        except (TypeError, ValueError):
            return None
        snapshot = AccuWeatherSnapshot(
            station_icao=station["icao"],
            city=station["market_label"],
            forecast_date=forecast_date,
            forecast_high=forecast_high,
            unit=unit,
        )
        ttl_seconds = self._cache_ttl_from_headers(response.headers)
        self._forecast_cache[station_icao] = {
            "cached_at_unix": int(time.time()),
            "ttl_seconds": ttl_seconds,
            "station_icao": snapshot.station_icao,
            "city": snapshot.city,
            "forecast_date": snapshot.forecast_date,
            "forecast_high": snapshot.forecast_high,
            "unit": snapshot.unit,
            "model_source": snapshot.model_source,
        }
        self._save_json(self.forecast_cache_path, self._forecast_cache)
        return snapshot

    async def _location_key_for_station(self, station: dict[str, Any]) -> str | None:
        station_icao = str(station.get("icao", ""))
        if station_icao in self._location_key_cache:
            return self._location_key_cache[station_icao]

        lat = station.get("lat")
        lon = station.get("lon")
        if lat is None or lon is None:
            return None

        response = await self.http.get(
            f"{ACCUWEATHER_API_BASE}/locations/v1/cities/geoposition/search",
            params={"apikey": self.api_key, "q": f"{lat},{lon}"},
        )
        if response.status_code in {403, 429}:
            self._mark_rate_limited()
            return None
        response.raise_for_status()
        payload = response.json()
        key = str(payload.get("Key", "")) if isinstance(payload, dict) else ""
        if not key:
            return None
        self._location_key_cache[station_icao] = key
        self._save_json(self.location_cache_path, self._location_key_cache)
        return key

    def _snapshot_from_cache(self, station_icao: str, expected_unit: str) -> AccuWeatherSnapshot | None:
        item = self._forecast_cache.get(station_icao)
        if not isinstance(item, dict):
            return None
        try:
            cached_at_unix = int(item.get("cached_at_unix", 0))
        except (TypeError, ValueError):
            return None
        if cached_at_unix <= 0:
            return None
        age_seconds = int(time.time()) - cached_at_unix
        ttl_seconds = self.forecast_ttl_seconds
        try:
            ttl_seconds = max(60, int(item.get("ttl_seconds", self.forecast_ttl_seconds)))
        except (TypeError, ValueError):
            ttl_seconds = self.forecast_ttl_seconds
        if age_seconds > ttl_seconds:
            return None
        unit = str(item.get("unit", ""))
        if unit != expected_unit:
            return None
        try:
            return AccuWeatherSnapshot(
                station_icao=str(item.get("station_icao", station_icao)),
                city=str(item.get("city", "")),
                forecast_date=str(item.get("forecast_date", "")),
                forecast_high=float(item.get("forecast_high")),
                unit=unit,
                model_source=str(item.get("model_source", "accuweather_daily_1day")),
            )
        except (TypeError, ValueError):
            return None

    def _load_json(self, path: Path) -> dict[str, Any]:
        try:
            if not path.exists():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _save_json(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp.replace(path)
        except OSError:
            # Cache persistence is optional; skip on filesystem errors.
            return

    def _is_rate_limited(self) -> bool:
        return int(time.time()) < self._rate_limited_until_unix

    def _mark_rate_limited(self) -> None:
        self._rate_limited_until_unix = int(time.time()) + self.rate_limit_cooldown_seconds
        self._meta_cache["rate_limited_until_unix"] = self._rate_limited_until_unix
        self._save_json(self.meta_cache_path, self._meta_cache)

    def _cache_ttl_from_headers(self, headers: httpx.Headers) -> int:
        expires = headers.get("Expires")
        if not expires:
            return self.forecast_ttl_seconds
        try:
            expires_at = parsedate_to_datetime(expires)
            now = datetime.now(expires_at.tzinfo)
            ttl = int((expires_at - now).total_seconds())
            return max(60, ttl)
        except (TypeError, ValueError, OverflowError):
            return self.forecast_ttl_seconds
