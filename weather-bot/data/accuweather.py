"""Optional AccuWeather side-channel logger client.

This client is intentionally read-only for snapshot logging and does not drive
live trading decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx


ACCUWEATHER_API_BASE = "https://dataservice.accuweather.com"


@dataclass(frozen=True)
class AccuWeatherSnapshot:
    station_icao: str
    city: str
    forecast_date: str
    forecast_high: float
    unit: str
    model_source: str = "accuweather_daily_1day"


class AccuWeatherClient:
    def __init__(self, api_key: str, timeout_seconds: float = 20.0) -> None:
        self.api_key = api_key
        self.http = httpx.AsyncClient(timeout=timeout_seconds, headers={"User-Agent": "WeatherBot/1.0"})
        self._location_key_cache: dict[str, str] = {}

    async def close(self) -> None:
        await self.http.aclose()

    async def get_daily_high_snapshot(self, station: dict[str, Any]) -> AccuWeatherSnapshot | None:
        location_key = await self._location_key_for_station(station)
        if not location_key:
            return None

        unit = "C" if station.get("resolution_unit") == "C" else "F"
        metric = "true" if unit == "C" else "false"
        response = await self.http.get(
            f"{ACCUWEATHER_API_BASE}/forecasts/v1/daily/1day/{location_key}",
            params={"apikey": self.api_key, "metric": metric},
        )
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
        return AccuWeatherSnapshot(
            station_icao=station["icao"],
            city=station["market_label"],
            forecast_date=forecast_date,
            forecast_high=forecast_high,
            unit=unit,
        )

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
        response.raise_for_status()
        payload = response.json()
        key = str(payload.get("Key", "")) if isinstance(payload, dict) else ""
        if not key:
            return None
        self._location_key_cache[station_icao] = key
        return key
