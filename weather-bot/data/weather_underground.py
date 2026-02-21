"""Weather Underground historical verification utilities."""

from __future__ import annotations

import re
from datetime import date

import httpx


HIGH_TEMP_PATTERN = re.compile(r'"temperatureMax":\{"value":(-?\d+(?:\.\d+)?)')


class WeatherUndergroundClient:
    def __init__(self) -> None:
        self.http = httpx.AsyncClient(timeout=20.0)

    async def close(self) -> None:
        await self.http.aclose()

    async def get_daily_high(self, base_url: str, day: date) -> int | None:
        """Fetch historical page and parse finalized daily max temperature."""
        url = f"{base_url}/date/{day.isoformat()}"
        response = await self.http.get(url, follow_redirects=True)
        response.raise_for_status()

        match = HIGH_TEMP_PATTERN.search(response.text)
        if not match:
            return None
        try:
            return int(round(float(match.group(1))))
        except ValueError:
            return None
