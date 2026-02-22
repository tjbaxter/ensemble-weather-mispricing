"""Station-first forecast client with NWS caching + METAR precision."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from config.cities import STATIONS
from config.settings import (
    ENABLE_ENSEMBLE_FORECASTS,
    ENSEMBLE_ADDITIONAL_MODELS,
    ENSEMBLE_CONFIDENCE_STD_HIGH,
    ENSEMBLE_CONFIDENCE_STD_LOW,
    ENSEMBLE_DAILY_API_URL,
    ENSEMBLE_PRIMARY_MODEL,
    ENSEMBLE_STD_SKIP_THRESHOLD,
    HIGH_DELTA_MEAN_SHIFT_DEG,
    HIGH_DELTA_THRESHOLD_DEG,
    MET_OFFICE_API_URL,
    NOAA_API_URL,
    NWS_CACHE_TTL_SECONDS,
)
from data.forecast_ensemble import EnsembleForecastClient
from data.metar_parser import parse_metar_temp_c
from data.probability import ensemble_to_bucket_probs
from data.wu_forecast import get_wu_daily_high
from data.wu_rounding import is_boundary_temperature, predict_wu_display_celsius, predict_wu_display_fahrenheit


AVIATION_WEATHER_API = "https://aviationweather.gov/api/data/metar"


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def _parse_bucket_bounds(bucket: str) -> tuple[float | None, float | None]:
    clean = bucket.replace("°F", "").replace("°C", "").strip()
    if "+" in clean:
        return float(clean.replace("+", "")), None
    if "-" in clean:
        left, right = clean.split("-", 1)
        return float(left.strip()), float(right.strip())
    value = float(clean)
    return value, value


def bucket_probabilities(bucket_labels: list[str], temp_mean: float, sigma: float) -> dict[str, float]:
    probs: dict[str, float] = {}
    total = 0.0
    for label in bucket_labels:
        low, high = _parse_bucket_bounds(label)
        if high is None:
            p = 1.0 - _normal_cdf(low - 0.5, temp_mean, sigma)
        else:
            p = _normal_cdf(high + 0.5, temp_mean, sigma) - _normal_cdf(low - 0.5, temp_mean, sigma)
        p = max(0.0, min(1.0, p))
        probs[label] = p
        total += p
    if total <= 0:
        return {b: 1.0 / len(bucket_labels) for b in bucket_labels} if bucket_labels else {}
    return {k: v / total for k, v in probs.items()}


def sigma_for_station(station: dict[str, Any], lead_hours: float) -> float:
    unit = station["resolution_unit"]
    if unit == "F":
        if lead_hours <= 12:
            return 2.0
        if lead_hours <= 24:
            return 3.5
        return 5.0
    if lead_hours <= 12:
        return 1.5
    if lead_hours <= 24:
        return 2.0
    return 3.0


class StationForecaster:
    def __init__(self, met_office_api_key: str | None = None) -> None:
        self.met_office_api_key = met_office_api_key
        self.http = httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "WeatherBot/1.0 (contact@example.com)"})
        self._nws_cache: dict[tuple[str, int, int], tuple[float, list[dict[str, Any]]]] = {}
        self.ensemble = EnsembleForecastClient()

    async def close(self) -> None:
        await self.ensemble.close()
        await self.http.aclose()

    async def get_latest_metar(self, icao: str) -> dict[str, Any] | None:
        response = await self.http.get(AVIATION_WEATHER_API, params={"ids": icao, "format": "json"})
        response.raise_for_status()
        data = response.json()
        return data[0] if data else None

    async def get_station_forecast(self, station_icao: str, target_date: date, bucket_labels: list[str]) -> dict[str, Any]:
        station = STATIONS[station_icao]
        models = [ENSEMBLE_PRIMARY_MODEL, *ENSEMBLE_ADDITIONAL_MODELS]
        if ENABLE_ENSEMBLE_FORECASTS:
            try:
                members, model_used, primary_temp, baseline_temp = await self.ensemble.get_weighted_daily_max_members(
                    station=station,
                    target_date=target_date.isoformat(),
                    models=models,
                )
            except httpx.HTTPError:
                members = []
                model_used = ""
                primary_temp = None
                baseline_temp = None
            if members:
                temp_mean = float(mean(members))
                ensemble_std = float(pstdev(members)) if len(members) > 1 else 0.0

                # Apply AI mean-reversion bias correction during high-delta regime.
                # AI models (trained with MSE loss) systematically under-predict rapid
                # warming peaks and over-predict rapid cooling troughs. Shift the
                # distribution centre to correct — letting EV logic pick ONE bucket,
                # not a spread. Factor starts at 1.0°C empirical; update from calibration.
                adjusted_mean = temp_mean
                if primary_temp is not None and baseline_temp is not None:
                    delta = primary_temp - baseline_temp
                    if abs(delta) >= HIGH_DELTA_THRESHOLD_DEG:
                        direction = 1.0 if delta > 0 else -1.0
                        adjusted_mean = temp_mean + direction * HIGH_DELTA_MEAN_SHIFT_DEG

                # Shift all ensemble members by the bias correction offset so
                # the KDE distribution centre moves without changing its shape.
                mean_shift = adjusted_mean - temp_mean
                shifted_members = [m + mean_shift for m in members] if mean_shift != 0.0 else members
                probs = ensemble_to_bucket_probs(shifted_members, bucket_labels)
                unit = station["resolution_unit"]
                display_temp = (
                    predict_wu_display_fahrenheit(temp_mean)
                    if unit == "F"
                    else predict_wu_display_celsius(temp_mean)
                )
                boundary_low = is_boundary_temperature(temp_mean, unit=unit)
                if ensemble_std <= ENSEMBLE_CONFIDENCE_STD_HIGH and not boundary_low:
                    confidence = "HIGH"
                elif ensemble_std <= ENSEMBLE_CONFIDENCE_STD_LOW:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"
                # Fetch WU crowd baseline (what retail traders see on wunderground.com)
                wu_crowd_temp = await get_wu_daily_high(
                    self.http,
                    lat=float(station["lat"]),
                    lon=float(station["lon"]),
                    target_date=target_date,
                    unit=station.get("resolution_unit", "C"),
                )

                return {
                    "probs": probs,
                    "rounding_confidence": confidence,
                    "predicted_display_temp": display_temp,
                    "forecast_temp_raw": temp_mean,
                    "ensemble_std": ensemble_std,
                    "ensemble_skip": ensemble_std > ENSEMBLE_STD_SKIP_THRESHOLD,
                    "forecast_model": model_used,
                    "ensemble_member_count": len(members),
                    "ensemble_endpoint": ENSEMBLE_DAILY_API_URL,
                    "primary_model_temp": primary_temp,    # ncep_aigfs025 raw mean
                    "baseline_model_temp": baseline_temp,  # GFS+ECMWF blend (approx IBM GRAF)
                    "wu_crowd_temp": wu_crowd_temp,        # WU actual forecast = exact crowd baseline
                }

        point_mean = await self._fetch_station_daily_high_forecast(station, target_date)
        if point_mean is None:
            return {"probs": {b: 0.0 for b in bucket_labels}, "rounding_confidence": "LOW", "predicted_display_temp": None}

        lead_hours = max((datetime.combine(target_date, datetime.min.time(), UTC) - datetime.now(UTC)).total_seconds() / 3600.0, 0.0)
        sigma = sigma_for_station(station, lead_hours)
        probs = bucket_probabilities(bucket_labels, point_mean, sigma)

        unit = station["resolution_unit"]
        display_temp = predict_wu_display_fahrenheit(point_mean) if unit == "F" else predict_wu_display_celsius(point_mean)
        low_conf = is_boundary_temperature(point_mean, unit=unit)
        return {
            "probs": probs,
            "rounding_confidence": "LOW" if low_conf else "HIGH",
            "predicted_display_temp": display_temp,
            "forecast_temp_raw": point_mean,
        }

    async def _fetch_station_daily_high_forecast(self, station: dict[str, Any], target_date: date) -> float | None:
        if "nws_grid_office" in station:
            return await self._fetch_nws_daily_high(station, target_date)
        if "met_office_lat" in station and self.met_office_api_key:
            return await self._fetch_met_office_daily_high(station, target_date)
        return await self._fetch_open_meteo_daily_high(station, target_date)

    async def _fetch_nws_daily_high(self, station: dict[str, Any], target_date: date) -> float | None:
        office = station["nws_grid_office"]
        gx = station["nws_grid_x"]
        gy = station["nws_grid_y"]
        periods = await self._get_nws_hourly_forecast_cached(office, gx, gy)
        highs: list[float] = []
        for period in periods:
            start = period.get("startTime")
            temp = period.get("temperature")
            if start is None or temp is None:
                continue
            dt = datetime.fromisoformat(start.replace("Z", "+00:00")).date()
            if dt == target_date:
                highs.append(float(temp))
        return max(highs) if highs else None

    async def _get_nws_hourly_forecast_cached(self, office: str, gx: int, gy: int) -> list[dict[str, Any]]:
        key = (office, gx, gy)
        now = datetime.now(UTC).timestamp()
        cached = self._nws_cache.get(key)
        if cached and now - cached[0] < NWS_CACHE_TTL_SECONDS:
            return cached[1]

        url = f"{NOAA_API_URL}/gridpoints/{office}/{gx},{gy}/forecast/hourly"
        headers = {"Accept": "application/geo+json", "User-Agent": "WeatherBot/1.0 (contact@example.com)"}
        response = await self.http.get(url, headers=headers)
        response.raise_for_status()
        periods = response.json().get("properties", {}).get("periods", [])
        self._nws_cache[key] = (now, periods)
        return periods

    async def _fetch_met_office_daily_high(self, station: dict[str, Any], target_date: date) -> float | None:
        params = {"latitude": station["met_office_lat"], "longitude": station["met_office_lon"]}
        headers = {"apikey": self.met_office_api_key}
        response = await self.http.get(MET_OFFICE_API_URL, params=params, headers=headers)
        response.raise_for_status()
        highs: list[float] = []
        for item in _iter_nested(response.json()):
            if isinstance(item, dict) and "time" in item and ("screenTemperature" in item or "temperature" in item):
                raw_temp = item.get("screenTemperature", item.get("temperature"))
                try:
                    dt = datetime.fromisoformat(str(item["time"]).replace("Z", "+00:00")).date()
                    if dt == target_date:
                        highs.append(float(raw_temp))
                except (ValueError, TypeError):
                    continue
        return max(highs) if highs else None

    async def _fetch_open_meteo_daily_high(self, station: dict[str, Any], target_date: date) -> float | None:
        lat = station.get("lat")
        lon = station.get("lon")
        if lat is None or lon is None:
            return None
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "timezone": "UTC",
        }
        response = await self.http.get(url, params=params)
        response.raise_for_status()
        hourly = response.json().get("hourly", {})
        values = hourly.get("temperature_2m", [])
        if not values:
            return None
        return float(max(values))

    def station_local_window_utc(self, station_icao: str, local_day: date) -> tuple[datetime, datetime]:
        """Return station-day start/end in UTC using station standard offset."""
        offset = STATIONS[station_icao]["station_standard_offset_hours"]
        start_local = datetime.combine(local_day, datetime.min.time())
        start_utc = (start_local - timedelta(hours=offset)).replace(tzinfo=UTC)
        end_utc = start_utc + timedelta(days=1)
        return start_utc, end_utc

    async def latest_observed_display_temp(self, station_icao: str) -> tuple[int | None, str]:
        obs = await self.get_latest_metar(station_icao)
        if not obs:
            return None, "LOW"
        temp_c, conf = parse_metar_temp_c(obs)
        if temp_c is None:
            return None, conf
        unit = STATIONS[station_icao]["resolution_unit"]
        if unit == "F":
            temp_f = (temp_c * 9.0 / 5.0) + 32.0
            return predict_wu_display_fahrenheit(temp_f), conf
        return predict_wu_display_celsius(temp_c), conf


ForecastClient = StationForecaster


def _iter_nested(value: Any):
    if isinstance(value, dict):
        for v in value.values():
            yield from _iter_nested(v)
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _iter_nested(v)
