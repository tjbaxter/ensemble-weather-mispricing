"""Open-Meteo ensemble and previous-runs forecast client."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from config.cities import STATIONS
from config.settings import (
    CALIBRATION_JSON_PATH,
    ENSEMBLE_BATCH_CACHE_TTL_SECONDS,
    ENSEMBLE_DAILY_API_URL,
    MODEL_RANKINGS_JSON_PATH,
)


class EnsembleForecastClient:
    def __init__(
        self,
        timeout_seconds: float = 20.0,
        calibration_path: str = CALIBRATION_JSON_PATH,
        rankings_path: str = MODEL_RANKINGS_JSON_PATH,
    ) -> None:
        self.http = httpx.AsyncClient(timeout=timeout_seconds)
        self.calibration_path = Path(calibration_path)
        self.rankings_path = Path(rankings_path)
        self._calibration_cache: dict[str, Any] | None = None
        self._rankings_cache: dict[str, Any] | None = None
        self._batch_cache: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
        self._batch_cache_ts: dict[tuple[str, str], float] = {}

    async def close(self) -> None:
        await self.http.aclose()

    def _load_calibration(self) -> dict[str, Any]:
        if self._calibration_cache is not None:
            return self._calibration_cache
        if not self.calibration_path.exists():
            self._calibration_cache = {}
            return self._calibration_cache
        try:
            self._calibration_cache = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._calibration_cache = {}
        return self._calibration_cache

    def _model_metrics(self, station_icao: str, model: str) -> dict[str, Any]:
        calibration = self._load_calibration()
        station_data = calibration.get("stations", {}).get(station_icao, {})
        return station_data.get(model, {})

    def _load_rankings(self) -> dict[str, Any]:
        if self._rankings_cache is not None:
            return self._rankings_cache
        if not self.rankings_path.exists():
            self._rankings_cache = {}
            return self._rankings_cache
        try:
            self._rankings_cache = json.loads(self.rankings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._rankings_cache = {}
        return self._rankings_cache

    def _ranking_metrics(self, station_icao: str, model: str) -> dict[str, Any]:
        rankings = self._load_rankings()
        station_data = rankings.get("stations", {}).get(station_icao, {})
        ranked_models = station_data.get("rankings", [])
        for row in ranked_models:
            if row.get("model") == model:
                return row
        return {}

    async def get_weighted_daily_max_members(
        self,
        station: dict[str, Any],
        target_date: str,
        models: list[str],
    ) -> tuple[list[float], str]:
        """Fetch model members and combine with calibration-based weighting."""
        unit = "fahrenheit" if station.get("resolution_unit") == "F" else "celsius"
        members: list[float] = []
        best_model = models[0]
        best_mae = float("inf")

        for model in models:
            model_members = await self._fetch_model_daily_members(
                station_icao=str(station["icao"]),
                target_date=target_date,
                model=model,
                unit=unit,
            )
            if not model_members:
                continue

            metrics = self._model_metrics(station["icao"], model)
            bias = float(metrics.get("mean_bias", 0.0) or 0.0)
            mae = float(metrics.get("mean_abs_error", 2.0) or 2.0)
            if mae < best_mae:
                best_mae = mae
                best_model = model

            # Base weighting from calibration MAE.
            weight = max(1, int(round(10.0 / max(mae, 0.5))))

            # Station-specific ranking boost from model_rankings.json.
            ranking = self._ranking_metrics(station["icao"], model)
            hit_rate = float(ranking.get("bucket_hit_rate", 0.0) or 0.0)
            is_best = ranking.get("model") == self._load_rankings().get("stations", {}).get(station["icao"], {}).get(
                "best_model"
            )
            if hit_rate > 0:
                weight = max(1, int(round(weight * (0.6 + hit_rate))))
            if is_best:
                weight = max(1, int(round(weight * 1.5)))

            corrected = [float(v - bias) for v in model_members]
            members.extend(corrected * weight)

        return members, best_model

    async def _fetch_model_daily_members(
        self,
        station_icao: str,
        target_date: str,
        model: str,
        unit: str,
    ) -> list[float]:
        cache_key = (model, unit)
        if self._is_batch_cache_stale(cache_key):
            await self._warm_batch_for_model_unit(model=model, unit=unit)
        station_dates = self._batch_cache.get(cache_key, {}).get(station_icao, {})
        members = station_dates.get(target_date)
        if members:
            return members

        # Fallback single-station fetch if batch parse failed.
        station = STATIONS.get(station_icao)
        if not station:
            return []
        params = {
            "latitude": float(station["lat"]),
            "longitude": float(station["lon"]),
            "daily": "temperature_2m_max",
            "models": model,
            "temperature_unit": unit,
            "timezone": "auto",
            "forecast_days": 4,
        }
        response = await self.http.get(ENSEMBLE_DAILY_API_URL, params=params)
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        return self._extract_members_for_date(payload if isinstance(payload, dict) else {}, target_date)

    def _is_batch_cache_stale(self, cache_key: tuple[str, str]) -> bool:
        ts = self._batch_cache_ts.get(cache_key)
        if ts is None:
            return True
        return (datetime.now(UTC).timestamp() - ts) > ENSEMBLE_BATCH_CACHE_TTL_SECONDS

    async def _warm_batch_for_model_unit(self, model: str, unit: str) -> None:
        eligible = [
            (icao, cfg)
            for icao, cfg in STATIONS.items()
            if cfg.get("lat") is not None
            and cfg.get("lon") is not None
            and ("fahrenheit" if cfg.get("resolution_unit") == "F" else "celsius") == unit
        ]
        cache_key = (model, unit)
        self._batch_cache[cache_key] = {}
        if not eligible:
            return

        lat_csv = ",".join(str(cfg["lat"]) for _, cfg in eligible)
        lon_csv = ",".join(str(cfg["lon"]) for _, cfg in eligible)
        params = {
            "latitude": lat_csv,
            "longitude": lon_csv,
            "daily": "temperature_2m_max",
            "models": model,
            "temperature_unit": unit,
            "timezone": "auto",
            "forecast_days": 4,
        }
        response = await self.http.get(ENSEMBLE_DAILY_API_URL, params=params)
        response.raise_for_status()
        payload = response.json()

        # Multi-location payload returns list aligned to input coordinate order.
        if isinstance(payload, list):
            for idx, (icao, _) in enumerate(eligible):
                city_payload = payload[idx] if idx < len(payload) and isinstance(payload[idx], dict) else {}
                self._batch_cache[cache_key][icao] = self._extract_members_by_date(city_payload)
            self._batch_cache_ts[cache_key] = datetime.now(UTC).timestamp()
            return

        # Single payload fallback.
        if isinstance(payload, dict):
            icao, _ = eligible[0]
            self._batch_cache[cache_key][icao] = self._extract_members_by_date(payload)
            self._batch_cache_ts[cache_key] = datetime.now(UTC).timestamp()

    def _extract_members_by_date(self, payload: dict[str, Any]) -> dict[str, list[float]]:
        daily = payload.get("daily", {}) if isinstance(payload, dict) else {}
        times = daily.get("time", [])
        if not isinstance(times, list):
            return {}
        out: dict[str, list[float]] = {}
        for day_idx, day in enumerate(times):
            if not isinstance(day, str):
                continue
            members: list[float] = []
            for key, values in daily.items():
                if not key.startswith("temperature_2m_max_member"):
                    continue
                if not isinstance(values, list) or day_idx >= len(values):
                    continue
                value = values[day_idx]
                if value is None:
                    continue
                try:
                    members.append(float(value))
                except (TypeError, ValueError):
                    continue
            if members:
                out[day] = members
        return out

    def _extract_members_for_date(self, payload: dict[str, Any], target_date: str) -> list[float]:
        return self._extract_members_by_date(payload).get(target_date, [])
