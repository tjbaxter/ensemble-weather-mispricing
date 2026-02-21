"""Backtest station/model forecast accuracy using Open-Meteo previous runs.

Uses:
- previous-runs-api hourly temperature_2m_previous_day1 as forecast proxy
- archive-api hourly temperature_2m as ground truth
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.cities import STATIONS
from config.settings import CALIBRATION_JSON_PATH, ENSEMBLE_PREVIOUS_RUNS_API_URL


MODELS = ("gfs_seamless", "ecmwf_ifs025", "icon_seamless_eps")
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"


def _extract_daily_max_map(payload: dict) -> dict[str, float]:
    daily = payload.get("daily", {})
    if not isinstance(daily, dict):
        return {}
    times = daily.get("time", [])
    values = daily.get("temperature_2m_max", [])
    if not isinstance(times, list) or not isinstance(values, list):
        return {}
    out: dict[str, float] = {}
    for idx, day in enumerate(times):
        if idx >= len(values):
            break
        v = values[idx]
        if v is None:
            continue
        try:
            out[str(day)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _rmse(errors: list[float]) -> float:
    if not errors:
        return 0.0
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def _extract_hourly_daily_max_map(payload: dict, variable: str) -> dict[str, float]:
    hourly = payload.get("hourly", {})
    if not isinstance(hourly, dict):
        return {}
    times = hourly.get("time", [])
    values = hourly.get(variable, [])
    if not isinstance(times, list) or not isinstance(values, list):
        return {}
    out: dict[str, float] = {}
    for idx, ts in enumerate(times):
        if idx >= len(values):
            break
        v = values[idx]
        if v is None:
            continue
        try:
            dt = datetime.fromisoformat(str(ts))
            val = float(v)
        except (TypeError, ValueError):
            continue
        day = dt.date().isoformat()
        out[day] = max(val, out.get(day, val))
    return out


def _resolution_bucket_id(temp: float, unit: str) -> str:
    if unit == "C":
        return f"C:{int(round(temp))}"
    display_f = int(round(temp))
    low = display_f if display_f % 2 == 0 else (display_f - 1)
    return f"F:{low}-{low + 1}"


async def _fetch_predicted_daily_max_map(
    client: httpx.AsyncClient,
    station: dict,
    model: str,
    past_days: int,
) -> dict[str, float]:
    unit = "fahrenheit" if station.get("resolution_unit") == "F" else "celsius"
    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "hourly": "temperature_2m_previous_day1",
        "models": model,
        "temperature_unit": unit,
        "timezone": "auto",
        "past_days": past_days,
    }
    response = await client.get(ENSEMBLE_PREVIOUS_RUNS_API_URL, params=params)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        return {}
    out = _extract_hourly_daily_max_map(payload, "temperature_2m_previous_day1")
    if out:
        return out

    # Fallback for older API behavior where previous_day is passed as parameter.
    fallback_params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "hourly": "temperature_2m",
        "models": model,
        "temperature_unit": unit,
        "timezone": "auto",
        "past_days": past_days,
        "previous_day": 1,
    }
    fallback_response = await client.get(ENSEMBLE_PREVIOUS_RUNS_API_URL, params=fallback_params)
    fallback_response.raise_for_status()
    fallback_payload = fallback_response.json()
    if isinstance(fallback_payload, list):
        fallback_payload = fallback_payload[0] if fallback_payload else {}
    return _extract_hourly_daily_max_map(fallback_payload if isinstance(fallback_payload, dict) else {}, "temperature_2m")


async def _fetch_actual_daily_max_map(
    client: httpx.AsyncClient,
    station: dict,
    past_days: int,
) -> dict[str, float]:
    unit = "fahrenheit" if station.get("resolution_unit") == "F" else "celsius"
    end_date = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    start_date = (datetime.now(UTC).date() - timedelta(days=past_days)).isoformat()
    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "hourly": "temperature_2m",
        "temperature_unit": unit,
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date,
    }
    response = await client.get(ARCHIVE_API_URL, params=params)
    response.raise_for_status()
    payload = response.json()
    return _extract_hourly_daily_max_map(payload if isinstance(payload, dict) else {}, "temperature_2m")


async def run_backtest(past_days: int, output_path: Path, rankings_output_path: Path) -> None:
    result: dict = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "past_days": past_days,
        "stations": {},
    }
    rankings: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for station_icao, station in STATIONS.items():
            station_out: dict = {}
            for model in MODELS:
                try:
                    predicted = await _fetch_predicted_daily_max_map(client, station, model, past_days=past_days)
                    actual = await _fetch_actual_daily_max_map(client, station, past_days=past_days)
                except httpx.HTTPError:
                    continue
                shared_days = sorted(set(predicted).intersection(actual))
                if not shared_days:
                    continue
                errors = [predicted[d] - actual[d] for d in shared_days]
                abs_errors = [abs(e) for e in errors]
                bucket_hits = sum(
                    1
                    for d in shared_days
                    if _resolution_bucket_id(predicted[d], station["resolution_unit"])
                    == _resolution_bucket_id(actual[d], station["resolution_unit"])
                )
                station_out[model] = {
                    "n_days": len(shared_days),
                    "mean_bias": mean(errors),
                    "mean_abs_error": mean(abs_errors),
                    "rmse": _rmse(errors),
                    "within_1deg": sum(1 for e in abs_errors if e <= 1.0) / len(abs_errors),
                    "within_2deg": sum(1 for e in abs_errors if e <= 2.0) / len(abs_errors),
                    "within_3deg": sum(1 for e in abs_errors if e <= 3.0) / len(abs_errors),
                    "bucket_hit_rate": bucket_hits / len(shared_days),
                    "max_error": max(abs_errors),
                }
            if station_out:
                result["stations"][station_icao] = station_out
                ranked = sorted(
                    station_out.items(),
                    key=lambda kv: (-kv[1]["bucket_hit_rate"], kv[1]["mean_abs_error"]),
                )
                rankings[station_icao] = {
                    "best_model": ranked[0][0],
                    "rankings": [
                        {
                            "model": model_name,
                            "mae": metrics["mean_abs_error"],
                            "bucket_hit_rate": metrics["bucket_hit_rate"],
                            "bias": metrics["mean_bias"],
                            "n_days": metrics["n_days"],
                        }
                        for model_name, metrics in ranked
                    ],
                }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rankings_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rankings_output_path.write_text(
        json.dumps(
            {
                "generated_at_utc": result["generated_at_utc"],
                "past_days": past_days,
                "stations": rankings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"calibration_written={output_path}")
    print(f"model_rankings_written={rankings_output_path}")
    print(f"stations_calibrated={len(result['stations'])}")
    for station_icao, rank_info in sorted(rankings.items()):
        best = rank_info["rankings"][0]
        print(
            f"{station_icao}: best_model={rank_info['best_model']} "
            f"bucket_hit_rate={best['bucket_hit_rate']:.2f} mae={best['mae']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest model forecast calibration.")
    parser.add_argument("--past-days", type=int, default=30, help="Number of historical days to compare.")
    parser.add_argument(
        "--output",
        default=CALIBRATION_JSON_PATH,
        help="Output JSON path for calibration metrics.",
    )
    parser.add_argument(
        "--rankings-output",
        default="logs/model_rankings.json",
        help="Output JSON path for per-station model rankings.",
    )
    args = parser.parse_args()
    asyncio.run(run_backtest(args.past_days, Path(args.output), Path(args.rankings_output)))


if __name__ == "__main__":
    main()
