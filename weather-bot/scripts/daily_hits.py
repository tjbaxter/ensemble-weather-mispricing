"""Build daily hit/miss logs from previous-day forecasts vs actuals.

This script does NOT use Polymarket prices. It answers:
"Did the model pick the right resolution bucket the day before?"
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.cities import STATIONS

MODELS = ("gfs_seamless", "ecmwf_ifs025", "icon_seamless_eps")
PREVIOUS_RUNS_API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"


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


def _bucket_id(temp: float, unit: str) -> str:
    if unit == "C":
        return str(int(round(temp)))
    display_f = int(round(temp))
    low = display_f if display_f % 2 == 0 else (display_f - 1)
    return f"{low}-{low + 1}"


def _longest_streak(bits: list[int]) -> int:
    best = 0
    cur = 0
    for b in bits:
        if b == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


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
    response = await client.get(PREVIOUS_RUNS_API_URL, params=params)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    return _extract_hourly_daily_max_map(payload if isinstance(payload, dict) else {}, "temperature_2m_previous_day1")


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


async def run_daily_hits(past_days: int, output_path: Path) -> None:
    rows: list[dict[str, str | int | float]] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for station_icao, station in STATIONS.items():
            actual = await _fetch_actual_daily_max_map(client, station, past_days=past_days)
            if not actual:
                continue
            for model in MODELS:
                try:
                    predicted = await _fetch_predicted_daily_max_map(client, station, model, past_days=past_days)
                except httpx.HTTPError:
                    continue
                shared_days = sorted(set(predicted).intersection(actual))
                for day in shared_days:
                    fc = predicted[day]
                    obs = actual[day]
                    forecast_bucket = _bucket_id(fc, station["resolution_unit"])
                    actual_bucket = _bucket_id(obs, station["resolution_unit"])
                    hit = 1 if forecast_bucket == actual_bucket else 0
                    rows.append(
                        {
                            "date": day,
                            "station": station_icao,
                            "model": model,
                            "forecast_high": round(fc, 3),
                            "actual_high": round(obs, 3),
                            "forecast_bucket": forecast_bucket,
                            "actual_bucket": actual_bucket,
                            "hit": hit,
                        }
                    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "station",
                "model",
                "forecast_high",
                "actual_high",
                "forecast_bucket",
                "actual_bucket",
                "hit",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"daily_hits_written={output_path}")
    print(f"rows={len(rows)}")

    # streak summary
    grouped: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for r in rows:
        key = (str(r["station"]), str(r["model"]))
        grouped.setdefault(key, []).append((str(r["date"]), int(r["hit"])))

    print("longest_streaks=")
    for (station, model), items in sorted(grouped.items()):
        items.sort(key=lambda x: x[0])
        bits = [hit for _, hit in items]
        streak = _longest_streak(bits)
        hit_rate = (sum(bits) / len(bits)) if bits else 0.0
        print(f"  {station} {model}: longest_hit_streak={streak} hit_rate={hit_rate:.2f} n_days={len(bits)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily model hit/miss logs.")
    parser.add_argument("--past-days", type=int, default=30)
    parser.add_argument("--output", default="logs/daily_hits.csv")
    args = parser.parse_args()
    asyncio.run(run_daily_hits(args.past_days, Path(args.output)))


if __name__ == "__main__":
    main()
