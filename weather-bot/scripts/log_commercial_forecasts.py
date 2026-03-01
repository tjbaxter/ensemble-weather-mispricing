#!/usr/bin/env python3
"""Daily logger for AccuWeather and Weather.com D+1 forecasts.

Captures what commercial providers predict for TOMORROW and writes it to
data/commercial_forecast_log.json.  Run once daily, ideally around 19:00 UTC
(after the 18:30 GFS_12Z model trigger, before Polymarket resolution windows).

Usage:
    python3 scripts/log_commercial_forecasts.py
    python3 scripts/log_commercial_forecasts.py --date 2026-02-25  # override target date
    python3 scripts/log_commercial_forecasts.py --dry-run           # fetch only, no disk write

Cron (VM):
    0 19 * * * /home/tombaxter/weather-bot/venv/bin/python3 \
               /home/tombaxter/weather-bot/scripts/log_commercial_forecasts.py \
               >> /home/tombaxter/weather-bot/logs/commercial_forecast.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "data" / "commercial_forecast_log.json"
ENV_PATH = ROOT / ".env"

_WU_API_KEY = "6532d6454b8aa370768e63d6ba5a832e"
_WU_FORECAST_URL = "https://api.weather.com/v3/wx/forecast/daily/10day"
_ACCU_API_BASE = "https://dataservice.accuweather.com"

_ACCU_LOCATION_KEYS: dict[str, str] = {
    "CYYZ": "55071",     # Toronto Pearson (Mississauga)
    "EGLC": "2532754",   # London City
    "LTAC": "1294331",   # Ankara Esenboğa
    "KATL": "2140212",   # Atlanta Hartsfield
    "KBOS": "338701",    # Boston Logan (Winthrop)
    "KDFW": "336107",    # Dallas/Fort Worth
    "KDEN": "2626644",   # Denver International
    "KLGA": "2627477",   # New York LaGuardia
    "KMIA": "3593859",   # Miami International
    "KORD": "2626577",   # Chicago O'Hare
    "KSEA": "341357",    # Seattle-Tacoma
    "LFPG": "159190",    # Paris CDG
    "RKSI": "2331998",   # Seoul Incheon
    "SBGR": "36369",     # São Paulo Guarulhos
    "SAEZ": "7894",      # Buenos Aires Ezeiza
    "NZWN": "250938",    # Wellington NZ
}

# All cities we track: name → (lat, lon, icao, unit)
# Keep in sync with config/cities.py STATIONS — add new markets here as they launch.
CITIES: dict[str, tuple[float, float, str, str]] = {
    "Seoul":        ( 37.4492,  126.4510, "RKSI", "C"),
    "London":       ( 51.5053,    0.0553, "EGLC", "C"),
    "Paris":        ( 49.0097,    2.5479, "LFPG", "C"),
    "Toronto":      ( 43.6777,  -79.6248, "CYYZ", "C"),
    "New York":     ( 40.7769,  -73.8740, "KLGA", "F"),
    "Chicago":      ( 41.9742,  -87.9073, "KORD", "F"),
    "Boston":       ( 42.3656,  -71.0096, "KBOS", "F"),
    "Denver":       ( 39.8561, -104.6737, "KDEN", "F"),
    "Atlanta":      ( 33.6407,  -84.4277, "KATL", "F"),
    "Dallas":       ( 32.8481,  -96.8512, "KDFW", "F"),
    "Miami":        ( 25.7959,  -80.2870, "KMIA", "F"),
    "Seattle":      ( 47.4502, -122.3088, "KSEA", "F"),
    "Sao Paulo":    (-23.4356,  -46.4731, "SBGR", "C"),
    "Ankara":       ( 40.1281,   32.9951, "LTAC", "C"),
    "Buenos Aires": (-34.8222,  -58.5358, "SAEZ", "C"),
    "Wellington":   (-41.3272,  174.8050, "NZWN", "C"),
}


def _read_env_key(name: str) -> str:
    for env_path in (ENV_PATH, Path("/etc/weather-bot.env")):
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip()
        except OSError:
            pass
    return os.environ.get(name, "")


def _hround(x: float) -> int:
    """Standard half-up rounding to nearest integer."""
    import math
    return math.floor(x + 0.5)


def fetch_wu_multi(lat: float, lon: float, unit: str, dates: list[str]) -> dict[str, float]:
    """Fetch Weather.com/IBM forecast highs for multiple target dates in one call."""
    wu_units = "m" if unit == "C" else "e"
    results: dict[str, float] = {}
    try:
        r = requests.get(
            _WU_FORECAST_URL,
            params={"geocode": f"{lat},{lon}", "format": "json",
                    "units": wu_units, "language": "en-US", "apiKey": _WU_API_KEY},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        r.raise_for_status()
        d = r.json()
        for ts, high in zip(d.get("validTimeLocal", []), d.get("calendarDayTemperatureMax", [])):
            if high is None:
                continue
            try:
                day = datetime.fromisoformat(str(ts)).date().isoformat()
            except (ValueError, TypeError):
                continue
            if day in dates:
                results[day] = float(high)
    except Exception as exc:
        print(f"    WU error: {exc}")
    return results


def fetch_wu(lat: float, lon: float, unit: str, target_date: str) -> float | None:
    """Fetch Weather.com/IBM D+1 forecast high for a given location."""
    return fetch_wu_multi(lat, lon, unit, [target_date]).get(target_date)


def fetch_accu_multi(icao: str, unit: str, dates: list[str], api_key: str) -> dict[str, float]:
    """Fetch AccuWeather forecast highs for multiple target dates in one 5-day call."""
    loc_key = _ACCU_LOCATION_KEYS.get(icao)
    if not loc_key or not api_key:
        return {}
    metric = "true" if unit == "C" else "false"
    results: dict[str, float] = {}
    try:
        r = requests.get(
            f"{_ACCU_API_BASE}/forecasts/v1/daily/5day/{loc_key}",
            params={"apikey": api_key, "metric": metric},
            timeout=12,
        )
        if r.status_code in (403, 429):
            print(f"    AccuWeather rate-limited ({r.status_code}) for {icao}")
            return {}
        r.raise_for_status()
        for fc in r.json().get("DailyForecasts", []):
            fc_date = str(fc.get("Date", ""))[:10]
            if fc_date in dates:
                temp = fc.get("Temperature", {}).get("Maximum", {}).get("Value")
                if temp is not None:
                    results[fc_date] = float(temp)
    except Exception as exc:
        print(f"    AccuWeather error: {exc}")
    return results


def fetch_accu(icao: str, unit: str, target_date: str, api_key: str) -> float | None:
    """Fetch AccuWeather forecast high for a single target date."""
    return fetch_accu_multi(icao, unit, [target_date], api_key).get(target_date)


def load_log() -> dict[str, Any]:
    try:
        if LOG_PATH.exists():
            data = json.loads(LOG_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def save_log(log: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(LOG_PATH)


def run(target_date: str, dry_run: bool = False) -> None:
    from datetime import date as _d, timedelta as _td
    # Derive D+2 and D+3 relative to target_date (which is normally D+1/tomorrow)
    try:
        _base = _d.fromisoformat(target_date)
        date_d2 = (_base + _td(days=1)).isoformat()
        date_d3 = (_base + _td(days=2)).isoformat()
    except ValueError:
        date_d2 = date_d3 = None

    print(f"\n{'DRY RUN — ' if dry_run else ''}Commercial Forecast Logger")
    print(f"Target D+1:  {target_date}  D+2: {date_d2}  D+3: {date_d3}")
    print(f"Logged at:   {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 70)

    accu_key = _read_env_key("ACCUWEATHER_API_KEY")
    if not accu_key:
        print("⚠ ACCUWEATHER_API_KEY not found in .env — AccuWeather skipped")

    log = load_log()
    any_new = False

    header = f"{'City':<16} {'ICAO':<6} {'Unit':<5} {'D+1 Accu':>10} {'D+1 WU':>8} {'D+2 Accu':>10} {'D+3 Accu':>10} Status"
    print(header)
    print("─" * len(header))

    for city, (lat, lon, icao, unit) in CITIES.items():
        city_log = log.setdefault(city, {})
        already = city_log.get(target_date, {})

        # Fetch all dates in one API call each
        accu_dates = [d for d in [target_date, date_d2, date_d3] if d]
        wu_dates   = [d for d in [target_date, date_d2, date_d3] if d]

        accu_map = fetch_accu_multi(icao, unit, accu_dates, accu_key) if accu_key else {}
        wu_map   = fetch_wu_multi(lat, lon, unit, wu_dates)

        accu_val  = accu_map.get(target_date)
        wu_val    = wu_map.get(target_date)
        accu_d2   = accu_map.get(date_d2) if date_d2 else None
        wu_d2     = wu_map.get(date_d2)   if date_d2 else None
        accu_d3   = accu_map.get(date_d3) if date_d3 else None
        wu_d3     = wu_map.get(date_d3)   if date_d3 else None

        u = f"°{unit}"
        a1 = f"{accu_val:.1f}{u}"  if accu_val  is not None else "—"
        w1 = f"{wu_val:.1f}{u}"    if wu_val    is not None else "—"
        a2 = f"{accu_d2:.1f}{u}"   if accu_d2   is not None else "—"
        a3 = f"{accu_d3:.1f}{u}"   if accu_d3   is not None else "—"

        status = "✅" if (accu_val is not None or wu_val is not None) else "⚠ no data"
        if target_date in city_log and not dry_run:
            status = "(already logged)"
        print(f"  {city:<14} {icao:<6} {u:<4} {a1:>10} {w1:>8} {a2:>10} {a3:>10}  {status}")

        if not dry_run and (accu_val is not None or wu_val is not None):
            existing = city_log.get(target_date, {})
            city_log[target_date] = {
                "accu":    accu_val  if existing.get("accu")    is None else existing["accu"],
                "wu":      wu_val    if existing.get("wu")      is None else existing["wu"],
                "accu_d2": accu_d2   if existing.get("accu_d2") is None else existing["accu_d2"],
                "wu_d2":   wu_d2     if existing.get("wu_d2")   is None else existing["wu_d2"],
                "accu_d3": accu_d3   if existing.get("accu_d3") is None else existing["accu_d3"],
                "wu_d3":   wu_d3     if existing.get("wu_d3")   is None else existing["wu_d3"],
                "unit":    unit,
                "logged_at": datetime.now(UTC).isoformat(),
            }
            any_new = True

    if not dry_run and any_new:
        save_log(log)
        print(f"\n✅ Saved to {LOG_PATH}")
    elif dry_run:
        print("\n[dry-run] Nothing written to disk.")
    else:
        print("\n(All cities already logged for this date.)")

    # Summary stats across all logged data
    print("\nLogging coverage summary:")
    log_current = load_log() if not dry_run else log
    for city in CITIES:
        entries = log_current.get(city, {})
        total = len(entries)
        accu_cnt = sum(1 for e in entries.values() if e.get("accu") is not None)
        wu_cnt   = sum(1 for e in entries.values() if e.get("wu")   is not None)
        if total:
            dates = sorted(entries.keys())
            print(f"  {city:<16} {total:>3} days logged  "
                  f"(AccuWeather: {accu_cnt}, WU: {wu_cnt})  "
                  f"{dates[0]} → {dates[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default=None,
                        help="Target date (YYYY-MM-DD). Defaults to tomorrow.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch data but do not write to disk.")
    args = parser.parse_args()

    if args.date:
        try:
            date.fromisoformat(args.date)
            target = args.date
        except ValueError:
            print(f"Invalid date: {args.date}. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        target = (date.today() + timedelta(days=1)).isoformat()

    run(target, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
