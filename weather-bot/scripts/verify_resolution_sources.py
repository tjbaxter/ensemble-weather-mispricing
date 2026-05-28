#!/usr/bin/env python3
"""Compare resolution sources against Polymarket-resolved bucket outcomes.

This script builds a deterministic parity report across three data views:
1) Polymarket resolved bucket label from data/polymarket_cache.json
2) Weather.com historical observations API daily max
3) AviationWeather METAR/SPECI-derived daily high (T-group aware parsing)

Usage:
    python scripts/verify_resolution_sources.py
    python scripts/verify_resolution_sources.py --cities "Seoul,Chicago,London"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.cities import STATIONS
from data.metar import parse_metar_temp

POLYMARKET_CACHE_PATH = ROOT / "data" / "polymarket_cache.json"
REPORT_PATH = ROOT / "data" / "resolution_source_parity.json"
WEATHERCOM_KEY = os.environ.get("WU_OBS_KEY", "")
WEATHERCOM_URL = "https://api.weather.com/v1/location/{station}/observations/historical.json"
AWC_METAR_URL = "https://aviationweather.gov/api/data/metar"
if not WEATHERCOM_KEY:
    raise RuntimeError("WU_OBS_KEY not set in environment (see weather-bot/.env.example)")


def _normalize_city_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _country_code_from_wu_url(url: str) -> str | None:
    # Example: https://www.wunderground.com/history/daily/us/il/chicago/KORD
    #                                   country segment ─^
    m = re.search(r"/history/daily/([a-z]{2})/", url.lower())
    if not m:
        return None
    return m.group(1).upper()


def _bucket_contains(bucket_label: str, temp_value: int) -> bool:
    clean = bucket_label.replace("°F", "").replace("°C", "").strip()
    clean = clean.replace("≤", "<=").replace("≥", ">=")

    # "73 or below" / "92 or higher"
    m = re.match(r"^\s*(-?\d+)\s*or\s*below\s*$", clean, flags=re.IGNORECASE)
    if m:
        return temp_value <= int(m.group(1))
    m = re.match(r"^\s*(-?\d+)\s*or\s*higher\s*$", clean, flags=re.IGNORECASE)
    if m:
        return temp_value >= int(m.group(1))

    # <=12 / >=14
    m = re.match(r"^\s*<=\s*(-?\d+)\s*$", clean)
    if m:
        return temp_value <= int(m.group(1))
    m = re.match(r"^\s*>=\s*(-?\d+)\s*$", clean)
    if m:
        return temp_value >= int(m.group(1))

    # "58-59"
    m = re.match(r"^\s*(-?\d+)\s*-\s*(-?\d+)\s*$", clean)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return lo <= temp_value <= hi

    # "14+"
    m = re.match(r"^\s*(-?\d+)\s*\+\s*$", clean)
    if m:
        return temp_value >= int(m.group(1))

    # "13"
    m = re.match(r"^\s*(-?\d+)\s*$", clean)
    if m:
        return temp_value == int(m.group(1))

    return False


def _parse_iso_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def _load_polymarket_cache() -> dict[str, dict[str, list[Any]]]:
    raw = json.loads(POLYMARKET_CACHE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, list[Any]]] = {}
    for city, rows in raw.items():
        if isinstance(rows, dict):
            out[city] = {k: v for k, v in rows.items() if isinstance(v, list)}
    return out


def _build_station_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for icao, cfg in STATIONS.items():
        city = str(cfg.get("market_label") or "").strip()
        if not city:
            continue
        key = _normalize_city_name(city)
        entry = {
            "icao": icao,
            "city": city,
            "resolution_unit": str(cfg.get("resolution_unit", "F")),
            "timezone": str(cfg.get("timezone", "UTC")),
            "wu_url": str(cfg.get("wu_url", "")),
        }
        lookup[key] = entry

    # Aliases used in cache/history
    if "nyc" in lookup and "newyork" not in lookup:
        lookup["newyork"] = lookup["nyc"]
    if "saopaulo" in lookup and "sãopaulo" not in lookup:
        lookup["sãopaulo"] = lookup["saopaulo"]
    return lookup


def _fetch_weathercom_daily_high(station_id: str, units: str, target_day: date) -> int | None:
    params = {
        "apiKey": WEATHERCOM_KEY,
        "units": units,
        "startDate": target_day.strftime("%Y%m%d"),
        "endDate": target_day.strftime("%Y%m%d"),
    }
    url = WEATHERCOM_URL.format(station=station_id) + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    obs = payload.get("observations", [])
    temps = [o.get("temp") for o in obs if o.get("temp") is not None]
    if not temps:
        return None
    return int(math.floor(float(max(temps)) + 0.5))


def _fetch_metar_daily_high(
    icao: str,
    resolution_unit: str,
    timezone_name: str,
    target_day: date,
) -> dict[str, Any] | None:
    local_start = datetime(target_day.year, target_day.month, target_day.day, tzinfo=ZoneInfo(timezone_name))
    local_end = local_start + timedelta(days=1)
    start_utc = local_start.astimezone(UTC)
    end_utc = local_end.astimezone(UTC)

    span_hours = max(26, int(math.ceil((end_utc - start_utc).total_seconds() / 3600.0)) + 2)
    params = {
        "ids": icao,
        "format": "json",
        "date": end_utc.strftime("%Y%m%d_%H%M"),
        "hours": str(span_hours),
    }
    url = AWC_METAR_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "PolymarketWeatherBot/1.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        records = json.loads(resp.read().decode("utf-8"))
    if not isinstance(records, list):
        return None

    high: dict[str, Any] | None = None
    for rec in records:
        report_time = rec.get("reportTime")
        if not report_time:
            continue
        t_utc = _parse_iso_utc(str(report_time))
        if t_utc < start_utc or t_utc >= end_utc:
            continue

        parsed = parse_metar_temp(
            raw_ob=str(rec.get("rawOb") or ""),
            awc_temp=rec.get("temp"),
            unit=resolution_unit,
        )
        if parsed is None:
            continue
        row = {
            "value": int(parsed.temp_resolution),
            "source": parsed.source,
            "confidence": parsed.confidence,
            "metar_type": str(rec.get("metarType") or "METAR"),
            "report_time_utc": t_utc.isoformat(),
        }
        if high is None or row["value"] > high["value"]:
            high = row
    return high


def _extract_polymarket_value(row: list[Any]) -> int | None:
    for item in row[1:]:
        if isinstance(item, (int, float)):
            return int(item)
    return None


def run_report(cities: set[str] | None, days_back: int, max_rows: int, output_path: Path) -> dict[str, Any]:
    station_lookup = _build_station_lookup()
    cache = _load_polymarket_cache()
    cutoff = date.today() - timedelta(days=max(1, days_back))

    rows: list[dict[str, Any]] = []
    for city_name, by_date in cache.items():
        if cities and city_name not in cities:
            continue
        city_key = _normalize_city_name(city_name)
        station = station_lookup.get(city_key)
        if not station:
            rows.append(
                {
                    "city": city_name,
                    "target_date": "",
                    "status": "no_station_mapping",
                }
            )
            continue

        icao = station["icao"]
        unit = station["resolution_unit"]
        tz_name = station["timezone"]
        country_code = _country_code_from_wu_url(station["wu_url"])
        if not country_code:
            continue
        station_id = f"{icao}:9:{country_code}"
        units_flag = "e" if unit == "F" else "m"

        for date_str, pm_row in sorted(by_date.items()):
            try:
                d = date.fromisoformat(date_str)
            except ValueError:
                continue
            if d < cutoff:
                continue
            if len(rows) >= max_rows:
                break
            if not pm_row:
                continue
            bucket_label = str(pm_row[0])
            pm_value = _extract_polymarket_value(pm_row)

            entry: dict[str, Any] = {
                "city": city_name,
                "target_date": date_str,
                "icao": icao,
                "unit": unit,
                "polymarket_bucket": bucket_label,
                "polymarket_value_hint": pm_value,
                "weathercom_station_id": station_id,
            }

            try:
                wu_temp = _fetch_weathercom_daily_high(station_id, units_flag, d)
                entry["weathercom_temp"] = wu_temp
                entry["weathercom_bucket_match"] = _bucket_contains(bucket_label, wu_temp) if wu_temp is not None else None
            except Exception as exc:
                entry["weathercom_temp"] = None
                entry["weathercom_error"] = str(exc)
                entry["weathercom_bucket_match"] = None

            try:
                metar = _fetch_metar_daily_high(icao=icao, resolution_unit=unit, timezone_name=tz_name, target_day=d)
                if metar is None:
                    entry["metar_temp"] = None
                    entry["metar_bucket_match"] = None
                else:
                    entry["metar_temp"] = metar["value"]
                    entry["metar_source"] = metar["source"]
                    entry["metar_confidence"] = metar["confidence"]
                    entry["metar_type"] = metar["metar_type"]
                    entry["metar_time_utc"] = metar["report_time_utc"]
                    entry["metar_bucket_match"] = _bucket_contains(bucket_label, metar["value"])
            except Exception as exc:
                entry["metar_temp"] = None
                entry["metar_error"] = str(exc)
                entry["metar_bucket_match"] = None

            rows.append(entry)

    def _score(key: str) -> dict[str, int]:
        valid = [r for r in rows if r.get(key) is not None]
        matched = [r for r in valid if r.get(key) is True]
        return {"matched": len(matched), "checked": len(valid)}

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(rows),
        "weathercom": _score("weathercom_bucket_match"),
        "metar": _score("metar_bucket_match"),
        "days_back": days_back,
        "max_rows": max_rows,
    }
    report = {"summary": summary, "rows": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Weather.com and METAR sources against Polymarket-resolved buckets")
    parser.add_argument("--cities", type=str, default="", help="Comma-separated city names as in polymarket_cache.json")
    parser.add_argument("--days-back", type=int, default=15, help="Only include target_date >= today-days_back")
    parser.add_argument("--max-rows", type=int, default=500, help="Maximum rows to process")
    parser.add_argument("--output", type=Path, default=REPORT_PATH, help="Output JSON report path")
    args = parser.parse_args()

    cities = {c.strip() for c in args.cities.split(",") if c.strip()} or None
    report = run_report(cities=cities, days_back=args.days_back, max_rows=max(1, args.max_rows), output_path=args.output)
    s = report["summary"]
    print(f"Wrote report: {args.output}")
    print(
        "Weather.com match: "
        f"{s['weathercom']['matched']}/{s['weathercom']['checked']} | "
        "METAR match: "
        f"{s['metar']['matched']}/{s['metar']['checked']}"
    )


if __name__ == "__main__":
    main()
