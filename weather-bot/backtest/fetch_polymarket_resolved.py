"""fetch_polymarket_resolved.py

Fetch all historically resolved Polymarket daily temperature markets via the
Gamma API, extract the winning bucket for each, and save to JSON + CSV.

Ground truth = the Polymarket winning bucket. We do NOT scrape Weather
Underground — WU is notorious for rounding discrepancies. Polymarket's
resolution IS the ground truth by definition.

Usage:
    python backtest/fetch_polymarket_resolved.py
    python backtest/fetch_polymarket_resolved.py --since 2026-01-01
    python backtest/fetch_polymarket_resolved.py --city Seoul
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, UTC
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GAMMA_URL = "https://gamma-api.polymarket.com/events"
OUT_DIR = ROOT / "backtest" / "data"
OUT_JSON = OUT_DIR / "resolved_markets.json"
OUT_CSV  = OUT_DIR / "resolved_markets.csv"

# City name (as it appears in Polymarket titles, lowercased) → station metadata
# Key = exact lowercase city name from title; also include common aliases
STATION_META: dict[str, dict] = {
    # ── North America ──────────────────────────────────────────────────────────
    "toronto":     {"icao": "CYYZ", "lat": 43.6772, "lon":  -79.6306, "unit": "C", "tz": "America/Toronto"},
    "nyc":         {"icao": "KLGA", "lat": 40.7743, "lon":  -73.8726, "unit": "F", "tz": "America/New_York"},
    "new york":    {"icao": "KLGA", "lat": 40.7743, "lon":  -73.8726, "unit": "F", "tz": "America/New_York"},
    "miami":       {"icao": "KMIA", "lat": 25.7959, "lon":  -80.2870, "unit": "F", "tz": "America/New_York"},
    "chicago":     {"icao": "KORD", "lat": 41.9742, "lon":  -87.9073, "unit": "F", "tz": "America/Chicago"},
    "dallas":      {"icao": "KDFW", "lat": 32.8998, "lon":  -97.0403, "unit": "F", "tz": "America/Chicago"},
    "seattle":     {"icao": "KSEA", "lat": 47.4502, "lon": -122.3088, "unit": "F", "tz": "America/Los_Angeles"},
    "atlanta":     {"icao": "KATL", "lat": 33.6407, "lon":  -84.4277, "unit": "F", "tz": "America/New_York"},
    "denver":      {"icao": "KDEN", "lat": 39.8561, "lon": -104.6737, "unit": "F", "tz": "America/Denver"},
    "los angeles": {"icao": "KLAX", "lat": 33.9425, "lon": -118.4081, "unit": "F", "tz": "America/Los_Angeles"},
    "boston":      {"icao": "KBOS", "lat": 42.3656, "lon":  -71.0096, "unit": "F", "tz": "America/New_York"},
    "houston":     {"icao": "KIAH", "lat": 29.9902, "lon":  -95.3368, "unit": "F", "tz": "America/Chicago"},
    "las vegas":   {"icao": "KLAS", "lat": 36.0840, "lon": -115.1537, "unit": "F", "tz": "America/Los_Angeles"},
    "phoenix":     {"icao": "KPHX", "lat": 33.4373, "lon": -111.9978, "unit": "F", "tz": "America/Phoenix"},
    # ── South America ──────────────────────────────────────────────────────────
    "sao paulo":   {"icao": "SBGR", "lat": -23.4356, "lon":  -46.4731, "unit": "C", "tz": "America/Sao_Paulo"},
    "rio de janeiro": {"icao":"SBGL","lat": -22.8099,"lon":  -43.2505, "unit": "C", "tz": "America/Sao_Paulo"},
    "buenos aires":{"icao": "SAEZ", "lat": -34.8222, "lon":  -58.5358, "unit": "C", "tz": "America/Argentina/Buenos_Aires"},
    "bogota":      {"icao": "SKBO", "lat":   4.7016, "lon":  -74.1469, "unit": "C", "tz": "America/Bogota"},
    # ── Europe ─────────────────────────────────────────────────────────────────
    "london":      {"icao": "EGLC", "lat": 51.5053, "lon":    0.0553, "unit": "C", "tz": "Europe/London"},
    "paris":       {"icao": "LFPG", "lat": 49.0097, "lon":    2.5479, "unit": "C", "tz": "Europe/Paris"},
    "berlin":      {"icao": "EDDB", "lat": 52.3667, "lon":   13.5033, "unit": "C", "tz": "Europe/Berlin"},
    "amsterdam":   {"icao": "EHAM", "lat": 52.3086, "lon":    4.7639, "unit": "C", "tz": "Europe/Amsterdam"},
    "madrid":      {"icao": "LEMD", "lat": 40.4719, "lon":   -3.5626, "unit": "C", "tz": "Europe/Madrid"},
    "rome":        {"icao": "LIRF", "lat": 41.7997, "lon":   12.2462, "unit": "C", "tz": "Europe/Rome"},
    "barcelona":   {"icao": "LEBL", "lat": 41.2974, "lon":    2.0834, "unit": "C", "tz": "Europe/Madrid"},
    "athens":      {"icao": "LGAV", "lat": 37.9364, "lon":   23.9445, "unit": "C", "tz": "Europe/Athens"},
    "istanbul":    {"icao": "LTFM", "lat": 41.2608, "lon":   28.7418, "unit": "C", "tz": "Europe/Istanbul"},
    "ankara":      {"icao": "LTAC", "lat": 40.1281, "lon":   32.9951, "unit": "C", "tz": "Europe/Istanbul"},
    "vienna":      {"icao": "LOWW", "lat": 48.1102, "lon":   16.5697, "unit": "C", "tz": "Europe/Vienna"},
    "warsaw":      {"icao": "EPWA", "lat": 52.1657, "lon":   20.9671, "unit": "C", "tz": "Europe/Warsaw"},
    "stockholm":   {"icao": "ESSA", "lat": 59.6519, "lon":   17.9186, "unit": "C", "tz": "Europe/Stockholm"},
    "zurich":      {"icao": "LSZH", "lat": 47.4647, "lon":    8.5492, "unit": "C", "tz": "Europe/Zurich"},
    "brussels":    {"icao": "EBBR", "lat": 50.9014, "lon":    4.4844, "unit": "C", "tz": "Europe/Brussels"},
    "lisbon":      {"icao": "LPPT", "lat": 38.7742, "lon":   -9.1342, "unit": "C", "tz": "Europe/Lisbon"},
    "prague":      {"icao": "LKPR", "lat": 50.1008, "lon":   14.2600, "unit": "C", "tz": "Europe/Prague"},
    # ── Asia-Pacific ───────────────────────────────────────────────────────────
    "seoul":       {"icao": "RKSI", "lat": 37.4492, "lon":  126.4510, "unit": "C", "tz": "Asia/Seoul"},
    "tokyo":       {"icao": "RJTT", "lat": 35.5494, "lon":  139.7798, "unit": "C", "tz": "Asia/Tokyo"},
    "beijing":     {"icao": "ZBAA", "lat": 40.0799, "lon":  116.5847, "unit": "C", "tz": "Asia/Shanghai"},
    "shanghai":    {"icao": "ZSPD", "lat": 31.1434, "lon":  121.8053, "unit": "C", "tz": "Asia/Shanghai"},
    "hong kong":   {"icao": "VHHH", "lat": 22.3080, "lon":  113.9185, "unit": "C", "tz": "Asia/Hong_Kong"},
    "singapore":   {"icao": "WSSS", "lat":  1.3502, "lon":  103.9940, "unit": "C", "tz": "Asia/Singapore"},
    "sydney":      {"icao": "YSSY", "lat": -33.9461,"lon":  151.1772, "unit": "C", "tz": "Australia/Sydney"},
    "melbourne":   {"icao": "YMML", "lat": -37.6690,"lon":  144.8410, "unit": "C", "tz": "Australia/Melbourne"},
    "auckland":    {"icao": "NZAA", "lat": -37.0082,"lon":  174.7917, "unit": "C", "tz": "Pacific/Auckland"},
    "wellington":  {"icao": "NZWN", "lat": -41.3272,"lon":  174.8052, "unit": "C", "tz": "Pacific/Auckland"},
    "dubai":       {"icao": "OMDB", "lat": 25.2523, "lon":   55.3644, "unit": "C", "tz": "Asia/Dubai"},
    "mumbai":      {"icao": "VABB", "lat": 19.0886, "lon":   72.8679, "unit": "C", "tz": "Asia/Kolkata"},
    "delhi":       {"icao": "VIDP", "lat": 28.5665, "lon":   77.1031, "unit": "C", "tz": "Asia/Kolkata"},
    "bangkok":     {"icao": "VTBS", "lat": 13.6811, "lon":  100.7476, "unit": "C", "tz": "Asia/Bangkok"},
    "jakarta":     {"icao": "WIII", "lat":  -6.1275, "lon": 106.6537, "unit": "C", "tz": "Asia/Jakarta"},
    "kuala lumpur":{"icao": "WMKK", "lat":  2.7456, "lon":  101.7099, "unit": "C", "tz": "Asia/Kuala_Lumpur"},
    # ── Africa / Middle East ──────────────────────────────────────────────────
    "cairo":       {"icao": "HECA", "lat": 30.1219, "lon":   31.4056, "unit": "C", "tz": "Africa/Cairo"},
    "johannesburg":{"icao": "FAOR", "lat": -26.1392,"lon":   28.2460, "unit": "C", "tz": "Africa/Johannesburg"},
    "nairobi":     {"icao": "HKJK", "lat":  -1.3192, "lon":  36.9275, "unit": "C", "tz": "Africa/Nairobi"},
    "lagos":       {"icao": "DNMM", "lat":   6.5774, "lon":   3.3212, "unit": "C", "tz": "Africa/Lagos"},
    "riyadh":      {"icao": "OERK", "lat": 24.9578, "lon":   46.6989, "unit": "C", "tz": "Asia/Riyadh"},
}

TITLE_PATTERN = re.compile(
    r"[Hh]ighest temperature in (.+?) on (.+)",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"(\w+)\s+(\d{1,2})(?:[,\s]+(\d{4}))?",
)
MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}


# ── Bucket parsing ────────────────────────────────────────────────────────────

def _parse_bucket_label(raw: str) -> tuple[float | None, str]:
    """Return (numeric_value, canonical_label).

    Handles:  "5°C", "82-83°F", "29°F or below", "12°C or higher", "36+"
    Returns the lower bound (or the single value) as the numeric anchor,
    plus a canonical string like "5", "82-83", "29-", "12+".
    """
    raw = raw.strip()

    # "X or higher / or above / +"
    m = re.match(r"(-?\d+(?:\.\d+)?)\s*[°º]?[A-Za-z]?\s*(?:or higher|or above|\+)", raw, re.I)
    if m:
        v = float(m.group(1))
        return v, f"{int(v) if v == int(v) else v}+"

    # "X or below / or lower"
    m = re.match(r"(-?\d+(?:\.\d+)?)\s*[°º]?[A-Za-z]?\s*(?:or below|or lower|and below|≤)", raw, re.I)
    if m:
        v = float(m.group(1))
        return v, f"{int(v) if v == int(v) else v}-"

    # Range "82-83°F" or "-5--4°C"
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)", raw)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return lo, f"{int(lo) if lo==int(lo) else lo}-{int(hi) if hi==int(hi) else hi}"

    # Single value "5°C" or "5"
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*[°º]?[A-Za-z]?$", raw, re.I)
    if m:
        v = float(m.group(1))
        return v, f"{int(v) if v == int(v) else v}"

    return None, raw


def _winning_bucket(markets: list[dict]) -> tuple[float | None, str | None]:
    """Return (numeric_temp, canonical_label) for the market that resolved YES.

    Uses groupItemTitle (e.g. "11°C", "12°C or higher", "4°C or below") which
    is already a clean bucket label — no regex parsing of the question needed.
    """
    for mkt in markets:
        prices_raw = mkt.get("outcomePrices", "[]")
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        except (ValueError, TypeError):
            continue
        if not prices:
            continue
        try:
            yes_price = float(str(prices[0]))
        except (ValueError, TypeError):
            continue
        if yes_price >= 0.95:
            # Prefer the clean groupItemTitle, fall back to parsing the question
            label = mkt.get("groupItemTitle") or ""
            if not label:
                q = mkt.get("question", "")
                label_m = re.search(r"be\s+(.+?)\s+on\b", q, re.I)
                label = label_m.group(1) if label_m else q
            val, canon = _parse_bucket_label(label.strip())
            return val, canon
    return None, None


# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_date(raw: str, fallback_year: int | None = None) -> str | None:
    """Parse 'February 21' or 'February 21, 2026' → '2026-02-21'.

    When no year is in the title (common in Polymarket weather markets), we
    infer year from the event's startDate or endDate.
    """
    # Strip trailing punctuation
    raw = raw.rstrip("?!.,")
    m = DATE_PATTERN.search(raw)
    if not m:
        return None
    month_name = m.group(1).lower()
    day = int(m.group(2))
    year_str = m.group(3)
    year = int(year_str) if year_str else fallback_year
    if year is None:
        return None
    month = MONTH_MAP.get(month_name)
    if not month:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


# ── City extraction ───────────────────────────────────────────────────────────

def _parse_city(raw: str) -> str | None:
    """Match city from event title to our station metadata.

    City names in titles are space-separated (e.g. "Sao Paulo", "NYC").
    Keys in STATION_META are lowercase canonical names.
    """
    cleaned = raw.lower().strip()
    # Exact match first
    if cleaned in STATION_META:
        return cleaned
    # Prefix match (handles "sao paulo" → "sao paulo")
    for key in STATION_META:
        if cleaned == key:
            return key
    # Substring: city key is contained in the title portion
    for key in STATION_META:
        if key in cleaned:
            return key
    return None


# ── API fetch ─────────────────────────────────────────────────────────────────

def _fetch_events_page(offset: int, limit: int = 50) -> list[dict]:
    """Fetch one page of closed weather events from the Gamma API."""
    params = {
        "closed":   "true",
        "limit":    limit,
        "offset":   offset,
        "tag_slug": "weather",      # narrows to weather events only
        "order":    "startDate",
        "ascending":"false",
    }
    for attempt in range(3):
        try:
            resp = requests.get(GAMMA_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return []


def fetch_all_resolved(
    since_date: str | None = None,
    city_filter: str | None = None,
) -> list[dict]:
    """Paginate through all closed weather events and return structured records."""
    records: list[dict] = []
    offset = 0
    limit = 50
    cutoff = datetime.fromisoformat(since_date).date() if since_date else date(2024, 1, 1)
    seen_ids: set[str] = set()

    print(f"Fetching resolved markets since {cutoff} …")

    while True:
        events = _fetch_events_page(offset, limit)
        if not events:
            break

        stop_early = False
        for event in events:
            # Only verified resolved events
            if event.get("status") not in ("resolved", None):
                continue
            if not event.get("closed"):
                continue

            title = event.get("title", "")
            m = TITLE_PATTERN.match(title)
            if not m:
                continue

            city_raw, date_raw = m.group(1), m.group(2)
            city_slug = _parse_city(city_raw)
            if city_slug is None:
                continue
            if city_filter and city_filter.lower() not in (city_slug, city_raw.lower()):
                continue

            # Infer year from event endDate / startDate when title lacks one
            fallback_year = None
            for date_field in ("endDate", "startDate", "createdAt"):
                raw_ts = event.get(date_field, "")
                if raw_ts and len(raw_ts) >= 4:
                    try:
                        fallback_year = int(raw_ts[:4])
                        break
                    except ValueError:
                        pass

            target_date = _parse_date(date_raw, fallback_year=fallback_year)
            if target_date is None:
                continue

            # Stop if we've gone past our cutoff
            if date.fromisoformat(target_date) < cutoff:
                stop_early = True
                continue

            event_id = str(event.get("id", ""))
            if event_id in seen_ids:
                continue
            seen_ids.add(event_id)

            markets = event.get("markets", [])
            resolved_temp, bucket_label = _winning_bucket(markets)
            if resolved_temp is None:
                continue

            meta = STATION_META[city_slug]
            records.append({
                "event_id":     event_id,
                "title":        title,
                "city":         city_raw.strip(),
                "city_slug":    city_slug,
                "icao":         meta["icao"],
                "lat":          meta["lat"],
                "lon":          meta["lon"],
                "unit":         meta["unit"],
                "timezone":     meta["tz"],
                "target_date":  target_date,
                "resolved_temp":resolved_temp,
                "bucket_label": bucket_label,
            })

        print(f"  offset={offset:4d}  events={len(events):3d}  records_so_far={len(records)}")
        offset += limit

        if stop_early or len(events) < limit:
            break

        time.sleep(0.3)

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2024-01-01", help="Earliest date (YYYY-MM-DD)")
    parser.add_argument("--city", default=None, help="Filter to one city slug")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = fetch_all_resolved(since_date=args.since, city_filter=args.city)
    print(f"\nTotal resolved markets found: {len(records)}")

    with OUT_JSON.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved → {OUT_JSON}")

    # CSV
    if records:
        import csv
        with OUT_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved → {OUT_CSV}")

        # Quick preview
        from collections import Counter
        city_counts = Counter(r["city_slug"] for r in records)
        print("\nMarkets per city:")
        for city, count in sorted(city_counts.items(), key=lambda x: -x[1]):
            print(f"  {city:<15} {count}")


if __name__ == "__main__":
    main()
