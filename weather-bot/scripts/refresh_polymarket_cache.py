#!/usr/bin/env python3
"""Refresh polymarket_cache.json with latest resolved temperatures.

Run on the VM to update the cache that the React dashboard reads:
    python3 scripts/refresh_polymarket_cache.py

This fetches from gamma-api.polymarket.com for any dates not already
in the disk cache, exactly like the Streamlit dashboard does server-side.
"""

import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Run: pip install requests")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "polymarket_cache.json"

PM_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

CITIES = {
    "London": {
        "slug": "highest-temperature-in-london-on",
        "bucket_style": "exact_1c",
        "alt_slugs": None,
    },
    "Paris": {
        "slug": "highest-temperature-in-paris-on",
        "bucket_style": "exact_1c",
        "alt_slugs": None,
    },
    "Seoul": {
        "slug": "highest-temperature-in-seoul-on",
        "bucket_style": "exact_1c",
        "alt_slugs": None,
    },
    "Toronto": {
        "slug": "highest-temperature-in-toronto-on",
        "bucket_style": "exact_1c",
        "alt_slugs": None,
    },
    "Seattle": {
        "slug": "highest-temperature-in-seattle-on",
        "bucket_style": "range_2f",
        "alt_slugs": None,
    },
    "Buenos Aires": {
        "slug": "highest-temperature-in-buenos-aires-on",
        "bucket_style": "exact_1c",
        "alt_slugs": None,
    },
    "Ankara": {
        "slug": "highest-temperature-in-ankara-on",
        "bucket_style": "exact_1c",
        "alt_slugs": None,
    },
    "Wellington": {
        "slug": "highest-temperature-in-wellington-on",
        "bucket_style": "exact_1c",
        "alt_slugs": None,
    },
    "New York": {
        "slug": "highest-temperature-in-new-york-on",
        "bucket_style": "range_2f",
        "alt_slugs": ["highest-temperature-in-nyc-on",
                       "high-temperature-at-laguardia-airport-on",
                       "will-the-high-temperature-at-laguardia-airport-on"],
    },
    "Chicago": {
        "slug": "highest-temperature-in-chicago-on",
        "bucket_style": "range_2f",
        "alt_slugs": ["high-temperature-at-ohare-airport-on",
                       "will-the-high-temperature-at-ohare-airport-on"],
    },
    "Miami": {
        "slug": "highest-temperature-in-miami-on",
        "bucket_style": "range_2f",
        "alt_slugs": ["high-temperature-at-miami-international-airport-on",
                       "will-the-high-temperature-at-miami-international-airport-on"],
    },
    "Dallas": {
        "slug": "highest-temperature-in-dallas-on",
        "bucket_style": "range_2f",
        "alt_slugs": ["high-temperature-at-dfw-airport-on",
                       "will-the-high-temperature-at-dfw-airport-on"],
    },
    "Atlanta": {
        "slug": "highest-temperature-in-atlanta-on",
        "bucket_style": "range_2f",
        "alt_slugs": ["high-temperature-at-hartsfield-jackson-airport-on",
                       "will-the-high-temperature-at-hartsfield-jackson-airport-on"],
    },
}


def parse_celsius(markets):
    for mkt in markets:
        raw = mkt.get("outcomePrices", "[]")
        prices = json.loads(raw) if isinstance(raw, str) else raw
        if not prices or float(prices[0]) < 0.9:
            continue
        q = mkt.get("question", "").lower()
        m = re.search(r"(\d+)\s*°c\s*or\s*(higher|above)", q)
        if m:
            return (f"≥{m.group(1)}°C", int(m.group(1)), True)
        m = re.search(r"(\d+)\s*°c\s*or\s*below", q)
        if m:
            return (f"≤{m.group(1)}°C", int(m.group(1)), None)
        m = re.search(r"be\s+(\d+)\s*°c\b", q)
        if m:
            return (f"{m.group(1)}°C", int(m.group(1)), False)
    return None


def parse_fahrenheit(markets):
    for mkt in markets:
        raw = mkt.get("outcomePrices", "[]")
        prices = json.loads(raw) if isinstance(raw, str) else raw
        if not prices or float(prices[0]) < 0.9:
            continue
        q = mkt.get("question", "").lower()
        m = re.search(r"(\d+)[-–](\d+)\s*°f", q)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            return (f"{lo}-{hi}°F", lo, hi, None, None)
        m = re.search(r"(\d+)\s*°f\s*or\s*below", q)
        if m:
            return (f"≤{m.group(1)}°F", None, int(m.group(1)), None, None)
        m = re.search(r"(\d+)\s*°f\s*or\s*(higher|above)", q)
        if m:
            return (f"≥{m.group(1)}°F", int(m.group(1)), None, None, None)
    return None


def fetch_resolutions(city_name, slug, bucket_style, from_date, to_date, alt_slugs=None):
    all_slugs = [slug] + (alt_slugs or [])
    resolved = {}
    d = from_date
    while d <= to_date:
        ds = d.strftime("%Y-%m-%d")
        mn = PM_MONTHS[d.month - 1]
        candidates = []
        for base in all_slugs:
            candidates.append(f"{base}-{mn}-{d.day}-{d.year}")
            candidates.append(f"{base}-{mn}-{d.day}")
        for sl in candidates:
            try:
                r = requests.get(
                    "https://gamma-api.polymarket.com/events",
                    params={"slug": sl},
                    timeout=8,
                )
                if r.status_code != 200 or not r.json():
                    continue
                e = r.json()[0]
                created = e.get("createdAt", "")[:10]
                if created:
                    cdate = datetime.strptime(created, "%Y-%m-%d").date()
                    if not (0 <= (d - cdate).days <= 7):
                        continue
                mkts = e.get("markets", [])
                parser = parse_fahrenheit if bucket_style == "range_2f" else parse_celsius
                result = parser(mkts)
                if result:
                    resolved[ds] = list(result)
                    print(f"  {ds}: {result[0]}")
                break
            except Exception as exc:
                print(f"  {ds}: fetch error - {exc}")
        time.sleep(0.12)
        d += timedelta(days=1)
    return resolved


def main():
    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            cache = {}

    today = datetime.now(timezone.utc).date()
    total_new = 0

    def _save():
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=2))

    for city_name, cfg in CITIES.items():
        existing = cache.get(city_name, {})
        if existing:
            last_cached = date.fromisoformat(max(existing.keys()))
        else:
            last_cached = today - timedelta(days=30)

        fetch_start = last_cached + timedelta(days=1)
        if fetch_start > today:
            print(f"{city_name}: up to date (last={last_cached})")
            continue

        print(f"{city_name}: fetching {fetch_start} → {today}")
        new_entries = fetch_resolutions(
            city_name,
            cfg["slug"],
            cfg["bucket_style"],
            fetch_start,
            today,
            alt_slugs=cfg.get("alt_slugs"),
        )

        if new_entries:
            if city_name not in cache:
                cache[city_name] = {}
            cache[city_name].update(new_entries)
            total_new += len(new_entries)
            _save()
            print(f"  → {len(new_entries)} new resolved dates (saved)")
        else:
            print(f"  → no new resolutions found")

    if total_new > 0:
        print(f"\nDone. {total_new} total new entries across all cities.")
    else:
        print("\nNo new entries to write.")


if __name__ == "__main__":
    main()
