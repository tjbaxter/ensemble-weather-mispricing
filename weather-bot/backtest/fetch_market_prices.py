"""fetch_market_prices.py

For each resolved Polymarket weather market, fetch the historical YES price
for every bucket at approximately T-24h before resolution (the moment a
day-before bettor would have placed their order).

Method: CLOB price history API — returns ~30 hourly price points per market.
We find the price closest to (endDate - 24h) for each bucket's YES token.

The bucket with the highest YES price at T-24h is the "crowd's pick" — what
the collective market was pricing most heavily before the model run came out.

Usage:
    python -m backtest.fetch_market_prices
    python -m backtest.fetch_market_prices --city seoul
    python -m backtest.fetch_market_prices --force-refetch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GAMMA_URL = "https://gamma-api.polymarket.com/events"
CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"

DATA_DIR = ROOT / "backtest" / "data"
IN_JSON  = DATA_DIR / "resolved_markets.json"
OUT_JSON = DATA_DIR / "market_prices.json"

BUCKET_CACHE = DATA_DIR / "bucket_cache.json"
_bucket_cache: dict[str, dict] = {}   # event_id → full bucket structure


# ── CLOB price history ─────────────────────────────────────────────────────────

def _fetch_clob_history(token_id: str) -> list[dict]:
    """Return [{t: unix_ts, p: price}, ...] or [] on failure."""
    for attempt in range(3):
        try:
            resp = requests.get(
                CLOB_HISTORY_URL,
                params={"market": token_id, "interval": "max", "fidelity": 60},
                timeout=20,
            )
            if resp.status_code == 200:
                return resp.json().get("history", [])
            if resp.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return []


def _price_at_t(history: list[dict], target_ts: float) -> float | None:
    """Find the price closest to target_ts in the CLOB history."""
    if not history:
        return None
    best = min(history, key=lambda h: abs(h["t"] - target_ts))
    # Only use a point if it's within 6 hours of target
    if abs(best["t"] - target_ts) > 6 * 3600:
        return None
    return float(best["p"])


# ── Gamma API — fetch bucket structure ────────────────────────────────────────

def _load_bucket_cache() -> None:
    global _bucket_cache
    if BUCKET_CACHE.exists():
        try:
            _bucket_cache = json.loads(BUCKET_CACHE.read_text())
        except (ValueError, json.JSONDecodeError):
            _bucket_cache = {}


def _save_bucket_cache() -> None:
    BUCKET_CACHE.write_text(json.dumps(_bucket_cache, indent=2))


def _fetch_event_buckets(event_id: str) -> dict | None:
    """Fetch full bucket structure for an event from Gamma API.

    Returns {
        "buckets": [
            {"label": "11°C", "token_id": "...", "end_date": "2026-02-22T12:00:00Z",
             "end_ts": 1234567890.0}
        ]
    }
    """
    if event_id in _bucket_cache:
        return _bucket_cache[event_id]

    for attempt in range(3):
        try:
            resp = requests.get(f"{GAMMA_URL}/{event_id}", timeout=20)
            if resp.status_code == 429:
                time.sleep(15)
                continue
            if resp.status_code != 200:
                return None
            event = resp.json()
            break
        except requests.RequestException:
            time.sleep(2 ** attempt)
    else:
        return None

    markets = event.get("markets", [])
    buckets = []
    for mkt in markets:
        label = mkt.get("groupItemTitle", "")
        token_ids_raw = mkt.get("clobTokenIds", "[]")
        try:
            token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
        except (ValueError, TypeError):
            token_ids = []
        yes_token = str(token_ids[0]) if token_ids else None

        end_date_str = mkt.get("endDate") or mkt.get("endDateIso", "")
        end_ts = None
        if end_date_str:
            try:
                dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                end_ts = dt.timestamp()
            except ValueError:
                pass

        if label and yes_token:
            buckets.append({
                "label":    label,
                "token_id": yes_token,
                "end_ts":   end_ts,
            })

    if not buckets:
        return None

    result = {"buckets": buckets}
    _bucket_cache[event_id] = result
    return result


# ── Crowd pick determination ──────────────────────────────────────────────────

def fetch_event_prices(
    event_id: str,
    target_date: str,
) -> dict | None:
    """Return price info for all buckets at T-24h.

    Returns {
        "crowd_bucket": "11",          # canonical label of crowd favourite
        "crowd_temp": 11.0,            # numeric anchor of crowd bucket
        "method": "clob_history",
        "bucket_prices": {"11": 0.45, "10": 0.30, ...}   # prices at T-24h
    }
    Or None if insufficient data.
    """
    structure = _fetch_event_buckets(event_id)
    if structure is None:
        return None

    buckets = structure["buckets"]
    if not buckets:
        return None

    # T-24h target timestamp (use the end_ts from first bucket)
    end_ts = next((b["end_ts"] for b in buckets if b["end_ts"]), None)
    if end_ts is None:
        return None
    t24_ts = end_ts - 86400  # exactly 24 hours before resolution

    # Fetch price for each bucket's YES token
    bucket_prices: dict[str, float] = {}
    from backtest.fetch_polymarket_resolved import _parse_bucket_label

    for bkt in buckets:
        label = bkt["label"]
        token = bkt["token_id"]
        _, canon = _parse_bucket_label(label)
        history = _fetch_clob_history(token)
        price = _price_at_t(history, t24_ts)
        if price is not None:
            bucket_prices[canon] = price
        time.sleep(0.2)

    if not bucket_prices:
        return None

    # Crowd's pick = bucket with highest price at T-24h
    crowd_canon = max(bucket_prices, key=lambda k: bucket_prices[k])

    # Parse numeric value from canonical label
    from backtest.fetch_polymarket_resolved import _parse_bucket_label as _pbl
    # canonical format: "11", "12+", "4-", "82-83"
    if crowd_canon.endswith("+"):
        crowd_temp = float(crowd_canon[:-1])
    elif crowd_canon.endswith("-"):
        crowd_temp = float(crowd_canon[:-1])
    elif "-" in crowd_canon and not crowd_canon.startswith("-"):
        parts = crowd_canon.split("-")
        crowd_temp = (float(parts[0]) + float(parts[1])) / 2
    else:
        try:
            crowd_temp = float(crowd_canon)
        except ValueError:
            crowd_temp = None

    return {
        "crowd_bucket": crowd_canon,
        "crowd_temp":   crowd_temp,
        "method":       "clob_history",
        "bucket_prices":bucket_prices,
        "t24_ts":       t24_ts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_all_prices(
    city_filter: str | None = None,
    force_refetch: bool = False,
) -> dict[str, dict]:
    """Fetch T-24h prices for all resolved markets. Returns {event_id: price_info}."""
    if not IN_JSON.exists():
        print(f"ERROR: {IN_JSON} not found. Run fetch_polymarket_resolved.py first.")
        sys.exit(1)

    records = json.loads(IN_JSON.read_text())
    _load_bucket_cache()

    # Load existing results for incremental updates
    existing: dict[str, dict] = {}
    if OUT_JSON.exists() and not force_refetch:
        try:
            existing = json.loads(OUT_JSON.read_text())
        except (ValueError, json.JSONDecodeError):
            pass

    results: dict[str, dict] = dict(existing)
    total = len(records)
    fetched = 0

    for i, rec in enumerate(records):
        eid  = rec["event_id"]
        city = rec["city_slug"]

        if city_filter and city_filter.lower() not in (city, rec["city"].lower()):
            continue
        if eid in results and not force_refetch:
            continue

        tdate = rec["target_date"]
        print(f"[{i+1:4d}/{total}] {tdate}  {rec['city']:<14}", end="  ", flush=True)

        info = fetch_event_prices(eid, tdate)
        if info:
            results[eid] = info
            crowd = info["crowd_bucket"]
            top_price = info["bucket_prices"].get(crowd, 0)
            print(f"crowd={crowd} @ {top_price:.0%}  ({len(info['bucket_prices'])} buckets)")
            fetched += 1
        else:
            results[eid] = {"crowd_bucket": None, "crowd_temp": None, "method": "failed"}
            print("FAILED")

        if (i + 1) % 20 == 0:
            _save_bucket_cache()
            with OUT_JSON.open("w") as f:
                json.dump(results, f, indent=2)

        time.sleep(0.3)

    _save_bucket_cache()
    with OUT_JSON.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone — fetched {fetched} new price records. Total: {len(results)}")
    print(f"Saved → {OUT_JSON}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",         default=None)
    parser.add_argument("--force-refetch",action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fetch_all_prices(city_filter=args.city, force_refetch=args.force_refetch)


if __name__ == "__main__":
    main()
