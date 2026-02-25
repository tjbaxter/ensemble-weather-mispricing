"""phase3_scout.py

Phase 3: Daily Seoul signal scanner.

Fetches the live Open-Meteo ensemble forecast and live Polymarket crowd pricing
for Seoul (RKSI), then outputs a clear BUY / SKIP trading decision.

Based on Phase 2 backtest findings:
  - ensemble_avg and ncep_aigfs025 show +30-33% ROI on Seoul disagreements
  - Average crowd buy price at T-24h: ~27-30¢ (break-even well below 50%)
  - Only trade when our bucket is priced < 40¢ and overround < 115%

Usage:
    python -m backtest.phase3_scout                  # tomorrow KST (default)
    python -m backtest.phase3_scout --date 2026-02-25
    python -m backtest.phase3_scout --date 2026-02-25 --all-cities
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

CITIES = {
    "seoul": {
        "label": "Seoul",
        "lat": 37.4691,
        "lon": 126.4510,
        "tz_offset": 9,
        "unit": "C",
        "gamma_keywords": ["seoul", "incheon"],
    },
    "london": {
        "label": "London",
        "lat": 51.4775,
        "lon": -0.4614,
        "tz_offset": 0,
        "unit": "C",
        "gamma_keywords": ["london", "heathrow"],
    },
    "nyc": {
        "label": "NYC",
        "lat": 40.7772,
        "lon": -73.8726,
        "tz_offset": -5,
        "unit": "F",
        "gamma_keywords": ["new york", "nyc", "laguardia"],
    },
}

MODELS = [
    "ncep_aigfs025",
    "gfs_graphcast025",
    "kma_gdps",
    "ecmwf_ifs025",
    "gem_global",
    "gfs_seamless",
    "icon_seamless",
]

# Trading thresholds (Phase 2 validated)
MAX_BUY_PRICE   = 0.40   # only buy if crowd prices our bucket below this
OVERROUND_CAP   = 1.15   # market overround must be ≤ 115%
TRADE_SIZE_USD  = 5.0    # fixed $5 per trade

GAMMA_API = "https://gamma-api.polymarket.com"


# ── Bucket helpers ─────────────────────────────────────────────────────────────

def _parse_bucket_label(label: str) -> str | None:
    """Convert a Polymarket bucket title to a canonical string.

    Examples:
        "12°C or higher"  → "12+"
        "-5°C or lower"   → "-5-"
        "below 0°C"       → "0-"
        "8°C"             → "8"
        "82-83°F"         → "82-83"
        "29°F or below"   → "29-"
        "44°F or higher"  → "44+"
    """
    label = label.strip()

    # "X or higher / X or above"
    m = re.match(r"^(-?\d+)°?[CF]?\s+or\s+(?:higher|above)$", label, re.I)
    if m:
        return f"{m.group(1)}+"

    # "X or lower / X or below"
    m = re.match(r"^(-?\d+)°?[CF]?\s+or\s+(?:lower|below)$", label, re.I)
    if m:
        return f"{m.group(1)}-"

    # "below X"
    m = re.match(r"^below\s+(-?\d+)", label, re.I)
    if m:
        return f"{m.group(1)}-"

    # "X-Y°F" range
    m = re.match(r"^(-?\d+)-(-?\d+)°?[CF]?$", label)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # Single "X°C" or "X°F"
    m = re.match(r"^(-?\d+)°?[CF]?$", label)
    if m:
        return m.group(1)

    return None


def _temp_in_bucket(temp: float, canon: str) -> bool:
    if canon.endswith("+"):
        try:
            return temp >= float(canon[:-1])
        except ValueError:
            return False
    if canon.endswith("-"):
        try:
            return temp <= float(canon[:-1])
        except ValueError:
            return False
    if "-" in canon and not canon.lstrip("-+").startswith(""):
        # range like "82-83"
        parts = canon.split("-", 1)
        try:
            lo, hi = float(parts[0]), float(parts[1])
            return lo <= temp < hi
        except ValueError:
            pass
    try:
        return round(temp) == int(canon)
    except ValueError:
        return False


def temp_to_bucket(temp: float, canons: list[str]) -> str | None:
    for c in canons:
        if _temp_in_bucket(temp, c):
            return c
    return str(int(round(temp)))


# ── Open-Meteo fetch ──────────────────────────────────────────────────────────

def fetch_forecasts(city_cfg: dict, target_date: str) -> dict[str, float | None]:
    """Fetch each model's predicted max temp at the city's ICAO coordinates."""
    tz_name = {9: "Asia/Seoul", 0: "Europe/London", -5: "America/New_York"}.get(
        city_cfg["tz_offset"], "UTC"
    )
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":   city_cfg["lat"],
        "longitude":  city_cfg["lon"],
        "daily":      "temperature_2m_max",
        "timezone":   tz_name,
        "start_date": target_date,
        "end_date":   target_date,
        "models":     ",".join(MODELS),
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    # When multiple models are requested in one call, Open-Meteo puts all data
    # in data["daily"] with keys like "temperature_2m_max_ncep_aigfs025".
    daily = data.get("daily", {})
    results: dict[str, float | None] = {}
    for model in MODELS:
        key = f"temperature_2m_max_{model}"
        try:
            val = daily.get(key, [None])[0]
            results[model] = float(val) if val is not None else None
        except (IndexError, TypeError, ValueError):
            results[model] = None
    return results


# ── Polymarket fetch ──────────────────────────────────────────────────────────

def fetch_markets(city_cfg: dict, target_date: str) -> list[dict]:
    """Fetch open temperature markets for this city/date from Gamma API."""
    url = f"{GAMMA_API}/events"
    params = {"tag_slug": "weather", "closed": "false", "limit": 200}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    events = resp.json()

    keywords = city_cfg["gamma_keywords"]
    results = []

    for event in events:
        title = event.get("title", "").lower()
        if not any(kw in title for kw in keywords):
            continue
        # Check date — endDate should match target_date
        end_date = (event.get("endDate") or "")[:10]
        if end_date != target_date:
            continue

        markets = event.get("markets", [])
        bucket_prices: dict[str, float] = {}
        bucket_tokens: dict[str, str] = {}

        for m in markets:
            raw_label = m.get("groupItemTitle") or m.get("question", "")
            canon = _parse_bucket_label(raw_label)
            if not canon:
                continue

            price_str = m.get("outcomePrices")
            yes_price: float | None = None
            if price_str:
                try:
                    prices = json.loads(price_str)
                    yes_price = float(prices[0]) if prices else None
                except (json.JSONDecodeError, ValueError, IndexError):
                    pass

            if yes_price is not None:
                bucket_prices[canon] = yes_price

            clob_str = m.get("clobTokenIds")
            if clob_str:
                try:
                    tokens = json.loads(clob_str)
                    if tokens:
                        bucket_tokens[canon] = tokens[0]
                except (json.JSONDecodeError, ValueError):
                    pass

        if bucket_prices:
            results.append({
                "event_id":     event.get("id"),
                "title":        event.get("title"),
                "bucket_prices": bucket_prices,
                "bucket_tokens": bucket_tokens,
                "overround":    sum(bucket_prices.values()),
            })

    return results


# ── Signal printer ────────────────────────────────────────────────────────────

def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9 / 5 + 32


def print_signal(city_key: str, target_date: str) -> None:
    cfg = CITIES[city_key]
    tz = timezone(timedelta(hours=cfg["tz_offset"]))
    now_local = datetime.now(tz)
    unit = cfg["unit"]

    print(f"\n{'═'*62}")
    print(f"  {cfg['label'].upper()} SIGNAL — {target_date}")
    print(f"{'═'*62}")
    print(f"  Scanned: {now_local.strftime('%Y-%m-%d %H:%M')} local\n")

    # --- Forecasts ---
    print("  [1/2] Open-Meteo ensemble forecasts …")
    try:
        preds = fetch_forecasts(cfg, target_date)
    except Exception as e:
        print(f"  ❌ ERROR fetching forecasts: {e}")
        return

    valid = {m: v for m, v in preds.items() if v is not None}
    if not valid:
        print("  ❌ No model predictions available.")
        return

    print(f"  {'Model':<22} {'°C':>6}  {'°F':>6}")
    print(f"  {'─'*36}")
    for model in MODELS:
        val = preds.get(model)
        if val is None:
            print(f"  {model:<22}   ---    ---")
        else:
            f_val = _celsius_to_fahrenheit(val) if unit == "F" else None
            f_str = f"{f_val:>5.1f}" if f_val is not None else "  ---"
            flag = " ← PRIMARY" if model == "ncep_aigfs025" else ""
            print(f"  {model:<22} {val:>5.1f}  {f_str}{flag}")

    ensemble_avg = sum(valid.values()) / len(valid)
    ensemble_avg_f = _celsius_to_fahrenheit(ensemble_avg)

    ai_models = [m for m in ["ncep_aigfs025", "gfs_graphcast025"] if preds.get(m) is not None]
    ai_duo = sum(preds[m] for m in ai_models) / len(ai_models) if ai_models else None

    print(f"\n  ensemble_avg  ({len(valid)}/7 models): {ensemble_avg:.1f}°C  /  {ensemble_avg_f:.1f}°F")
    if ai_duo:
        ai_f = _celsius_to_fahrenheit(ai_duo)
        print(f"  AI duo        (ncep + graphcast): {ai_duo:.1f}°C  /  {ai_f:.1f}°F")
    primary = preds.get("ncep_aigfs025")
    if primary:
        print(f"  ncep_aigfs025 (primary model):    {primary:.1f}°C  /  {_celsius_to_fahrenheit(primary):.1f}°F")

    std_vals = list(valid.values())
    std = (sum((v - ensemble_avg) ** 2 for v in std_vals) / len(std_vals)) ** 0.5
    print(f"  Model spread (σ): {std:.1f}°C  {'⚠ HIGH UNCERTAINTY' if std > 2.0 else '✅ Models agree'}")

    # Convert ensemble to market unit for bucket mapping
    ensemble_for_buckets = ensemble_avg_f if unit == "F" else ensemble_avg

    # --- Polymarket ---
    print(f"\n  [2/2] Polymarket crowd prices …")
    try:
        markets = fetch_markets(cfg, target_date)
    except Exception as e:
        print(f"  ❌ ERROR fetching Polymarket: {e}")
        markets = []

    if not markets:
        ensemble_bucket_est = str(int(round(ensemble_for_buckets)))
        print(f"  ⚠  No {cfg['label']} market found for {target_date}.")
        print(f"     Market may open later or date is too far ahead.")
        print(f"\n  MANUAL CHECK: https://polymarket.com/markets?tag=weather")
        print(f"  Your ensemble_avg → bucket {ensemble_bucket_est}{'°F' if unit=='F' else '°C'}")
        print(f"{'═'*62}\n")
        return

    for mkt in markets:
        canons = list(mkt["bucket_prices"].keys())
        ensemble_bucket = temp_to_bucket(ensemble_for_buckets, canons)
        crowd_bucket    = max(mkt["bucket_prices"], key=lambda k: mkt["bucket_prices"][k])
        buy_price       = mkt["bucket_prices"].get(ensemble_bucket)
        overround_ok    = mkt["overround"] <= OVERROUND_CAP
        disagree        = ensemble_bucket != crowd_bucket
        price_ok        = buy_price is not None and buy_price < MAX_BUY_PRICE

        print(f"\n  Market: {mkt['title']}")
        print(f"  Overround: {mkt['overround']*100:.0f}%  {'✅ Tradeable' if overround_ok else '❌ Untradeable (>115%)'}\n")

        print(f"  {'Bucket':>8}  {'Crowd':>6}   Notes")
        print(f"  {'─'*46}")
        for bucket, price in sorted(mkt["bucket_prices"].items(),
                                    key=lambda x: -x[1]):
            notes = []
            if bucket == crowd_bucket:
                notes.append("CROWD PICK")
            if bucket == ensemble_bucket:
                notes.append("← OUR MODEL")
            bar = "█" * max(1, int(price * 25))
            note_str = "  ".join(notes)
            print(f"  {bucket:>8}  {price:>5.0%}   {bar}  {note_str}")

        # ── Decision ──────────────────────────────────────────────────────────
        print(f"\n  ── DECISION ({'ensemble_avg'} → {ensemble_bucket}{'°F' if unit=='F' else '°C'}) ──")

        if not overround_ok:
            print(f"  ⛔ SKIP — overround {mkt['overround']*100:.0f}% exceeds 115% cap.")

        elif not disagree:
            print(f"  😐 PASS — model agrees with crowd on '{crowd_bucket}'.")
            print(f"     Crowd pricing: {buy_price:.0%}. No disagreement = no alpha signal.")

        elif buy_price is None:
            print(f"  ❓ SKIP — bucket '{ensemble_bucket}' not found in market pricing.")

        elif not price_ok:
            print(f"  ⚠️  SKIP — our bucket '{ensemble_bucket}' already priced at {buy_price:.0%}.")
            print(f"     Need < 40¢ to have positive expected value.")

        else:
            shares   = TRADE_SIZE_USD / buy_price
            win_pnl  = shares * (1.0 - buy_price)
            be_rate  = buy_price
            token    = mkt["bucket_tokens"].get(ensemble_bucket, "—")

            print(f"  ✅✅ BUY SIGNAL")
            print(f"")
            print(f"     Bucket to buy:    '{ensemble_bucket}' YES")
            print(f"     Buy price:         {buy_price:.0%}  (crowd favours '{crowd_bucket}')")
            print(f"     Trade size:       ${TRADE_SIZE_USD:.0f}  →  {shares:.1f} shares")
            print(f"     Break-even:        {be_rate:.0%} win rate")
            print(f"     Backtest hit rate: ~37-42% (Seoul, 30-day sample)")
            print(f"     Expected PnL:     +${win_pnl:.2f} if correct  /  -${TRADE_SIZE_USD:.2f} if wrong")
            print(f"     Model σ warning:  {'⚠ Spread > 2°C — consider halving size' if std > 2.0 else '✅ Models agree, full size OK'}")
            print(f"")
            print(f"     Token ID:  {token}")
            print(f"     Event:     https://polymarket.com/event/{mkt['event_id']}")

        print(f"\n{'═'*62}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Daily city signal scanner")
    parser.add_argument("--date",       default=None,
                        help="Target date YYYY-MM-DD (default: tomorrow local time)")
    parser.add_argument("--city",       default="seoul",
                        choices=list(CITIES.keys()),
                        help="City to scan (default: seoul)")
    parser.add_argument("--all-cities", action="store_true",
                        help="Scan all cities")
    args = parser.parse_args()

    targets = list(CITIES.keys()) if args.all_cities else [args.city]

    for city_key in targets:
        cfg = CITIES[city_key]
        if args.date:
            target_date = args.date
        else:
            tz = timezone(timedelta(hours=cfg["tz_offset"]))
            target_date = (datetime.now(tz) + timedelta(days=1)).strftime("%Y-%m-%d")

        print_signal(city_key, target_date)


if __name__ == "__main__":
    main()
