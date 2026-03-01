#!/usr/bin/env python3
"""Daily resolver — scrapes Wunderground for actual temps, marks trades WIN/LOSS/MISS.

This is a standalone safety net that runs independently of the main bot.
It reads signals.csv for all 'trade' rows from previous days, fetches the
finalized daily max from WU, and writes outcomes to logs/resolved.csv.

Run daily at 10:00 UTC — WU typically finalises overnight data by then.

Cron (VM):
    0 10 * * * /home/tombaxter/weather-bot/venv/bin/python3 \
               /home/tombaxter/weather-bot/scripts/daily_resolver.py \
               >> /home/tombaxter/weather-bot/logs/resolver.log 2>&1
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.cities import STATIONS

SIGNALS_CSV = ROOT / "logs" / "signals.csv"
TRADES_CSV  = ROOT / "logs" / "trades.csv"
RESOLVED_CSV = ROOT / "logs" / "resolved.csv"

WU_HIGH_PATTERN = re.compile(r'"temperatureMax":\{"value":(-?\d+(?:\.\d+)?)')

RESOLVED_HEADER = [
    "resolved_at",
    "target_date",
    "city",
    "station_icao",
    "bucket",
    "side",
    "entry_price",
    "size_usd",
    "ev_per_bet",
    "spread_colour",
    "det_spread",
    "model_values_json",
    "actual_temp",
    "outcome",        # WIN / LOSS / PUSH
    "pnl_usd",
    "miss_distance",  # actual_temp - ensemble_mean (signed)
    "signal_timestamp",
]


def _ensure_resolved_csv() -> None:
    if not RESOLVED_CSV.exists():
        with RESOLVED_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(RESOLVED_HEADER)


def _load_resolved_keys() -> set[tuple[str, str, str]]:
    """Return set of (target_date, city, bucket) already in resolved.csv."""
    keys: set[tuple[str, str, str]] = set()
    if not RESOLVED_CSV.exists():
        return keys
    with RESOLVED_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys.add((row["target_date"], row["city"], row["bucket"]))
    return keys


def _load_pending_trades() -> list[dict]:
    """Read signals.csv, return rows where action='trade' and date < today."""
    if not SIGNALS_CSV.exists():
        print("signals.csv not found — nothing to resolve.")
        return []

    today = date.today().isoformat()
    pending: dict[tuple[str, str, str], dict] = {}  # (date, city, bucket) → row

    with SIGNALS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("action_taken") != "trade":
                continue
            target_date = row.get("date", "")
            if not target_date or target_date >= today:
                continue
            key = (target_date, row.get("city", ""), row.get("bucket", ""))
            # Keep the earliest signal per (date, city, bucket)
            if key not in pending:
                pending[key] = row

    return list(pending.values())


def _parse_bucket_win(bucket_label: str, actual_temp: int) -> bool:
    """Return True if actual_temp falls in the bucket."""
    clean = bucket_label.replace("°F", "").replace("°C", "").strip()
    try:
        if "+" in clean:
            lower = int(clean.replace("+", ""))
            return actual_temp >= lower
        if "-" in clean:
            parts = clean.split("-", 1)
            return int(parts[0].strip()) <= actual_temp <= int(parts[1].strip())
        return actual_temp == int(clean)
    except (ValueError, IndexError):
        return False


def _compute_pnl(side: str, entry_price: float, size_usd: float, won: bool) -> float:
    """Compute realised P&L on a resolved position.

    For BUY_YES:
        win  → receive $1/share × (size_usd/entry_price) → net = size_usd*(1-entry_price)/entry_price
        loss → lose cost → net = -size_usd
    For BUY_NO:
        entry_price is the NO price = (1 - YES price)
        win  → same formula applied to NO price
        loss → -size_usd
    """
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    if won:
        return round(size_usd * (1.0 - entry_price) / entry_price, 4)
    return round(-size_usd, 4)


async def _fetch_wu_high(http: httpx.AsyncClient, wu_url: str, day: date) -> int | None:
    url = f"{wu_url}/date/{day.isoformat()}"
    try:
        resp = await http.get(url, follow_redirects=True, timeout=20.0)
        resp.raise_for_status()
        match = WU_HIGH_PATTERN.search(resp.text)
        if match:
            return int(round(float(match.group(1))))
    except Exception as exc:
        print(f"  WU fetch error for {wu_url} {day}: {exc}")
    return None


async def resolve_all() -> dict:
    _ensure_resolved_csv()
    already_resolved = _load_resolved_keys()
    pending = _load_pending_trades()

    if not pending:
        print("No pending trades to resolve.")
        return {"resolved": 0, "wins": 0, "losses": 0, "pnl": 0.0}

    # Group by station+date to minimise WU HTTP requests
    station_date_cache: dict[tuple[str, str], int | None] = {}

    stats = {"resolved": 0, "wins": 0, "losses": 0, "pnl": 0.0, "rows": []}

    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as http:
        for row in pending:
            target_date_str = row["date"]
            city = row.get("city", "")
            bucket = row.get("bucket", "")
            station_icao = row.get("station_icao", "")
            side = row.get("side", "BUY_YES")

            key = (target_date_str, city, bucket)
            if key in already_resolved:
                continue

            # Look up WU URL for this station
            station_cfg = STATIONS.get(station_icao)
            if not station_cfg:
                # Try to find station by city slug
                for icao, cfg in STATIONS.items():
                    if cfg.get("market_label", "").lower() == city.lower():
                        station_cfg = cfg
                        station_icao = icao
                        break

            if not station_cfg:
                print(f"  [{city}] No station config for ICAO '{station_icao}' — skipping.")
                continue

            wu_url = station_cfg["wu_url"]
            target_date = date.fromisoformat(target_date_str)
            cache_key = (wu_url, target_date_str)

            if cache_key not in station_date_cache:
                print(f"  [{city}] Fetching WU for {target_date_str}...")
                actual = await _fetch_wu_high(http, wu_url, target_date)
                station_date_cache[cache_key] = actual
            else:
                actual = station_date_cache[cache_key]

            if actual is None:
                print(f"  [{city}] WU returned no data for {target_date_str} — will retry tomorrow.")
                continue

            won = _parse_bucket_win(bucket, actual)
            # For BUY_NO, invert the win condition
            if side == "BUY_NO":
                won = not won

            try:
                entry_price = float(row.get("market_prob") or 0.0)
                size_usd = float(row.get("size_usd") or 0.0)
                ev_per_bet = float(row.get("ev_per_bet") or 0.0)
            except (ValueError, TypeError):
                entry_price, size_usd, ev_per_bet = 0.0, 0.0, 0.0

            pnl = _compute_pnl(side, entry_price, size_usd, won)
            outcome = "WIN" if won else "LOSS"

            # Miss distance: difference between ensemble mean and actual
            miss_distance = ""
            try:
                model_vals = json.loads(row.get("model_values_json") or "{}")
                if model_vals:
                    ensemble_mean = sum(model_vals.values()) / len(model_vals)
                    miss_distance = round(actual - ensemble_mean, 2)
            except (json.JSONDecodeError, ZeroDivisionError):
                pass

            result_row = {
                "resolved_at": datetime.now(UTC).isoformat(),
                "target_date": target_date_str,
                "city": city,
                "station_icao": station_icao,
                "bucket": bucket,
                "side": side,
                "entry_price": entry_price,
                "size_usd": size_usd,
                "ev_per_bet": ev_per_bet,
                "spread_colour": row.get("spread_colour", ""),
                "det_spread": row.get("det_spread", ""),
                "model_values_json": row.get("model_values_json", "{}"),
                "actual_temp": actual,
                "outcome": outcome,
                "pnl_usd": pnl,
                "miss_distance": miss_distance,
                "signal_timestamp": row.get("timestamp", ""),
            }

            with RESOLVED_CSV.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RESOLVED_HEADER)
                writer.writerow(result_row)

            emoji = "✅" if won else "❌"
            print(f"  {emoji} [{city}] {bucket} | actual={actual} | {outcome} | pnl={pnl:+.2f}")

            stats["resolved"] += 1
            stats["pnl"] = round(stats["pnl"] + pnl, 4)
            stats["wins" if won else "losses"] += 1
            stats["rows"].append(result_row)

    return stats


def print_summary(stats: dict) -> None:
    resolved = stats["resolved"]
    if resolved == 0:
        print("\nNothing new resolved today.")
        return

    wins = stats["wins"]
    losses = stats["losses"]
    pnl = stats["pnl"]
    acc = wins / resolved * 100 if resolved else 0

    print(f"\n{'='*50}")
    print(f"  RESOLVER SUMMARY — {date.today()}")
    print(f"  Resolved:  {resolved}  ({wins}W / {losses}L)  {acc:.0f}% accuracy")
    print(f"  P&L:       ${pnl:+.2f}")
    print(f"{'='*50}")

    # Running totals from resolved.csv
    if RESOLVED_CSV.exists():
        all_rows = []
        with RESOLVED_CSV.open(encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))
        if all_rows:
            total_wins = sum(1 for r in all_rows if r.get("outcome") == "WIN")
            total_pnl = sum(float(r.get("pnl_usd") or 0) for r in all_rows)
            total = len(all_rows)
            print(f"\n  ALL-TIME: {total} resolved | {total_wins}W/{total-total_wins}L | "
                  f"{total_wins/total*100:.0f}% acc | ${total_pnl:+.2f} cumulative P&L")

            # Breakdown by spread colour
            green = [r for r in all_rows if r.get("spread_colour") == "GREEN"]
            red   = [r for r in all_rows if r.get("spread_colour") == "RED"]
            if green:
                g_wins = sum(1 for r in green if r["outcome"] == "WIN")
                print(f"  GREEN days: {g_wins}/{len(green)} = {g_wins/len(green)*100:.0f}% "
                      f"(target 75%)")
            if red:
                r_wins = sum(1 for r in red if r["outcome"] == "WIN")
                print(f"  RED days:   {r_wins}/{len(red)} = {r_wins/len(red)*100:.0f}% "
                      f"(target 55%)")

            # Calibration check: group by city
            cities: dict[str, list] = {}
            for r in all_rows:
                cities.setdefault(r["city"], []).append(r)
            print(f"\n  PER-CITY ACCURACY:")
            for city_name, rows in sorted(cities.items()):
                cw = sum(1 for r in rows if r["outcome"] == "WIN")
                cp = sum(float(r.get("pnl_usd") or 0) for r in rows)
                print(f"    {city_name:<16} {cw}/{len(rows)} = "
                      f"{cw/len(rows)*100:.0f}%  P&L ${cp:+.2f}")


if __name__ == "__main__":
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}] Running daily resolver...")
    stats = asyncio.run(resolve_all())
    print_summary(stats)
