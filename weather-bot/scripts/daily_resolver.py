#!/usr/bin/env python3
"""Daily resolver — fetches actual temps from IEM ASOS, marks trades WIN/LOSS/MISS.

IEM (Iowa Environmental Mesonet) reads from the same METAR station observations
that Wunderground displays, so the daily max matches what Polymarket resolves to.
Using IEM rather than WU avoids the JS-rendering problem (WU pages return no data
to headless scrapers).

Run daily at 10:00 UTC — IEM typically finalises overnight data by then.

Cron (VM):
    0 10 * * * /home/tombaxter/weather-bot/venv/bin/python3 \
               /home/tombaxter/weather-bot/scripts/daily_resolver.py \
               >> /home/tombaxter/weather-bot/logs/resolver.log 2>&1
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.cities import STATIONS

SIGNALS_CSV    = ROOT / "logs" / "signals.csv"
TRADES_CSV     = ROOT / "logs" / "trades.csv"
RESOLVED_CSV   = ROOT / "logs" / "resolved.csv"
POSITIONS_JSON = ROOT / "data" / "positions.json"

IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"


def _round_half_up(x: float) -> int:
    """Round to nearest integer, halves away from zero (standard weather convention)."""
    return math.floor(x + 0.5)

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
    "strategy",       # LADDER | CONVICTION | SINGLE
]


def _ensure_resolved_csv() -> None:
    if not RESOLVED_CSV.exists():
        with RESOLVED_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(RESOLVED_HEADER)


def _load_resolved_keys() -> set[tuple[str, str, str, str]]:
    """Return set of (target_date, city, bucket, strategy) already in resolved.csv."""
    keys: set[tuple[str, str, str, str]] = set()
    if not RESOLVED_CSV.exists():
        return keys
    with RESOLVED_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys.add((row["target_date"], row["city"], row["bucket"],
                       row.get("strategy", "LADDER")))
    return keys


def _load_pending_trades() -> list[dict]:
    """Read signals.csv for rows to resolve:
    - action_taken='trade'             → LADDER / SINGLE actual bets
    - action_taken='conviction_signal' → CONVICTION shadow picks (never executed)
    Both are scored against actuals for A/B comparison.
    """
    if not SIGNALS_CSV.exists():
        print("signals.csv not found — nothing to resolve.")
        return []

    today = date.today().isoformat()
    # Key = (date, city, bucket, strategy) → keep earliest signal
    pending: dict[tuple[str, str, str, str], dict] = {}

    with SIGNALS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            action = row.get("action_taken", "")
            if action not in ("trade", "conviction_signal"):
                continue
            target_date = row.get("date", "")
            if not target_date or target_date >= today:
                continue
            strategy = row.get("strategy", "LADDER" if action == "trade" else "CONVICTION")
            key = (target_date, row.get("city", ""), row.get("bucket", ""), strategy)
            if key not in pending:
                pending[key] = {**row, "strategy": strategy}

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


async def _fetch_iem_high(
    http: httpx.AsyncClient,
    iem_network: str,
    iem_station: str,
    resolution_unit: str,
    day: date,
) -> int | None:
    """Fetch daily max temperature from IEM ASOS (same METAR data that WU shows).

    IEM returns max_tmpf (°F) for all stations regardless of country.
    For °F markets: round and return directly.
    For °C markets: convert then round with round_half_up.
    """
    params = {
        "network": iem_network,
        "stations": iem_station,
        "year1": str(day.year),
        "month1": str(day.month),
        "day1": str(day.day),
        "year2": str(day.year),
        "month2": str(day.month),
        "day2": str(day.day),
        "vars[]": "max_tmpf",
        "what": "view",
        "delim": "comma",
        "gis": "no",
    }
    try:
        resp = await http.get(IEM_DAILY_URL, params=params, timeout=20.0)
        resp.raise_for_status()
        for line in resp.text.strip().splitlines():
            if line.startswith("station") or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            raw = parts[2].strip()
            if raw in ("", "None", "M", "null"):
                print(f"  IEM: no data yet for {iem_station} on {day} (got '{raw}')")
                return None
            max_tmpf = float(raw)
            if resolution_unit == "F":
                return _round_half_up(max_tmpf)
            else:
                return _round_half_up((max_tmpf - 32.0) * 5.0 / 9.0)
    except Exception as exc:
        print(f"  IEM fetch error for {iem_station} {day}: {exc}")
    return None


async def resolve_all() -> dict:
    _ensure_resolved_csv()
    already_resolved = _load_resolved_keys()
    pending = _load_pending_trades()

    if not pending:
        print("No pending trades to resolve.")
        return {"resolved": 0, "wins": 0, "losses": 0, "pnl": 0.0}

    # Group by station+date to minimise IEM HTTP requests
    station_date_cache: dict[tuple[str, str], int | None] = {}

    stats = {"resolved": 0, "wins": 0, "losses": 0, "pnl": 0.0, "rows": []}

    async with httpx.AsyncClient() as http:
        for row in pending:
            target_date_str = row["date"]
            city = row.get("city", "")
            bucket = row.get("bucket", "")
            station_icao = row.get("station_icao", "")
            side = row.get("side", "BUY_YES")

            strategy = row.get("strategy", "LADDER")
            key = (target_date_str, city, bucket, strategy)
            if key in already_resolved:
                continue

            # Look up IEM config for this station
            station_cfg = STATIONS.get(station_icao)
            if not station_cfg:
                for icao, cfg in STATIONS.items():
                    if cfg.get("market_label", "").lower() == city.lower():
                        station_cfg = cfg
                        station_icao = icao
                        break

            if not station_cfg:
                print(f"  [{city}] No station config for ICAO '{station_icao}' — skipping.")
                continue

            iem_network = station_cfg.get("iem_network")
            iem_station = station_cfg.get("iem_station")
            if not iem_network or not iem_station:
                print(f"  [{city}] No iem_network/iem_station in config — skipping.")
                continue

            resolution_unit = station_cfg.get("resolution_unit", "F")
            target_date = date.fromisoformat(target_date_str)
            cache_key = (iem_station, target_date_str)

            if cache_key not in station_date_cache:
                print(f"  [{city}] Fetching IEM ({iem_network}/{iem_station}) for {target_date_str}...")
                actual = await _fetch_iem_high(http, iem_network, iem_station, resolution_unit, target_date)
                station_date_cache[cache_key] = actual
            else:
                actual = station_date_cache[cache_key]

            if actual is None:
                print(f"  [{city}] IEM returned no data for {target_date_str} — will retry tomorrow.")
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
                "strategy": strategy,
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
        if not all_rows:
            return

        # ── Overall stats (LADDER + SINGLE only — real money) ────────────────
        real_rows = [r for r in all_rows if r.get("strategy", "") != "CONVICTION"]
        total_wins = sum(1 for r in real_rows if r.get("outcome") == "WIN")
        total_pnl = sum(float(r.get("pnl_usd") or 0) for r in real_rows)
        total = len(real_rows)
        if total:
            print(f"\n  ALL-TIME (executed): {total} resolved | {total_wins}W/{total-total_wins}L | "
                  f"{total_wins/total*100:.0f}% acc | ${total_pnl:+.2f} cumulative P&L")

        # ── LADDER vs CONVICTION head-to-head ─────────────────────────────────
        ladder_rows     = [r for r in all_rows if r.get("strategy") == "LADDER"]
        conviction_rows = [r for r in all_rows if r.get("strategy") == "CONVICTION"]
        if ladder_rows or conviction_rows:
            print(f"\n  ┌─────────────── A/B STRATEGY COMPARISON ──────────────┐")

            def _strat_summary(rows: list[dict], label: str) -> None:
                if not rows:
                    print(f"  │  {label:<12} — no data yet")
                    return
                w  = sum(1 for r in rows if r.get("outcome") == "WIN")
                n  = len(rows)
                p  = sum(float(r.get("pnl_usd") or 0) for r in rows)
                s  = sum(float(r.get("size_usd") or 0) for r in rows)
                roi = p / s * 100 if s else 0
                print(f"  │  {label:<12} {w}/{n} ({w/n*100:.0f}% acc)  P&L ${p:+.2f}  "
                      f"ROI {roi:+.1f}%  (${s:.0f} staked)")

            _strat_summary(ladder_rows,     "LADDER")
            _strat_summary(conviction_rows, "CONVICTION")

            # Head-to-head: only days where both strategies resolved
            l_dict = {(r["target_date"], r["city"]): r for r in ladder_rows}
            c_dict = {(r["target_date"], r["city"]): r for r in conviction_rows}
            shared = set(l_dict) & set(c_dict)
            if len(shared) >= 3:
                l_pnl = sum(float(l_dict[k]["pnl_usd"] or 0) for k in shared)
                c_pnl = sum(float(c_dict[k]["pnl_usd"] or 0) for k in shared)
                winner = "CONVICTION" if c_pnl > l_pnl else "LADDER"
                print(f"  │  Head-to-head ({len(shared)} city-days): LADDER ${l_pnl:+.2f} vs "
                      f"CONVICTION ${c_pnl:+.2f}  → {winner} wins")
            print(f"  └───────────────────────────────────────────────────────┘")

        # ── Breakdown by spread colour ────────────────────────────────────────
        green = [r for r in real_rows if r.get("spread_colour") == "GREEN"]
        red   = [r for r in real_rows if r.get("spread_colour") == "RED"]
        if green:
            g_wins = sum(1 for r in green if r["outcome"] == "WIN")
            print(f"  GREEN days: {g_wins}/{len(green)} = {g_wins/len(green)*100:.0f}% "
                  f"(target 75%)")
        if red:
            r_wins = sum(1 for r in red if r["outcome"] == "WIN")
            print(f"  RED days:   {r_wins}/{len(red)} = {r_wins/len(red)*100:.0f}% "
                  f"(target 55%)")

        # ── Calibration check: group by city ─────────────────────────────────
        cities: dict[str, list] = {}
        for r in real_rows:
            cities.setdefault(r["city"], []).append(r)
        print(f"\n  PER-CITY ACCURACY (executed bets):")
        for city_name, rows in sorted(cities.items()):
            cw = sum(1 for r in rows if r["outcome"] == "WIN")
            cp = sum(float(r.get("pnl_usd") or 0) for r in rows)
            print(f"    {city_name:<16} {cw}/{len(rows)} = "
                  f"{cw/len(rows)*100:.0f}%  P&L ${cp:+.2f}")


def prune_expired_positions() -> int:
    """Remove positions from positions.json whose target date is strictly before today,
    BUT ONLY if they are already captured in resolved.csv.

    Positions not in resolved.csv are left in place so the dashboard can still
    compute their live P&L until they are formally recovered/resolved.

    Returns number of positions removed.
    """
    if not POSITIONS_JSON.exists():
        return 0
    try:
        positions = json.loads(POSITIONS_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [prune] Could not load positions.json: {exc}")
        return 0

    # Build set of (city, date, bucket, side) already in resolved.csv
    resolved_keys: set[tuple[str, str, str, str]] = set()
    if RESOLVED_CSV.exists():
        try:
            import csv as _csv
            with RESOLVED_CSV.open(encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    resolved_keys.add((
                        row.get("city", ""),
                        row.get("target_date", ""),
                        row.get("bucket", ""),
                        row.get("side", ""),
                    ))
        except Exception:
            pass

    today_str = date.today().isoformat()
    before = len(positions)

    def _safe_to_prune(p: dict) -> bool:
        if p.get("date", "9999") >= today_str:
            return False  # not expired yet
        key = (p.get("city",""), p.get("date",""), p.get("bucket",""), p.get("side",""))
        return key in resolved_keys  # only prune if resolved

    active  = [p for p in positions if not _safe_to_prune(p)]
    removed = before - len(active)

    if removed:
        POSITIONS_JSON.write_text(
            json.dumps(active, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  [prune] Removed {removed} expired position(s) from positions.json "
              f"(kept {len(active)} active).")
    else:
        print("  [prune] No expired positions to remove.")

    return removed


if __name__ == "__main__":
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}] Running daily resolver...")
    stats = asyncio.run(resolve_all())
    print_summary(stats)
    print()
    prune_expired_positions()
