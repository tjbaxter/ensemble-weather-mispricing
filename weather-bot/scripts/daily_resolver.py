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
from strategy.model_weights import log_actual_temperature

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
    "strategy",       # LADDER | CONVICTION | SINGLE | TOP2_* | ACE | etc.
    "forecast_prob",
    "edge",
    "roi_pct",
    "days_ahead",
    "kelly_fraction_used",
]


def _ensure_resolved_csv() -> None:
    """Create resolved.csv with the correct header if missing or header is stale.

    Guards against old-format headers (written by a previous version of this
    script) that would cause column misalignment and make the dashboard show 0
    for all settled trades.
    """
    correct_header = ",".join(RESOLVED_HEADER)
    if RESOLVED_CSV.exists():
        with RESOLVED_CSV.open(encoding="utf-8") as f:
            existing_header = f.readline().strip()
        if existing_header == correct_header:
            return  # file is fine, nothing to do
        # Header is stale — rewrite it in-place, preserving data rows
        print(f"  [resolver] Updating stale CSV header in {RESOLVED_CSV.name}")
        lines = RESOLVED_CSV.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[0] = correct_header + "\n"
        RESOLVED_CSV.write_text("".join(lines), encoding="utf-8")
        return
    with RESOLVED_CSV.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(RESOLVED_HEADER)


_SHADOW_STRATEGIES = {
    "TOP2_EQUAL", "TOP2_COND", "TOP2_PROP",
    "CAVENDISH_MK1", "CAVENDISH_MK3",
}


def _load_resolved_keys() -> set[tuple[str, str, str, str, str]]:
    """Return set of already-resolved keys from resolved.csv.

    For real trades (PAPER / METAR / WS_PRICE): key = (date, city, bucket, side, "")
    so the strategy doesn't matter — one real position per bucket.

    For shadow models (CONVICTION / TOP2_*): key includes strategy so each
    model gets its own row in resolved.csv and its own independent P&L tracking.
    Without strategy in the key, the second shadow model to resolve the same bucket
    would be skipped as a duplicate of the first.
    """
    keys: set[tuple[str, str, str, str, str]] = set()
    if not RESOLVED_CSV.exists():
        return keys
    with RESOLVED_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            strategy = row.get("strategy", "")
            strategy_key = strategy if strategy in _SHADOW_STRATEGIES else ""
            keys.add((
                row.get("target_date", ""),
                row.get("city", ""),
                row.get("bucket", ""),
                row.get("side", "BUY_YES"),
                strategy_key,
            ))
    return keys


def _load_pending_trades() -> list[dict]:
    """Collect all paper trades that need resolving.

    Two sources:
    1. positions.json  — canonical source for ALL executed paper trades from all
       three strategies (Strategy 1 paper_trader, Strategy 2 METAR scanner,
       Strategy 3 WS price monitor).  Field names differ from signals.csv so we
       normalise them here.
    2. signals.csv     — used only for CONVICTION shadow picks (action_taken=
       'conviction_signal'), which are never in positions.json.

    De-duplication key: (target_date, city, bucket, side) — ensures that if
    the same trade appears in both files we don't double-count it.
    """
    today = date.today().isoformat()
    pending: dict[tuple[str, str, str, str], dict] = {}

    # ── 1. positions.json — actual executed paper trades ──────────────────────
    if POSITIONS_JSON.exists():
        try:
            raw_positions = json.loads(POSITIONS_JSON.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [resolver] Could not load positions.json: {exc}")
            raw_positions = []

        for p in raw_positions:
            target_date = p.get("date", "")
            if not target_date or target_date >= today:
                continue

            strategy = p.get("strategy", "") or ""
            # Normalise strategy tag → one of LADDER | SINGLE | METAR | WS_PRICE | PAPER
            if strategy in ("LADDER", "CONVICTION", "SINGLE"):
                pass
            elif strategy == "WS_PRICE_MONITOR":
                strategy = "WS_PRICE"
            elif strategy in ("METAR_SCANNER", "METAR"):
                strategy = "METAR"
            elif strategy == "":
                # Positions written before strategy tagging was added belong to
                # the live strategy that was running at the time. SINGLE has been
                # the only live strategy since ENABLE_LADDER_STRATEGY=False was set.
                strategy = "SINGLE"
            else:
                strategy = "PAPER"

            city   = p.get("city", "")
            bucket = p.get("bucket", "")
            side   = p.get("side", "BUY_YES")
            key    = (target_date, city, bucket, side)

            if key not in pending:
                pending[key] = {
                    "date":               target_date,
                    "city":               city,
                    "station_icao":       p.get("station_icao", ""),
                    "bucket":             bucket,
                    "side":               side,
                    "market_prob":        p.get("fill_price", p.get("entry_price", 0.0)),
                    "size_usd":           p.get("cost",       p.get("size_usd",    0.0)),
                    "ev_per_bet":         p.get("ev_at_entry", p.get("ev_per_bet", 0.0)),
                    "spread_colour":      p.get("spread_colour", ""),
                    "det_spread":         p.get("det_spread", ""),
                    "model_values_json":  p.get("model_values_json", "{}"),
                    "timestamp":          p.get("timestamp", ""),
                    "strategy":           strategy,
                    "forecast_prob":      p.get("forecast_prob", 0.0),
                    "edge":              p.get("edge", 0.0),
                    "days_ahead":        p.get("days_ahead", ""),
                    "kelly_fraction_used": p.get("kelly_fraction_used", ""),
                }
    else:
        print("positions.json not found — will fall back to signals.csv only.")

        # ── 2. signals.csv — CONVICTION shadow picks only ─────────────────────────
    # CONVICTION signals are logged to the main signals.csv (not a sub-directory)
    # with action_taken="conviction_signal".  TOP2_* shadow models use their own
    # positions files (see step 3 below) so we only filter CONVICTION here.
    if SIGNALS_CSV.exists():
        with SIGNALS_CSV.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("action_taken", "") != "conviction_signal":
                    continue
                target_date = row.get("date", "")
                if not target_date or target_date >= today:
                    continue
                side     = row.get("side", "BUY_YES")
                strategy = row.get("strategy", "CONVICTION")
                if strategy not in _SHADOW_STRATEGIES:
                    continue
                # Strategy is part of the key so each shadow model resolves separately
                key = (target_date, row.get("city", ""), row.get("bucket", ""), side, strategy)
                if key not in pending:
                    pending[key] = {
                        **row,
                        "strategy":    strategy,
                        "market_prob": row.get("market_prob", "0"),
                        "size_usd":    row.get("size_usd", "0"),
                        "ev_per_bet":  row.get("ev_per_bet", "0"),
                    }

    # ── 3. Shadow positions files (TOP2_EQUAL / TOP2_COND / TOP2_PROP) ────────
    # ShadowTrader writes executed positions to data/positions_shadow_*.json.
    # These are independent portfolios; each gets its own strategy tag in resolved.csv
    # so the dashboard can track them separately.
    _SHADOW_FILES: list[tuple[Path, str]] = [
        (ROOT / "data" / "positions_shadow_2a.json",        "TOP2_EQUAL"),
        (ROOT / "data" / "positions_shadow_2b.json",        "TOP2_COND"),
        (ROOT / "data" / "positions_shadow_2c.json",        "TOP2_PROP"),
        (ROOT / "data" / "positions_shadow_purdey.json",    "PURDEY_MK1"),
        (ROOT / "data" / "positions_shadow_cavendish.json", "CAVENDISH_MK1"),
        (ROOT / "data" / "positions_shadow_purdey2.json",   "PURDEY_MK2"),
        (ROOT / "data" / "positions_shadow_cavendish2.json","CAVENDISH_MK2"),
        (ROOT / "data" / "positions_shadow_ace.json",       "ACE"),
        (ROOT / "data" / "positions_shadow_props_kelly.json", "PROPS_KELLY"),
    ]
    for shadow_path, shadow_strategy in _SHADOW_FILES:
        if not shadow_path.exists():
            continue
        try:
            raw_shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [resolver] Could not load {shadow_path.name}: {exc}")
            continue
        for p in raw_shadow:
            target_date = p.get("date", "")
            if not target_date or target_date >= today:
                continue
            city   = p.get("city", "")
            bucket = p.get("bucket", "")
            side   = p.get("side", "BUY_YES")
            # 5-tuple key so each shadow strategy tracks independently
            key = (target_date, city, bucket, side, shadow_strategy)
            if key not in pending:
                pending[key] = {
                    "date":              target_date,
                    "city":              city,
                    "station_icao":      p.get("station_icao", ""),
                    "bucket":            bucket,
                    "side":              side,
                    "market_prob":       p.get("fill_price", p.get("entry_price", 0.0)),
                    "size_usd":          p.get("cost",       p.get("size_usd",    0.0)),
                    "ev_per_bet":        p.get("ev_at_entry", p.get("ev_per_bet", 0.0)),
                    "spread_colour":     p.get("spread_colour", ""),
                    "det_spread":        p.get("det_spread", ""),
                    "model_values_json": p.get("model_values_json", "{}"),
                    "timestamp":         p.get("timestamp", ""),
                    "strategy":          shadow_strategy,
                    "forecast_prob":     p.get("forecast_prob", 0.0),
                    "edge":             p.get("edge", 0.0),
                    "days_ahead":       p.get("days_ahead", ""),
                    "kelly_fraction_used": p.get("kelly_fraction_used", ""),
                }

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

            strategy = row.get("strategy", "")
            strategy_key = strategy if strategy in _SHADOW_STRATEGIES else ""
            key = (target_date_str, city, bucket, side, strategy_key)
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

            if station_icao:
                try:
                    log_actual_temperature(station_icao, target_date_str, float(actual))
                except Exception as exc:
                    print(f"  [{city}] Failed to log actual temp for model weights: {exc}")

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

            forecast_prob = 0.0
            edge = 0.0
            try:
                forecast_prob = float(row.get("forecast_prob") or 0.0)
                edge = float(row.get("edge") or 0.0)
            except (ValueError, TypeError):
                pass
            roi_pct = round(pnl / size_usd * 100, 2) if size_usd > 0 else 0.0

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
                "forecast_prob": forecast_prob,
                "edge": edge,
                "roi_pct": roi_pct,
                "days_ahead": row.get("days_ahead", ""),
                "kelly_fraction_used": row.get("kelly_fraction_used", ""),
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

        # ── Overall stats (executed real trades only — not shadow models) ────
        real_rows = [r for r in all_rows if r.get("strategy", "") not in _SHADOW_STRATEGIES]
        total_wins = sum(1 for r in real_rows if r.get("outcome") == "WIN")
        total_pnl = sum(float(r.get("pnl_usd") or 0) for r in real_rows)
        total = len(real_rows)
        if total:
            print(f"\n  ALL-TIME (executed): {total} resolved | {total_wins}W/{total-total_wins}L | "
                  f"{total_wins/total*100:.0f}% acc | ${total_pnl:+.2f} cumulative P&L")

        # ── Shadow model comparison ────────────────────────────────────────────
        shadow_labels = {
            "TOP2_EQUAL":    "TOP2_EQUAL    (2A equal sizing)",
            "TOP2_COND":     "TOP2_COND     (2B conditional)",
            "TOP2_PROP":     "TOP2_PROP     (2C proportional)",
            "CAVENDISH_MK1": "CAVENDISH_MK1 (peak + earned flanks, 50/25/25)",
            "CAVENDISH_MK3": "CAVENDISH_MK3 (weighted peak + earned flanks)",
        }

        def _strat_summary(rows: list[dict], label: str) -> None:
            if not rows:
                print(f"  │  {label:<32} — no data yet")
                return
            w   = sum(1 for r in rows if r.get("outcome") == "WIN")
            n   = len(rows)
            p   = sum(float(r.get("pnl_usd") or 0) for r in rows)
            s   = sum(float(r.get("size_usd") or 0) for r in rows)
            roi = p / s * 100 if s else 0
            print(f"  │  {label:<32} {w}/{n} ({w/n*100:.0f}%)  P&L ${p:+.2f}  ROI {roi:+.1f}%")

        any_shadow = any(
            [r for r in all_rows if r.get("strategy") == s]
            for s in _SHADOW_STRATEGIES
        )
        if any_shadow:
            print(f"\n  ┌──────────────── SHADOW MODEL COMPARISON ─────────────────┐")
            for strat_key, strat_label in shadow_labels.items():
                rows = [r for r in all_rows if r.get("strategy") == strat_key]
                _strat_summary(rows, strat_label)
            print(f"  └───────────────────────────────────────────────────────────┘")

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


def _build_resolved_keys() -> tuple[
    set[tuple[str, str, str, str]],
    set[tuple[str, str, str, str, str]],
]:
    """Build resolved key sets from resolved.csv.

    Returns (simple_keys, strategy_keys) where:
      simple_keys   = (date, city, bucket, side)          — for main positions
      strategy_keys = (date, city, bucket, side, strategy) — for shadow positions
    """
    simple: set[tuple[str, str, str, str]] = set()
    strat: set[tuple[str, str, str, str, str]] = set()
    if not RESOLVED_CSV.exists():
        return simple, strat
    try:
        with RESOLVED_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = row.get("target_date", "")
                c = row.get("city", "")
                b = row.get("bucket", "")
                s = row.get("side", "BUY_YES")
                simple.add((d, c, b, s))
                strategy = row.get("strategy", "")
                if strategy:
                    strat.add((d, c, b, s, strategy))
    except Exception:
        pass
    return simple, strat


def _prune_file(path: Path, today_str: str, resolved_keys: set, use_strategy: str = "") -> int:
    """Prune resolved positions from a single positions file. Returns count removed."""
    if not path.exists():
        return 0
    try:
        positions = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [prune] Could not load {path.name}: {exc}")
        return 0

    if not isinstance(positions, list):
        return 0

    before = len(positions)

    def _safe_to_prune(p: dict) -> bool:
        if p.get("date", "9999") >= today_str:
            return False
        d = p.get("date", "")
        c = p.get("city", "")
        b = p.get("bucket", "")
        s = p.get("side", "BUY_YES")
        if use_strategy:
            return (d, c, b, s, use_strategy) in resolved_keys
        return (d, c, b, s) in resolved_keys

    active = [p for p in positions if not _safe_to_prune(p)]
    removed = before - len(active)

    if removed:
        path.write_text(json.dumps(active, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [prune] {path.name}: removed {removed} resolved (kept {len(active)} active).")

    return removed


def prune_expired_positions() -> int:
    """Remove resolved positions from ALL position files (main + shadows).

    Returns total number of positions removed.
    """
    simple_keys, strat_keys = _build_resolved_keys()
    today_str = date.today().isoformat()
    total_removed = 0

    total_removed += _prune_file(POSITIONS_JSON, today_str, simple_keys)

    _SHADOW_PRUNE: list[tuple[Path, str]] = [
        (ROOT / "data" / "positions_shadow_2a.json",        "TOP2_EQUAL"),
        (ROOT / "data" / "positions_shadow_2b.json",        "TOP2_COND"),
        (ROOT / "data" / "positions_shadow_2c.json",        "TOP2_PROP"),
        (ROOT / "data" / "positions_shadow_purdey.json",    "PURDEY_MK1"),
        (ROOT / "data" / "positions_shadow_cavendish.json", "CAVENDISH_MK1"),
        (ROOT / "data" / "positions_shadow_purdey2.json",   "PURDEY_MK2"),
        (ROOT / "data" / "positions_shadow_cavendish2.json","CAVENDISH_MK2"),
        (ROOT / "data" / "positions_shadow_ace.json",       "ACE"),
        (ROOT / "data" / "positions_shadow_props_kelly.json","PROPS_KELLY"),
    ]
    for shadow_path, shadow_strategy in _SHADOW_PRUNE:
        total_removed += _prune_file(shadow_path, today_str, strat_keys, use_strategy=shadow_strategy)

    if total_removed == 0:
        print("  [prune] No expired positions to remove from any file.")

    return total_removed


if __name__ == "__main__":
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}] Running daily resolver...")
    stats = asyncio.run(resolve_all())
    print_summary(stats)
    print()
    prune_expired_positions()
