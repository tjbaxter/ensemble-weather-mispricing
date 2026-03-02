#!/usr/bin/env python3
"""METAR resolution-day scanner — Strategy 2 (information speed edge).

Runs always-on alongside the forecast bot (Strategy 1).  Polls
aviationweather.gov every 60 s for all configured stations, tracks the running
daily high, and on resolution day detects when the live temperature already
confirms which Polymarket bucket will win — before the market has priced it in.

Behaviour by flag
-----------------
    Default (logging only)
        Logs every observation + any actionable signals to metar_signals.csv.
        Sends Telegram alert on BUY / STRONG_BUY / NO_TRADE_NEAR_BOUNDARY.

    PAPER_TRADING=true (in .env)
        On STRONG_BUY signals: appends a paper position to data/positions.json
        (same format as the forecast bot) if:
          - the winning bucket is NOT already held
          - market ask < 0.90
          - daily METAR notional has not exceeded METAR_MAX_DAILY_NOTIONAL_USD

    LIVE_TRADING=true (in .env)
        NOT implemented in this module — live orders go through the forecast
        bot's order_manager.  This module never places live orders.

Signal actions
--------------
    OBSERVE              Early morning / no positions today — just watch.
    BUY                  Midday, daily high suggests winning bucket, temp may climb.
    STRONG_BUY           Post-peak, daily high very likely the final high.
    NO_TRADE_NEAR_BOUNDARY  Temperature within ±guard of a bucket boundary.
    HOLD_WINNER          Already holding the confirmed winning bucket.
    HEDGE_WARNING        Holding a bucket that the live high has ruled out.
    CLI_BOUNDARY_RESOLVED  NWS CLI confirmed; overrides earlier BOUNDARY flag.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytz
import requests
from dotenv import load_dotenv

# ── Path bootstrap ────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent
_ROOT    = _SCRIPTS.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from config.cities import STATIONS
from config.settings import (
    INITIAL_BANKROLL,
    KELLY_FRACTION,
    KELLY_MAX_BET_USD,
    KELLY_MIN_BET_USD,
)
from strategy.kelly import kelly_size as _kelly_size
from data.metar import (
    AWCClient,
    DailyHighRecord,
    DailyHighTracker,
    Observation,
    check_boundary,
    find_winning_bucket,
)
from data.cli_checker import fetch_cli_max_temp_f

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("weather-bot.metar-scanner")

# ── Config ────────────────────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS       = 60
PEAK_HEAT_LOCAL_HOUR        = 15   # >= this hour → "past peak" → STRONG_BUY eligible
MIDDAY_LOCAL_HOUR           = 12   # >= this hour → BUY eligible
METAR_MAX_DAILY_NOTIONAL    = float(os.getenv("METAR_MAX_DAILY_NOTIONAL_USD", "50"))
# Kelly sizing: STRONG_BUY (post-peak) = HIGH confidence; BUY (midday) = MEDIUM.
# Win-probability priors are conservative and direction-only — the METAR tells us
# which bucket is winning, but not with certainty (another SPECI could still move
# the daily high).  Post-peak past 15:00 local the daily high is effectively set.
_METAR_WIN_PROB = {"STRONG_BUY": 0.75, "BUY": 0.62, "CLI_BOUNDARY_RESOLVED": 0.82}
_METAR_KELLY_CONFIDENCE = {"STRONG_BUY": "HIGH", "BUY": "MEDIUM", "CLI_BOUNDARY_RESOLVED": "HIGH"}
METAR_MAX_LIVE_PRICE        = 0.90              # don't paper-trade above 90¢
PAPER_TRADING               = os.getenv("PAPER_TRADING",  "false").lower() in ("1", "true", "yes")
METAR_SCANNER_ENABLED       = os.getenv("METAR_SCANNER_ENABLED", "true").lower() in ("1", "true", "yes")

POSITIONS_JSON  = _ROOT / "data"  / "positions.json"
LIVE_STATE_JSON = _ROOT / "data"  / "metar_live_state.json"
SIGNALS_CSV     = _ROOT / "logs"  / "metar_signals.csv"
LOGS_DIR        = _ROOT / "logs"

# Telegram
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# State-change tracker: only alert when action changes for a (icao, date) pair.
# Always-alert actions bypass this gate regardless.
_ALWAYS_ALERT = {"STRONG_BUY", "CLI_BOUNDARY_RESOLVED"}
_last_telegram_action: dict[str, str] = {}   # key: "ICAO:YYYY-MM-DD"
_last_hedge_alert_time: dict[str, float] = {}  # key: "ICAO:YYYY-MM-DD" → epoch seconds
HEDGE_WARNING_COOLDOWN_SECONDS = 3600  # re-alert at most once per hour

# ── Positions helpers ─────────────────────────────────────────────────────────

def _load_positions() -> list[dict]:
    if not POSITIONS_JSON.exists():
        return []
    try:
        return json.loads(POSITIONS_JSON.read_text())
    except Exception:
        return []


def _save_positions(positions: list[dict]) -> None:
    POSITIONS_JSON.write_text(json.dumps(positions, indent=2, default=str))


def _local_date_for_station(icao: str) -> str:
    tz_name = STATIONS.get(icao, {}).get("timezone", "UTC")
    tz = pytz.timezone(tz_name)
    return datetime.now(tz).strftime("%Y-%m-%d")


def get_today_positions(icao: str) -> list[dict]:
    """Return open positions for this station that resolve today (local date)."""
    today = _local_date_for_station(icao)
    return [
        p for p in _load_positions()
        if p.get("station_icao") == icao and p.get("date") == today
    ]


def already_holding(icao: str, bucket: str) -> bool:
    today = _local_date_for_station(icao)
    positions = _load_positions()
    return any(
        p.get("station_icao") == icao
        and p.get("date") == today
        and p.get("bucket") == bucket
        and p.get("side") == "BUY_YES"
        for p in positions
    )


# ── METAR daily notional tracking ─────────────────────────────────────────────
# Simple in-memory accumulator — resets each calendar day.
_metar_notional_today: dict[str, float] = {}   # "YYYY-MM-DD" → cumulative USD


def _metar_notional_used() -> float:
    today = date.today().isoformat()
    return _metar_notional_today.get(today, 0.0)


def _add_metar_notional(amount: float) -> None:
    today = date.today().isoformat()
    _metar_notional_today[today] = _metar_notional_today.get(today, 0.0) + amount


# ── Signal generation ─────────────────────────────────────────────────────────

def local_hour(icao: str) -> int:
    tz_name = STATIONS.get(icao, {}).get("timezone", "UTC")
    tz = pytz.timezone(tz_name)
    return datetime.now(tz).hour


def generate_signal(
    rec: DailyHighRecord,
    today_positions: list[dict],
    all_bucket_labels: list[str],
) -> dict:
    """Compare daily high against open positions and market buckets.

    Returns a signal dict with keys: action, winning_bucket, boundary_status,
    held_bucket_status, reason.
    """
    icao     = rec.icao
    temp_res = rec.high_resolution
    unit     = rec.unit
    conf     = rec.confidence

    winning_bucket = find_winning_bucket(temp_res, all_bucket_labels)
    boundary_status = check_boundary(temp_res, all_bucket_labels, conf) if winning_bucket else "NO_BUCKET"

    hour_local = local_hour(icao)

    # What buckets do we hold (BUY_YES) for today?
    held_buckets = {p["bucket"] for p in today_positions if p.get("side") == "BUY_YES"}

    # Determine held-bucket status
    held_bucket_status = "NONE_HELD"
    if held_buckets:
        if winning_bucket in held_buckets:
            held_bucket_status = "HOLDING_WINNER"
        elif all(find_winning_bucket(temp_res, [b]) is None for b in held_buckets):
            # Held bucket is ruled out — temp already past it
            held_bucket_status = "HEDGING_NEEDED"
        else:
            held_bucket_status = "UNCLEAR"

    # Build base signal
    sig: dict = {
        "icao": icao,
        "city": STATIONS.get(icao, {}).get("market_label", icao),
        "local_date": rec.local_date,
        "unit": unit,
        "daily_high_res": temp_res,
        "daily_high_c": rec.high_c,
        "confidence": conf,
        "source": rec.source,
        "obs_count": rec.obs_count,
        "winning_bucket": winning_bucket,
        "boundary_status": boundary_status,
        "held_buckets": list(held_buckets),
        "held_bucket_status": held_bucket_status,
        "local_hour": hour_local,
        "action": "OBSERVE",
        "reason": "",
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # ── Derive action ─────────────────────────────────────────────────────────
    if not today_positions:
        sig["action"] = "OBSERVE"
        sig["reason"] = "No positions resolving today for this station"
        return sig

    if winning_bucket is None:
        sig["action"] = "OBSERVE"
        sig["reason"] = f"Daily high {temp_res}°{unit} not in any known bucket"
        return sig

    if boundary_status == "NO_TRADE_NEAR_BOUNDARY":
        sig["action"] = "NO_TRADE_NEAR_BOUNDARY"
        sig["reason"] = (
            f"High {temp_res}°{unit} ({conf}) within ±1° of bucket boundary. "
            "Wait for CLI confirmation (US) or next METAR."
        )
        return sig

    if held_bucket_status == "HOLDING_WINNER":
        sig["action"] = "HOLD_WINNER"
        sig["reason"] = f"Already holding {winning_bucket} — confirmed winning"
        return sig

    if held_bucket_status == "HEDGING_NEEDED":
        sig["action"] = "HEDGE_WARNING"
        sig["reason"] = (
            f"Daily high {temp_res}°{unit} suggests {winning_bucket} wins. "
            f"Held buckets {held_buckets} appear LOSING."
        )
        return sig

    # Look for new buy opportunity
    if hour_local >= PEAK_HEAT_LOCAL_HOUR and conf in ("HIGH", "MEDIUM"):
        sig["action"] = "STRONG_BUY"
        sig["reason"] = (
            f"Past peak heat ({hour_local}:xx local). "
            f"Daily high {temp_res}°{unit} [{conf}] in bucket {winning_bucket}."
        )
    elif hour_local >= MIDDAY_LOCAL_HOUR:
        sig["action"] = "BUY"
        sig["reason"] = (
            f"Midday ({hour_local}:xx local). "
            f"Daily high {temp_res}°{unit} [{conf}] in bucket {winning_bucket}. "
            "Temp may still climb."
        )
    else:
        sig["action"] = "OBSERVE"
        sig["reason"] = (
            f"Morning ({hour_local}:xx local). "
            f"High {temp_res}°{unit} so far — too early to trade."
        )

    return sig


# ── Paper trading ─────────────────────────────────────────────────────────────

async def _fetch_market_for_bucket(
    icao: str, local_date: str, winning_bucket: str
) -> tuple[str | None, str | None, float | None]:
    """Return (market_id, token_id, ask_price) for the winning bucket, or Nones."""
    from data.polymarket import PolymarketDataClient
    client = PolymarketDataClient()
    try:
        markets = await client.discover_weather_markets()
        hydrated = await client.hydrate_prices(markets)
        city = STATIONS.get(icao, {}).get("market_label", "")
        for m in hydrated:
            if m.get("station_icao") != icao:
                continue
            if m.get("date") != local_date:
                continue
            buckets: dict = m.get("buckets", {})
            if winning_bucket not in buckets:
                continue
            bkt = buckets[winning_bucket]
            ask = float(bkt.get("best_ask") or bkt.get("price") or 0.0)
            if ask <= 0:
                continue
            # token_id lives on the market level when there's one bucket per market,
            # or per-bucket if the market has multiple buckets.
            token_id = bkt.get("token_id") or m.get("token_id")
            market_id = m.get("market_id") or m.get("condition_id")
            return market_id, token_id, ask
    except Exception as exc:
        log.warning("Could not fetch market for %s %s %s: %s", icao, local_date, winning_bucket, exc)
    finally:
        await client.close()
    return None, None, None


async def paper_trade(sig: dict) -> bool:
    """Append a paper position to positions.json.  Returns True if placed."""
    if not PAPER_TRADING:
        return False

    icao           = sig["icao"]
    winning_bucket = sig["winning_bucket"]
    local_date     = sig["local_date"]

    if already_holding(icao, winning_bucket):
        log.debug("Already hold %s %s — skip paper trade", icao, winning_bucket)
        return False

    market_id, token_id, ask = await _fetch_market_for_bucket(icao, local_date, winning_bucket)
    if ask is None:
        log.warning("Could not find market price for %s %s — skip", icao, winning_bucket)
        return False

    if ask > METAR_MAX_LIVE_PRICE:
        log.info("Ask %.2f > max %.2f for %s %s — market already pricing it in", ask, METAR_MAX_LIVE_PRICE, icao, winning_bucket)
        return False

    # Kelly sizing: scale by signal confidence (STRONG_BUY → HIGH, BUY → MEDIUM)
    action     = sig["action"]
    win_prob   = _METAR_WIN_PROB.get(action, 0.62)
    confidence = _METAR_KELLY_CONFIDENCE.get(action, "MEDIUM")
    bet_size   = _kelly_size(
        market_price=ask,
        win_prob=win_prob,
        bankroll=INITIAL_BANKROLL,
        edge=win_prob - ask,
        kelly_fraction=KELLY_FRACTION,
        max_position=KELLY_MAX_BET_USD,
        rounding_confidence=confidence,
    )
    bet_size = max(bet_size, KELLY_MIN_BET_USD) if bet_size > 0 else KELLY_MIN_BET_USD

    used = _metar_notional_used()
    if used + bet_size > METAR_MAX_DAILY_NOTIONAL:
        log.warning(
            "METAR daily notional cap reached ($%.2f / $%.2f) — skip %s %s",
            used, METAR_MAX_DAILY_NOTIONAL, icao, winning_bucket,
        )
        return False

    fill_size = bet_size / ask if ask > 0 else 0.0

    position = {
        "market_id":        market_id or "",
        "token_id":         token_id  or "",
        "side":             "BUY_YES",
        "city":             STATIONS.get(icao, {}).get("market_label", icao),
        "station_icao":     icao,
        "date":             local_date,
        "bucket":           winning_bucket,
        "fill_price":       ask,
        "fill_size":        fill_size,
        "cost":             bet_size,
        "timestamp":        datetime.now(UTC).isoformat(),
        "strategy":         "METAR_SCANNER",
        "metar_signal":     action,
        "metar_confidence": sig["confidence"],
        "kelly_confidence": confidence,
        "kelly_win_prob":   win_prob,
    }

    positions = _load_positions()
    positions.append(position)
    _save_positions(positions)
    _add_metar_notional(bet_size)

    log.info(
        "📡 METAR PAPER TRADE: %s %s %s @ %.2f  kelly=%s  cost=$%.2f",
        STATIONS.get(icao, {}).get("market_label", icao),
        winning_bucket, "BUY_YES", ask, confidence, bet_size,
    )
    return True


# ── Telegram alerts ───────────────────────────────────────────────────────────

def _send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception as exc:
        log.debug("Telegram send failed: %s", exc)


def _telegram_signal(sig: dict, traded: bool) -> None:
    action     = sig["action"]
    icao       = sig.get("icao", "")
    local_date = sig.get("local_date", "")
    city       = sig["city"]
    high       = sig["daily_high_res"]
    unit       = sig["unit"]
    bkt        = sig["winning_bucket"] or "?"
    conf       = sig["confidence"]

    # Skip silent actions
    if action in ("OBSERVE", "HOLD_WINNER"):
        return

    # Rate-gate: only fire when action changes for this station+date,
    # unless it's an always-alert action (STRONG_BUY, CLI confirmed).
    gate_key = f"{icao}:{local_date}"
    if action == "HEDGE_WARNING":
        import time
        last = _last_hedge_alert_time.get(gate_key, 0.0)
        if time.time() - last < HEDGE_WARNING_COOLDOWN_SECONDS:
            return  # Already alerted recently — suppress repeat
        _last_hedge_alert_time[gate_key] = time.time()
    elif action not in _ALWAYS_ALERT:
        if _last_telegram_action.get(gate_key) == action:
            return  # Same action as last alert — suppress repeat
    _last_telegram_action[gate_key] = action

    if action == "STRONG_BUY":
        emoji = "🟢🟢"
        trade_note = " — <b>PAPER TRADE PLACED</b>" if traded else " — paper disabled"
    elif action == "BUY":
        emoji = "🟡"
        trade_note = ""
    elif action == "NO_TRADE_NEAR_BOUNDARY":
        emoji = "⚠️"
        trade_note = " — watching, waiting for CLI or next METAR"
    elif action == "HEDGE_WARNING":
        emoji = "🔴"
        trade_note = ""
    elif action == "CLI_BOUNDARY_RESOLVED":
        emoji = "✅"
        trade_note = " — <b>CLI confirmed</b>"
    else:
        return

    reason = sig.get("reason", "")
    msg = (
        f"{emoji} <b>METAR {action}</b>\n"
        f"  {city}  ·  Daily high: {high}°{unit} [{conf}]\n"
        f"  Winning bucket: <b>{bkt}</b>{trade_note}\n"
        f"  {reason}"
    )
    _send_telegram(msg)


# ── Signal CSV logging ────────────────────────────────────────────────────────

_CSV_HEADER = [
    "timestamp", "icao", "city", "local_date", "unit",
    "daily_high_res", "daily_high_c", "confidence", "source", "obs_count",
    "winning_bucket", "boundary_status", "held_buckets", "held_bucket_status",
    "local_hour", "action", "reason", "paper_traded",
]


def _init_csv() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not SIGNALS_CSV.exists():
        with SIGNALS_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_CSV_HEADER)


def _log_signal_csv(sig: dict, paper_traded: bool) -> None:
    with SIGNALS_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            sig.get("timestamp", ""),
            sig.get("icao", ""),
            sig.get("city", ""),
            sig.get("local_date", ""),
            sig.get("unit", ""),
            sig.get("daily_high_res", ""),
            sig.get("daily_high_c", ""),
            sig.get("confidence", ""),
            sig.get("source", ""),
            sig.get("obs_count", ""),
            sig.get("winning_bucket", ""),
            sig.get("boundary_status", ""),
            ";".join(sig.get("held_buckets", [])),
            sig.get("held_bucket_status", ""),
            sig.get("local_hour", ""),
            sig.get("action", ""),
            sig.get("reason", ""),
            "yes" if paper_traded else "no",
        ])


# ── Live state JSON (read by dashboard) ───────────────────────────────────────

_live_state: dict[str, Any] = {"last_updated": "", "stations": {}}


def _update_live_state(icao: str, rec: DailyHighRecord, sig: dict) -> None:
    _live_state["last_updated"] = datetime.now(UTC).isoformat()
    _live_state["stations"][icao] = {
        "city":            STATIONS.get(icao, {}).get("market_label", icao),
        "local_date":      rec.local_date,
        "unit":            rec.unit,
        "daily_high_res":  rec.high_resolution,
        "daily_high_c":    rec.high_c,
        "confidence":      rec.confidence,
        "source":          rec.source,
        "obs_count":       rec.obs_count,
        "last_obs_time":   rec.last_obs_time,
        "winning_bucket":  sig["winning_bucket"],
        "action":          sig["action"],
        "boundary_status": sig["boundary_status"],
        "held_buckets":    sig["held_buckets"],
        "held_status":     sig["held_bucket_status"],
        "local_hour":      sig["local_hour"],
        "reason":          sig["reason"],
        "signal_time":     sig["timestamp"],
    }
    try:
        LIVE_STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
        LIVE_STATE_JSON.write_text(json.dumps(_live_state, indent=2))
    except Exception as exc:
        log.debug("Could not write live state: %s", exc)


# ── CLI boundary resolver ─────────────────────────────────────────────────────

# Track which boundary cases have been escalated to CLI this day.
_cli_checked_today: set[str] = set()   # "ICAO:YYYY-MM-DD"


async def maybe_check_cli(sig: dict, tracker: DailyHighTracker) -> dict | None:
    """If signal is NO_TRADE_NEAR_BOUNDARY for a US station, try CLI confirmation.

    Returns a new signal with action=CLI_BOUNDARY_RESOLVED (or None).
    """
    if sig["action"] != "NO_TRADE_NEAR_BOUNDARY":
        return None

    icao = sig["icao"]
    station = STATIONS.get(icao, {})
    if station.get("resolution_unit") != "F":
        return None   # CLI only for °F markets

    dedup_key = f"{icao}:{sig['local_date']}"
    if dedup_key in _cli_checked_today:
        return None   # Already checked this boundary today

    _cli_checked_today.add(dedup_key)
    log.info("Fetching NWS CLI for %s boundary case...", icao)
    cli_max = await fetch_cli_max_temp_f(icao)

    if cli_max is None:
        log.info("CLI not yet available for %s", icao)
        return None

    # Check if CLI max resolves the boundary
    all_positions = get_today_positions(icao)
    all_buckets = list({p["bucket"] for p in all_positions})
    winning = find_winning_bucket(cli_max, all_buckets)

    if winning is None:
        log.info("CLI max %d°F for %s not in any held bucket", cli_max, icao)
        return None

    resolved_sig = dict(sig)
    resolved_sig.update({
        "action":           "CLI_BOUNDARY_RESOLVED",
        "winning_bucket":   winning,
        "daily_high_res":   cli_max,
        "confidence":       "HIGH",
        "source":           "NWS_CLI",
        "boundary_status":  "CLEAR",
        "reason":           f"NWS CLI max = {cli_max}°F → bucket {winning} confirmed",
        "timestamp":        datetime.now(UTC).isoformat(),
    })
    log.info("CLI resolved %s boundary: %d°F → %s", icao, cli_max, winning)
    return resolved_sig


# ── Bucket discovery (from today's positions) ──────────────────────────────────

def all_bucket_labels_for_station(icao: str) -> list[str]:
    """Return all unique bucket labels from today's positions for this station.

    Since we don't fetch all Polymarket buckets in this module, we use the
    held positions as a proxy — any buckets not held won't generate BUY
    signals anyway.  The boundary guard still works correctly because we know
    the bounds of held buckets.
    """
    today = _local_date_for_station(icao)
    positions = _load_positions()
    return list({
        p["bucket"]
        for p in positions
        if p.get("station_icao") == icao and p.get("date") == today
    })


# ── Main scanner loop ─────────────────────────────────────────────────────────

async def run_metar_scanner() -> None:
    """Always-on async loop.  Runs alongside PaperTrader.run_forever()."""
    if not METAR_SCANNER_ENABLED:
        log.info("METAR scanner disabled (METAR_SCANNER_ENABLED=false)")
        return

    log.info("METAR scanner starting  poll=%ds  paper=%s  daily_cap=$%.0f",
             POLL_INTERVAL_SECONDS, PAPER_TRADING, METAR_MAX_DAILY_NOTIONAL)

    _init_csv()
    tracker = DailyHighTracker()
    tracker.cleanup_old(keep_days=7)

    awc = AWCClient()
    all_icaos = list(STATIONS.keys())

    try:
        while True:
            tick_start = datetime.now(UTC)
            log.debug("METAR poll tick — %d stations", len(all_icaos))

            # ── 1. Fetch METARs ───────────────────────────────────────────────
            observations: list[Observation] = await awc.fetch_observations(all_icaos)
            log.debug("AWC returned %d observations", len(observations))

            # ── 2. Update daily highs ─────────────────────────────────────────
            updated: dict[str, DailyHighRecord] = {}
            for obs in observations:
                rec = tracker.update(obs)
                if rec:
                    updated[obs.icao] = rec
                    unit = obs.parsed.unit if obs.parsed else "?"
                    temp_res = obs.parsed.temp_resolution if obs.parsed else "?"
                    log.debug(
                        "%s: obs %s°%s [%s] | daily_high=%s°%s",
                        obs.icao, temp_res, unit,
                        obs.parsed.source if obs.parsed else "N/A",
                        rec.high_resolution, rec.unit,
                    )

            # ── 3. Generate signals for stations with positions today ──────────
            for icao, rec in updated.items():
                today_positions = get_today_positions(icao)
                if not today_positions:
                    continue   # No positions resolving today — skip signal logic

                bucket_labels = all_bucket_labels_for_station(icao)
                if not bucket_labels:
                    continue

                sig = generate_signal(rec, today_positions, bucket_labels)

                # Log every actionable or notable signal
                if sig["action"] != "OBSERVE":
                    log.info(
                        "[%s] %s | high=%s°%s [%s] | bucket=%s | %s",
                        sig["city"], sig["action"],
                        rec.high_resolution, rec.unit, rec.confidence,
                        sig["winning_bucket"] or "?",
                        sig["reason"][:80],
                    )

                # ── CLI boundary check (US only) ──────────────────────────────
                paper_traded = False
                cli_sig = await maybe_check_cli(sig, tracker)
                if cli_sig:
                    _log_signal_csv(cli_sig, False)
                    _telegram_signal(cli_sig, False)
                    # Promote to STRONG_BUY if CLI is past-peak confirmed
                    if local_hour(icao) >= PEAK_HEAT_LOCAL_HOUR:
                        cli_sig["action"] = "STRONG_BUY"
                    sig = cli_sig   # use CLI signal from here on

                # ── Paper trade ───────────────────────────────────────────────
                if sig["action"] in ("STRONG_BUY", "BUY") and PAPER_TRADING:
                    if not already_holding(icao, sig["winning_bucket"] or ""):
                        paper_traded = await paper_trade(sig)

                # ── Log + Telegram ────────────────────────────────────────────
                _log_signal_csv(sig, paper_traded)
                _telegram_signal(sig, paper_traded)
                _update_live_state(icao, rec, sig)

            # ── 4. Sleep until next poll ──────────────────────────────────────
            elapsed = (datetime.now(UTC) - tick_start).total_seconds()
            sleep_for = max(0.0, POLL_INTERVAL_SECONDS - elapsed)
            await asyncio.sleep(sleep_for)

    except asyncio.CancelledError:
        log.info("METAR scanner cancelled — shutting down cleanly")
    finally:
        await awc.close()


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        asyncio.run(run_metar_scanner())
    except KeyboardInterrupt:
        print("\nMETAR scanner stopped.")
