"""Strategy 3 — Continuous Price Scanner.

Runs alongside the forecast bot (Strategy 1) and METAR scanner (Strategy 2).

The forecast bot wakes at 5 NWP trigger times and caches model probabilities to
data/cached_signals.json.  Prices on Polymarket move continuously between those
model runs — other traders dumping positions, market makers refreshing quotes,
liquidity thin patches.  This scanner polls orderbook ask prices every 5 minutes
and fires a paper trade whenever:

    EV  =  model_prob  -  live_ask  >=  PRICE_SCAN_MIN_EV

…and we don't already hold that token.

Rate limits: Polymarket CLOB /books allows 500 batch requests / 10 s.  At one
batch call every 5 minutes we use ~0.001 % of capacity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import requests
from dotenv import load_dotenv

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent
_ROOT    = _SCRIPTS.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from config.settings import (
    ALPHA_THRESHOLD,
    CLOB_API_URL,
    FIXED_ORDER_USD,
    HARD_MAX_YES_ENTRY_PRICE,
    HARD_MIN_YES_ENTRY_PRICE,
    PRACTICAL_MIN_ORDER_USD,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("weather-bot.price-scanner")

# ── Config ─────────────────────────────────────────────────────────────────────
PRICE_SCAN_INTERVAL      = int(os.getenv("PRICE_SCAN_INTERVAL", "300"))   # 5 min
PRICE_SCAN_MIN_EV        = float(os.getenv("PRICE_SCAN_MIN_EV", str(ALPHA_THRESHOLD)))
PRICE_SCAN_MAX_DAYS_AHEAD = int(os.getenv("PRICE_SCAN_MAX_DAYS_AHEAD", "2"))  # skip D+3
PRICE_SCAN_ENABLED       = os.getenv("PRICE_SCAN_ENABLED", "true").lower() in ("1", "true", "yes")
PAPER_TRADING            = os.getenv("PAPER_TRADING", "false").lower() in ("1", "true", "yes")

CACHED_SIGNALS_PATH = _ROOT / "data" / "cached_signals.json"
POSITIONS_PATH      = _ROOT / "data" / "positions.json"

# Telegram
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_cached_signals() -> list[dict]:
    try:
        payload = json.loads(CACHED_SIGNALS_PATH.read_text(encoding="utf-8"))
        return payload.get("signals", [])
    except Exception:
        return []


def _load_positions() -> list[dict]:
    try:
        return json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_positions(positions: list[dict]) -> None:
    POSITIONS_PATH.write_text(json.dumps(positions, indent=2, default=str), encoding="utf-8")


def _already_holding(token_id: str) -> bool:
    return any(p.get("token_id") == token_id for p in _load_positions())


def _is_expired(end_date_iso: str) -> bool:
    """Return True if the market has already resolved."""
    if not end_date_iso:
        return False
    try:
        end_dt = datetime.fromisoformat(str(end_date_iso).replace("Z", "+00:00"))
        return datetime.now(UTC) >= end_dt
    except Exception:
        return False


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


# ── Orderbook price fetching ───────────────────────────────────────────────────

async def _fetch_ask_prices(token_ids: list[str], http: httpx.AsyncClient) -> dict[str, float]:
    """Batch-fetch best ask prices for a list of YES token IDs.

    Returns {token_id: best_ask} for tokens where a valid ask exists.
    Uses the same CLOB /books endpoint the main bot uses.
    """
    if not token_ids:
        return {}

    results: dict[str, float] = {}
    chunk_size = 100  # CLOB supports up to ~100 per batch call

    for start in range(0, len(token_ids), chunk_size):
        chunk = token_ids[start:start + chunk_size]
        try:
            resp = await http.post(
                f"{CLOB_API_URL}/books",
                json=[{"token_id": tid} for tid in chunk],
                timeout=15.0,
            )
            resp.raise_for_status()
            books = resp.json()
            if not isinstance(books, list):
                continue
            for book in books:
                tid = str(book.get("asset_id") or book.get("token_id") or "")
                if not tid:
                    continue
                # Best ask: prefer top-level field, fall back to asks list
                best_ask = float(book.get("bestAsk", 0.0) or 0.0)
                if best_ask <= 0:
                    asks = book.get("asks", [])
                    if isinstance(asks, list) and asks:
                        try:
                            best_ask = min(
                                float(a.get("price", 0.0) or 0.0)
                                for a in asks
                                if float(a.get("price", 0.0) or 0.0) > 0.0
                            )
                        except (TypeError, ValueError):
                            best_ask = 0.0
                if best_ask > 0:
                    results[tid] = round(best_ask, 4)
        except httpx.HTTPError as exc:
            log.warning("Batch /books failed (chunk %d-%d): %s", start, start + chunk_size, exc)
            # Fallback: single requests for this chunk
            for tid in chunk:
                try:
                    r = await http.get(
                        f"{CLOB_API_URL}/price",
                        params={"token_id": tid, "side": "buy"},
                        timeout=10.0,
                    )
                    r.raise_for_status()
                    price_str = r.json().get("price", "0")
                    p = float(price_str)
                    if p > 0:
                        results[tid] = round(p, 4)
                except Exception:
                    pass

    return results


# ── Paper trade placement ──────────────────────────────────────────────────────

def _place_paper_trade(sig: dict, live_ask: float, ev: float) -> None:
    """Append a paper position to positions.json."""
    if not PAPER_TRADING:
        return
    if live_ask < HARD_MIN_YES_ENTRY_PRICE or live_ask > HARD_MAX_YES_ENTRY_PRICE:
        log.info(
            "PRICE_SCANNER skip %s %s — ask %.3f outside hard price guards [%.2f, %.2f]",
            sig["city"], sig["bucket"], live_ask,
            HARD_MIN_YES_ENTRY_PRICE, HARD_MAX_YES_ENTRY_PRICE,
        )
        return

    cost      = FIXED_ORDER_USD
    fill_size = round(cost / live_ask, 4) if live_ask > 0 else 0.0

    position = {
        "market_id":    sig["condition_id"],
        "token_id":     sig["token_id"],
        "side":         "BUY_YES",
        "city":         sig["city"],
        "station_icao": sig["station_icao"],
        "date":         sig["date"],
        "bucket":       sig["bucket"],
        "fill_price":   live_ask,
        "fill_size":    fill_size,
        "cost":         cost,
        "timestamp":    datetime.now(UTC).isoformat(),
        "strategy":     "PRICE_SCANNER",
        "model_prob":   sig["model_prob"],
        "ev_at_entry":  round(ev, 4),
        "spread_colour": sig.get("spread_colour", ""),
        "model_values_json": sig.get("model_values_json", "{}"),
    }

    positions = _load_positions()
    positions.append(position)
    _save_positions(positions)

    log.info(
        "💰 PRICE_SCANNER PAPER TRADE: %s %s BUY_YES @ %.3f  model=%.0f%%  ev=+%.3f  cost=$%.2f",
        sig["city"], sig["bucket"], live_ask,
        sig["model_prob"] * 100, ev, cost,
    )

    _send_telegram(
        f"💰 <b>PRICE DIP TRADE</b>\n"
        f"  {sig['city']}  ·  {sig['bucket']}\n"
        f"  Model: <b>{sig['model_prob']:.0%}</b>  →  Ask dipped to <b>{live_ask:.3f}</b>\n"
        f"  EV: <b>+{ev:.3f}</b>  (was {sig['market_prob_at_scan']:.3f} at last scan)\n"
        f"  <i>Spread: {sig.get('spread_colour', '?')} · {sig.get('det_spread', '?')}°</i>"
    )


# ── Main scanner loop ──────────────────────────────────────────────────────────

async def run_price_scanner() -> None:
    """Always-on async loop.  Polls Polymarket ask prices every PRICE_SCAN_INTERVAL seconds."""
    if not PRICE_SCAN_ENABLED:
        log.info("Price scanner disabled (PRICE_SCAN_ENABLED=false)")
        return

    log.info(
        "Price scanner starting  interval=%ds  min_ev=%.3f  max_days_ahead=%d  paper=%s",
        PRICE_SCAN_INTERVAL, PRICE_SCAN_MIN_EV, PRICE_SCAN_MAX_DAYS_AHEAD, PAPER_TRADING,
    )

    async with httpx.AsyncClient(timeout=20.0) as http:
        while True:
            tick_start = datetime.now(UTC)

            signals = _load_cached_signals()
            if not signals:
                log.debug("No cached signals yet — waiting for first forecast scan")
                await asyncio.sleep(PRICE_SCAN_INTERVAL)
                continue

            # Filter: skip expired markets and D+3+ signals
            active = [
                s for s in signals
                if not _is_expired(s.get("end_date_iso", ""))
                and s.get("days_ahead", 1) <= PRICE_SCAN_MAX_DAYS_AHEAD
                and s.get("side") == "BUY_YES"   # only YES side for now
            ]

            if not active:
                log.debug("No active signals to scan prices for")
                await asyncio.sleep(PRICE_SCAN_INTERVAL)
                continue

            token_ids = [s["token_id"] for s in active if s.get("token_id")]
            log.debug("Fetching prices for %d tokens", len(token_ids))

            ask_prices = await _fetch_ask_prices(token_ids, http)

            trades = 0
            for sig in active:
                tid = sig.get("token_id", "")
                if not tid or tid not in ask_prices:
                    continue

                live_ask   = ask_prices[tid]
                model_prob = sig.get("model_prob", 0.0)
                ev         = model_prob - live_ask

                if ev < PRICE_SCAN_MIN_EV:
                    continue

                if _already_holding(tid):
                    log.debug(
                        "PRICE_SCANNER already hold %s %s — skip",
                        sig["city"], sig["bucket"],
                    )
                    continue

                # Verify the dip is real — live ask must be meaningfully below the scan price
                scan_ask = sig.get("market_prob_at_scan", live_ask)
                dip_magnitude = scan_ask - live_ask
                if dip_magnitude < 0.01:
                    # Price hasn't materially moved since last model scan — not a dip, just noise
                    continue

                log.info(
                    "🎯 PRICE DIP: %s %s  model=%.0f%%  was=%.3f  now=%.3f  dip=%.3f  ev=+%.3f",
                    sig["city"], sig["bucket"],
                    model_prob * 100, scan_ask, live_ask, dip_magnitude, ev,
                )

                _place_paper_trade(sig, live_ask, ev)
                trades += 1

            elapsed = (datetime.now(UTC) - tick_start).total_seconds()
            log.info(
                "PRICE_SCAN_TICK tokens=%d checked=%d trades=%d elapsed=%.1fs",
                len(token_ids), len(ask_prices), trades, elapsed,
            )

            sleep_for = max(0.0, PRICE_SCAN_INTERVAL - elapsed)
            await asyncio.sleep(sleep_for)


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        asyncio.run(run_price_scanner())
    except KeyboardInterrupt:
        print("\nPrice scanner stopped.")
