"""Strategy 3 (upgraded) — WebSocket Price Monitor.

Replaces the REST-polling price scanner with a persistent WebSocket connection
to Polymarket's market channel:

    wss://ws-subscriptions-clob.polymarket.com/ws/market

The forecast bot writes cached model probabilities to data/cached_signals.json
after each NWP model run.  This monitor subscribes to those token IDs and
receives real-time price_change events the moment any order is placed or
cancelled.  When best_ask drops enough for EV = model_prob - ask >= threshold,
it fires a paper trade — zero latency vs up to 5 minutes with REST polling.

Connection details (Polymarket docs, March 2026):
  - No auth needed, public read-only channel
  - Up to ~500 instruments per connection
  - Reconnection with exponential backoff handles network blips and maintenance
  - First 'book' event after subscribe is a full snapshot; 'price_change' is incremental
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests
import websockets
from dotenv import load_dotenv

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent
_ROOT    = _SCRIPTS.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from config.settings import (
    ALPHA_THRESHOLD,
    HARD_MAX_YES_ENTRY_PRICE,
    HARD_MIN_YES_ENTRY_PRICE,
    INITIAL_BANKROLL,
    KELLY_FRACTION,
    KELLY_MAX_BET_USD,
    KELLY_MIN_BET_USD,
)
from strategy.kelly import kelly_size

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("websockets").setLevel(logging.WARNING)
log = logging.getLogger("weather-bot.ws-price-monitor")

# ── Config ─────────────────────────────────────────────────────────────────────
WS_URL               = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
WS_ENABLED           = os.getenv("WS_PRICE_MONITOR_ENABLED", "true").lower() in ("1", "true", "yes")
PAPER_TRADING        = os.getenv("PAPER_TRADING", "false").lower() in ("1", "true", "yes")
WS_MIN_EV            = float(os.getenv("WS_MIN_EV", str(ALPHA_THRESHOLD)))
WS_MIN_DIP           = float(os.getenv("WS_MIN_DIP", "0.01"))   # price must have moved ≥ 1¢
WS_CACHE_REFRESH_SEC = int(os.getenv("WS_CACHE_REFRESH_SEC", "120"))  # re-read cache every 2 min
WS_RECONNECT_BASE    = 2    # seconds — exponential backoff base
WS_RECONNECT_MAX     = 60   # seconds — max backoff
WS_PING_INTERVAL     = 20   # seconds

CACHED_SIGNALS_PATH = _ROOT / "data" / "cached_signals.json"
POSITIONS_PATH      = _ROOT / "data" / "positions.json"

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ── File-backed signal cache ───────────────────────────────────────────────────

class SignalCache:
    """Reads cached model probabilities from data/cached_signals.json.

    The forecast bot writes this file after each NWP model run.
    The WS monitor calls refresh() periodically to pick up new model data.
    """

    def __init__(self) -> None:
        self._signals: dict[str, dict] = {}   # token_id → signal dict
        self._computed_at: str = ""
        self.refresh()

    def refresh(self) -> set[str]:
        """Re-read cache file. Returns the set of token IDs (possibly updated)."""
        try:
            payload  = json.loads(CACHED_SIGNALS_PATH.read_text(encoding="utf-8"))
            old_ids  = set(self._signals.keys())
            new_sigs = {}
            for s in payload.get("signals", []):
                tid = s.get("token_id", "")
                if tid:
                    new_sigs[tid] = s
            self._signals     = new_sigs
            self._computed_at = payload.get("computed_at", "")
            new_ids = set(self._signals.keys())
            added   = new_ids - old_ids
            if added:
                log.info("SignalCache refreshed: %d signals (%d new tokens)", len(new_sigs), len(added))
            return new_ids
        except Exception:
            return set(self._signals.keys())

    def get_model_prob(self, token_id: str) -> float | None:
        sig = self._signals.get(token_id)
        return sig["model_prob"] if sig else None

    def get_signal(self, token_id: str) -> dict:
        return self._signals.get(token_id, {})

    def all_token_ids(self) -> set[str]:
        return set(self._signals.keys())

    def market_prob_at_scan(self, token_id: str) -> float:
        return float(self._signals.get(token_id, {}).get("market_prob_at_scan", 1.0))

    def is_expired(self, token_id: str) -> bool:
        end_iso = self._signals.get(token_id, {}).get("end_date_iso", "")
        if not end_iso:
            return False
        try:
            end_dt = datetime.fromisoformat(str(end_iso).replace("Z", "+00:00"))
            return datetime.now(UTC) >= end_dt
        except Exception:
            return False


# ── Position helpers ───────────────────────────────────────────────────────────

def _load_positions() -> list[dict]:
    try:
        return json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_positions(positions: list[dict]) -> None:
    POSITIONS_PATH.write_text(json.dumps(positions, indent=2, default=str), encoding="utf-8")


def _already_holding(token_id: str) -> bool:
    return any(p.get("token_id") == token_id for p in _load_positions())


# ── Telegram ───────────────────────────────────────────────────────────────────

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


# ── Trade placement ────────────────────────────────────────────────────────────

def _kelly_size_ws(model_prob: float, ask: float) -> float:
    """Fractional Kelly for WS dip trades — MEDIUM confidence by default.

    WS trades use the same model probability from the morning forecast scan
    but at a better live price.  Confidence is MEDIUM because we trust the
    model but the dip could be noise.  HIGH is reserved for multi-model
    consensus signals (those come with their own sizing from Strategy 1).
    """
    size = kelly_size(
        market_price=ask,
        win_prob=model_prob,
        bankroll=INITIAL_BANKROLL,
        edge=model_prob - ask,
        kelly_fraction=KELLY_FRACTION,
        max_position=KELLY_MAX_BET_USD,
        rounding_confidence="MEDIUM",
    )
    return max(size, KELLY_MIN_BET_USD) if size > 0 else KELLY_MIN_BET_USD


def _place_paper_trade(sig: dict, live_ask: float, ev: float) -> None:
    if not PAPER_TRADING:
        return
    if live_ask < HARD_MIN_YES_ENTRY_PRICE or live_ask > HARD_MAX_YES_ENTRY_PRICE:
        log.info(
            "WS_MONITOR skip %s %s — ask %.3f outside hard price guards",
            sig.get("city", "?"), sig.get("bucket", "?"), live_ask,
        )
        return

    model_prob = float(sig.get("model_prob", 0.5))
    cost      = _kelly_size_ws(model_prob, live_ask)
    fill_size = round(cost / live_ask, 4) if live_ask > 0 else 0.0

    position = {
        "market_id":    sig.get("condition_id", ""),
        "token_id":     sig.get("token_id", ""),
        "side":         "BUY_YES",
        "city":         sig.get("city", ""),
        "station_icao": sig.get("station_icao", ""),
        "date":         sig.get("date", ""),
        "bucket":       sig.get("bucket", ""),
        "fill_price":   live_ask,
        "fill_size":    fill_size,
        "cost":         cost,
        "timestamp":    datetime.now(UTC).isoformat(),
        "strategy":     "WS_PRICE_MONITOR",
        "model_prob":   sig.get("model_prob", 0.0),
        "ev_at_entry":  round(ev, 4),
        "spread_colour": sig.get("spread_colour", ""),
        "model_values_json": sig.get("model_values_json", "{}"),
    }

    positions = _load_positions()
    positions.append(position)
    _save_positions(positions)

    log.info(
        "⚡ WS PAPER TRADE: %s %s BUY_YES @ %.3f  model=%.0f%%  ev=+%.3f  cost=$%.2f",
        sig.get("city"), sig.get("bucket"), live_ask,
        sig.get("model_prob", 0) * 100, ev, cost,
    )

    _send_telegram(
        f"⚡ <b>WS PRICE DIP TRADE</b>\n"
        f"  {sig.get('city')}  ·  {sig.get('bucket')}\n"
        f"  Model: <b>{sig.get('model_prob', 0):.0%}</b>  "
        f"→  Ask dipped to <b>{live_ask:.3f}</b>\n"
        f"  EV: <b>+{ev:.3f}</b>  "
        f"(was {sig.get('market_prob_at_scan', '?'):.3f} at last scan)\n"
        f"  <i>Spread: {sig.get('spread_colour', '?')} · {sig.get('det_spread', '?')}°</i>"
    )


# ── WebSocket monitor ──────────────────────────────────────────────────────────

class WebSocketPriceMonitor:
    """Persistent Polymarket market-channel WebSocket with auto-reconnect.

    Receives real-time book and price_change events for all tracked weather
    bucket tokens.  On every price_change, evaluates EV against cached model
    probabilities and fires a paper trade if the opportunity threshold is met.
    """

    def __init__(self, cache: SignalCache) -> None:
        self.cache            = cache
        self.subscribed_ids:  set[str] = set()
        self.best_asks:       dict[str, float] = {}   # token_id → current best ask
        self._ws              = None
        self._reconnect_count = 0
        self._trade_count_today = 0

    async def run(self) -> None:
        """Main loop: connect → listen → reconnect with backoff."""
        while True:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                log.info("WS monitor cancelled — shutting down.")
                return
            except Exception as exc:
                self._reconnect_count += 1
                delay = min(WS_RECONNECT_BASE ** self._reconnect_count, WS_RECONNECT_MAX)
                log.warning(
                    "WS disconnected: %s — reconnecting in %.0fs (attempt %d)",
                    exc, delay, self._reconnect_count,
                )
                await asyncio.sleep(delay)

    async def _connect_and_listen(self) -> None:
        log.info("Connecting to Polymarket WebSocket: %s", WS_URL)

        async with websockets.connect(
            WS_URL,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_INTERVAL * 2,
            close_timeout=5,
            additional_headers={"User-Agent": "PolymarketWeatherBot/2.0"},
        ) as ws:
            self._ws = ws
            self._reconnect_count = 0
            log.info("WS connected — subscribing to %d tokens", len(self.cache.all_token_ids()))

            await self._subscribe(self.cache.all_token_ids())

            # Run message loop + periodic cache refresh concurrently
            await asyncio.gather(
                self._message_loop(ws),
                self._cache_refresh_loop(ws),
            )

    async def _subscribe(self, token_ids: set[str]) -> None:
        if not token_ids or not self._ws:
            return
        msg = {"assets_ids": sorted(token_ids), "type": "market"}
        await self._ws.send(json.dumps(msg))
        self.subscribed_ids |= token_ids
        log.info("Subscribed to %d token IDs", len(token_ids))

    async def _cache_refresh_loop(self, ws) -> None:
        """Periodically refresh the signal cache and subscribe to any new tokens."""
        while True:
            await asyncio.sleep(WS_CACHE_REFRESH_SEC)
            new_ids = self.cache.refresh()
            to_add  = new_ids - self.subscribed_ids
            if to_add:
                log.info("New tokens from model run — subscribing to %d more", len(to_add))
                await self._subscribe(to_add)

    async def _message_loop(self, ws) -> None:
        async for raw in ws:
            try:
                data   = json.loads(raw)
                events = data if isinstance(data, list) else [data]
                for event in events:
                    await self._dispatch(event)
            except json.JSONDecodeError:
                log.debug("Non-JSON WS message: %s", str(raw)[:80])

    async def _dispatch(self, event: dict) -> None:
        etype = event.get("event_type")
        if etype == "book":
            self._handle_book(event)
        elif etype == "price_change":
            self._handle_price_change(event)
        # last_trade_price and market_resolved are logged at debug level only
        elif etype:
            log.debug("WS event: %s", etype)

    def _handle_book(self, event: dict) -> None:
        """Full orderbook snapshot — extract best ask."""
        tid  = event.get("asset_id", "")
        asks = event.get("asks", [])
        if not tid or not asks:
            return
        try:
            best_ask = min(float(a["price"]) for a in asks if float(a["price"]) > 0)
        except (KeyError, TypeError, ValueError):
            return
        self.best_asks[tid] = best_ask
        self._evaluate(tid, best_ask)

    def _handle_price_change(self, event: dict) -> None:
        """Incremental update — the hot path. Fires on every order placed/cancelled."""
        for change in event.get("price_changes", []):
            tid      = change.get("asset_id", "")
            ask_str  = change.get("best_ask")
            if not tid or ask_str is None:
                continue
            try:
                best_ask = float(ask_str)
            except (TypeError, ValueError):
                continue
            if best_ask <= 0:
                continue
            self.best_asks[tid] = best_ask
            self._evaluate(tid, best_ask)

    def _evaluate(self, token_id: str, live_ask: float) -> None:
        """Core EV check: if price dropped enough → paper trade."""
        model_prob = self.cache.get_model_prob(token_id)
        if model_prob is None:
            return   # token not in our signal universe

        if self.cache.is_expired(token_id):
            return

        ev = model_prob - live_ask
        if ev < WS_MIN_EV:
            return

        # Price must have moved meaningfully below the scan price — not just noise
        scan_ask = self.cache.market_prob_at_scan(token_id)
        if scan_ask - live_ask < WS_MIN_DIP:
            return

        if _already_holding(token_id):
            return

        sig = self.cache.get_signal(token_id)
        log.info(
            "🎯 WS DIP: %s %s  model=%.0f%%  was=%.3f  now=%.3f  ev=+%.3f",
            sig.get("city", "?"), sig.get("bucket", "?"),
            model_prob * 100, scan_ask, live_ask, ev,
        )
        _place_paper_trade(sig, live_ask, ev)


# ── Entry point ────────────────────────────────────────────────────────────────

async def run_ws_price_monitor() -> None:
    if not WS_ENABLED:
        log.info("WS price monitor disabled (WS_PRICE_MONITOR_ENABLED=false)")
        return

    log.info(
        "WS price monitor starting  min_ev=%.3f  min_dip=%.3f  paper=%s",
        WS_MIN_EV, WS_MIN_DIP, PAPER_TRADING,
    )

    cache   = SignalCache()
    monitor = WebSocketPriceMonitor(cache)
    await monitor.run()


if __name__ == "__main__":
    try:
        asyncio.run(run_ws_price_monitor())
    except KeyboardInterrupt:
        print("\nWS price monitor stopped.")
