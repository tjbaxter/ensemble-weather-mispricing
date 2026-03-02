"""NWS Daily Climate Report (CLI) — tie-breaker for US boundary cases.

The CLI gives the OFFICIAL daily maximum temperature in whole °F computed from
raw sensor data (nearest whole °F), which is exactly what Weather Underground
displays and what Polymarket resolves against for US markets.

CLI publishes after midnight local time (typically 00:30–09:30 UTC depending on
the WFO), so it is only useful as FINAL confirmation, not intraday signal.

Usage
-----
    from data.cli_checker import fetch_cli_max_temp_f
    max_f = await fetch_cli_max_temp_f("KLGA")   # returns int or None

Source URLs (both tried, IEM first):
    IEM: https://mesonet.agron.iastate.edu/api/1/nws/text?pil=CLI{ID}&limit=1
    NWS: https://forecast.weather.gov/product.php?format=txt&issuedby={ID}&product=CLI&site={WFO}
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

log = logging.getLogger("weather-bot.cli")

# ── WFO / CLI config for supported US stations ───────────────────────────────
# cli_id : used in PIL code (e.g. "LGA" → PIL "CLILGA")
# wfo    : NWS Weather Forecast Office (fallback URL path)
# Published times are approximate UTC — CLI comes out overnight / early morning.
US_CLI_CONFIG: dict[str, dict] = {
    "KLGA": {"cli_id": "LGA", "wfo": "OKX", "approx_utc": "06:17"},
    "KORD": {"cli_id": "ORD", "wfo": "LOT", "approx_utc": "06:33"},
    "KATL": {"cli_id": "ATL", "wfo": "FFC", "approx_utc": "08:18"},
    "KMIA": {"cli_id": "MIA", "wfo": "MFL", "approx_utc": "08:22"},
    "KDFW": {"cli_id": "DFW", "wfo": "FWD", "approx_utc": "06:40"},
    "KSEA": {"cli_id": "SEA", "wfo": "SEW", "approx_utc": "09:28"},
    "KDEN": {"cli_id": "DEN", "wfo": "BOU", "approx_utc": "08:00"},
    "KBOS": {"cli_id": "BOS", "wfo": "BOX", "approx_utc": "07:00"},
}

_TIMEOUT = 12.0

# ── Parsing ───────────────────────────────────────────────────────────────────
# Matches:
#   "MAXIMUM          54   351 PM    67  1976 ..."
#   "MAXIMUM     -2   ..."  (negative temps in winter)
_MAX_TEMP_RE = re.compile(r"MAXIMUM\s+(-?\d+)", re.IGNORECASE)


def parse_cli_max_temp_f(text: str) -> int | None:
    """Extract the daily maximum temperature (°F) from a CLI text product."""
    m = _MAX_TEMP_RE.search(text)
    if m:
        return int(m.group(1))
    return None


# ── Fetch helpers ─────────────────────────────────────────────────────────────

async def _fetch_iem(cli_id: str, client: httpx.AsyncClient) -> str | None:
    """Try IEM API for CLI text.  Returns raw text or None."""
    pil = f"CLI{cli_id}"
    url = f"https://mesonet.agron.iastate.edu/api/1/nws/text?pil={pil}&limit=1"
    try:
        resp = await client.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # IEM response: {"data": [{"data": "...<CLI text>..."}]}
        items = data.get("data", [])
        if items:
            return items[0].get("data", "")
    except Exception as exc:
        log.debug("IEM CLI fetch failed for %s: %s", cli_id, exc)
    return None


async def _fetch_nws(cli_id: str, wfo: str, client: httpx.AsyncClient) -> str | None:
    """Try NWS product.php for CLI text.  Returns raw text or None."""
    url = (
        "https://forecast.weather.gov/product.php"
        f"?format=txt&issuedby={cli_id}&product=CLI&site={wfo}"
    )
    try:
        resp = await client.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        log.debug("NWS CLI fetch failed for %s: %s", cli_id, exc)
    return None


# ── Public API ────────────────────────────────────────────────────────────────

async def fetch_cli_max_temp_f(icao: str) -> int | None:
    """Return the official NWS daily maximum temperature in °F for a US station.

    Tries IEM first (cleaner), falls back to NWS product.php.
    Returns None if the station is not in US_CLI_CONFIG, or if the CLI has not
    yet been published (it comes out after midnight local time).

    Only call this for US °F stations when METAR flagged NO_TRADE_NEAR_BOUNDARY.
    """
    cfg = US_CLI_CONFIG.get(icao)
    if cfg is None:
        log.debug("No CLI config for %s (non-US station?)", icao)
        return None

    cli_id = cfg["cli_id"]
    wfo    = cfg["wfo"]

    async with httpx.AsyncClient(headers={"User-Agent": "PolymarketWeatherBot/1.0"}) as client:
        text = await _fetch_iem(cli_id, client)
        if text:
            result = parse_cli_max_temp_f(text)
            if result is not None:
                log.info("CLI (IEM): %s daily max = %d°F", icao, result)
                return result

        text = await _fetch_nws(cli_id, wfo, client)
        if text:
            result = parse_cli_max_temp_f(text)
            if result is not None:
                log.info("CLI (NWS): %s daily max = %d°F", icao, result)
                return result

    log.warning("CLI unavailable for %s (not yet published or parse failed)", icao)
    return None


async def fetch_all_cli(icaos: list[str]) -> dict[str, int | None]:
    """Fetch CLI max temps for multiple US stations concurrently."""
    import asyncio
    tasks = {icao: asyncio.create_task(fetch_cli_max_temp_f(icao)) for icao in icaos}
    return {icao: await task for icao, task in tasks.items()}
