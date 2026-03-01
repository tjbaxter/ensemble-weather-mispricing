#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Run with: python3 scanner.py  (not python — requires Python 3.9+)
"""
One-shot Polymarket weather market scanner.

Fetches current bucket prices and model forecasts for all active markets,
then prints a decision table: which bucket to buy, at what price, Kelly size.

Usage:
    cd weather-bot && python scanner.py
    python scanner.py --cities Chicago NYC
    python scanner.py --bankroll 500 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from config.cities import STATIONS
from config.settings import (
    ENSEMBLE_PRIMARY_MODEL,
    INITIAL_BANKROLL,
)
from data.forecast import ForecastClient
from data.polymarket import PolymarketDataClient
from strategy.signals import generate_signals

# ── Per-model display config ─────────────────────────────────────────────────
# Maps station ICAO → list of models to display in the table (for readability).
# These are the same models tracked by the dashboard and log_model_snapshots.py.
ICAO_DISPLAY_MODELS: dict[str, dict[str, Any]] = {
    "KLGA": {
        "models": ["ncep_nbm_conus", "gfs_graphcast025", "icon_seamless", "kma_seamless", "gem_seamless", "ncep_aigfs025"],
        "unit": "fahrenheit",
    },
    "KORD": {
        "models": ["ncep_nbm_conus", "gfs_seamless", "ecmwf_ifs025", "icon_seamless"],
        "unit": "fahrenheit",
    },
    "KDEN": {
        "models": ["ncep_nbm_conus", "gfs_seamless", "ecmwf_ifs025", "icon_seamless"],
        "unit": "fahrenheit",
    },
    "KBOS": {
        "models": ["ncep_nbm_conus", "gfs_seamless", "ecmwf_ifs025", "icon_seamless"],
        "unit": "fahrenheit",
    },
    "KMIA": {
        "models": ["ncep_nbm_conus", "gfs_seamless", "ecmwf_ifs025", "icon_seamless"],
        "unit": "fahrenheit",
    },
    "KSEA": {
        "models": ["ncep_nbm_conus", "gem_seamless", "knmi_seamless", "dmi_seamless", "icon_seamless", "ecmwf_ifs025"],
        "unit": "fahrenheit",
    },
    "KDFW": {
        "models": ["ncep_nbm_conus", "gfs_seamless", "ecmwf_ifs025", "icon_seamless"],
        "unit": "fahrenheit",
    },
    "KATL": {
        "models": ["ncep_nbm_conus", "gfs_seamless", "ecmwf_ifs025", "icon_seamless"],
        "unit": "fahrenheit",
    },
    "EGLC": {
        "models": [
            "meteofrance_arome_france", "meteofrance_seamless", "meteofrance_arome_france_hd",
            "icon_seamless", "dmi_seamless", "ecmwf_ifs025", "ukmo_seamless", "ncep_aigfs025",
        ],
        "unit": "celsius",
    },
    "LFPG": {
        "models": ["meteofrance_arome_france", "meteofrance_seamless", "icon_seamless", "ecmwf_ifs025", "dmi_seamless"],
        "unit": "celsius",
    },
    "RKSI": {
        "models": ["kma_seamless", "kma_gdps", "ncep_aigfs025", "gfs_graphcast025", "icon_seamless", "ecmwf_ifs025"],
        "unit": "celsius",
    },
    "SBGR": {
        "models": ["meteofrance_seamless", "icon_seamless", "ecmwf_ifs025", "gfs_seamless"],
        "unit": "celsius",
    },
}

MODEL_SHORT_NAMES: dict[str, str] = {
    "ncep_nbm_conus":               "NCEP NBM",
    "ncep_aigfs025":                "AI GFS 0.25°",
    "gfs_graphcast025":             "GraphCast",
    "gfs_seamless":                 "GFS Seamless",
    "ecmwf_ifs025":                 "ECMWF IFS",
    "icon_seamless":                "ICON Seamless",
    "kma_seamless":                 "KMA Seamless",
    "kma_gdps":                     "Korea GDPS",
    "gem_global":                   "GEM Global",
    "gem_seamless":                 "GEM Seamless",
    "gem_hrdps_continental":        "GEM HRDPS",
    "dmi_seamless":                 "DMI Seamless",
    "knmi_seamless":                "KNMI Seamless",
    "meteofrance_arome_france":     "MF AROME",
    "meteofrance_seamless":         "MF Seamless",
    "meteofrance_arome_france_hd":  "MF AROME HD",
    "meteofrance_arpege_world":     "MF ARPEGE",
    "ukmo_seamless":                "UKMO Seamless",
    "ukmo_uk_deterministic_2km":    "UKMO 2km",
    "ukmo_global_deterministic_10km": "UKMO Global",
}

WIDTH = 64


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hround(x: float) -> int:
    """Round 0.5 UP (matching Polymarket resolution), not banker's rounding."""
    return math.floor(x + 0.5)


def fmt_temp(val: float, unit: str) -> str:
    sym = "°F" if unit == "fahrenheit" else "°C"
    return f"{val:.1f}{sym}"


def fmt_price(p: float | None) -> str:
    return f"${p:.2f}" if p is not None else "  —  "


def _bucket_sort_key(b: str) -> float:
    clean = b.replace("°F", "").replace("°C", "").strip()
    if "+" in clean:
        return float(clean.replace("+", ""))
    if "-" in clean:
        try:
            return float(clean.split("-")[0].strip())
        except ValueError:
            pass
    try:
        return float(clean)
    except ValueError:
        return 9999.0


async def _fetch_one_model_temp(
    http: httpx.AsyncClient,
    params_base: dict[str, Any],
    model: str,
) -> tuple[str, float | None]:
    """Fetch a single model's daily high temp. Returns (model, temp_or_None)."""
    try:
        resp = await http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={**params_base, "models": model},
            timeout=15.0,
        )
        data = resp.json()
        if "error" in data:
            return model, None
        vals = [v for v in data.get("hourly", {}).get("temperature_2m", []) if v is not None]
        if vals:
            return model, round(max(vals) * 10) / 10
    except Exception:
        pass
    return model, None


async def fetch_per_model_temps(
    http: httpx.AsyncClient,
    station: dict[str, Any],
    target_date_str: str,
    models: list[str],
    unit: str,
) -> dict[str, float]:
    """Fetch individual model D+1 daily-high temps concurrently."""
    params_base: dict[str, Any] = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "hourly": "temperature_2m",
        "start_date": target_date_str,
        "end_date": target_date_str,
        "timezone": station["timezone"],
    }
    if unit == "fahrenheit":
        params_base["temperature_unit"] = "fahrenheit"

    tasks = [_fetch_one_model_temp(http, params_base, m) for m in models]
    pairs = await asyncio.gather(*tasks)
    return {model: temp for model, temp in pairs if temp is not None}


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_market(
    station_icao: str,
    date_str: str,
    bundle: dict[str, Any],
    market_buckets: dict[str, dict[str, Any]],
    signals_for_market: list,
    model_temps: dict[str, float],
    unit: str,
    bankroll: float,
    now_utc: datetime,
) -> None:
    station = STATIONS[station_icao]
    city = station["market_label"]
    unit_sym = "°F" if unit == "fahrenheit" else "°C"
    in_12z = now_utc.hour * 60 + now_utc.minute >= 18 * 60 + 30
    timing = "✅ post-12Z" if in_12z else "⚠  pre-12Z (stale until 18:30 UTC)"

    bar = "═" * WIDTH
    print(bar)
    print(f"  {city.upper()} — {date_str}   {timing}")
    print(bar)

    # Ensemble summary from bot's forecast engine
    ensemble_mean   = bundle.get("forecast_temp_raw")
    ensemble_std    = bundle.get("ensemble_std")
    predicted_disp  = bundle.get("predicted_display_temp")
    wu_crowd        = bundle.get("wu_crowd_temp")
    primary_temp    = bundle.get("primary_model_temp")
    baseline_temp   = bundle.get("baseline_model_temp")
    confidence      = bundle.get("rounding_confidence", "?")
    skip            = bundle.get("ensemble_skip", False)
    member_count    = bundle.get("ensemble_member_count", 0)

    if ensemble_mean is not None:
        std_str = f"  σ={ensemble_std:.1f}{unit_sym}" if ensemble_std is not None else ""
        skip_warn = "  ⚠ HIGH SPREAD — SKIP" if skip else ""
        print(f"  Ensemble mean:     {fmt_temp(ensemble_mean, unit)}{std_str}  [{confidence} conf, n={member_count}{skip_warn}]")

    if predicted_disp is not None:
        print(f"  Predicted display: {predicted_disp}{unit_sym}  (bucket: {_hround(float(predicted_disp))})")

    if primary_temp is not None:
        ai_short = MODEL_SHORT_NAMES.get(ENSEMBLE_PRIMARY_MODEL, ENSEMBLE_PRIMARY_MODEL)
        delta_str = ""
        if baseline_temp is not None:
            d = primary_temp - baseline_temp
            regime = "  HIGH-DELTA REGIME" if abs(d) >= 3.0 else ""
            delta_str = f"  (Δ {d:+.1f} vs GFS/ECMWF){regime}"
        print(f"  {ai_short:<22}  {fmt_temp(primary_temp, unit)}{delta_str}")

    if wu_crowd is not None:
        print(f"  WU crowd baseline: {fmt_temp(wu_crowd, unit)}  (what retail traders see)")

    # Individual model temps (from direct Open-Meteo fetch, for display only)
    if model_temps:
        print()
        print("  Model temps (live D+1):")
        primary_key = ENSEMBLE_PRIMARY_MODEL
        sorted_models = sorted(model_temps.items(), key=lambda kv: kv[1])
        for model, temp in sorted_models:
            short = MODEL_SHORT_NAMES.get(model, model)
            marker = "  ◄ primary" if model == primary_key else ""
            print(f"    {short:<24} {fmt_temp(temp, unit)}{marker}")

    # Bucket prices
    if market_buckets:
        print()
        print("  Bucket prices:")
        signal_buckets = {s.bucket for s in signals_for_market}
        sorted_buckets = sorted(market_buckets.items(), key=lambda kv: _bucket_sort_key(kv[0]))
        for bucket, info in sorted_buckets:
            bid = float(info.get("best_bid") or 0.0)
            ask = float(info.get("best_ask") or 0.0)
            price = ask if ask > 0 else float(info.get("price") or 0.0)
            signal_tag = "  ◄ SIGNAL" if bucket in signal_buckets else ""
            spread_str = ""
            if bid > 0 and ask > 0:
                spread_str = f"  (bid {bid:.2f} / ask {ask:.2f})"
            print(f"    {bucket:<16}  {fmt_price(price)}{spread_str}{signal_tag}")

    # Decision
    print()
    print("  ── DECISION " + "─" * (WIDTH - 14))
    if skip:
        print("  SKIP — ensemble spread too high (unreliable forecast)")
    elif not signals_for_market:
        print("  SKIP — no edge above alpha threshold at current prices")
    else:
        for sig in signals_for_market:
            is_yes = sig.side == "BUY_YES"
            side_label = "BUY YES" if is_yes else "BUY NO"
            trade_price = sig.market_prob if is_yes else (1.0 - sig.market_prob)
            shares = sig.size_usd / trade_price if trade_price > 0 else 0.0
            profit = shares * (1.0 - trade_price)
            pct = sig.size_usd / bankroll * 100
            print(f"  {side_label}  {sig.bucket}  @  ${trade_price:.2f}")
            print(f"    Size:   ${sig.size_usd:.2f}  ({pct:.1f}% of ${bankroll:.0f})  |  edge={sig.edge:.3f}  p_win={sig.forecast_prob:.1%}  conf={sig.rounding_confidence}")
            print(f"    Profit: +${profit:.2f} if win  |  -${sig.size_usd:.2f} if lose")

    print("═" * WIDTH)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

async def scan(
    cities: list[str] | None = None,
    bankroll: float = INITIAL_BANKROLL,
    verbose: bool = False,
) -> None:
    now_utc = datetime.now(UTC)
    tomorrow = (now_utc + timedelta(days=1)).date()

    print(f"\nPolymarket Weather Scanner  ─  {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Target date:  {tomorrow.isoformat()}")
    print(f"Bankroll:     ${bankroll:.2f}\n")

    http = httpx.AsyncClient(timeout=20.0)
    market_client = PolymarketDataClient()
    forecast_client = ForecastClient(met_office_api_key=os.getenv("MET_OFFICE_API_KEY"))

    try:
        # 1. Discover + hydrate markets
        print("Discovering markets...", end="", flush=True)
        markets = await market_client.discover_weather_markets()
        markets = await market_client.hydrate_prices(markets)
        stats = market_client.last_discovery_stats
        print(f" {len(markets)} hydrated markets")
        if verbose:
            print(f"  Stats: {stats}")

        if not markets:
            print("No markets found. Check STATION_PRIORITY_FILTER or network.")
            return

        # Filter to requested cities
        if cities:
            cities_lower = [c.lower() for c in cities]
            markets = [
                m for m in markets
                if m["city"].lower() in cities_lower
                or m["station_icao"].lower() in cities_lower
            ]
            if not markets:
                print(f"No markets found for cities: {cities}")
                return

        # 2. Group by (station_icao, date) — merge buckets across markets for same event
        grouped: dict[tuple[str, str], dict[str, dict]] = {}
        market_meta: dict[tuple[str, str], dict] = {}
        for m in markets:
            key = (m["station_icao"], m["date"])
            grouped.setdefault(key, {}).update(m["buckets"])
            market_meta[key] = m

        # 3. Fetch ensemble forecasts + per-model display temps — all concurrently
        print("Fetching forecasts...", end="", flush=True)
        forecasts: dict[str, dict[str, dict]] = {}
        model_temps_cache: dict[tuple[str, str], dict[str, float]] = {}

        async def _fetch_one_station(station_icao: str, date_str: str) -> None:
            bucket_labels = sorted(grouped[(station_icao, date_str)].keys())
            target_date = datetime.fromisoformat(date_str).date()
            station = STATIONS[station_icao]

            bundle = await forecast_client.get_station_forecast(
                station_icao=station_icao,
                target_date=target_date,
                bucket_labels=bucket_labels,
            )
            forecasts.setdefault(station_icao, {})[date_str] = bundle

            display_cfg = ICAO_DISPLAY_MODELS.get(station_icao)
            if display_cfg:
                temps = await fetch_per_model_temps(
                    http=http,
                    station=station,
                    target_date_str=date_str,
                    models=display_cfg["models"],
                    unit=display_cfg["unit"],
                )
                model_temps_cache[(station_icao, date_str)] = temps

        await asyncio.gather(*[
            _fetch_one_station(icao, ds) for icao, ds in grouped
        ])
        print(f" done ({len(grouped)} markets)")

        # 4. Generate trading signals
        signals = generate_signals(markets, forecasts, bankroll)
        print(f"Signals generated: {len(signals)}\n")

        # 5. Print one decision table per market
        for (station_icao, date_str) in sorted(
            grouped.keys(), key=lambda k: (k[1], STATIONS[k[0]]["market_label"])
        ):
            station = STATIONS[station_icao]
            display_cfg = ICAO_DISPLAY_MODELS.get(station_icao, {})
            unit = display_cfg.get("unit", "fahrenheit" if station["resolution_unit"] == "F" else "celsius")
            bundle = forecasts.get(station_icao, {}).get(date_str, {})
            market_buckets = grouped[(station_icao, date_str)]
            model_temps = model_temps_cache.get((station_icao, date_str), {})
            sigs = [s for s in signals if s.station_icao == station_icao and s.date == date_str]

            render_market(
                station_icao=station_icao,
                date_str=date_str,
                bundle=bundle,
                market_buckets=market_buckets,
                signals_for_market=sigs,
                model_temps=model_temps,
                unit=unit,
                bankroll=bankroll,
                now_utc=now_utc,
            )

        # Summary footer
        if signals:
            print(f"{'─' * WIDTH}")
            print(f"  Total signals: {len(signals)}")
            from collections import Counter
            side_counts = Counter(s.side for s in signals)
            for side, count in sorted(side_counts.items()):
                total_usd = sum(s.size_usd for s in signals if s.side == side)
                print(f"    {side}: {count}  (${total_usd:.2f} total)")
            print(f"{'─' * WIDTH}\n")

    finally:
        await http.aclose()
        await market_client.close()
        await forecast_client.close()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="One-shot Polymarket weather market scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scanner.py                       # all active markets
  python scanner.py --cities Chicago NYC  # specific cities
  python scanner.py --bankroll 500        # custom bankroll
  python scanner.py -v                    # verbose output
        """,
    )
    parser.add_argument("--cities", nargs="*", metavar="CITY",
                        help="Filter: city names or ICAO codes (e.g. Chicago NYC EGLC)")
    parser.add_argument("--bankroll", type=float, default=float(os.getenv("BANKROLL", str(INITIAL_BANKROLL))),
                        help=f"Bankroll for Kelly sizing (default: from $BANKROLL env or {INITIAL_BANKROLL})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show extra diagnostic info (discovery stats, etc.)")
    args = parser.parse_args()
    asyncio.run(scan(cities=args.cities, bankroll=args.bankroll, verbose=args.verbose))


if __name__ == "__main__":
    main()
