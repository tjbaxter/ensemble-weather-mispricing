"""Main runtime entrypoint for weather trading bot."""

from __future__ import annotations

import asyncio
import argparse
import os
import sys

from dotenv import load_dotenv

from backtest.paper_trader import PaperTrader
from config.cities import STATIONS
from config.settings import load_runtime_overrides
from data.forecast import StationForecaster
from data.polymarket import PolymarketDataClient


async def startup_checks() -> None:
    print(f"Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10+ is required.")

    forecaster = StationForecaster(met_office_api_key=os.getenv("MET_OFFICE_API_KEY"))
    market_client = PolymarketDataClient()
    try:
        print("Testing aviationweather.gov connectivity...")
        metar = await forecaster.get_latest_metar("KLGA")
        if not metar:
            raise RuntimeError("AviationWeather API returned no data for KLGA.")
        print("  OK: AviationWeather returned METAR for KLGA.")

        print("Testing Polymarket Gamma connectivity...")
        markets = await market_client.discover_weather_markets()
        print(f"  OK: discovered {len(markets)} weather markets.")

        print("Active stations:")
        for icao, cfg in STATIONS.items():
            print(f"  - {icao}: {cfg['name']} [{cfg['priority']}]")
    finally:
        await forecaster.close()
        await market_client.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Weather bot runtime")
    parser.add_argument("--diagnostic", action="store_true", help="Run startup and discovery diagnostics then exit.")
    args = parser.parse_args()

    load_dotenv()
    await startup_checks()
    if args.diagnostic:
        diag_client = PolymarketDataClient(diagnostic=True)
        try:
            markets = await diag_client.discover_weather_markets()
            hydrated = await diag_client.hydrate_prices(markets)
            print(f"DIAGNOSTIC discovered={len(markets)} hydrated={len(hydrated)}")
            print(f"DIAGNOSTIC stats={diag_client.last_discovery_stats}")
            for sample in hydrated[:5]:
                print(f"DIAGNOSTIC sample {sample['station_icao']} {sample['date']} {sample['question']}")
        finally:
            await diag_client.close()
        return
    runtime = load_runtime_overrides()

    live = bool(runtime["LIVE_TRADING"])
    paper = bool(runtime["PAPER_TRADING"])

    if live and paper:
        raise RuntimeError("Set either LIVE_TRADING=true or PAPER_TRADING=true, not both.")
    if live:
        raise RuntimeError(
            "Live mode is intentionally disabled in this starter. "
            "Run paper mode for 3-5 days and implement explicit live safety checks first."
        )
    if not paper:
        raise RuntimeError("No execution mode selected. Enable PAPER_TRADING=true in .env.")

    trader = PaperTrader()
    await trader.run_forever()


if __name__ == "__main__":
    # Guard rails for accidental execution in constrained geographies.
    if os.getenv("REQUIRE_VPN", "true").strip().lower() in {"1", "true", "yes"}:
        print("Reminder: run only in a legally permitted region with VPN/VPS configured.")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down gracefully.")
