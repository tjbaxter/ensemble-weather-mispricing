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
from execution.order_manager import OrderManager
from strategy.signals import generate_signals
from scripts.metar_scanner import run_metar_scanner
from scripts.ws_price_monitor import run_ws_price_monitor


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
            ensemble_models = cfg.get("ensemble_models", [])
            p_win_green = cfg.get("p_win_green", "—")
            kelly = cfg.get("kelly_fraction", "—")
            bn = cfg.get("backtest_n", 0)
            print(
                f"  - {icao}: {cfg['name']} [{cfg['priority']}] "
                f"models={len(ensemble_models)} p_win_green={p_win_green} kelly={kelly} bt_n={bn}"
            )
    finally:
        await forecaster.close()
        await market_client.close()


async def run_live(bankroll: float) -> None:
    """Single-pass live trading run. Called by cron at 18:30 UTC."""
    pk = os.getenv("POLYMARKET_PK")
    funder = os.getenv("POLYMARKET_FUNDER")
    api_key = os.getenv("POLYMARKET_API_KEY")

    if not all([pk, funder]):
        raise RuntimeError(
            "Live trading requires POLYMARKET_PK and POLYMARKET_FUNDER in .env"
        )

    order_manager = OrderManager(
        live_trading=True,
        api_key=api_key or pk,
        private_key=pk,
        wallet_address=funder,
    )

    forecaster = StationForecaster(met_office_api_key=os.getenv("MET_OFFICE_API_KEY"))
    market_client = PolymarketDataClient()

    try:
        print("Discovering live markets...")
        markets = await market_client.discover_weather_markets()
        hydrated = await market_client.hydrate_prices(markets)
        print(f"  Found {len(hydrated)} markets with prices.")

        print("Fetching forecasts for all cities...")
        from datetime import date, timedelta
        target_date = date.today() + timedelta(days=1)
        forecasts: dict = {}
        for icao in STATIONS:
            bucket_labels = [
                b
                for m in hydrated
                if m.get("station_icao") == icao
                for b in m.get("buckets", {})
            ]
            if not bucket_labels:
                continue
            try:
                bundle = await forecaster.get_station_forecast(icao, target_date, list(bucket_labels))
                forecasts.setdefault(icao, {})[target_date.isoformat()] = bundle
            except Exception as e:
                print(f"  ⚠ Forecast error for {icao}: {e}")

        signals = generate_signals(hydrated, forecasts, bankroll)
        print(f"  Generated {len(signals)} signals.")

        placed = 0
        for signal in signals:
            result = order_manager.place_order(signal.to_dict())
            print(
                f"  [{signal.city}] {signal.bucket} {signal.side} "
                f"${signal.size_usd:.2f} @ {signal.market_prob:.2f} → {result.status}"
            )
            placed += 1

        print(f"Done. {placed} orders placed.")

    finally:
        await forecaster.close()
        await market_client.close()


async def run_paper_once(bankroll: float) -> None:
    """Single-pass paper trading scan across all 6 strategies. Safe to call on demand."""
    trader = PaperTrader()
    await trader.run_once()
    print("Paper scan complete.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Weather bot runtime")
    parser.add_argument("--diagnostic", action="store_true", help="Run startup and discovery diagnostics then exit.")
    parser.add_argument("--live-once", action="store_true", help="Run one live trading pass and exit (for cron).")
    parser.add_argument("--paper-once", action="store_true", help="Run one paper trading scan across all strategies and exit.")
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
    bankroll = float(os.getenv("BANKROLL", "300"))

    if live and paper:
        raise RuntimeError("Set either LIVE_TRADING=true or PAPER_TRADING=true, not both.")

    if args.paper_once:
        print(f"Running single paper scan. Bankroll: ${bankroll}")
        await run_paper_once(bankroll)
        return

    if live or args.live_once:
        print(f"Starting live trading run. Bankroll: ${bankroll}")
        await run_live(bankroll)
        return

    if not paper:
        raise RuntimeError("No execution mode selected. Enable PAPER_TRADING=true or LIVE_TRADING=true in .env.")

    trader = PaperTrader()
    # Run all three strategies concurrently:
    #   1. Forecast bot       — wakes at 5 NWP trigger times, caches model probs, executes on mispriced buckets
    #   2. METAR scanner      — polls every 60s on resolution days, trades on confirmed temps
    #   3. WS price monitor   — persistent WebSocket to Polymarket market channel; real-time price dip detection
    # Either task crashing propagates and restarts the whole service via systemd.
    await asyncio.gather(
        trader.run_forever(),
        run_metar_scanner(),
        run_ws_price_monitor(),
    )


if __name__ == "__main__":
    if os.getenv("REQUIRE_VPN", "true").strip().lower() in {"1", "true", "yes"}:
        print("Reminder: run only in a legally permitted region with VPN/VPS configured.")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down gracefully.")
