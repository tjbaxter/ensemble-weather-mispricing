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
from monitoring.deep_observability import get_deep_observability
from monitoring.trade_audit import TradeAuditStore, forecast_bundle_for_signal, market_snapshot_for_signal
from strategy.signals import generate_signals, set_signal_observability_context
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
    from datetime import UTC, date, datetime, timedelta
    from execution.portfolio import Portfolio
    from config.settings import (
        MAX_DAILY_EXPOSURE, MAX_DRAWDOWN_PCT, MAX_POSITIONS_PER_MARKET,
    )
    from monitoring.logger import BotLogger
    from execution.risk_engine import apply_risk_controls

    pk = os.getenv("POLYMARKET_PK")
    funder = os.getenv("POLYMARKET_FUNDER")
    api_key = os.getenv("POLYMARKET_API_KEY")

    if not all([pk, funder]):
        raise RuntimeError(
            "Live trading requires POLYMARKET_PK and POLYMARKET_FUNDER in .env"
        )

    _KILL_FILE = "data/.kill_switch"
    if os.path.exists(_KILL_FILE):
        print("KILL SWITCH ACTIVE — refusing to trade. Remove data/.kill_switch to resume.")
        return

    order_manager = OrderManager(
        live_trading=True,
        api_key=api_key or pk,
        private_key=pk,
        wallet_address=funder,
    )

    logger = BotLogger(output_dir="logs")
    audit = TradeAuditStore()
    deep_obs = get_deep_observability()
    run_id = audit.new_run_id("live-main")
    portfolio = Portfolio(initial_bankroll=bankroll, positions_path="data/positions_live.json")

    if portfolio.max_drawdown_pct() >= MAX_DRAWDOWN_PCT:
        print(f"DRAWDOWN LIMIT HIT ({portfolio.max_drawdown_pct():.1%} >= {MAX_DRAWDOWN_PCT:.1%}) — refusing to trade.")
        return

    forecaster = StationForecaster(met_office_api_key=os.getenv("MET_OFFICE_API_KEY"))
    market_client = PolymarketDataClient()

    try:
        print("Discovering live markets...")
        markets = await market_client.discover_weather_markets()
        hydrated = await market_client.hydrate_prices(markets)
        print(f"  Found {len(hydrated)} markets with prices.")
        market_scan_id = deep_obs.log_market_state(
            {
                "market_scan_id": "",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "scan_trigger": "live_run_once",
                "markets_count": len(hydrated),
                "markets": hydrated,
            },
            mode="live",
        )
        set_signal_observability_context(market_scan_id, mode="live")

        print("Fetching forecasts for active cities...")
        target_date = date.today() + timedelta(days=1)
        forecasts: dict = {}
        for icao in STATIONS:
            if STATIONS[icao].get("paused"):
                continue
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
                print(f"  Forecast error for {icao}: {e}")

        signals = generate_signals(hydrated, forecasts, portfolio.current_cash)
        print(f"  Generated {len(signals)} signals.")

        placed = 0
        daily_deployed = 0.0
        skipped_reasons: dict[str, int] = {}

        for signal in signals:
            sig = signal.to_dict() if hasattr(signal, "to_dict") else signal
            size_usd = float(sig.get("size_usd", 0))

            if os.path.exists(_KILL_FILE):
                print("KILL SWITCH activated mid-run — stopping.")
                break

            market_id = sig.get("market_id", "")
            bucket = sig.get("bucket", "")

            if portfolio.holds_market_bucket(market_id, bucket):
                audit.log_event(
                    run_id=run_id,
                    engine="live:main",
                    action="skip_already_held",
                    signal=sig,
                    reason="already_held",
                    forecast_bundle=forecast_bundle_for_signal(sig, forecasts),
                    market_snapshot=market_snapshot_for_signal(sig, hydrated),
                )
                skipped_reasons["duplicate"] = skipped_reasons.get("duplicate", 0) + 1
                continue

            existing = sum(1 for p in portfolio.positions if p.market_id == market_id)
            if existing >= MAX_POSITIONS_PER_MARKET:
                audit.log_event(
                    run_id=run_id,
                    engine="live:main",
                    action="skip_position_limit",
                    signal=sig,
                    reason="max_positions_per_market",
                    forecast_bundle=forecast_bundle_for_signal(sig, forecasts),
                    market_snapshot=market_snapshot_for_signal(sig, hydrated),
                )
                skipped_reasons["max_per_market"] = skipped_reasons.get("max_per_market", 0) + 1
                continue

            decision = apply_risk_controls(
                requested_size_usd=size_usd,
                signal=sig,
                cash_usd=portfolio.current_cash,
                active_exposure_usd=portfolio.active_exposure(),
                deployed_today_usd=daily_deployed,
            )
            if decision.skipped:
                audit.log_event(
                    run_id=run_id,
                    engine="live:main",
                    action="skip_risk",
                    signal=sig,
                    reason=decision.reason,
                    risk_decision={
                        "size_usd": decision.size_usd,
                        "daily_budget_usd": decision.daily_budget_usd,
                        "position_cap_usd": decision.position_cap_usd,
                        "quality_mult": decision.quality_mult,
                        "skipped": decision.skipped,
                        "reason": decision.reason,
                    },
                    forecast_bundle=forecast_bundle_for_signal(sig, forecasts),
                    market_snapshot=market_snapshot_for_signal(sig, hydrated),
                )
                skipped_reasons[decision.reason] = skipped_reasons.get(decision.reason, 0) + 1
                continue
            if decision.daily_budget_usd <= 0 and daily_deployed + decision.size_usd > MAX_DAILY_EXPOSURE:
                audit.log_event(
                    run_id=run_id,
                    engine="live:main",
                    action="skip_daily_exposure",
                    signal=sig,
                    reason="daily_cap",
                    risk_decision={
                        "size_usd": decision.size_usd,
                        "daily_budget_usd": decision.daily_budget_usd,
                        "position_cap_usd": decision.position_cap_usd,
                        "quality_mult": decision.quality_mult,
                        "skipped": decision.skipped,
                        "reason": decision.reason,
                    },
                    forecast_bundle=forecast_bundle_for_signal(sig, forecasts),
                    market_snapshot=market_snapshot_for_signal(sig, hydrated),
                )
                skipped_reasons["daily_cap"] = skipped_reasons.get("daily_cap", 0) + 1
                continue

            sig["size_usd"] = decision.size_usd
            result = order_manager.place_order(sig)
            execution_id = deep_obs.log_execution(
                {
                    "execution_id": "",
                    "decision_id": sig.get("decision_id", ""),
                    "market_scan_id": sig.get("market_scan_id", market_scan_id),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "strategy": sig.get("strategy", ""),
                    "city": sig.get("city", ""),
                    "target_date": sig.get("date", ""),
                    "bucket": sig.get("bucket", ""),
                    "side": sig.get("side", ""),
                    "price": result.fill_price,
                    "size_usd": result.size_usd,
                    "status": result.status,
                    "details": result.details,
                },
                mode="live",
            )

            if result.status in ("filled", "submitted", "matched", "live"):
                pos = portfolio.open_position(sig, result.fill_price)
                daily_deployed += decision.size_usd
                logger.log_signal(sig, "live_trade")
                audit.log_event(
                    run_id=run_id,
                    engine="live:main",
                    action="trade_executed",
                    signal=sig,
                    risk_decision={
                        "size_usd": decision.size_usd,
                        "daily_budget_usd": decision.daily_budget_usd,
                        "position_cap_usd": decision.position_cap_usd,
                        "quality_mult": decision.quality_mult,
                        "skipped": decision.skipped,
                        "reason": decision.reason,
                    },
                    execution_result={
                        "execution_id": execution_id,
                        "status": result.status,
                        "fill_price": result.fill_price,
                        "size_usd": result.size_usd,
                        "details": result.details,
                    },
                    forecast_bundle=forecast_bundle_for_signal(sig, forecasts),
                    market_snapshot=market_snapshot_for_signal(sig, hydrated),
                    context={
                        "requested_size_usd": size_usd,
                        "approved_size_usd": decision.size_usd,
                        "fill_price": result.fill_price,
                    },
                )
                print(
                    f"  [{sig.get('city')}] {bucket} {sig.get('side')} "
                    f"${decision.size_usd:.2f} @ {result.fill_price:.3f} -> {result.status} "
                    f"(q={decision.quality_mult:.2f} cap={decision.position_cap_usd:.2f} day={decision.daily_budget_usd:.2f})"
                )
                placed += 1
            else:
                logger.log_signal(sig, f"live_rejected_{result.status}")
                audit.log_event(
                    run_id=run_id,
                    engine="live:main",
                    action="skip_execution",
                    signal=sig,
                    reason=result.status,
                    risk_decision={
                        "size_usd": decision.size_usd,
                        "daily_budget_usd": decision.daily_budget_usd,
                        "position_cap_usd": decision.position_cap_usd,
                        "quality_mult": decision.quality_mult,
                        "skipped": decision.skipped,
                        "reason": decision.reason,
                    },
                    execution_result={
                        "execution_id": execution_id,
                        "status": result.status,
                        "fill_price": result.fill_price,
                        "size_usd": result.size_usd,
                        "details": result.details,
                    },
                    forecast_bundle=forecast_bundle_for_signal(sig, forecasts),
                    market_snapshot=market_snapshot_for_signal(sig, hydrated),
                )
                print(f"  [{sig.get('city')}] {bucket} REJECTED: {result.status}")

        print(f"Done. {placed} orders placed, ${daily_deployed:.2f} deployed.")
        if skipped_reasons:
            print(f"  Skipped: {skipped_reasons}")

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
    parser.add_argument("--force-shadows", action="store_true", help="Immediately execute shadow strategies using cached data and exit.")
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

    if args.force_shadows:
        print(f"Force-executing shadow strategies using cached data. Bankroll: ${bankroll}")
        trader = PaperTrader()
        try:
            markets = await trader.market_client.discover_weather_markets()
            markets = await trader.market_client.hydrate_prices(markets)
            trader._cached_markets = markets
            print(f"  Discovered {len(markets)} markets.")
            await trader._refresh_forecasts(markets)
            print(f"  Forecasts loaded for {sum(len(v) for v in trader.forecasts.values())} station-dates.")
            trader.run_shadows_now()
            print("Force shadow run complete.")
        finally:
            await trader.close()
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
