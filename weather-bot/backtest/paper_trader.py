"""Paper-trading runner for end-to-end validation."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from config.cities import STATIONS
from config.settings import (
    ACCUWEATHER_SNAPSHOT_LOGGING_ENABLED,
    FORECAST_REFRESH_SECONDS,
    INITIAL_BANKROLL,
    MAX_DAILY_EXPOSURE,
    MAX_DRAWDOWN_PCT,
    MAX_POSITIONS_PER_MARKET,
    MODEL_RUN_BOOST_SCAN_INTERVAL_SECONDS,
    MODEL_RUN_BOOST_WINDOW_MINUTES,
    MODEL_RUN_TRIGGER_TIMES_UTC,
    SCAN_INTERVAL_SECONDS,
)
from data.accuweather import AccuWeatherClient
from data.forecast import ForecastClient
from data.polymarket import PolymarketDataClient
from data.weather_underground import WeatherUndergroundClient
from execution.order_manager import OrderManager
from execution.portfolio import Portfolio
from monitoring.dashboard import render_dashboard
from monitoring.logger import BotLogger
from strategy.signals import generate_signals, summarize_top_missed_edges


class PaperTrader:
    def __init__(self) -> None:
        self.logger = BotLogger(output_dir="logs")
        self.portfolio = Portfolio(initial_bankroll=INITIAL_BANKROLL)
        self.order_manager = OrderManager(live_trading=False)
        self.forecast_client = ForecastClient(met_office_api_key=os.getenv("MET_OFFICE_API_KEY"))
        self.market_client = PolymarketDataClient(diagnostic=os.getenv("DIAGNOSTIC_MODE", "").lower() in {"1", "true", "yes"})
        self.wu_client = WeatherUndergroundClient()
        accuweather_key = os.getenv("ACCUWEATHER_API_KEY", "").strip()
        self.accuweather_client = (
            AccuWeatherClient(api_key=accuweather_key)
            if ACCUWEATHER_SNAPSHOT_LOGGING_ENABLED and accuweather_key
            else None
        )
        self.forecasts: dict[str, dict[str, dict]] = {}
        self.intraday_observed_highs: dict[str, dict[str, float]] = {}
        self.last_forecast_refresh = 0.0

    async def close(self) -> None:
        if self.accuweather_client is not None:
            await self.accuweather_client.close()
        await self.forecast_client.close()
        await self.market_client.close()
        await self.wu_client.close()

    async def run_once(self) -> None:
        if self.portfolio.max_drawdown_pct() >= MAX_DRAWDOWN_PCT:
            self.logger.warning("Drawdown limit reached; skipping scan.")
            return

        discovery_stats: dict = {}
        try:
            markets = await self.market_client.discover_weather_markets()
            markets = await self.market_client.hydrate_prices(markets)
            stats = self.market_client.last_discovery_stats
            discovery_stats = stats
            self.logger.info(
                "DISCOVERY "
                f"found={stats.get('discovered_markets', 0)} "
                f"slugs_checked={stats.get('slugs_checked', 0)} "
                f"search_hits={stats.get('search_hits', 0)} "
                f"paged_events={stats.get('paginated_events_considered', 0)} "
                f"rejects={stats.get('reject_stats', {})}"
            )
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            self.logger.warning(f"Market data fetch failed: {exc}. Retrying next scan.")
            render_dashboard(
                bankroll=self.portfolio.initial_bankroll,
                cash=self.portfolio.current_cash,
                active_exposure=self.portfolio.active_exposure(),
                signals_count=0,
                open_positions=len(self.portfolio.positions),
                stats=self.portfolio.stats(),
            )
            return

        await self._log_accuweather_snapshots(markets)

        now_ts = datetime.now(UTC).timestamp()
        if now_ts - self.last_forecast_refresh > FORECAST_REFRESH_SECONDS:
            try:
                await self._refresh_forecasts(markets)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                self.logger.warning(f"Forecast refresh failed: {exc}. Using last cached forecasts.")
            self.last_forecast_refresh = now_ts

        signals = generate_signals(markets, self.forecasts, self.portfolio.current_cash)
        missed_summary = summarize_top_missed_edges(markets, self.forecasts, self.portfolio.current_cash)

        deployed = 0.0
        trades_executed = 0
        skipped_position_limit = 0
        skipped_daily_exposure = 0
        skipped_execution = 0
        skipped_already_held = 0
        for signal in signals:
            if self.portfolio.holds_market_bucket(signal.market_id, signal.bucket):
                self.logger.log_signal(signal.to_dict(), "already_held")
                skipped_already_held += 1
                continue
            if self._existing_positions(signal.market_id) >= MAX_POSITIONS_PER_MARKET:
                self.logger.log_signal(signal.to_dict(), "skip_position_limit")
                skipped_position_limit += 1
                continue
            if deployed + signal.size_usd > MAX_DAILY_EXPOSURE:
                self.logger.log_signal(signal.to_dict(), "skip_daily_exposure")
                skipped_daily_exposure += 1
                continue
            result = self.order_manager.place_order(signal.to_dict())
            if result.status.startswith("skipped"):
                self.logger.log_signal(signal.to_dict(), result.status)
                skipped_execution += 1
                continue
            position = self.portfolio.open_position(signal.to_dict(), result.fill_price)
            deployed += signal.size_usd
            trades_executed += 1
            self.logger.log_signal(signal.to_dict(), "trade")
            self.logger.info(
                f"PAPER TRADE {signal.city} {signal.date} {signal.bucket} "
                f"{signal.side} ${signal.size_usd:.2f} @ {result.fill_price:.3f}"
            )
            # Keep unresolved in paper mode until external resolver marks outcome.
            _ = position

        try:
            await self._resolve_matured_positions()
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            self.logger.warning(f"Position resolution check failed: {exc}. Will retry next scan.")

        intraday_adjusted = sum(
            1
            for station_days in self.forecasts.values()
            for bundle in station_days.values()
            if bundle.get("intraday_adjusted")
        )
        stats = self.portfolio.stats()
        self.logger.info(
            "HEARTBEAT "
            f"discovered={discovery_stats.get('discovered_markets', 0)} "
            f"hydrated={len(markets)} "
            f"signals={len(signals)} "
            f"trades_executed={trades_executed} "
            f"skip_already_held={skipped_already_held} "
            f"skip_position_limit={skipped_position_limit} "
            f"skip_daily_exposure={skipped_daily_exposure} "
            f"skip_execution={skipped_execution} "
            f"open_positions={len(self.portfolio.positions)} "
            f"cash={self.portfolio.current_cash:.2f} "
            f"exposure={self.portfolio.active_exposure():.2f} "
            f"intraday_adjusted={intraday_adjusted} "
            f"missed={missed_summary}"
        )
        render_dashboard(
            bankroll=self.portfolio.initial_bankroll,
            cash=self.portfolio.current_cash,
            active_exposure=self.portfolio.active_exposure(),
            signals_count=len(signals),
            open_positions=len(self.portfolio.positions),
            stats=stats,
        )

    async def _refresh_forecasts(self, markets: list[dict]) -> None:
        grouped: dict[tuple[str, str], set[str]] = {}
        observed_display_cache: dict[str, float | None] = {}
        for market in markets:
            station_icao = market["station_icao"]
            date_str = market["date"]
            key = (station_icao, date_str)
            grouped.setdefault(key, set()).update(market["buckets"].keys())

        for (station_icao, date_str), bucket_set in grouped.items():
            target_date = datetime.fromisoformat(date_str).date()
            forecast_bundle = await self.forecast_client.get_station_forecast(
                station_icao=station_icao,
                target_date=target_date,
                bucket_labels=sorted(bucket_set),
            )
            await self._apply_intraday_observed_high_adjustment(
                station_icao=station_icao,
                target_date=target_date,
                forecast_bundle=forecast_bundle,
                observed_display_cache=observed_display_cache,
            )
            self.forecasts.setdefault(station_icao, {})[date_str] = forecast_bundle

    async def _log_accuweather_snapshots(self, markets: list[dict]) -> None:
        if self.accuweather_client is None:
            return
        active_stations = sorted({m["station_icao"] for m in markets if m.get("station_icao") in STATIONS})
        if not active_stations:
            return
        logged = 0
        for station_icao in active_stations:
            station = STATIONS[station_icao]
            try:
                snapshot = await self.accuweather_client.get_daily_high_snapshot(station)
            except httpx.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", "n/a")
                self.logger.warning(f"AccuWeather fetch failed for {station_icao}: status={status}")
                continue
            if snapshot is None:
                continue
            self.logger.log_accuweather_snapshot(
                station_icao=snapshot.station_icao,
                city=snapshot.city,
                forecast_date=snapshot.forecast_date,
                forecast_high=snapshot.forecast_high,
                unit=snapshot.unit,
                model_source=snapshot.model_source,
            )
            logged += 1
        if logged:
            self.logger.info(f"ACCUWEATHER snapshots_logged={logged}")

    async def _apply_intraday_observed_high_adjustment(
        self,
        station_icao: str,
        target_date: date,
        forecast_bundle: dict,
        observed_display_cache: dict[str, float | None],
    ) -> None:
        station = STATIONS.get(station_icao)
        if not station:
            return
        probs = forecast_bundle.get("probs")
        if not isinstance(probs, dict) or not probs:
            return

        local_now = datetime.now(ZoneInfo(station["timezone"])).date()
        if target_date != local_now:
            return

        if station_icao in observed_display_cache:
            observed_display = observed_display_cache[station_icao]
        else:
            try:
                observed_display, _ = await self.forecast_client.latest_observed_display_temp(station_icao)
            except Exception:
                observed_display = None
            observed_display_cache[station_icao] = observed_display

        if observed_display is None:
            return

        day_key = target_date.isoformat()
        current_high = self.intraday_observed_highs.get(station_icao, {}).get(day_key, float("-inf"))
        observed_high = max(current_high, float(observed_display))
        self.intraday_observed_highs.setdefault(station_icao, {})[day_key] = observed_high

        adjusted: dict[str, float] = {}
        for bucket, prob in probs.items():
            bounds = _parse_bucket_bounds(bucket)
            if bounds is None:
                adjusted[bucket] = float(prob)
                continue
            low, high = bounds
            if high is not None and high < observed_high:
                adjusted[bucket] = 0.0
            else:
                adjusted[bucket] = float(prob)
        total = sum(adjusted.values())
        if total > 0:
            forecast_bundle["probs"] = {k: (v / total) for k, v in adjusted.items()}
            forecast_bundle["observed_high_display"] = observed_high
            forecast_bundle["intraday_adjusted"] = True

    async def _trigger_run(self, label: str) -> None:
        """Force a forecast refresh and full scan at a model-run boundary."""
        self.last_forecast_refresh = 0.0  # bypass cache — always pull fresh model data
        self.logger.info(f"MODEL_RUN_TRIGGER source={label} forcing full forecast refresh")
        await self.run_once()

    async def run_forever(self) -> None:
        """
        Event-driven main loop.

        Sleeps until the next model-run availability window, fires immediately
        when fresh Open-Meteo data is expected, then runs fast follow-up scans
        for MODEL_RUN_BOOST_WINDOW_MINUTES to catch market repricing lag.
        Between windows the bot sleeps entirely — no wasted API quota.
        """
        self.logger.info("Starting paper trader — event-driven scheduler active.")
        try:
            # Immediate startup scan so the bot is live right away
            await self._trigger_run("STARTUP")

            while True:
                next_dt, label = _next_model_run_trigger(datetime.now(UTC))
                sleep_secs = max(0.0, (next_dt - datetime.now(UTC)).total_seconds())
                self.logger.info(
                    f"SCHEDULER next={label} "
                    f"at={next_dt.strftime('%Y-%m-%d %H:%M UTC')} "
                    f"sleep={sleep_secs/60:.1f}min"
                )
                await asyncio.sleep(sleep_secs)

                # Trigger: force-refresh forecasts and scan immediately
                await self._trigger_run(label)

                # Boost window: fast follow-up scans to catch market repricing lag
                boost_end = datetime.now(UTC) + timedelta(minutes=MODEL_RUN_BOOST_WINDOW_MINUTES)
                scan_num = 1
                while datetime.now(UTC) < boost_end:
                    await asyncio.sleep(MODEL_RUN_BOOST_SCAN_INTERVAL_SECONDS)
                    self.logger.info(
                        f"BOOST_SCAN source={label} scan={scan_num} "
                        f"remaining={int((boost_end - datetime.now(UTC)).total_seconds() / 60)}min"
                    )
                    await self.run_once()
                    scan_num += 1

        finally:
            await self.close()

    def _existing_positions(self, market_id: str) -> int:
        return sum(1 for p in self.portfolio.positions if p.market_id == market_id)

    async def _resolve_matured_positions(self) -> None:
        today = date.today()
        for position in list(self.portfolio.positions):
            target = datetime.fromisoformat(position.date).date()
            if target >= today:
                continue
            station_icao = position.station_icao
            station = STATIONS.get(station_icao)
            if not station:
                continue
            observed = await self.wu_client.get_daily_high(station["wu_url"], target)
            if observed is None:
                continue

            won = _is_winning_bucket(position.bucket, observed)
            closed = self.portfolio.resolve_position(position, won)
            row = {
                "city": closed.city,
                "date": closed.date,
                "bucket": closed.bucket,
                "side": closed.side,
                "forecast_prob": 0.0,
                "market_prob": closed.fill_price if closed.side == "BUY_YES" else (1.0 - closed.fill_price),
                "edge": 0.0,
                "size_usd": closed.cost,
                "fill_price": closed.fill_price,
            }
            self.logger.log_trade(row, "won" if won else "lost", closed.pnl)
            self.logger.info(
                f"RESOLVED {closed.city} {closed.date} {closed.bucket} "
                f"{'WIN' if won else 'LOSS'} pnl={closed.pnl:.2f}"
            )


def _parse_bucket_bounds(bucket_label: str) -> tuple[float, float | None] | None:
    clean = bucket_label.replace("°F", "").replace("°C", "").strip()
    try:
        if "+" in clean:
            return float(clean.replace("+", "")), None
        if "-" in clean:
            left, right = clean.split("-", 1)
            return float(left.strip()), float(right.strip())
        value = float(clean)
        return value, value
    except ValueError:
        return None


def _is_winning_bucket(bucket_label: str, observed_temp: int) -> bool:
    clean = bucket_label.replace("°F", "").replace("°C", "").strip()
    if "+" in clean:
        lower = int(clean.replace("+", ""))
        return observed_temp >= lower
    if "-" in clean:
        left, right = clean.split("-", 1)
        return int(left.strip()) <= observed_temp <= int(right.strip())
    return observed_temp == int(clean)


def _next_model_run_trigger(now_utc: datetime) -> tuple[datetime, str]:
    """
    Return the next (datetime, label) from MODEL_RUN_TRIGGER_TIMES_UTC.

    Searches today and tomorrow so we always find a future trigger even if
    called in the last seconds before midnight UTC.
    """
    candidates: list[tuple[datetime, str]] = []
    for delta_days in (0, 1):
        base = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=delta_days)
        for hour, minute, label in MODEL_RUN_TRIGGER_TIMES_UTC:
            candidate = base.replace(hour=hour, minute=minute)
            # Only include strictly future triggers
            if candidate > now_utc:
                candidates.append((candidate, label))
    candidates.sort(key=lambda x: x[0])
    return candidates[0]
