"""Paper-trading runner for end-to-end validation."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
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
from strategy.signals import (
    generate_signals,
    generate_mk2_ace_signals,
    generate_purdey_cavendish_signals,
    generate_top2_shadow_signals,
    summarize_top_missed_edges,
)


class ShadowTrader:
    """Fully independent paper book for one TOP2 shadow variant (2A / 2B / 2C).

    Receives pre-fetched markets + forecasts from the main PaperTrader so it adds
    zero extra API calls.  Each instance maintains its own Portfolio (separate
    positions file), its own BotLogger (separate CSV directory), and its own
    OrderManager — giving a fully isolated P&L track comparable to the live book.
    """

    _SLUGS: dict[str, str] = {
        "TOP2_EQUAL":    "shadow_2a",
        "TOP2_COND":     "shadow_2b",
        "TOP2_PROP":     "shadow_2c",
        "CAVENDISH_MK1": "shadow_cavendish",
        "CAVENDISH_MK3": "shadow_cavendish3",
    }

    def __init__(self, variant: str) -> None:
        if variant not in self._SLUGS:
            raise ValueError(f"Unknown shadow variant: {variant}")
        self.variant = variant
        slug = self._SLUGS[variant]
        self.logger = BotLogger(output_dir=f"logs/{slug}")
        self.portfolio = Portfolio(
            initial_bankroll=INITIAL_BANKROLL,
            positions_path=f"data/positions_{slug}.json",
        )
        self.order_manager = OrderManager(live_trading=False)
        self._log = logging.getLogger(f"weather-bot.{slug}")

    _PURDEY_CAVENDISH_VARIANTS = {"CAVENDISH_MK1"}
    _MK2_ACE_VARIANTS = {"CAVENDISH_MK3"}

    def run_once(self, markets: list[dict], forecasts: dict, bankroll: float) -> None:
        """Execute one scan using shared market + forecast data."""
        if self.variant in self._MK2_ACE_VARIANTS:
            all_shadows = generate_mk2_ace_signals(markets, forecasts, bankroll)
        elif self.variant in self._PURDEY_CAVENDISH_VARIANTS:
            all_shadows = generate_purdey_cavendish_signals(markets, forecasts, bankroll)
        else:
            all_shadows = generate_top2_shadow_signals(markets, forecasts, bankroll)
        signals = [s for s in all_shadows if s.strategy == self.variant]

        deployed = 0.0
        executed = 0
        for signal in signals:
            if self.portfolio.holds_market_bucket(signal.market_id, signal.bucket):
                continue
            if self._existing_positions(signal.market_id) >= MAX_POSITIONS_PER_MARKET:
                continue
            if deployed + signal.size_usd > MAX_DAILY_EXPOSURE:
                continue
            result = self.order_manager.place_order(signal.to_dict())
            if result.status.startswith("skipped"):
                continue
            self.portfolio.open_position(signal.to_dict(), result.fill_price)
            deployed += signal.size_usd
            executed += 1
            self.logger.log_signal(signal.to_dict(), "trade")
            self._log.info(
                f"SHADOW_TRADE {self.variant} {signal.city} {signal.date} "
                f"{signal.bucket} {signal.side} ${signal.size_usd:.2f} @ {result.fill_price:.3f}"
            )

        self._log.info(
            f"SHADOW_HEARTBEAT variant={self.variant} executed={executed} "
            f"open_positions={len(self.portfolio.positions)} "
            f"cash={self.portfolio.current_cash:.2f} "
            f"exposure={self.portfolio.active_exposure():.2f}"
        )

    def _existing_positions(self, market_id: str) -> int:
        return sum(1 for p in self.portfolio.positions if p.market_id == market_id)

    async def resolve_matured_positions(self, wu_client: "WeatherUndergroundClient") -> None:
        today = date.today()
        for position in list(self.portfolio.positions):
            target = datetime.fromisoformat(position.date).date()
            if target >= today:
                continue
            station = STATIONS.get(position.station_icao)
            if not station:
                continue
            observed = await wu_client.get_daily_high(station["wu_url"], target)
            if observed is None:
                continue
            won = _is_winning_bucket(position.bucket, observed)
            closed = self.portfolio.resolve_position(position, won)
            row = {
                "city": closed.city, "date": closed.date, "bucket": closed.bucket,
                "side": closed.side, "forecast_prob": closed.forecast_prob,
                "market_prob": closed.market_prob, "edge": closed.edge,
                "size_usd": closed.cost, "fill_price": closed.fill_price,
                "strategy": closed.strategy, "station_icao": closed.station_icao,
                "model_values_json": closed.model_values_json,
            }
            self.logger.log_trade(row, "won" if won else "lost", closed.pnl)
            self._log.info(
                f"SHADOW_RESOLVED {self.variant} {closed.city} {closed.date} "
                f"{closed.bucket} {'WIN' if won else 'LOSS'} pnl={closed.pnl:.2f}"
            )


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
        # Latest hydrated markets — updated after every successful discovery run.
        # Shared with _run_shadows_from_cache so shadows can execute independently.
        self._cached_markets: list[dict] = []
        self.shadows = [
            ShadowTrader("TOP2_EQUAL"),
            ShadowTrader("TOP2_COND"),
            ShadowTrader("TOP2_PROP"),
            ShadowTrader("PURDEY_MK1"),
            ShadowTrader("CAVENDISH_MK1"),
            ShadowTrader("PURDEY_MK2"),
            ShadowTrader("CAVENDISH_MK2"),
            ShadowTrader("ACE"),
            ShadowTrader("PROPS_KELLY"),
        ]

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

        # Store latest markets so the decoupled shadow loop can use them.
        self._cached_markets = markets

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

        # Cache signals so the price scanner can re-check prices between model runs.
        _write_signal_cache(signals, markets)

        deployed = 0.0
        trades_executed = 0
        skipped_position_limit = 0
        skipped_daily_exposure = 0
        skipped_execution = 0
        skipped_already_held = 0
        for signal in signals:
            # Shadow CONVICTION signals are never executed — just logged for A/B scoring.
            if signal.strategy == "CONVICTION":
                self.logger.log_signal(signal.to_dict(), "conviction_signal")
                continue

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

            # Log individual model predictions for the accuracy tracker
            det_mv = forecast_bundle.get("det_model_values")
            if det_mv:
                from strategy.model_weights import log_predictions
                log_predictions(station_icao, date_str, det_mv)

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

    def run_shadows_now(self) -> None:
        """Execute all shadow traders synchronously using currently cached data.
        Safe to call at any time — skips silently if no market data cached yet."""
        if not self._cached_markets or not self.forecasts:
            self.logger.info("SHADOW_SKIP no cached markets or forecasts yet")
            return
        for shadow in self.shadows:
            try:
                shadow.run_once(
                    self._cached_markets,
                    self.forecasts,
                    shadow.portfolio.current_cash,
                )
            except Exception as exc:
                self.logger.warning(f"Shadow {shadow.variant} failed: {exc}")

    async def _continuous_resolver(self) -> None:
        """Resolve matured positions as soon as actual temps become available.

        Runs every 30 min, checking ALL position files (main + shadows) for
        trades whose target date has passed. Uses IEM ASOS for actual temps —
        the same source Polymarket uses to settle.

        This replaces the old daily cron approach so resolution happens within
        ~30 min of data availability, regardless of timezone.
        """
        import csv
        import json
        import subprocess
        from scripts.daily_resolver import (
            _fetch_iem_high, _parse_bucket_win, _compute_pnl,
            RESOLVED_CSV, RESOLVED_HEADER, _ensure_resolved_csv,
            _SHADOW_STRATEGIES,
        )
        from strategy.model_weights import log_actual_temperature

        _RESOLVER_INTERVAL = 1800  # 30 min

        await asyncio.sleep(120)  # let bot start up first

        while True:
            try:
                _ensure_resolved_csv()
                today_str = date.today().isoformat()

                # Build set of already-resolved keys
                resolved_keys: set[tuple[str, str, str, str, str]] = set()
                if RESOLVED_CSV.exists():
                    with RESOLVED_CSV.open(encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            strategy = row.get("strategy", "")
                            sk = strategy if strategy in _SHADOW_STRATEGIES else ""
                            resolved_keys.add((
                                row.get("target_date", ""),
                                row.get("city", ""),
                                row.get("bucket", ""),
                                row.get("side", "BUY_YES"),
                                sk,
                            ))

                # Collect pending positions from all files
                _ROOT = Path(__file__).resolve().parents[1]
                _ALL_FILES: list[tuple[Path, str]] = [
                    (_ROOT / "data" / "positions.json", ""),
                    (_ROOT / "data" / "positions_shadow_2a.json", "TOP2_EQUAL"),
                    (_ROOT / "data" / "positions_shadow_2b.json", "TOP2_COND"),
                    (_ROOT / "data" / "positions_shadow_2c.json", "TOP2_PROP"),
                    (_ROOT / "data" / "positions_shadow_cavendish.json", "CAVENDISH_MK1"),
                    (_ROOT / "data" / "positions_shadow_cavendish3.json", "CAVENDISH_MK3"),
                    (_ROOT / "data" / "positions_shadow_props_kelly.json", "PROPS_KELLY"),
                ]

                pending: list[dict] = []
                for pos_path, strat_override in _ALL_FILES:
                    if not pos_path.exists():
                        continue
                    try:
                        raw = json.loads(pos_path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    for p in raw:
                        td = p.get("date", "")
                        if not td or td >= today_str:
                            continue
                        strategy = strat_override or p.get("strategy", "PAPER")
                        sk = strategy if strategy in _SHADOW_STRATEGIES else ""
                        key = (td, p.get("city", ""), p.get("bucket", ""), p.get("side", "BUY_YES"), sk)
                        if key in resolved_keys:
                            continue
                        pending.append({**p, "strategy": strategy})

                if not pending:
                    await asyncio.sleep(_RESOLVER_INTERVAL)
                    continue

                self.logger.info(f"AUTO_RESOLVER checking {len(pending)} pending positions")

                # Fetch actuals from IEM
                iem_cache: dict[tuple[str, str], int | None] = {}
                resolved_count = 0
                new_rows: list[dict] = []

                async with httpx.AsyncClient() as http:
                    for p in pending:
                        td = p["date"]
                        city = p.get("city", "")
                        bucket = p.get("bucket", "")
                        station_icao = p.get("station_icao", "")
                        side = p.get("side", "BUY_YES")
                        strategy = p.get("strategy", "")

                        station_cfg = STATIONS.get(station_icao)
                        if not station_cfg:
                            for icao, cfg in STATIONS.items():
                                if cfg.get("market_label", "").lower() == city.lower():
                                    station_cfg = cfg
                                    station_icao = icao
                                    break
                        if not station_cfg:
                            continue

                        iem_net = station_cfg.get("iem_network")
                        iem_stn = station_cfg.get("iem_station")
                        if not iem_net or not iem_stn:
                            continue

                        res_unit = station_cfg.get("resolution_unit", "F")
                        cache_key = (iem_stn, td)

                        if cache_key not in iem_cache:
                            target_d = date.fromisoformat(td)
                            actual = await _fetch_iem_high(http, iem_net, iem_stn, res_unit, target_d)
                            iem_cache[cache_key] = actual
                        else:
                            actual = iem_cache[cache_key]

                        if actual is None:
                            continue

                        try:
                            log_actual_temperature(station_icao, td, float(actual))
                        except Exception:
                            pass

                        won = _parse_bucket_win(bucket, actual)
                        if side == "BUY_NO":
                            won = not won

                        entry_price = float(p.get("fill_price", p.get("entry_price", 0)) or 0)
                        size_usd = float(p.get("cost", p.get("size_usd", 0)) or 0)
                        ev_per_bet = float(p.get("ev_at_entry", p.get("ev_per_bet", 0)) or 0)

                        pnl = _compute_pnl(side, entry_price, size_usd, won)
                        outcome = "WIN" if won else "LOSS"

                        miss_distance = ""
                        try:
                            mv = json.loads(p.get("model_values_json") or "{}")
                            if mv:
                                miss_distance = round(actual - sum(mv.values()) / len(mv), 2)
                        except (json.JSONDecodeError, ZeroDivisionError):
                            pass

                        forecast_prob = float(p.get("forecast_prob", 0) or 0)
                        edge = float(p.get("edge", 0) or 0)
                        roi_pct = round(pnl / size_usd * 100, 2) if size_usd > 0 else 0.0

                        result_row = {
                            "resolved_at": datetime.now(UTC).isoformat(),
                            "target_date": td,
                            "city": city,
                            "station_icao": station_icao,
                            "bucket": bucket,
                            "side": side,
                            "entry_price": entry_price,
                            "size_usd": size_usd,
                            "ev_per_bet": ev_per_bet,
                            "spread_colour": p.get("spread_colour", ""),
                            "det_spread": p.get("det_spread", ""),
                            "model_values_json": p.get("model_values_json", "{}"),
                            "actual_temp": actual,
                            "outcome": outcome,
                            "pnl_usd": pnl,
                            "miss_distance": miss_distance,
                            "signal_timestamp": p.get("timestamp", ""),
                            "strategy": strategy,
                            "forecast_prob": forecast_prob,
                            "edge": edge,
                            "roi_pct": roi_pct,
                            "days_ahead": p.get("days_ahead", ""),
                            "kelly_fraction_used": p.get("kelly_fraction_used", ""),
                        }

                        with RESOLVED_CSV.open("a", newline="", encoding="utf-8") as f:
                            csv.DictWriter(f, fieldnames=RESOLVED_HEADER).writerow(result_row)

                        new_rows.append(result_row)
                        resolved_count += 1
                        emoji = "✅" if won else "❌"
                        self.logger.info(
                            f"AUTO_RESOLVED {emoji} [{strategy}] {city} {td} {bucket} "
                            f"actual={actual} {outcome} pnl={pnl:+.2f}"
                        )

                if resolved_count > 0:
                    self.logger.info(f"AUTO_RESOLVER resolved {resolved_count} positions")
                    # Prune resolved positions from all files
                    from scripts.daily_resolver import prune_expired_positions
                    prune_expired_positions()
                    # Push data to git so dashboard updates
                    try:
                        git_script = _ROOT / "scripts" / "git_push_data.sh"
                        if git_script.exists():
                            subprocess.run(
                                ["bash", str(git_script)],
                                cwd=str(_ROOT.parent),
                                timeout=120,
                                capture_output=True,
                            )
                            self.logger.info("AUTO_RESOLVER git push complete")
                    except Exception as exc:
                        self.logger.warning(f"AUTO_RESOLVER git push failed: {exc}")

            except Exception as exc:
                self.logger.error(f"AUTO_RESOLVER error: {exc}")

            await asyncio.sleep(_RESOLVER_INTERVAL)

    async def _run_shadows_from_cache(self) -> None:
        """Independent shadow execution loop — runs every 5 min using cached data.

        Decoupled from the main forecast scan so shadows execute even when the
        scan is in the middle of a slow forecast refresh.  On first startup,
        waits up to 10 min for the initial discovery to populate the cache.
        """
        # Wait for initial discovery (max 10 min, check every 15s)
        for _ in range(40):
            if self._cached_markets and self.forecasts:
                break
            await asyncio.sleep(15)
        else:
            self.logger.warning("SHADOW_LOOP no data after 10 min — will keep retrying")

        while True:
            self.run_shadows_now()
            await asyncio.sleep(300)  # re-run every 5 min

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
        asyncio.get_event_loop().create_task(self._run_shadows_from_cache())
        asyncio.get_event_loop().create_task(self._continuous_resolver())
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
                "forecast_prob": closed.forecast_prob,
                "market_prob": closed.market_prob,
                "edge": closed.edge,
                "size_usd": closed.cost,
                "fill_price": closed.fill_price,
                "strategy": closed.strategy,
                "station_icao": closed.station_icao,
                "model_values_json": closed.model_values_json,
            }
            self.logger.log_trade(row, "won" if won else "lost", closed.pnl)
            self.logger.info(
                f"RESOLVED {closed.city} {closed.date} {closed.bucket} "
                f"{'WIN' if won else 'LOSS'} pnl={closed.pnl:.2f}"
            )

        for shadow in self.shadows:
            try:
                await shadow.resolve_matured_positions(self.wu_client)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                self.logger.warning(f"Shadow resolution failed for {shadow.variant}: {exc}")


_CACHED_SIGNALS_PATH = Path(__file__).resolve().parent.parent / "data" / "cached_signals.json"


def _write_signal_cache(signals: list, markets: list[dict]) -> None:
    """Write all evaluated signals to disk so the price scanner can poll prices between model runs.

    Includes every signal above the alpha threshold (LADDER + CONVICTION + SINGLE),
    keyed by token_id. The price scanner reads this, fetches live ask prices, and fires
    a trade when EV = model_prob - live_ask exceeds the threshold.
    """
    # Build a market end_date lookup so we can store expiry in the cache
    end_dates: dict[str, str] = {}
    for m in markets:
        for bucket, info in m.get("buckets", {}).items():
            tid = info.get("yes_token_id", "")
            if tid:
                end_dates[tid] = str(m.get("end_date_iso", ""))

    cache_entries: list[dict] = []
    seen: set[str] = set()
    for sig in signals:
        tid = sig.token_id
        if not tid or tid in seen:
            continue
        seen.add(tid)
        cache_entries.append({
            "condition_id":    sig.market_id,
            "token_id":        tid,
            "station_icao":    sig.station_icao,
            "city":            sig.city,
            "date":            sig.date,
            "bucket":          sig.bucket,
            "side":            sig.side,
            "model_prob":      round(sig.forecast_prob, 4),
            "market_prob_at_scan": round(sig.market_prob, 4),
            "edge_at_scan":    round(sig.edge, 4),
            "ev_per_bet":      round(sig.ev_per_bet, 4),
            "spread_colour":   sig.spread_colour,
            "det_spread":      sig.det_spread,
            "model_values_json": sig.model_values_json,
            "days_ahead":      sig.days_ahead,
            "end_date_iso":    end_dates.get(tid, ""),
            "strategy":        sig.strategy,
        })

    payload = {
        "computed_at": datetime.now(UTC).isoformat(),
        "signal_count": len(cache_entries),
        "signals": cache_entries,
    }
    try:
        _CACHED_SIGNALS_PATH.write_text(
            __import__("json").dumps(payload, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        import logging
        logging.getLogger("weather-bot.paper-trader").warning(
            "Could not write signal cache: %s", exc
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
