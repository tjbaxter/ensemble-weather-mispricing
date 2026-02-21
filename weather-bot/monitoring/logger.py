"""Structured logging to console + CSV outputs."""

from __future__ import annotations

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path


class BotLogger:
    def __init__(self, output_dir: str = "logs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trades_file = self.output_dir / "trades.csv"
        self.signals_file = self.output_dir / "signals.csv"
        self.accuweather_file = self.output_dir / "accuweather_snapshots.csv"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        # Prevent third-party HTTP client logs from leaking query-string secrets.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        self.log = logging.getLogger("weather-bot")
        self._init_csv_headers()

    def _init_csv_headers(self) -> None:
        if not self.trades_file.exists():
            with self.trades_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "city",
                        "date",
                        "bucket",
                        "side",
                        "forecast_prob",
                        "market_prob",
                        "edge",
                        "size_usd",
                        "fill_price",
                        "outcome",
                        "pnl",
                    ]
                )
        if not self.signals_file.exists():
            with self.signals_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "city",
                        "date",
                        "bucket",
                        "forecast_prob",
                        "market_prob",
                        "edge",
                        "action_taken",
                    ]
                )
        if not self.accuweather_file.exists():
            with self.accuweather_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "station_icao",
                        "city",
                        "forecast_date",
                        "forecast_high",
                        "unit",
                        "model_source",
                    ]
                )

    def info(self, msg: str) -> None:
        self.log.info(msg)

    def warning(self, msg: str) -> None:
        self.log.warning(msg)

    def error(self, msg: str) -> None:
        self.log.error(msg)

    def log_signal(self, signal: dict, action_taken: str) -> None:
        ts = datetime.now(UTC).isoformat()
        with self.signals_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ts,
                    signal["city"],
                    signal["date"],
                    signal["bucket"],
                    signal["forecast_prob"],
                    signal["market_prob"],
                    signal["edge"],
                    action_taken,
                ]
            )

    def log_trade(self, trade: dict, outcome: str, pnl: float) -> None:
        ts = datetime.now(UTC).isoformat()
        with self.trades_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ts,
                    trade["city"],
                    trade["date"],
                    trade["bucket"],
                    trade["side"],
                    trade["forecast_prob"],
                    trade["market_prob"],
                    trade["edge"],
                    trade["size_usd"],
                    trade.get("fill_price", ""),
                    outcome,
                    pnl,
                ]
            )

    def log_accuweather_snapshot(
        self,
        station_icao: str,
        city: str,
        forecast_date: str,
        forecast_high: float,
        unit: str,
        model_source: str = "accuweather_daily_1day",
    ) -> None:
        ts = datetime.now(UTC).isoformat()
        with self.accuweather_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ts,
                    station_icao,
                    city,
                    forecast_date,
                    forecast_high,
                    unit,
                    model_source,
                ]
            )
