"""Persistent trade observability storage (SQLite + JSONL mirror)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


class TradeAuditStore:
    """Append-only audit store for every trade decision and outcome."""

    def __init__(
        self,
        db_path: str = "data/trade_observability.db",
        jsonl_path: str = "data/trade_observability.jsonl",
    ) -> None:
        self.db_path = Path(db_path)
        self.jsonl_path = Path(jsonl_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_decisions (
                    event_id TEXT PRIMARY KEY,
                    event_ts TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT,

                    city TEXT,
                    station_icao TEXT,
                    target_date TEXT,
                    strategy TEXT,
                    market_id TEXT,
                    token_id TEXT,
                    bucket TEXT,
                    side TEXT,

                    forecast_prob REAL,
                    market_prob REAL,
                    edge REAL,
                    requested_size_usd REAL,
                    approved_size_usd REAL,
                    fill_price REAL,

                    days_ahead INTEGER,
                    hours_to_resolution REAL,
                    temporal_discount REAL,
                    spread_colour TEXT,
                    det_spread REAL,
                    ev_per_bet REAL,

                    risk_skipped INTEGER,
                    risk_quality_mult REAL,
                    risk_position_cap_usd REAL,
                    risk_daily_budget_usd REAL,
                    risk_reason TEXT,

                    execution_status TEXT,
                    execution_details_json TEXT,

                    model_values_json TEXT,
                    forecast_bundle_json TEXT,
                    market_snapshot_json TEXT,
                    context_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_decisions_date ON trade_decisions(target_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_decisions_city ON trade_decisions(city)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_decisions_action ON trade_decisions(action)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_decisions_strategy ON trade_decisions(strategy)"
            )

    def new_run_id(self, engine: str) -> str:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{engine}-{ts}-{uuid4().hex[:8]}"

    def log_event(
        self,
        *,
        run_id: str,
        engine: str,
        action: str,
        signal: dict[str, Any] | None = None,
        reason: str = "",
        risk_decision: dict[str, Any] | None = None,
        execution_result: dict[str, Any] | None = None,
        forecast_bundle: dict[str, Any] | None = None,
        market_snapshot: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        event_id = uuid4().hex
        event_ts = datetime.now(UTC).isoformat()
        sig = signal or {}
        risk = risk_decision or {}
        exe = execution_result or {}
        f_bundle = forecast_bundle or {}
        m_snap = market_snapshot or {}
        ctx = context or {}

        model_values_json = sig.get("model_values_json", "{}")
        if not isinstance(model_values_json, str):
            model_values_json = json.dumps(model_values_json, separators=(",", ":"))

        row = {
            "event_id": event_id,
            "event_ts": event_ts,
            "run_id": run_id,
            "engine": engine,
            "action": action,
            "reason": reason,
            "city": sig.get("city", ""),
            "station_icao": sig.get("station_icao", ""),
            "target_date": sig.get("date", ""),
            "strategy": sig.get("strategy", ""),
            "market_id": sig.get("market_id", ""),
            "token_id": sig.get("token_id", ""),
            "bucket": sig.get("bucket", ""),
            "side": sig.get("side", ""),
            "forecast_prob": _as_float(sig.get("forecast_prob")),
            "market_prob": _as_float(sig.get("market_prob")),
            "edge": _as_float(sig.get("edge")),
            "requested_size_usd": _as_float(ctx.get("requested_size_usd", sig.get("size_usd"))),
            "approved_size_usd": _as_float(ctx.get("approved_size_usd", sig.get("size_usd"))),
            "fill_price": _as_float(ctx.get("fill_price", exe.get("fill_price"))),
            "days_ahead": _as_int(sig.get("days_ahead")),
            "hours_to_resolution": _as_float(sig.get("hours_to_resolution")),
            "temporal_discount": _as_float(sig.get("temporal_discount")),
            "spread_colour": str(sig.get("spread_colour", "")),
            "det_spread": _as_float(sig.get("det_spread")),
            "ev_per_bet": _as_float(sig.get("ev_per_bet")),
            "risk_skipped": 1 if bool(risk.get("skipped", False)) else 0,
            "risk_quality_mult": _as_float(risk.get("quality_mult")),
            "risk_position_cap_usd": _as_float(risk.get("position_cap_usd")),
            "risk_daily_budget_usd": _as_float(risk.get("daily_budget_usd")),
            "risk_reason": str(risk.get("reason", "")),
            "execution_status": str(exe.get("status", "")),
            "execution_details_json": json.dumps(exe.get("details", {}), separators=(",", ":")),
            "model_values_json": model_values_json,
            "forecast_bundle_json": json.dumps(f_bundle, separators=(",", ":")),
            "market_snapshot_json": json.dumps(m_snap, separators=(",", ":")),
            "context_json": json.dumps(ctx, separators=(",", ":")),
        }

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trade_decisions (
                    event_id, event_ts, run_id, engine, action, reason,
                    city, station_icao, target_date, strategy, market_id, token_id, bucket, side,
                    forecast_prob, market_prob, edge, requested_size_usd, approved_size_usd, fill_price,
                    days_ahead, hours_to_resolution, temporal_discount, spread_colour, det_spread, ev_per_bet,
                    risk_skipped, risk_quality_mult, risk_position_cap_usd, risk_daily_budget_usd, risk_reason,
                    execution_status, execution_details_json, model_values_json, forecast_bundle_json,
                    market_snapshot_json, context_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["event_id"],
                    row["event_ts"],
                    row["run_id"],
                    row["engine"],
                    row["action"],
                    row["reason"],
                    row["city"],
                    row["station_icao"],
                    row["target_date"],
                    row["strategy"],
                    row["market_id"],
                    row["token_id"],
                    row["bucket"],
                    row["side"],
                    row["forecast_prob"],
                    row["market_prob"],
                    row["edge"],
                    row["requested_size_usd"],
                    row["approved_size_usd"],
                    row["fill_price"],
                    row["days_ahead"],
                    row["hours_to_resolution"],
                    row["temporal_discount"],
                    row["spread_colour"],
                    row["det_spread"],
                    row["ev_per_bet"],
                    row["risk_skipped"],
                    row["risk_quality_mult"],
                    row["risk_position_cap_usd"],
                    row["risk_daily_budget_usd"],
                    row["risk_reason"],
                    row["execution_status"],
                    row["execution_details_json"],
                    row["model_values_json"],
                    row["forecast_bundle_json"],
                    row["market_snapshot_json"],
                    row["context_json"],
                ),
            )

        # JSONL mirror for easy grep/forensics and git diffs.
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

        return event_id


def market_snapshot_for_signal(signal: dict[str, Any], markets: list[dict]) -> dict[str, Any]:
    station = str(signal.get("station_icao", ""))
    target_date = str(signal.get("date", ""))
    condition_id = str(signal.get("market_id", ""))
    out: dict[str, Any] = {"station_icao": station, "date": target_date, "buckets": {}}
    for m in markets:
        if m.get("station_icao") != station or m.get("date") != target_date:
            continue
        if condition_id and m.get("condition_id") != condition_id:
            continue
        for bucket, info in (m.get("buckets") or {}).items():
            out["buckets"][bucket] = {
                "price": info.get("price"),
                "yes_token_id": info.get("yes_token_id"),
                "condition_id": m.get("condition_id", ""),
            }
        break
    return out


def forecast_bundle_for_signal(signal: dict[str, Any], forecasts: dict[str, dict[str, dict]]) -> dict[str, Any]:
    station = str(signal.get("station_icao", ""))
    target_date = str(signal.get("date", ""))
    return forecasts.get(station, {}).get(target_date, {})


def _as_float(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> int | None:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except (TypeError, ValueError):
        return None
