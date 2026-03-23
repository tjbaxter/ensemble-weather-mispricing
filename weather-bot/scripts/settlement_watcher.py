#!/usr/bin/env python3
"""Read-only realtime settlement watcher for dashboard display.

This service never mutates positions files or resolver CSVs. It watches those
files, observes Polymarket resolution state, and writes an independent
authoritative settlement store for dashboard use.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (  # noqa: E402
    GAMMA_API_URL,
    SETTLEMENT_WATCHER_FINAL_PRICE_THRESHOLD,
    SETTLEMENT_WATCHER_OFFICIAL_REFRESH_SECONDS,
    SETTLEMENT_WATCHER_POLL_SECONDS,
    SETTLEMENT_WATCHER_SPLIT_PRICE_TOLERANCE,
)
from monitoring.settlement_common import (  # noqa: E402
    PORTFOLIO_SOURCES,
    SETTLEMENT_DB,
    SETTLEMENT_SNAPSHOT_JSON,
    SETTLEMENT_STATUS_JSON,
    SETTLEMENT_SUMMARY_JSON,
    build_event_slug,
    build_position_key,
    coerce_float,
    compute_pnl_for_outcome,
    iter_all_positions,
    normalize_bucket_label,
    parse_market_prices,
    portfolio_source_map,
    strategy_mode_for_display,
)

DB_PATH = ROOT / SETTLEMENT_DB
SNAPSHOT_PATH = ROOT / SETTLEMENT_SNAPSHOT_JSON
STATUS_PATH = ROOT / SETTLEMENT_STATUS_JSON
SUMMARY_PATH = ROOT / SETTLEMENT_SUMMARY_JSON


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tracked_positions (
            position_key TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            portfolio_slug TEXT NOT NULL,
            source_file TEXT NOT NULL,
            strategy TEXT NOT NULL,
            market_id TEXT,
            token_id TEXT,
            city TEXT,
            station_icao TEXT,
            target_date TEXT,
            bucket TEXT,
            side TEXT,
            fill_price REAL,
            fill_size REAL,
            cost REAL,
            signal_timestamp TEXT,
            forecast_prob REAL,
            edge REAL,
            model_values_json TEXT,
            raw_json TEXT NOT NULL,
            first_seen_utc TEXT NOT NULL,
            last_seen_utc TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS display_settlements (
            display_key TEXT PRIMARY KEY,
            position_key TEXT,
            mode TEXT NOT NULL,
            portfolio_slug TEXT NOT NULL,
            strategy TEXT NOT NULL,
            city TEXT NOT NULL,
            station_icao TEXT,
            target_date TEXT NOT NULL,
            bucket TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            size_usd REAL NOT NULL,
            outcome TEXT NOT NULL,
            pnl_usd REAL NOT NULL,
            forecast_prob REAL,
            edge REAL,
            model_values_json TEXT,
            signal_timestamp TEXT,
            resolved_at TEXT NOT NULL,
            settlement_phase TEXT NOT NULL,
            source TEXT NOT NULL,
            official_resolved INTEGER NOT NULL DEFAULT 0,
            challenge_window INTEGER NOT NULL DEFAULT 0,
            market_slug TEXT,
            market_condition_id TEXT,
            market_question TEXT,
            yes_price REAL,
            no_price REAL,
            market_closed INTEGER,
            accepting_orders INTEGER,
            uma_resolution_status TEXT,
            latest_observed_utc TEXT NOT NULL,
            extra_json TEXT
        );

        CREATE TABLE IF NOT EXISTS settlement_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_key TEXT NOT NULL,
            observed_at_utc TEXT NOT NULL,
            settlement_phase TEXT NOT NULL,
            source TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_display_settlements_strategy
            ON display_settlements(strategy, mode, target_date);
        CREATE INDEX IF NOT EXISTS idx_tracked_positions_match
            ON tracked_positions(strategy, city, target_date, bucket, side, signal_timestamp);
        """
    )
    conn.commit()


def load_official_resolved_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "logs").glob("resolved*.csv")):
        try:
            with path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if not isinstance(row, dict):
                        continue
                    outcome = str(row.get("outcome", "") or "").strip().upper()
                    if outcome not in {"WIN", "LOSS", "HALF_WIN"}:
                        continue
                    row["bucket"] = normalize_bucket_label(row.get("bucket"))
                    rows.append(row)
        except Exception:
            continue
    return rows


def official_match_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("strategy", "") or ""),
        str(row.get("city", "") or ""),
        str(row.get("target_date", "") or row.get("date", "") or ""),
        normalize_bucket_label(row.get("bucket")),
        str(row.get("side", "BUY_YES") or "BUY_YES"),
        str(row.get("signal_timestamp", "") or ""),
    )


def tracked_match_key(row: sqlite3.Row | dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    getter = row.__getitem__ if isinstance(row, sqlite3.Row) else row.get
    return (
        str(getter("strategy") or ""),
        str(getter("city") or ""),
        str(getter("target_date") or getter("date") or ""),
        normalize_bucket_label(getter("bucket")),
        str(getter("side") or "BUY_YES"),
        str(getter("signal_timestamp") or getter("timestamp") or ""),
    )


def load_tracked_lookup(conn: sqlite3.Connection) -> dict[tuple[str, str, str, str, str, str], sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT position_key, mode, portfolio_slug, strategy, city, target_date, bucket, side, signal_timestamp
        FROM tracked_positions
        """
    ).fetchall()
    return {tracked_match_key(row): row for row in rows}


def upsert_tracked_positions(conn: sqlite3.Connection, positions: list[dict[str, Any]], observed_at: str) -> None:
    conn.execute("UPDATE tracked_positions SET active = 0")
    for position in positions:
        payload = json.dumps(position, sort_keys=True)
        conn.execute(
            """
            INSERT INTO tracked_positions (
                position_key, mode, portfolio_slug, source_file, strategy, market_id, token_id,
                city, station_icao, target_date, bucket, side, fill_price, fill_size, cost,
                signal_timestamp, forecast_prob, edge, model_values_json, raw_json,
                first_seen_utc, last_seen_utc, active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(position_key) DO UPDATE SET
                mode=excluded.mode,
                portfolio_slug=excluded.portfolio_slug,
                source_file=excluded.source_file,
                strategy=excluded.strategy,
                market_id=excluded.market_id,
                token_id=excluded.token_id,
                city=excluded.city,
                station_icao=excluded.station_icao,
                target_date=excluded.target_date,
                bucket=excluded.bucket,
                side=excluded.side,
                fill_price=excluded.fill_price,
                fill_size=excluded.fill_size,
                cost=excluded.cost,
                signal_timestamp=excluded.signal_timestamp,
                forecast_prob=excluded.forecast_prob,
                edge=excluded.edge,
                model_values_json=excluded.model_values_json,
                raw_json=excluded.raw_json,
                last_seen_utc=excluded.last_seen_utc,
                active=1
            """,
            (
                position["_position_key"],
                position["_mode"],
                position["_portfolio_slug"],
                position["_source_rel_path"],
                str(position.get("strategy", "") or ""),
                str(position.get("market_id", "") or ""),
                str(position.get("token_id", "") or ""),
                str(position.get("city", "") or ""),
                str(position.get("station_icao", "") or ""),
                str(position.get("date", "") or ""),
                normalize_bucket_label(position.get("bucket")),
                str(position.get("side", "BUY_YES") or "BUY_YES"),
                coerce_float(position.get("fill_price")),
                coerce_float(position.get("fill_size")),
                coerce_float(position.get("cost")),
                str(position.get("timestamp", "") or ""),
                coerce_float(position.get("forecast_prob")),
                coerce_float(position.get("edge")),
                str(position.get("model_values_json", "{}") or "{}"),
                payload,
                observed_at,
                observed_at,
            ),
        )
    conn.commit()


def upsert_display_row(conn: sqlite3.Connection, row: dict[str, Any], observed_at: str) -> None:
    existing = conn.execute(
        """
        SELECT settlement_phase, outcome, pnl_usd, source, official_resolved, challenge_window,
               yes_price, no_price, uma_resolution_status
        FROM display_settlements
        WHERE display_key = ?
        """,
        (row["display_key"],),
    ).fetchone()

    compare_tuple = (
        row["settlement_phase"],
        row["outcome"],
        round(float(row["pnl_usd"]), 6),
        row["source"],
        int(row["official_resolved"]),
        int(row["challenge_window"]),
        None if row.get("yes_price") is None else round(float(row["yes_price"]), 6),
        None if row.get("no_price") is None else round(float(row["no_price"]), 6),
        row.get("uma_resolution_status", ""),
    )
    existing_tuple = None
    if existing is not None:
        existing_tuple = (
            existing["settlement_phase"],
            existing["outcome"],
            round(float(existing["pnl_usd"]), 6),
            existing["source"],
            int(existing["official_resolved"]),
            int(existing["challenge_window"]),
            None if existing["yes_price"] is None else round(float(existing["yes_price"]), 6),
            None if existing["no_price"] is None else round(float(existing["no_price"]), 6),
            existing["uma_resolution_status"],
        )

    conn.execute(
        """
        INSERT INTO display_settlements (
            display_key, position_key, mode, portfolio_slug, strategy, city, station_icao,
            target_date, bucket, side, entry_price, size_usd, outcome, pnl_usd,
            forecast_prob, edge, model_values_json, signal_timestamp, resolved_at,
            settlement_phase, source, official_resolved, challenge_window, market_slug,
            market_condition_id, market_question, yes_price, no_price, market_closed,
            accepting_orders, uma_resolution_status, latest_observed_utc, extra_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(display_key) DO UPDATE SET
            position_key=excluded.position_key,
            mode=excluded.mode,
            portfolio_slug=excluded.portfolio_slug,
            strategy=excluded.strategy,
            city=excluded.city,
            station_icao=excluded.station_icao,
            target_date=excluded.target_date,
            bucket=excluded.bucket,
            side=excluded.side,
            entry_price=excluded.entry_price,
            size_usd=excluded.size_usd,
            outcome=excluded.outcome,
            pnl_usd=excluded.pnl_usd,
            forecast_prob=excluded.forecast_prob,
            edge=excluded.edge,
            model_values_json=excluded.model_values_json,
            signal_timestamp=excluded.signal_timestamp,
            resolved_at=excluded.resolved_at,
            settlement_phase=excluded.settlement_phase,
            source=excluded.source,
            official_resolved=excluded.official_resolved,
            challenge_window=excluded.challenge_window,
            market_slug=excluded.market_slug,
            market_condition_id=excluded.market_condition_id,
            market_question=excluded.market_question,
            yes_price=excluded.yes_price,
            no_price=excluded.no_price,
            market_closed=excluded.market_closed,
            accepting_orders=excluded.accepting_orders,
            uma_resolution_status=excluded.uma_resolution_status,
            latest_observed_utc=excluded.latest_observed_utc,
            extra_json=excluded.extra_json
        """,
        (
            row["display_key"],
            row.get("position_key"),
            row["mode"],
            row["portfolio_slug"],
            row["strategy"],
            row["city"],
            row.get("station_icao", ""),
            row["target_date"],
            row["bucket"],
            row["side"],
            row["entry_price"],
            row["size_usd"],
            row["outcome"],
            row["pnl_usd"],
            row.get("forecast_prob", 0.0),
            row.get("edge", 0.0),
            row.get("model_values_json", "{}"),
            row.get("signal_timestamp", ""),
            row["resolved_at"],
            row["settlement_phase"],
            row["source"],
            int(row.get("official_resolved", 0)),
            int(row.get("challenge_window", 0)),
            row.get("market_slug", ""),
            row.get("market_condition_id", ""),
            row.get("market_question", ""),
            row.get("yes_price"),
            row.get("no_price"),
            row.get("market_closed"),
            row.get("accepting_orders"),
            row.get("uma_resolution_status", ""),
            observed_at,
            json.dumps(row.get("extra", {}), sort_keys=True),
        ),
    )

    if existing_tuple != compare_tuple:
        conn.execute(
            """
            INSERT INTO settlement_events (
                display_key, observed_at_utc, settlement_phase, source, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["display_key"],
                observed_at,
                row["settlement_phase"],
                row["source"],
                json.dumps(row, sort_keys=True),
            ),
        )
    conn.commit()


def remove_non_official_row(conn: sqlite3.Connection, display_key: str, observed_at: str, reason: str) -> None:
    row = conn.execute(
        "SELECT * FROM display_settlements WHERE display_key = ? AND official_resolved = 0",
        (display_key,),
    ).fetchone()
    if row is None:
        return
    payload = dict(row)
    payload["removal_reason"] = reason
    conn.execute("DELETE FROM display_settlements WHERE display_key = ?", (display_key,))
    conn.execute(
        """
        INSERT INTO settlement_events (
            display_key, observed_at_utc, settlement_phase, source, payload_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            display_key,
            observed_at,
            "unresolved",
            "watcher_unset",
            json.dumps(payload, sort_keys=True),
        ),
    )
    conn.commit()


def infer_market_bucket(market: dict[str, Any]) -> str:
    for candidate in (market.get("groupItemTitle"), market.get("question"), market.get("description")):
        normalized = normalize_bucket_label(candidate)
        if normalized:
            return normalized
    return ""


def assess_market(market: dict[str, Any]) -> dict[str, Any] | None:
    yes_price, no_price = parse_market_prices(market)
    if yes_price is None or no_price is None:
        return None

    status = str(market.get("umaResolutionStatus", "") or "").strip().lower()
    market_closed = bool(market.get("closed"))
    accepting_orders = market.get("acceptingOrders")
    binary_threshold = SETTLEMENT_WATCHER_FINAL_PRICE_THRESHOLD
    split_tol = SETTLEMENT_WATCHER_SPLIT_PRICE_TOLERANCE

    market_outcome: str | None = None
    if yes_price >= binary_threshold and no_price <= (1.0 - binary_threshold):
        market_outcome = "YES"
    elif no_price >= binary_threshold and yes_price <= (1.0 - binary_threshold):
        market_outcome = "NO"
    elif abs(yes_price - 0.5) <= split_tol and abs(no_price - 0.5) <= split_tol:
        market_outcome = "SPLIT"

    if market_outcome is None:
        return None

    if status == "resolved" or (market_closed and accepting_orders is False):
        phase = "resolved"
        challenge_window = 0
    elif status == "proposed":
        phase = "proposed"
        challenge_window = 1
    else:
        return None

    return {
        "phase": phase,
        "challenge_window": challenge_window,
        "market_outcome": market_outcome,
        "yes_price": yes_price,
        "no_price": no_price,
        "market_closed": market_closed,
        "accepting_orders": accepting_orders,
        "uma_resolution_status": status,
    }


def build_display_row_from_position(
    position: dict[str, Any],
    market: dict[str, Any],
    settlement: dict[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    market_outcome = settlement["market_outcome"]
    if market_outcome == "SPLIT":
        our_outcome = "HALF_WIN"
    else:
        yes_won = market_outcome == "YES"
        our_outcome = "WIN" if (position.get("side", "BUY_YES") == "BUY_YES") == yes_won else "LOSS"

    return {
        "display_key": position["_position_key"],
        "position_key": position["_position_key"],
        "mode": position["_mode"],
        "portfolio_slug": position["_portfolio_slug"],
        "strategy": str(position.get("strategy", "") or ""),
        "city": str(position.get("city", "") or ""),
        "station_icao": str(position.get("station_icao", "") or ""),
        "target_date": str(position.get("date", "") or ""),
        "bucket": normalize_bucket_label(position.get("bucket")),
        "side": str(position.get("side", "BUY_YES") or "BUY_YES"),
        "entry_price": coerce_float(position.get("fill_price")),
        "size_usd": coerce_float(position.get("cost")),
        "outcome": our_outcome,
        "pnl_usd": compute_pnl_for_outcome(position, our_outcome),
        "forecast_prob": coerce_float(position.get("forecast_prob")),
        "edge": coerce_float(position.get("edge")),
        "model_values_json": str(position.get("model_values_json", "{}") or "{}"),
        "signal_timestamp": str(position.get("timestamp", "") or ""),
        "resolved_at": observed_at,
        "settlement_phase": settlement["phase"],
        "source": f"polymarket_{settlement['phase']}",
        "official_resolved": 0,
        "challenge_window": settlement["challenge_window"],
        "market_slug": str(market.get("slug", "") or ""),
        "market_condition_id": str(market.get("conditionId") or market.get("condition_id") or ""),
        "market_question": str(market.get("question", "") or ""),
        "yes_price": settlement["yes_price"],
        "no_price": settlement["no_price"],
        "market_closed": int(bool(settlement["market_closed"])),
        "accepting_orders": None if settlement["accepting_orders"] is None else int(bool(settlement["accepting_orders"])),
        "uma_resolution_status": settlement["uma_resolution_status"],
        "extra": {
            "resolved_by": market.get("resolvedBy"),
            "closed_time": market.get("closedTime"),
        },
    }


def build_display_row_from_official(
    row: dict[str, Any],
    tracked: sqlite3.Row | None,
    observed_at: str,
) -> dict[str, Any]:
    mode = "paper"
    portfolio_slug = "legacy_paper"
    position_key = None
    if tracked is not None:
        mode = str(tracked["mode"] or "paper")
        portfolio_slug = str(tracked["portfolio_slug"] or "paper_main")
        position_key = str(tracked["position_key"] or "")

    display_key = position_key or (
        "legacy::"
        + "::".join(
            [
                mode,
                str(row.get("strategy", "") or ""),
                str(row.get("city", "") or ""),
                str(row.get("target_date", "") or row.get("date", "") or ""),
                normalize_bucket_label(row.get("bucket")),
                str(row.get("side", "BUY_YES") or "BUY_YES"),
                str(row.get("signal_timestamp", "") or ""),
            ]
        )
    )

    return {
        "display_key": display_key,
        "position_key": position_key,
        "mode": mode,
        "portfolio_slug": portfolio_slug,
        "strategy": str(row.get("strategy", "") or ""),
        "city": str(row.get("city", "") or ""),
        "station_icao": str(row.get("station_icao", "") or ""),
        "target_date": str(row.get("target_date", "") or row.get("date", "") or ""),
        "bucket": normalize_bucket_label(row.get("bucket")),
        "side": str(row.get("side", "BUY_YES") or "BUY_YES"),
        "entry_price": coerce_float(row.get("entry_price")),
        "size_usd": coerce_float(row.get("size_usd")),
        "outcome": str(row.get("outcome", "") or ""),
        "pnl_usd": coerce_float(row.get("pnl_usd")),
        "forecast_prob": coerce_float(row.get("forecast_prob")),
        "edge": coerce_float(row.get("edge")),
        "model_values_json": str(row.get("model_values_json", "{}") or "{}"),
        "signal_timestamp": str(row.get("signal_timestamp", "") or ""),
        "resolved_at": str(row.get("resolved_at", "") or observed_at),
        "settlement_phase": "official",
        "source": "official_resolved_csv",
        "official_resolved": 1,
        "challenge_window": 0,
        "market_slug": "",
        "market_condition_id": "",
        "market_question": "",
        "yes_price": None,
        "no_price": None,
        "market_closed": None,
        "accepting_orders": None,
        "uma_resolution_status": "official",
        "extra": {
            "actual_temp": row.get("actual_temp"),
            "roi_pct": row.get("roi_pct"),
        },
    }


def build_market_indexes(event: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_condition: dict[str, dict[str, Any]] = {}
    by_bucket: dict[str, dict[str, Any]] = {}
    for market in event.get("markets", []):
        if not isinstance(market, dict):
            continue
        condition_id = str(market.get("conditionId") or market.get("condition_id") or "")
        if condition_id:
            by_condition[condition_id] = market
        bucket = infer_market_bucket(market)
        if bucket:
            by_bucket[bucket] = market
    return by_condition, by_bucket


def fetch_event_by_slug(session: requests.Session, slug: str) -> dict[str, Any] | None:
    try:
        response = session.get(f"{GAMMA_API_URL}/events/slug/{slug}", timeout=20)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def fetch_market_by_condition_id(session: requests.Session, condition_id: str) -> dict[str, Any] | None:
    try:
        response = session.get(
            f"{GAMMA_API_URL}/markets",
            params={"condition_ids": condition_id},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list) and payload:
            first = payload[0]
            return first if isinstance(first, dict) else None
    except Exception:
        return None
    return None


def fetch_live_reconciliation(positions: list[dict[str, Any]]) -> dict[str, Any]:
    wallet = (
        os.getenv("POLYMARKET_FUNDER", "").strip()
        or os.getenv("WALLET_ADDRESS", "").strip()
    )
    if not wallet:
        return {"enabled": False, "reason": "wallet_not_configured"}

    local_live = [p for p in positions if p.get("_mode") == "live"]
    local_pairs = {
        (str(p.get("market_id", "") or ""), str(p.get("token_id", "") or ""))
        for p in local_live
        if p.get("market_id") and p.get("token_id")
    }

    try:
        response = requests.get(
            "https://data-api.polymarket.com/positions",
            params={"user": wallet, "sizeThreshold": 0, "limit": 500},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("positions payload was not a list")
        remote_pairs = {
            (str(item.get("conditionId", "") or ""), str(item.get("asset", "") or ""))
            for item in payload
            if isinstance(item, dict) and item.get("conditionId") and item.get("asset")
        }
    except Exception as exc:
        return {"enabled": True, "ok": False, "error": str(exc)}

    missing_local = sorted(remote_pairs - local_pairs)[:10]
    missing_remote = sorted(local_pairs - remote_pairs)[:10]
    return {
        "enabled": True,
        "ok": not missing_local and not missing_remote,
        "wallet": wallet,
        "local_count": len(local_pairs),
        "remote_count": len(remote_pairs),
        "missing_local_samples": missing_local,
        "missing_remote_samples": missing_remote,
    }


def write_snapshot_files(conn: sqlite3.Connection, observed_at: str, status_payload: dict[str, Any]) -> None:
    rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT *
            FROM display_settlements
            ORDER BY target_date ASC, strategy ASC, city ASC, bucket ASC
            """
        ).fetchall()
    ]

    snapshot_payload = {
        "generated_at": observed_at,
        "rows": rows,
    }
    write_json_atomic(SNAPSHOT_PATH, snapshot_payload)

    by_strategy: dict[str, dict[str, Any]] = {}
    for row in rows:
        strategy = str(row.get("strategy", "") or "")
        entry = by_strategy.setdefault(
            strategy,
            {
                "strategy": strategy,
                "mode": row.get("mode", "paper"),
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "half_wins": 0,
                "pnl_usd": 0.0,
                "provisional": 0,
                "official": 0,
            },
        )
        entry["trades"] += 1
        outcome = str(row.get("outcome", "") or "")
        if outcome == "WIN":
            entry["wins"] += 1
        elif outcome == "LOSS":
            entry["losses"] += 1
        elif outcome == "HALF_WIN":
            entry["half_wins"] += 1
        entry["pnl_usd"] += coerce_float(row.get("pnl_usd"))
        if str(row.get("settlement_phase", "")) == "proposed":
            entry["provisional"] += 1
        if int(row.get("official_resolved", 0)):
            entry["official"] += 1

    summary_payload = {
        "generated_at": observed_at,
        "strategies": sorted(by_strategy.values(), key=lambda item: item["strategy"]),
    }
    write_json_atomic(SUMMARY_PATH, summary_payload)
    write_json_atomic(STATUS_PATH, status_payload)


def run_loop() -> None:
    conn = connect_db()
    init_db(conn)

    official_last_loaded = 0.0
    session = requests.Session()
    source_lookup = portfolio_source_map()

    while True:
        observed_at = _utcnow()
        today_str = date.today().isoformat()
        status_payload: dict[str, Any] = {
            "service": "settlement_watcher",
            "source": "polymarket+official-resolver",
            "last_heartbeat_utc": observed_at,
            "last_error": "",
            "tracked_positions": 0,
            "active_candidates": 0,
            "watcher_rows": 0,
            "official_rows": 0,
        }
        try:
            positions = iter_all_positions(ROOT)
            upsert_tracked_positions(conn, positions, observed_at)
            status_payload["tracked_positions"] = len(positions)

            tracked_lookup = load_tracked_lookup(conn)

            if (time.time() - official_last_loaded) >= SETTLEMENT_WATCHER_OFFICIAL_REFRESH_SECONDS:
                official_rows = load_official_resolved_rows()
                for row in official_rows:
                    tracked = tracked_lookup.get(official_match_key(row))
                    display_row = build_display_row_from_official(row, tracked, observed_at)
                    upsert_display_row(conn, display_row, observed_at)
                status_payload["official_rows"] = len(official_rows)
                official_last_loaded = time.time()

            candidate_positions = [
                p for p in positions
                if str(p.get("date", "")) and str(p.get("date", "")) <= today_str
            ]
            status_payload["active_candidates"] = len(candidate_positions)

            slug_map: dict[str, list[dict[str, Any]]] = {}
            fallback_positions: list[dict[str, Any]] = []
            for position in candidate_positions:
                slug = build_event_slug(
                    str(position.get("station_icao", "") or ""),
                    str(position.get("city", "") or ""),
                    str(position.get("date", "") or ""),
                )
                if slug:
                    slug_map.setdefault(slug, []).append(position)
                else:
                    fallback_positions.append(position)

            watcher_display_keys: set[str] = set()
            for slug, slug_positions in slug_map.items():
                event = fetch_event_by_slug(session, slug)
                if event is None:
                    fallback_positions.extend(slug_positions)
                    continue
                by_condition, by_bucket = build_market_indexes(event)
                for position in slug_positions:
                    market = by_condition.get(str(position.get("market_id", "") or ""))
                    if market is None:
                        market = by_bucket.get(normalize_bucket_label(position.get("bucket")))
                    if market is None:
                        continue
                    settlement = assess_market(market)
                    if settlement is None:
                        continue
                    display_row = build_display_row_from_position(position, market, settlement, observed_at)
                    upsert_display_row(conn, display_row, observed_at)
                    watcher_display_keys.add(display_row["display_key"])

            for position in fallback_positions:
                condition_id = str(position.get("market_id", "") or "")
                if not condition_id:
                    continue
                market = fetch_market_by_condition_id(session, condition_id)
                if market is None:
                    continue
                settlement = assess_market(market)
                if settlement is None:
                    continue
                display_row = build_display_row_from_position(position, market, settlement, observed_at)
                upsert_display_row(conn, display_row, observed_at)
                watcher_display_keys.add(display_row["display_key"])

            active_non_official_keys = [
                row["display_key"]
                for row in conn.execute(
                    """
                    SELECT display_key
                    FROM display_settlements
                    WHERE official_resolved = 0 AND position_key IS NOT NULL
                    """
                ).fetchall()
            ]
            for display_key in active_non_official_keys:
                if display_key not in watcher_display_keys:
                    remove_non_official_row(conn, display_key, observed_at, "market_no_longer_in_resolution_phase")

            status_payload["watcher_rows"] = len(watcher_display_keys)
            status_payload["last_success_utc"] = observed_at
            status_payload["live_reconciliation"] = fetch_live_reconciliation(positions)
        except Exception as exc:
            status_payload["last_error"] = str(exc)

        write_snapshot_files(conn, observed_at, status_payload)
        time.sleep(SETTLEMENT_WATCHER_POLL_SECONDS)


if __name__ == "__main__":
    run_loop()
