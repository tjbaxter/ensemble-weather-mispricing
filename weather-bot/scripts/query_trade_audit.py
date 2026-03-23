#!/usr/bin/env python3
"""Query the trade observability SQLite database without hand-written SQL."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.trade_audit import TradeAuditStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(ROOT / "data" / "trade_observability.db"), help="SQLite DB path")
    parser.add_argument("--city", help="Filter by city")
    parser.add_argument("--station", help="Filter by station ICAO")
    parser.add_argument("--date", help="Filter by target date (YYYY-MM-DD)")
    parser.add_argument("--strategy", help="Filter by strategy")
    parser.add_argument("--engine", help="Filter by engine")
    parser.add_argument("--action", help="Filter by action")
    parser.add_argument("--run-id", help="Filter by run_id")
    parser.add_argument("--decision-id", help="Filter by decision_id")
    parser.add_argument("--market-scan-id", help="Filter by market_scan_id")
    parser.add_argument("--execution-id", help="Filter by execution_id")
    parser.add_argument("--limit", type=int, default=50, help="Max rows to return")
    parser.add_argument("--ascending", action="store_true", help="Sort oldest first")
    parser.add_argument("--json", action="store_true", help="Emit full rows as JSON")
    parser.add_argument(
        "--show-payloads",
        action="store_true",
        help="Include JSON payload columns in table mode",
    )
    return parser


def ensure_schema(db_path: str) -> None:
    TradeAuditStore(db_path=db_path, jsonl_path=str(ROOT / "data" / "trade_observability.jsonl"))


def query_rows(args: argparse.Namespace) -> list[sqlite3.Row]:
    ensure_schema(args.db)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    where: list[str] = []
    values: list[Any] = []

    for column, value in (
        ("city", args.city),
        ("station_icao", args.station),
        ("target_date", args.date),
        ("strategy", args.strategy),
        ("engine", args.engine),
        ("action", args.action),
        ("run_id", args.run_id),
        ("decision_id", args.decision_id),
        ("market_scan_id", args.market_scan_id),
        ("execution_id", args.execution_id),
    ):
        if value:
            where.append(f"{column} = ?")
            values.append(value)

    order_dir = "ASC" if args.ascending else "DESC"
    sql = """
        SELECT
            event_ts,
            run_id,
            engine,
            action,
            reason,
            city,
            station_icao,
            target_date,
            strategy,
            bucket,
            side,
            decision_id,
            market_scan_id,
            snapshot_id,
            prob_calc_id,
            execution_id,
            forecast_prob,
            market_prob,
            edge,
            requested_size_usd,
            approved_size_usd,
            fill_price,
            execution_status,
            model_values_json,
            forecast_bundle_json,
            market_snapshot_json,
            context_json
        FROM trade_decisions
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY event_ts {order_dir} LIMIT ?"
    values.append(max(1, int(args.limit)))

    try:
        rows = list(conn.execute(sql, values))
    finally:
        conn.close()
    return rows


def print_table(rows: list[sqlite3.Row], show_payloads: bool) -> None:
    if not rows:
        print("No matching rows.")
        return

    columns = [
        "event_ts",
        "engine",
        "action",
        "strategy",
        "city",
        "target_date",
        "bucket",
        "side",
        "forecast_prob",
        "market_prob",
        "edge",
        "approved_size_usd",
        "fill_price",
        "execution_status",
        "decision_id",
        "market_scan_id",
    ]
    if show_payloads:
        columns.extend(
            [
                "model_values_json",
                "forecast_bundle_json",
                "market_snapshot_json",
                "context_json",
            ]
        )

    widths = {col: len(col) for col in columns}
    rendered: list[dict[str, str]] = []
    for row in rows:
        item: dict[str, str] = {}
        for col in columns:
            raw = row[col]
            text = "" if raw is None else str(raw)
            if not show_payloads and len(text) > 36:
                text = text[:33] + "..."
            item[col] = text
            widths[col] = min(max(widths[col], len(text)), 72 if show_payloads else 36)
        rendered.append(item)

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    print(header)
    print(sep)
    for item in rendered:
        print(" | ".join(item[col].ljust(widths[col]) for col in columns))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rows = query_rows(args)
    if args.json:
        print(json.dumps([dict(row) for row in rows], indent=2))
        return
    print_table(rows, show_payloads=args.show_payloads)


if __name__ == "__main__":
    main()
