"""Shared helpers for read-only settlement monitoring and dashboard overlays."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from config.cities import STATIONS

PRIMARY_STRATEGIES = {"SINGLE", "LADDER", "CONVICTION"}
SHADOW_STRATEGIES = {
    "TOP2_EQUAL",
    "TOP2_COND",
    "TOP2_PROP",
    "PURDEY_MK1",
    "CAVENDISH_MK1",
    "PURDEY_MK2",
    "CAVENDISH_MK3",
    "TRUE_ALPHA",
    "PRIME_ALPHA",
    "PROPS_KELLY",
}

PORTFOLIO_SOURCES: list[dict[str, str]] = [
    {
        "portfolio_slug": "paper_main",
        "mode": "paper",
        "rel_path": "data/positions.json",
        "strategy_fallback": "LADDER",
    },
    {
        "portfolio_slug": "live_main",
        "mode": "live",
        "rel_path": "data/positions_live.json",
        "strategy_fallback": "LADDER",
    },
    {
        "portfolio_slug": "shadow_2a",
        "mode": "paper",
        "rel_path": "data/positions_shadow_2a.json",
        "strategy_fallback": "TOP2_EQUAL",
    },
    {
        "portfolio_slug": "shadow_2b",
        "mode": "paper",
        "rel_path": "data/positions_shadow_2b.json",
        "strategy_fallback": "TOP2_COND",
    },
    {
        "portfolio_slug": "shadow_2c",
        "mode": "paper",
        "rel_path": "data/positions_shadow_2c.json",
        "strategy_fallback": "TOP2_PROP",
    },
    {
        "portfolio_slug": "shadow_purdey",
        "mode": "paper",
        "rel_path": "data/positions_shadow_purdey.json",
        "strategy_fallback": "PURDEY_MK1",
    },
    {
        "portfolio_slug": "shadow_cavendish",
        "mode": "paper",
        "rel_path": "data/positions_shadow_cavendish.json",
        "strategy_fallback": "CAVENDISH_MK1",
    },
    {
        "portfolio_slug": "shadow_purdey2",
        "mode": "paper",
        "rel_path": "data/positions_shadow_purdey2.json",
        "strategy_fallback": "PURDEY_MK2",
    },
    {
        "portfolio_slug": "shadow_cavendish3",
        "mode": "paper",
        "rel_path": "data/positions_shadow_cavendish3.json",
        "strategy_fallback": "CAVENDISH_MK3",
    },
    {
        "portfolio_slug": "shadow_true_alpha",
        "mode": "paper",
        "rel_path": "data/positions_shadow_true_alpha.json",
        "strategy_fallback": "TRUE_ALPHA",
    },
    {
        "portfolio_slug": "shadow_prime_alpha",
        "mode": "paper",
        "rel_path": "data/positions_shadow_prime_alpha.json",
        "strategy_fallback": "PRIME_ALPHA",
    },
    {
        "portfolio_slug": "shadow_props_kelly",
        "mode": "paper",
        "rel_path": "data/positions_shadow_props_kelly.json",
        "strategy_fallback": "PROPS_KELLY",
    },
]

SETTLEMENT_DB = "data/settlement_watcher.db"
SETTLEMENT_SNAPSHOT_JSON = "data/settlement_snapshot.json"
SETTLEMENT_STATUS_JSON = "data/settlement_status.json"
SETTLEMENT_SUMMARY_JSON = "data/settlement_summary.json"

_RANGE_PATTERN = re.compile(r"(-?\d+)\s*-\s*(-?\d+)")
_PLUS_PATTERN = re.compile(r"(-?\d+)\s*\+")
_ABOVE_PATTERN = re.compile(r"(-?\d+)\s*(?:or higher|or above|and above|or more)")
_BELOW_PATTERN = re.compile(r"(-?\d+)\s*(?:or below|or lower|and below|and lower|or less)")


def normalize_bucket_label(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    text = text.replace("°f", "").replace("°c", "")
    text = text.replace("ºf", "").replace("ºc", "")
    text = re.sub(r"\s+", " ", text)

    range_match = _RANGE_PATTERN.search(text)
    if range_match:
        return f"{range_match.group(1)}-{range_match.group(2)}"

    plus_match = _PLUS_PATTERN.search(text)
    if plus_match:
        return f"{plus_match.group(1)}+"

    above_match = _ABOVE_PATTERN.search(text)
    if above_match:
        return f"{above_match.group(1)}+"

    below_match = _BELOW_PATTERN.search(text)
    if below_match:
        return f"{below_match.group(1)} or below"

    return text


def coerce_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return default


def portfolio_source_map() -> dict[str, dict[str, str]]:
    return {source["portfolio_slug"]: source for source in PORTFOLIO_SOURCES}


def load_position_payload(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def position_identity_fields(position: dict[str, Any], portfolio_slug: str) -> dict[str, Any]:
    return {
        "portfolio_slug": portfolio_slug,
        "market_id": str(position.get("market_id", "") or ""),
        "token_id": str(position.get("token_id", "") or ""),
        "strategy": str(position.get("strategy", "") or ""),
        "side": str(position.get("side", "") or ""),
        "city": str(position.get("city", "") or ""),
        "station_icao": str(position.get("station_icao", "") or ""),
        "date": str(position.get("date", "") or ""),
        "bucket": normalize_bucket_label(position.get("bucket")),
        "fill_price": round(coerce_float(position.get("fill_price")), 8),
        "fill_size": round(coerce_float(position.get("fill_size")), 8),
        "cost": round(coerce_float(position.get("cost")), 8),
        "timestamp": str(position.get("timestamp", "") or ""),
    }


def build_position_key(position: dict[str, Any], portfolio_slug: str) -> str:
    payload = position_identity_fields(position, portfolio_slug)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def annotate_position(position: dict[str, Any], source: dict[str, str]) -> dict[str, Any]:
    out = dict(position)
    out["bucket"] = normalize_bucket_label(out.get("bucket"))
    out["strategy"] = str(out.get("strategy", "") or source.get("strategy_fallback", ""))
    out["_portfolio_slug"] = source["portfolio_slug"]
    out["_mode"] = source["mode"]
    out["_source_rel_path"] = source["rel_path"]
    out["_position_key"] = build_position_key(out, source["portfolio_slug"])
    return out


def iter_all_positions(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_rel_paths: set[str] = set()
    for source in PORTFOLIO_SOURCES:
        path = root / source["rel_path"]
        seen_rel_paths.add(source["rel_path"])
        if not path.exists():
            continue
        for position in load_position_payload(path):
            rows.append(annotate_position(position, source))

    for path in sorted((root / "data").glob("positions_shadow_*.json")):
        rel_path = str(path.relative_to(root))
        if rel_path in seen_rel_paths:
            continue
        slug = path.stem.replace("positions_", "", 1)
        source = {
            "portfolio_slug": slug,
            "mode": "paper",
            "rel_path": rel_path,
            "strategy_fallback": slug.replace("shadow_", "", 1).upper(),
        }
        for position in load_position_payload(path):
            rows.append(annotate_position(position, source))
    return rows


def choose_primary_positions_path(root: Path, live_mode: bool) -> Path:
    rel_path = "data/positions_live.json" if live_mode else "data/positions.json"
    return root / rel_path


def build_event_slug(station_icao: str, city: str, target_date_str: str) -> str | None:
    station_cfg = STATIONS.get(station_icao)
    if station_cfg is None and city:
        city_lower = city.strip().lower()
        for icao, cfg in STATIONS.items():
            if str(cfg.get("market_label", "")).strip().lower() == city_lower:
                station_cfg = cfg
                break

    city_slug = str((station_cfg or {}).get("city_slug", "")).strip()
    if not city_slug:
        return None

    try:
        year, month, day = [int(part) for part in target_date_str.split("-")]
    except Exception:
        return None

    import datetime as _dt

    target_date = _dt.date(year, month, day)
    month_name = target_date.strftime("%B").lower()
    return f"highest-temperature-in-{city_slug}-on-{month_name}-{target_date.day}-{target_date.year}"


def parse_json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
            return payload if isinstance(payload, list) else []
        except json.JSONDecodeError:
            return []
    return []


def parse_market_prices(market: dict[str, Any]) -> tuple[float | None, float | None]:
    outcomes = parse_json_list(market.get("outcomes"))
    prices = parse_json_list(market.get("outcomePrices"))
    yes_price = None
    no_price = None
    for idx, outcome in enumerate(outcomes):
        if idx >= len(prices):
            continue
        label = str(outcome).strip().lower()
        try:
            price = float(prices[idx])
        except (TypeError, ValueError):
            continue
        if label == "yes":
            yes_price = price
        elif label == "no":
            no_price = price
    return yes_price, no_price


def compute_pnl_for_outcome(position: dict[str, Any], outcome: str) -> float:
    fill_size = coerce_float(position.get("fill_size"))
    cost = coerce_float(position.get("cost"))
    if outcome == "WIN":
        return round(fill_size - cost, 4)
    if outcome == "HALF_WIN":
        return round((0.5 * fill_size) - cost, 4)
    return round(-cost, 4)


def strategy_mode_for_display(strategy: str, live_mode: bool) -> str:
    if strategy in SHADOW_STRATEGIES:
        return "paper"
    if strategy in PRIMARY_STRATEGIES:
        return "live" if live_mode else "paper"
    return "live" if live_mode else "paper"
