from __future__ import annotations

import io
import json
import os
from datetime import UTC, date as _date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from monitoring.settlement_common import (
    build_position_key,
    choose_primary_positions_path,
    coerce_float,
    normalize_bucket_label,
)

TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_ENV = ".env"

MODEL_COLORS: dict[str, str] = {
    "SINGLE": "#00FF88",
    "LADDER": "#4CC9F0",
    "CONVICTION": "#F72585",
    "TOP2_EQUAL": "#FFB347",
    "TOP2_COND": "#9B59B6",
    "TOP2_PROP": "#F1C40F",
    "PURDEY_MK1": "#FFD166",
    "CAVENDISH_MK1": "#2ECC71",
    "PURDEY_MK2": "#F9844A",
    "CAVENDISH_MK3": "#1ABC9C",
    "TRUE_ALPHA": "#F0C040",
    "PRIME_ALPHA": "#FF6B6B",
    "PROPS_KELLY": "#7C3AED",
}

STRATEGY_SPECS: list[dict[str, str]] = [
    {
        "label": "⚡ SINGLE",
        "strategy_key": "SINGLE",
        "source_kind": "main",
        "source_value": "SINGLE",
    },
    {
        "label": "🪜 LADDER",
        "strategy_key": "LADDER",
        "source_kind": "main",
        "source_value": "LADDER",
    },
    {
        "label": "🎯 CONVICTION",
        "strategy_key": "CONVICTION",
        "source_kind": "main",
        "source_value": "CONVICTION",
    },
    {
        "label": "2A Equal",
        "strategy_key": "TOP2_EQUAL",
        "source_kind": "shadow",
        "source_value": "shadow_2a",
    },
    {
        "label": "2B Cond",
        "strategy_key": "TOP2_COND",
        "source_kind": "shadow",
        "source_value": "shadow_2b",
    },
    {
        "label": "2C Prop",
        "strategy_key": "TOP2_PROP",
        "source_kind": "shadow",
        "source_value": "shadow_2c",
    },
    {
        "label": "🎯 PURDEY",
        "strategy_key": "PURDEY_MK1",
        "source_kind": "shadow",
        "source_value": "shadow_purdey",
    },
    {
        "label": "🌿 CAVENDISH",
        "strategy_key": "CAVENDISH_MK1",
        "source_kind": "shadow",
        "source_value": "shadow_cavendish",
    },
    {
        "label": "🎯 PURDEY MK2",
        "strategy_key": "PURDEY_MK2",
        "source_kind": "shadow",
        "source_value": "shadow_purdey2",
    },
    {
        "label": "🌱 CAVENDISH MK3",
        "strategy_key": "CAVENDISH_MK3",
        "source_kind": "shadow",
        "source_value": "shadow_cavendish3",
    },
    {
        "label": "💎 True Alpha",
        "strategy_key": "TRUE_ALPHA",
        "source_kind": "shadow",
        "source_value": "shadow_true_alpha",
    },
    {
        "label": "🧭 Prime Alpha",
        "strategy_key": "PRIME_ALPHA",
        "source_kind": "shadow",
        "source_value": "shadow_prime_alpha",
    },
    {
        "label": "🎲 Props Kelly",
        "strategy_key": "PROPS_KELLY",
        "source_kind": "shadow",
        "source_value": "shadow_props_kelly",
    },
]

SHADOW_STRATEGY_FALLBACKS = {
    "shadow_2a": "TOP2_EQUAL",
    "shadow_2b": "TOP2_COND",
    "shadow_2c": "TOP2_PROP",
    "shadow_purdey": "PURDEY_MK1",
    "shadow_cavendish": "CAVENDISH_MK1",
    "shadow_purdey2": "PURDEY_MK2",
    "shadow_cavendish3": "CAVENDISH_MK3",
    "shadow_true_alpha": "TRUE_ALPHA",
    "shadow_prime_alpha": "PRIME_ALPHA",
    "shadow_props_kelly": "PROPS_KELLY",
}


def _parse_truthy(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in TRUTHY


def _load_mode_from_env(path: Path) -> tuple[bool, bool]:
    paper = True
    live = False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return paper, live
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        truthy = _parse_truthy(value.strip().strip('"').strip("'"), default=False)
        if key == "PAPER_TRADING":
            paper = truthy
        elif key == "LIVE_TRADING":
            live = truthy
    return paper, live


def current_mode_name(root: Path) -> str:
    paper_raw = os.getenv("PAPER_TRADING")
    live_raw = os.getenv("LIVE_TRADING")
    if paper_raw is not None or live_raw is not None:
        paper = _parse_truthy(paper_raw, default=True)
        live = _parse_truthy(live_raw, default=False)
    else:
        paper, live = _load_mode_from_env(root / DEFAULT_ENV)
    if live and not paper:
        return "live"
    return "paper"


def _read_json_list(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [row for row in payload.values() if isinstance(row, dict)]
    return []


def _read_json_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_primary_positions(root: Path) -> list[dict]:
    live_mode = current_mode_name(root) == "live"
    path = choose_primary_positions_path(root, live_mode=live_mode)
    mode = "live" if live_mode else "paper"
    out: list[dict] = []
    for row in _read_json_list(path):
        item = dict(row)
        item["bucket"] = normalize_bucket_label(item.get("bucket"))
        item["strategy"] = str(item.get("strategy", "") or "LADDER")
        item["_mode"] = mode
        item["_portfolio_slug"] = "live_main" if live_mode else "paper_main"
        item["_position_key"] = build_position_key(item, item["_portfolio_slug"])
        out.append(item)
    return out


def _load_shadow_positions(root: Path, slug: str) -> list[dict]:
    path = root / "data" / f"positions_{slug}.json"
    out: list[dict] = []
    for row in _read_json_list(path):
        item = dict(row)
        item["bucket"] = normalize_bucket_label(item.get("bucket"))
        item["strategy"] = str(item.get("strategy", "") or SHADOW_STRATEGY_FALLBACKS.get(slug, ""))
        item["_mode"] = "paper"
        item["_portfolio_slug"] = slug
        item["_position_key"] = build_position_key(item, slug)
        out.append(item)
    return out


def _load_settlement_snapshot_df(root: Path) -> pd.DataFrame:
    payload = _read_json_object(root / "data" / "settlement_snapshot.json")
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    df = pd.DataFrame([row for row in rows if isinstance(row, dict)])
    if df.empty:
        return df
    defaults = {
        "resolved_at": "",
        "target_date": "",
        "city": "",
        "bucket": "",
        "side": "BUY_YES",
        "strategy": "",
        "mode": "paper",
        "settlement_phase": "",
        "outcome": "",
        "entry_price": 0.0,
        "size_usd": 0.0,
        "pnl_usd": 0.0,
        "official_resolved": 0,
        "challenge_window": 0,
        "position_key": "",
        "portfolio_slug": "",
        "signal_timestamp": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce", utc=True)
    df["target_date_dt"] = pd.to_datetime(df["target_date"], errors="coerce", utc=True)
    df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"], errors="coerce", utc=True)
    df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce").fillna(0.0)
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce").fillna(0.0)
    df["size_usd"] = pd.to_numeric(df["size_usd"], errors="coerce").fillna(0.0)
    df["official_resolved"] = pd.to_numeric(df["official_resolved"], errors="coerce").fillna(0).astype(int)
    df["challenge_window"] = pd.to_numeric(df["challenge_window"], errors="coerce").fillna(0).astype(int)
    df["bucket"] = df["bucket"].map(normalize_bucket_label)
    return df.sort_values(["target_date_dt", "resolved_at", "signal_timestamp"], ascending=True)


def _load_resolved_df(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((root / "logs").glob("resolved*.csv")):
        try:
            df_part = pd.read_csv(path, index_col=False)
        except Exception:
            continue
        if not df_part.empty:
            frames.append(df_part)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    dedup_cols = [c for c in ("resolved_at", "city", "target_date", "bucket", "side") if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last")
    for col in ("resolved_at", "target_date", "city", "bucket", "outcome"):
        if col not in df.columns:
            df[col] = ""
    for col in ("pnl_usd", "entry_price", "size_usd", "ev_per_bet"):
        if col not in df.columns:
            df[col] = 0.0
    df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce", utc=True)
    df["target_date_dt"] = pd.to_datetime(df["target_date"], errors="coerce", utc=True)
    if "signal_timestamp" in df.columns:
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"], errors="coerce", utc=True)
    df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce").fillna(0.0)
    df["bucket"] = df["bucket"].map(normalize_bucket_label)
    if "strategy" not in df.columns:
        df["strategy"] = "LADDER"
    else:
        df["strategy"] = df["strategy"].fillna("LADDER")
    sort_cols = ["target_date_dt"]
    if "signal_timestamp" in df.columns:
        sort_cols.append("signal_timestamp")
    df = df.sort_values(sort_cols, ascending=True)
    df["cum_pnl"] = df["pnl_usd"].cumsum()
    return df


def _resolved_match_key_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)
    cols = {
        "strategy": "",
        "city": "",
        "target_date": "",
        "bucket": "",
        "side": "",
        "signal_timestamp": "",
    }
    parts: list[pd.Series] = []
    for col, default in cols.items():
        if col in df.columns:
            parts.append(df[col].fillna(default).astype(str))
        else:
            parts.append(pd.Series([default] * len(df), index=df.index, dtype=str))
    return parts[0] + "|" + parts[1] + "|" + parts[2] + "|" + parts[3] + "|" + parts[4] + "|" + parts[5]


def _load_effective_resolved_df(root: Path) -> pd.DataFrame:
    legacy = _load_resolved_df(root).copy()
    if not legacy.empty:
        legacy["mode"] = "paper"
        legacy["settlement_phase"] = "official"
        legacy["official_resolved"] = 1
    snapshot_df = _load_settlement_snapshot_df(root).copy()
    if snapshot_df.empty:
        return legacy
    if legacy.empty:
        combined = snapshot_df
    else:
        legacy_keys = set(_resolved_match_key_series(legacy).tolist())
        snapshot_keys = _resolved_match_key_series(snapshot_df)
        snapshot_overlay = snapshot_df.loc[~snapshot_keys.isin(legacy_keys)].copy()
        combined = pd.concat([legacy, snapshot_overlay], ignore_index=True, sort=False) if not snapshot_overlay.empty else legacy.copy()
    if combined.empty:
        return combined
    combined = combined.sort_values(["target_date_dt", "resolved_at", "signal_timestamp"], ascending=True)
    combined["cum_pnl"] = combined["pnl_usd"].cumsum()
    return combined


def _filter_rows_for_strategy(df: pd.DataFrame, strategy: str, main_mode: str) -> pd.DataFrame:
    if df.empty or "strategy" not in df.columns:
        return pd.DataFrame()
    subset = df[df["strategy"] == strategy].copy()
    if strategy in {"SINGLE", "LADDER", "CONVICTION"} and "mode" in subset.columns:
        subset = subset[subset["mode"].isin({main_mode, ""})].copy()
    elif "mode" in subset.columns and strategy not in {"SINGLE", "LADDER", "CONVICTION"}:
        subset = subset[subset["mode"].isin({"paper", ""})].copy()
    return subset


def _settlement_lookup_by_position_key(snapshot_df: pd.DataFrame) -> dict[str, dict]:
    if snapshot_df.empty or "position_key" not in snapshot_df.columns:
        return {}
    ordered = snapshot_df.sort_values(["resolved_at", "signal_timestamp"], ascending=True)
    latest = ordered.drop_duplicates(subset=["position_key"], keep="last")
    return {
        str(row["position_key"]): row.to_dict()
        for _, row in latest.iterrows()
        if str(row.get("position_key", "") or "")
    }


def split_positions_for_display(
    positions: list[dict],
    lookup: dict[str, dict],
) -> tuple[list[dict], list[dict], int]:
    today_str = _date.today().isoformat()
    still_open: list[dict] = []
    settled_rows: list[dict] = []
    stale_count = 0
    for position in positions:
        key = str(position.get("_position_key", "") or "")
        settlement = lookup.get(key)
        if settlement is not None:
            settled_rows.append({**position, **settlement})
            continue
        if str(position.get("date", "9999")) < today_str:
            stale_count += 1
            continue
        still_open.append(position)
    return still_open, settled_rows, stale_count


def _strat_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "n": 0,
            "wr": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "staked": 0.0,
            "roi": 0.0,
            "provisional": 0,
            "official": 0,
        }
    wins = int((df["outcome"] == "WIN").sum())
    losses = int((df["outcome"] == "LOSS").sum())
    n = wins + losses
    pnl = float(df["pnl_usd"].sum())
    staked = float(df["size_usd"].fillna(0).sum()) if "size_usd" in df.columns else 0.0
    win_pnls = df.loc[df["outcome"] == "WIN", "pnl_usd"]
    loss_pnls = df.loc[df["outcome"] == "LOSS", "pnl_usd"]
    provisional = int((df.get("settlement_phase", pd.Series(dtype=str)) == "proposed").sum()) if "settlement_phase" in df.columns else 0
    official = int((pd.to_numeric(df.get("official_resolved", 0), errors="coerce").fillna(0) == 1).sum()) if "official_resolved" in df.columns else 0
    return {
        "pnl": pnl,
        "wins": wins,
        "losses": losses,
        "n": n,
        "wr": wins / n if n else 0.0,
        "avg_win": float(win_pnls.mean()) if len(win_pnls) else 0.0,
        "avg_loss": float(loss_pnls.mean()) if len(loss_pnls) else 0.0,
        "staked": staked,
        "roi": pnl / staked * 100 if staked else 0.0,
        "provisional": provisional,
        "official": official,
    }


def _compute_live_stats(open_positions: list[dict], live_prices: dict[str, float]) -> dict[str, Any]:
    total_staked = 0.0
    unrealized_pnl = 0.0
    open_count = 0
    n_priced = 0
    for position in open_positions:
        token_id = position.get("token_id")
        cur = live_prices.get(token_id) if token_id else None
        fill = float(position.get("fill_price", 0) or 0)
        size = float(position.get("fill_size", 0) or 0)
        cost = float(position.get("cost", 0) or 0)
        open_count += 1
        total_staked += cost
        if cur is not None:
            unrealized_pnl += round((cur - fill) * size, 2)
            n_priced += 1
    return {
        "unrealized_pnl": round(unrealized_pnl, 2),
        "open_count": open_count,
        "n_priced": n_priced,
        "total_staked": round(total_staked, 2),
        "roi": round(unrealized_pnl / total_staked * 100, 1) if total_staked else 0.0,
    }


def fetch_live_position_prices(token_ids: tuple[str, ...]) -> dict[str, float]:
    prices: dict[str, float] = {}
    if not token_ids:
        return prices

    token_ids = tuple(dict.fromkeys(token_ids))
    session = requests.Session()

    batch_size = 100
    for idx in range(0, len(token_ids), batch_size):
        batch = list(token_ids)[idx : idx + batch_size]
        try:
            response = session.post(
                "https://clob.polymarket.com/prices",
                json=[{"token_id": token_id} for token_id in batch],
                timeout=15,
            )
            if not response.ok:
                continue
            data = response.json()
            for token_id, info in data.items():
                try:
                    sell = info.get("SELL") or info.get("BUY")
                    if sell is not None:
                        prices[token_id] = float(sell)
                except (ValueError, TypeError, AttributeError):
                    pass
        except Exception:
            pass

    missing = [token_id for token_id in token_ids if token_id not in prices]
    for token_id in missing:
        try:
            response = session.get(
                "https://gamma-api.polymarket.com/markets",
                params={"clob_token_ids": token_id},
                timeout=10,
            )
            if not response.ok:
                continue
            markets = response.json()
            if not isinstance(markets, list) or not markets:
                continue
            market = markets[0]
            outcome_prices = market.get("outcomePrices")
            clob_ids = market.get("clobTokenIds")
            if not outcome_prices or not clob_ids:
                continue
            clob_list = json.loads(clob_ids) if isinstance(clob_ids, str) else clob_ids
            price_list = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
            for clob_token_id, price in zip(clob_list, price_list):
                if clob_token_id == token_id:
                    prices[token_id] = float(price)
                    break
        except Exception:
            pass

    still_missing = [token_id for token_id in token_ids if token_id not in prices]
    for token_id in still_missing:
        try:
            response = session.get(
                "https://clob.polymarket.com/last-trade-price",
                params={"token_id": token_id},
                timeout=10,
            )
            if not response.ok:
                continue
            data = response.json()
            price = data.get("price")
            if price is None:
                continue
            value = float(price)
            if not (value == 0.5 and data.get("side", "") == ""):
                prices[token_id] = value
        except Exception:
            pass

    return prices


def build_dashboard_overview_payload(root: Path) -> dict[str, Any]:
    generated_at = datetime.now(UTC)
    main_mode = current_mode_name(root)
    resolved_df = _load_effective_resolved_df(root)
    snapshot_df = _load_settlement_snapshot_df(root)
    lookup = _settlement_lookup_by_position_key(snapshot_df)

    primary_positions = _load_primary_positions(root)
    shadow_positions = {
        spec["source_value"]: _load_shadow_positions(root, spec["source_value"])
        for spec in STRATEGY_SPECS
        if spec["source_kind"] == "shadow"
    }

    def _main_positions_for_strategy(strategy: str) -> list[dict]:
        tagged = [position for position in primary_positions if position.get("strategy") == strategy]
        if strategy == "LADDER":
            tagged = [position for position in primary_positions if position.get("strategy", "") in ("LADDER", "")]
        return tagged if tagged else primary_positions

    def _resolved_slice(strategy_key: str) -> pd.DataFrame:
        return _filter_rows_for_strategy(resolved_df, strategy_key, main_mode)

    strategy_rows: list[dict[str, Any]] = []
    all_open_positions: list[dict] = []

    for spec in STRATEGY_SPECS:
        if spec["source_kind"] == "main":
            positions = _main_positions_for_strategy(spec["source_value"])
        else:
            positions = shadow_positions.get(spec["source_value"], [])
        open_positions, _, _ = split_positions_for_display(positions, lookup)
        all_open_positions.extend(open_positions)
        df = _resolved_slice(spec["strategy_key"])
        strategy_rows.append(
            {
                "label": spec["label"],
                "strategy_key": spec["strategy_key"],
                "color": MODEL_COLORS.get(spec["strategy_key"], "#4DA6FF"),
                "positions": positions,
                "open_positions": open_positions,
                "settled_df": df,
                "settled_stats": _strat_stats(df),
            }
        )

    open_token_ids = tuple(
        dict.fromkeys(
            str(position.get("token_id", "") or "")
            for position in all_open_positions
            if position.get("token_id")
        )
    )
    live_prices = fetch_live_position_prices(open_token_ids) if open_token_ids else {}
    live_ts = datetime.now(UTC).strftime("%H:%M UTC")

    strategies: list[dict[str, Any]] = []
    chart_series: list[dict[str, Any]] = []
    for row in strategy_rows:
        live_stats = _compute_live_stats(row["open_positions"], live_prices)
        strategies.append(
            {
                "label": row["label"],
                "strategy_key": row["strategy_key"],
                "color": row["color"],
                "settled": row["settled_stats"],
                "live": live_stats,
            }
        )

        settled_df: pd.DataFrame = row["settled_df"]
        settled_points: list[dict[str, Any]] = []
        last_cum = 0.0
        if not settled_df.empty:
            df_s = (
                settled_df.sort_values("target_date_dt")
                .groupby("target_date_dt", as_index=False)["pnl_usd"]
                .sum()
                .sort_values("target_date_dt")
            )
            df_s["_cum"] = df_s["pnl_usd"].cumsum()
            settled_points = [
                {"x": value.isoformat(), "y": float(cum)}
                for value, cum in zip(df_s["target_date_dt"], df_s["_cum"])
                if pd.notna(value)
            ]
            if settled_points:
                last_cum = float(settled_points[-1]["y"])

        live_total = None
        if live_stats["open_count"] > 0:
            live_total = round(last_cum + live_stats["unrealized_pnl"], 2)

        chart_series.append(
            {
                "label": row["label"],
                "strategy_key": row["strategy_key"],
                "color": row["color"],
                "settled_points": settled_points,
                "live_total": live_total,
            }
        )

    strategies.sort(key=lambda item: item["settled"]["pnl"] + item["live"]["unrealized_pnl"], reverse=True)

    main_open_positions, main_settled_rows, stale_count = split_positions_for_display(primary_positions, lookup)
    live_book_rows: list[dict[str, Any]] = []
    for position in main_open_positions:
        token_id = position.get("token_id")
        current = live_prices.get(token_id)
        fill = float(position.get("fill_price", 0) or 0)
        size = float(position.get("fill_size", 0) or 0)
        cost = float(position.get("cost", 0) or 0)
        unreal = round((current - fill) * size, 2) if current is not None else None
        live_book_rows.append(
            {
                "City": position.get("city", ""),
                "Date": position.get("date", ""),
                "Bucket": position.get("bucket", ""),
                "Side": position.get("side", ""),
                "Entry": round(fill, 3),
                "Live": round(current, 3) if current is not None else "—",
                "Cost": round(cost, 2),
                "Unreal P&L": f"${unreal:+.2f}" if unreal is not None else "—",
            }
        )

    provisional_count = sum(
        1
        for row in main_settled_rows
        if str(row.get("settlement_phase", "") or "") == "proposed"
    )

    return {
        "generated_at_utc": generated_at.isoformat(),
        "live_ts": live_ts,
        "main_mode": main_mode,
        "strategies": strategies,
        "chart": {
            "series": chart_series,
            "live_point_x": generated_at.isoformat(),
            "live_ts": live_ts,
        },
        "live_book": {
            "rows": live_book_rows,
            "open_count": len(main_open_positions),
            "settled_count": len(main_settled_rows),
            "provisional_count": provisional_count,
            "stale_count": stale_count,
            "live_ts": live_ts,
        },
    }
