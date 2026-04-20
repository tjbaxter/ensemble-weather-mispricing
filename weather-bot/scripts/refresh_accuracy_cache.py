#!/usr/bin/env python3
"""Standalone accuracy-cache builder — runs independently of Streamlit.

Refreshes data/accuracy_rows_cache.json with the latest Open-Meteo
previous_day1/day2 predictions for every configured accuracy city.

Usage:
    python3 scripts/refresh_accuracy_cache.py
    python3 scripts/refresh_accuracy_cache.py --city "New York"
    python3 scripts/refresh_accuracy_cache.py --dry-run

Cron (VM) — hourly:
    15 * * * * /home/tombaxter/weather-bot/venv/bin/python3 \
               /home/tombaxter/weather-bot/scripts/refresh_accuracy_cache.py \
               >> /home/tombaxter/weather-bot/logs/accuracy_refresh.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.cities import canonical_dashboard_city  # noqa: E402

CACHE_PATH = ROOT / "data" / "accuracy_rows_cache.json"
SNAPSHOT_LOG_PATH = ROOT / "data" / "model_snapshot_log.json"
PM_CACHE_PATH = ROOT / "data" / "polymarket_cache.json"
OM_PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# ---------------------------------------------------------------------------
# City configuration — imported from dashboard.py's ACCURACY_CITIES.
# This avoids duplicating the large config dict.
# ---------------------------------------------------------------------------

def _load_accuracy_cities() -> dict:
    from config.accuracy_cities import ACCURACY_CITIES  # noqa: E402
    return ACCURACY_CITIES


# ---------------------------------------------------------------------------
# Scoring helpers (standalone copies — no Streamlit dependency)
# ---------------------------------------------------------------------------

def _hround(x: float) -> int:
    return math.floor(x + 0.5)


def _wins(pred: float, res_int: int, is_plus) -> bool:
    p = _hround(pred)
    if is_plus is True:
        return p >= res_int
    if is_plus is None:
        return p <= res_int
    return p == res_int


def _wins_range(pred_f: float, low, high, _bottom_thresh, _top_thresh) -> bool:
    p = _hround(pred_f)
    if low is None:
        return p <= (_bottom_thresh or high or 999)
    if high is None:
        return p >= (_top_thresh or low or -999)
    return low <= p <= high


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_cache() -> dict:
    return _load_json(CACHE_PATH)


def _load_snapshot_log() -> dict:
    return _load_json(SNAPSHOT_LOG_PATH)


def _load_pm_cache() -> dict:
    return _load_json(PM_CACHE_PATH)


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(CACHE_PATH.parent), suffix=".json.tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
        os.replace(tmp, str(CACHE_PATH))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _merge_aliased(data: dict, city: str) -> dict:
    """Read-side alias merge: combine data from canonical + alias keys."""
    from config.cities import _DASHBOARD_CITY_ALIASES
    primary = data.get(city, {})
    for alias, canonical in _DASHBOARD_CITY_ALIASES.items():
        if canonical == city and alias in data:
            alias_data = data[alias]
            for k, v in alias_data.items():
                if k not in primary:
                    primary[k] = v
    return primary


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def refresh_city(
    city: str,
    cfg: dict,
    cache: dict,
    snapshot_log: dict,
    pm_cache: dict,
    *,
    dry_run: bool = False,
) -> list[dict]:
    """Build/refresh accuracy rows for one city. Returns the full row list."""

    bucket_style = cfg.get("bucket_style", "exact_1c")
    temp_unit = cfg.get("temperature_unit", "celsius")
    model_keys = list(cfg.get("models", {}).keys())
    ens_keys = cfg["best_ensemble"]["model_keys"]
    n_models = len(model_keys)
    min_preds = max(1, (n_models * 2 + 2) // 3)

    dashboard_city = canonical_dashboard_city(city)

    pm_entries = _merge_aliased(pm_cache, dashboard_city)
    city_snap = _merge_aliased(snapshot_log, dashboard_city)

    hardcoded_pm = cfg.get("polymarket", {})
    all_pm = dict(hardcoded_pm)
    for ds, entry in pm_entries.items():
        all_pm[ds] = tuple(entry) if isinstance(entry, list) else entry

    cached_rows = cache.get(dashboard_city, [])
    cached_rows = [
        r for r in cached_rows
        if sum(1 for mk in model_keys if r.get(f"{mk}_d1") is not None) >= min_preds
    ]
    cached_dates = {r["date"] for r in cached_rows}

    now = datetime.now(UTC)
    today_str = now.strftime("%Y-%m-%d")
    yesterday_str = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    end = today_str if today_str in all_pm else yesterday_str
    all_pm_dates = sorted(all_pm.keys())
    new_dates = [d for d in all_pm_dates if d not in cached_dates and d <= end]

    if not new_dates:
        print(f"  {dashboard_city}: up to date ({len(cached_rows)} rows cached)")
        return cached_rows

    start = min(new_dates)
    print(f"  {dashboard_city}: fetching {len(new_dates)} new dates ({start} → {max(new_dates)})")

    raw: dict[str, tuple[dict, dict]] = {}
    for model_key in model_keys:
        params = {
            "latitude": cfg["lat"],
            "longitude": cfg["lon"],
            "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
            "models": model_key,
            "timezone": cfg["timezone"],
            "start_date": start,
            "end_date": end,
        }
        if temp_unit != "celsius":
            params["temperature_unit"] = temp_unit
        try:
            r = requests.get(OM_PREV_URL, params=params, timeout=30)
            d = r.json()
            if "error" in d:
                raw[model_key] = ({}, {})
                continue
            times = d["hourly"]["time"]
            v1 = d["hourly"].get("temperature_2m_previous_day1", [])
            v2 = d["hourly"].get("temperature_2m_previous_day2", [])
            daily1: dict[str, list] = defaultdict(list)
            daily2: dict[str, list] = defaultdict(list)
            for t, a, b in zip(times, v1, v2):
                dt = t[:10]
                if a is not None:
                    daily1[dt].append(a)
                if b is not None:
                    daily2[dt].append(b)
            raw[model_key] = (dict(daily1), dict(daily2))
        except Exception as exc:
            print(f"    {model_key}: {exc}")
            raw[model_key] = ({}, {})

    win_fn = _wins_range if bucket_style == "range_2f" else None
    rows: list[dict] = []

    for dt in sorted(new_dates):
        pm_entry = all_pm.get(dt)
        if pm_entry is None:
            continue

        if bucket_style == "range_2f":
            lbl, low, high, bottom_thresh, top_thresh = pm_entry
            row: dict = {
                "date": dt,
                "resolved": lbl,
                "range_low": low,
                "range_high": high,
                "bottom_thresh": bottom_thresh,
                "top_thresh": top_thresh,
            }
        else:
            lbl, res_int, is_plus = pm_entry[:3]
            row = {"date": dt, "resolved": lbl, "res_int": res_int, "is_plus": is_plus}

        def compute_win(pred):
            if pred is None:
                return None
            if bucket_style == "range_2f":
                if low is None and high is None:
                    return None
                return _wins_range(pred, low, high, bottom_thresh, top_thresh)
            if res_int is None:
                return None
            return _wins(pred, res_int, is_plus)

        snap_preds: dict = {}
        snap_entry = city_snap.get(dt)
        if isinstance(snap_entry, dict):
            snap_preds = snap_entry.get("preds", {})

        for mk in model_keys:
            d1_map, d2_map = raw.get(mk, ({}, {}))
            p1 = _hround(max(d1_map[dt]) * 10) / 10 if d1_map.get(dt) else None
            p2 = _hround(max(d2_map[dt]) * 10) / 10 if d2_map.get(dt) else None
            if p1 is None and mk in snap_preds:
                try:
                    p1 = _hround(float(snap_preds[mk]) * 10) / 10
                except (TypeError, ValueError):
                    pass
            row[f"{mk}_d1"] = p1
            row[f"{mk}_d2"] = p2
            row[f"{mk}_d1_win"] = compute_win(p1)
            row[f"{mk}_d2_win"] = compute_win(p2)

        ens_d1 = [row[f"{k}_d1"] for k in ens_keys if row.get(f"{k}_d1") is not None]
        best_ens_d1 = (_hround(sum(ens_d1) / len(ens_d1) * 10) / 10) if len(ens_d1) == len(ens_keys) else None
        row["best_ens_d1"] = best_ens_d1
        row["best_ens_d1_win"] = compute_win(best_ens_d1)

        ens_d2 = [row[f"{k}_d2"] for k in ens_keys if row.get(f"{k}_d2") is not None]
        best_ens_d2 = (_hround(sum(ens_d2) / len(ens_d2) * 10) / 10) if len(ens_d2) == len(ens_keys) else None
        row["best_ens_d2"] = best_ens_d2
        row["best_ens_d2_win"] = compute_win(best_ens_d2)

        for hyp in cfg.get("hypothesis_ensembles", []):
            hkeys = hyp["model_keys"]
            hweights = hyp.get("weights")
            hpreds = [row[f"{k}_d1"] for k in hkeys if row.get(f"{k}_d1") is not None]
            if len(hpreds) == len(hkeys):
                if hweights:
                    wavg = sum(p * w for p, w in zip(hpreds, hweights))
                else:
                    wavg = sum(hpreds) / len(hpreds)
                hval = _hround(wavg * 10) / 10
            else:
                hval = None
            row[f"{hyp['key']}_d1"] = hval
            row[f"{hyp['key']}_d1_win"] = compute_win(hval)

        sf = cfg.get("spread_filter")
        if sf:
            sf_preds = [row[f"{k}_d1"] for k in sf["model_keys"] if row.get(f"{k}_d1") is not None]
            if len(sf_preds) == len(sf["model_keys"]):
                row["spread_d1"] = round(max(sf_preds) - min(sf_preds), 1)
                row["spread_green"] = row["spread_d1"] <= sf["threshold"]
            else:
                row["spread_d1"] = None
                row["spread_green"] = None

        rows.append(row)

    merged: dict[str, dict] = {}
    for r in cached_rows + rows:
        d = r["date"]
        existing = merged.get(d)
        if existing is None:
            merged[d] = r
        else:
            n_new = sum(1 for k, v in r.items() if k.endswith("_d1") and v is not None)
            n_old = sum(1 for k, v in existing.items() if k.endswith("_d1") and v is not None)
            if n_new > n_old:
                merged[d] = r

    all_rows = sorted(merged.values(), key=lambda r: r["date"])
    n_new = len(rows)
    n_pop = sum(
        1 for r in rows
        if sum(1 for k, v in r.items() if k.endswith("_d1") and v is not None) >= min_preds
    )
    print(f"    → {n_new} new rows ({n_pop} fully populated), {len(all_rows)} total")
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh accuracy_rows_cache.json independently of Streamlit."
    )
    parser.add_argument("--city", default=None, help="Refresh only this city.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but do not write.")
    args = parser.parse_args()

    print(f"\nAccuracy Cache Refresh — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    accuracy_cities = _load_accuracy_cities()
    cache = _load_cache()
    snapshot_log = _load_snapshot_log()
    pm_cache = _load_pm_cache()

    cities = [args.city] if args.city else list(accuracy_cities.keys())

    for city in cities:
        cfg = accuracy_cities.get(city)
        if cfg is None:
            print(f"  {city}: not in ACCURACY_CITIES, skipping")
            continue
        dashboard_city = canonical_dashboard_city(city)
        try:
            updated_rows = refresh_city(
                city, cfg, cache, snapshot_log, pm_cache, dry_run=args.dry_run
            )
            if not args.dry_run:
                cache[dashboard_city] = updated_rows
        except Exception as exc:
            print(f"  {dashboard_city}: ERROR — {exc}")

    if not args.dry_run:
        _save_cache(cache)
        print(f"\n✅ Saved to {CACHE_PATH}")
    else:
        print("\n[dry-run] Nothing written.")


if __name__ == "__main__":
    main()
