#!/usr/bin/env python3
"""Daily logger for Open-Meteo D+1 model prediction snapshots.

Captures what each model predicts for TOMORROW at the 12Z window (19:05 UTC)
and writes it to data/model_snapshot_log.json.  This is the canonical 12Z
snapshot — the same run that historical accuracy rows use (previous_day1).

The dashboard uses this file to show consistent 12Z predictions in the
morning when the live API is serving the less accurate 00Z run.

Usage:
    python3 scripts/log_model_snapshots.py
    python3 scripts/log_model_snapshots.py --dry-run

Cron (VM) — run 5 min after commercial forecasts, both post-18:30 trigger:
    5 19 * * * /home/tombaxter/weather-bot/venv/bin/python3 \
               /home/tombaxter/weather-bot/scripts/log_model_snapshots.py \
               >> /home/tombaxter/weather-bot/logs/model_snapshot.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests

ROOT     = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "data" / "model_snapshot_log.json"
LOCK_HOUR_UTC = 19  # After this hour the snapshot is frozen (canonical 12Z)

sys.path.insert(0, str(ROOT))
from config.cities import STATIONS, canonical_dashboard_city  # noqa: E402


def _build_cities() -> dict[str, dict]:
    """Auto-generate snapshot cities from the canonical STATIONS config."""
    cities: dict[str, dict] = {}
    for _icao, cfg in STATIONS.items():
        city = canonical_dashboard_city(cfg.get("market_label", ""))
        if not city or not cfg.get("ensemble_models"):
            continue
        cities[city] = {
            "lat": cfg["lat"],
            "lon": cfg["lon"],
            "timezone": cfg["timezone"],
            "unit": "fahrenheit" if cfg.get("resolution_unit") == "F" else "celsius",
            "models": list(cfg["ensemble_models"]),
        }
    return cities


CITIES = _build_cities()


def _hround(x: float) -> float:
    import math
    return math.floor(x * 10 + 0.5) / 10


def fetch_city_preds(city: str, cfg: dict, target_date: str) -> dict[str, float]:
    """Fetch D+1 predictions for all models for a city."""
    preds: dict[str, float] = {}
    params_base = {
        "latitude":  cfg["lat"],
        "longitude": cfg["lon"],
        "hourly":    "temperature_2m",
        "start_date": target_date,
        "end_date":   target_date,
        "timezone":  cfg["timezone"],
    }
    if cfg.get("unit") == "fahrenheit":
        params_base["temperature_unit"] = "fahrenheit"

    for mk in cfg["models"]:
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={**params_base, "models": mk},
                timeout=15,
            )
            d = r.json()
            if "error" in d:
                print(f"    {mk}: API error — {d['error']}")
                continue
            vals = [v for v in d.get("hourly", {}).get("temperature_2m", []) if v is not None]
            if vals:
                preds[mk] = _hround(max(vals))
        except Exception as exc:
            print(f"    {mk}: {exc}")

    return preds


def load_log() -> dict:
    try:
        if LOG_PATH.exists():
            return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_log(log: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(LOG_PATH)


def run(target_date: str, dry_run: bool = False) -> None:
    now_utc = datetime.now(UTC)
    print(f"\n{'DRY RUN — ' if dry_run else ''}Model Snapshot Logger")
    print(f"Target date: {target_date}")
    print(f"Logged at:   {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    log      = load_log()
    any_new  = False

    for city, cfg in CITIES.items():
        city_log = log.setdefault(city, {})
        existing = city_log.get(target_date)
        n_expected = len(cfg["models"])
        min_complete = max(1, (n_expected * 2 + 2) // 3)  # ceil(2/3)

        # Lock check: only skip if entry has sufficient model coverage.
        # Incomplete entries (API failures) are always eligible for backfill.
        if existing and not dry_run:
            n_have = len(existing.get("preds", {}))
            try:
                logged_hour = datetime.fromisoformat(existing["logged_at"]).hour
                if logged_hour >= LOCK_HOUR_UTC and n_have >= min_complete:
                    print(f"  {city:<16} locked ({n_have}/{n_expected} models @ {existing['logged_at'][11:16]} UTC)")
                    continue
            except Exception:
                pass
            if n_have < min_complete:
                missing = [m for m in cfg["models"] if m not in existing.get("preds", {})]
                print(f"  {city:<16} incomplete ({n_have}/{n_expected}), backfilling {len(missing)} models...")

        if not existing:
            print(f"  {city:<16} fetching {n_expected} models...")

        preds = fetch_city_preds(city, cfg, target_date)

        if not preds:
            print(f"    ⚠ no predictions returned")
            continue

        # Merge with existing predictions so we never lose already-captured data
        if existing and existing.get("preds"):
            merged_preds = dict(existing["preds"])
            merged_preds.update(preds)
            preds = merged_preds

        # Compute spread for spread-filter models if applicable
        spread_models = {
            "London": ["meteofrance_arome_france", "meteofrance_seamless",
                       "meteofrance_arome_france_hd", "icon_seamless", "dmi_seamless"],
        }
        sf_keys = spread_models.get(city, [])
        sf_vals = [preds[k] for k in sf_keys if k in preds]
        spread  = round(max(sf_vals) - min(sf_vals), 1) if len(sf_vals) >= 2 else None

        unit_sym = "°F" if cfg.get("unit") == "fahrenheit" else "°C"
        top_pred = preds.get(sf_keys[0] if sf_keys else cfg["models"][0])
        spread_s = f"  spread={spread}{unit_sym}" if spread is not None else ""
        print(f"    → {len(preds)}/{n_expected} models  top={top_pred}{unit_sym}{spread_s}")

        if not dry_run:
            city_log[target_date] = {
                "preds":     preds,
                "spread":    spread,
                "logged_at": now_utc.isoformat(),
            }
            any_new = True

    if not dry_run and any_new:
        save_log(log)
        print(f"\n✅ Saved to {LOG_PATH}")
    elif dry_run:
        print("\n[dry-run] Nothing written.")
    else:
        print("\n(All cities already locked for this date.)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default=None,
                        help="Target date (YYYY-MM-DD). Defaults to tomorrow.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch but do not write to disk.")
    args = parser.parse_args()

    if args.date:
        try:
            date.fromisoformat(args.date)
            targets = [args.date]
        except ValueError:
            print(f"Invalid date: {args.date}")
            sys.exit(1)
    else:
        today = date.today()
        targets = [
            (today + timedelta(days=1)).isoformat(),
            (today + timedelta(days=2)).isoformat(),
            (today + timedelta(days=3)).isoformat(),
        ]

    for target in targets:
        run(target, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
