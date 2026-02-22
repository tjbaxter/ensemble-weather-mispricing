"""fetch_model_predictions.py

For each resolved Polymarket market, fetch what each weather model predicted
the day before (previous_day1) using the Open-Meteo Previous Runs API.

DATA LEAKAGE PREVENTION — CRITICAL:
  ✅ ONLY uses temperature_2m_previous_day1_* (24h lead time)
  ❌ NEVER uses temperature_2m_* (Day 0 — intraday / quasi-observed)

The daily max of the previous_day1 hourly series for the target date is what
the model would have been showing you at bet time (day before resolution).

Usage:
    python backtest/fetch_model_predictions.py
    python backtest/fetch_model_predictions.py --city Seoul
    python backtest/fetch_model_predictions.py --model ncep_aigfs025
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
DATA_DIR = ROOT / "backtest" / "data"
IN_JSON  = DATA_DIR / "resolved_markets.json"
OUT_JSON = DATA_DIR / "model_predictions.json"

# Models to evaluate, in priority order
MODELS = [
    "ncep_aigfs025",       # ELITE: only day-before model that hit 14.2°C Seoul Feb 21
    "gfs_graphcast025",    # ELITE: DeepMind GraphCast, strong convergence
    "kma_gdps",            # ELITE: Korean NWP, home-turf advantage for Seoul
    "ecmwf_ifs025",        # SECONDARY: traditional ECMWF
    "gem_global",          # SECONDARY: Canadian GEM
    "gfs_seamless",        # BASELINE: worst on Feb 21, kept as reference
    "icon_seamless",       # REFERENCE: DWD Germany
]

CACHE_PATH = DATA_DIR / "prediction_cache.json"
_cache: dict[str, float | None] = {}


def _load_cache() -> None:
    global _cache
    if CACHE_PATH.exists():
        try:
            _cache = json.loads(CACHE_PATH.read_text())
            print(f"Loaded prediction cache: {len(_cache)} entries")
        except (ValueError, json.JSONDecodeError):
            _cache = {}


def _save_cache() -> None:
    CACHE_PATH.write_text(json.dumps(_cache, indent=2))


def _cache_key(lat: float, lon: float, model: str, target_date: str, unit: str) -> str:
    return f"{lat:.4f}|{lon:.4f}|{model}|{target_date}|{unit}"


def fetch_previous_day1_max(
    lat: float,
    lon: float,
    model: str,
    target_date: str,
    timezone: str,
    unit: str = "C",
) -> float | None:
    """Return the max of previous_day1 hourly temps for target_date.

    This is what the model was predicting 24 hours before — the ONLY valid
    signal for day-before betting. No data leakage.
    """
    ck = _cache_key(lat, lon, model, target_date, unit)
    if ck in _cache:
        return _cache[ck]

    temp_unit = "fahrenheit" if unit == "F" else "celsius"

    # We need enough past_days to reach the target_date.
    # Open-Meteo previous-runs only goes back ~60 days via past_days.
    # Use start_date / end_date for older data.
    today = date.today()
    target = date.fromisoformat(target_date)
    days_back = (today - target).days + 2  # +2 buffer

    params: dict = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m_previous_day1",
        "models": model,
        "temperature_unit": temp_unit,
        "timezone": timezone,
        "forecast_days": 1,
    }

    if days_back <= 92:
        params["past_days"] = days_back
    else:
        # Use explicit date range for older data
        start = (target - timedelta(days=1)).isoformat()
        end = (target + timedelta(days=1)).isoformat()
        params["start_date"] = start
        params["end_date"] = end
        del params["forecast_days"]

    for attempt in range(3):
        try:
            resp = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"    [rate-limit] 429 on {model}. Waiting {wait}s …")
                time.sleep(wait)
                continue
            if resp.status_code == 400:
                # Model may not support this date range
                _cache[ck] = None
                return None
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.RequestException:
            time.sleep(2 ** attempt)
    else:
        _cache[ck] = None
        return None

    if isinstance(payload, list):
        payload = payload[0] if payload else {}

    hourly = payload.get("hourly", {})
    times  = hourly.get("time", [])
    # The variable name may include the model suffix
    temps_key = next(
        (k for k in hourly if k.startswith("temperature_2m_previous_day1")), None
    )
    if temps_key is None:
        _cache[ck] = None
        return None

    temps = hourly.get(temps_key, [])

    day_vals: list[float] = []
    for ts, val in zip(times, temps):
        if val is None:
            continue
        try:
            if str(ts)[:10] == target_date:
                day_vals.append(float(val))
        except (ValueError, TypeError):
            continue

    result = max(day_vals) if day_vals else None
    _cache[ck] = result
    return result


def fetch_all_predictions(
    records: list[dict],
    model_filter: str | None = None,
    city_filter: str | None = None,
) -> dict[str, dict]:
    """Fetch previous_day1 predictions for all records and all models.

    Returns:
        {event_id: {model_id: predicted_max_temp_or_None}}
    """
    models = [model_filter] if model_filter else MODELS
    results: dict[str, dict] = {}

    total = len(records)
    for i, rec in enumerate(records):
        eid   = rec["event_id"]
        city  = rec["city_slug"]
        lat   = rec["lat"]
        lon   = rec["lon"]
        tz    = rec["timezone"]
        unit  = rec["unit"]
        tdate = rec["target_date"]

        if city_filter and city_filter.lower() not in (city, rec["city"].lower()):
            continue

        print(f"[{i+1:3d}/{total}] {tdate}  {rec['city']:<15}", end="  ")
        results[eid] = {}

        for model in models:
            pred = fetch_previous_day1_max(lat, lon, model, tdate, tz, unit)
            results[eid][model] = pred
            marker = f"{pred:.1f}" if pred is not None else "---"
            print(f"{model}={marker}", end="  ")
            time.sleep(0.3)  # gentle rate limiting

        print()

        # Save cache periodically
        if (i + 1) % 10 == 0:
            _save_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",  default=None, help="Filter to one city")
    parser.add_argument("--model", default=None, help="Run only one model")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _load_cache()

    if not IN_JSON.exists():
        print(f"ERROR: {IN_JSON} not found. Run fetch_polymarket_resolved.py first.")
        sys.exit(1)

    records = json.loads(IN_JSON.read_text())
    print(f"Loaded {len(records)} resolved markets\n")

    predictions = fetch_all_predictions(records, model_filter=args.model, city_filter=args.city)

    _save_cache()
    print(f"\nSaved cache: {len(_cache)} entries → {CACHE_PATH}")

    # Load existing enriched data so city-by-city fetches are cumulative.
    # Previously fetched cities keep their predictions; only newly fetched
    # cities (or models) are updated.
    existing: dict[str, dict] = {}
    if OUT_JSON.exists():
        try:
            for row in json.loads(OUT_JSON.read_text()):
                existing[str(row["event_id"])] = row
        except (ValueError, json.JSONDecodeError):
            pass

    enriched: list[dict] = []
    for rec in records:
        eid = rec["event_id"]
        row = dict(existing.get(eid, rec))  # start from existing data if available
        new_preds = predictions.get(eid, {})
        for model in MODELS:
            pred_key = f"pred_{model}"
            new_val = new_preds.get(model)
            # Only overwrite if we actually fetched this record this run
            if eid in predictions:
                row[pred_key] = new_val
            elif pred_key not in row:
                row[pred_key] = None
        enriched.append(row)

    with OUT_JSON.open("w") as f:
        json.dump(enriched, f, indent=2)

    filled = sum(
        1 for r in enriched
        if any(r.get(f"pred_{m}") is not None for m in MODELS)
    )
    print(f"Saved → {OUT_JSON}  ({filled}/{len(enriched)} records have ≥1 prediction)")


if __name__ == "__main__":
    main()
