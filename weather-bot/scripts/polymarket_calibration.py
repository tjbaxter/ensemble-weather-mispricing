"""Polymarket-grounded model accuracy calibration.

The ONLY question that matters for calibration:
  Did our model's forecast for Day D land in the bucket that Polymarket resolved to?

No market price. No edge calculation. Just:
  model_forecast_temp → which bucket → did Polymarket resolve there?

We fetch:
  1. Open-Meteo previous-runs-api: what each model said on Day D-1 about Day D max temp
  2. Polymarket resolved event: which bucket won on Day D

That's it. This tells us which model / station combinations are actually accurate.

Usage:
    python3 scripts/polymarket_calibration.py --past-days 60
    python3 scripts/polymarket_calibration.py --past-days 60 --station EGLC
    python3 scripts/polymarket_calibration.py --past-days 60 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.cities import STATIONS
from config.settings import ENSEMBLE_PREVIOUS_RUNS_API_URL, GAMMA_API_URL

OUTPUT_PATH = ROOT / "logs" / "polymarket_calibration.json"
# Forecast cache: avoids re-hitting the rate-limited previous-runs-api on re-runs
FORECAST_CACHE_PATH = ROOT / "logs" / "forecast_cache.json"

# All Open-Meteo models we want to score
MODELS = ("gfs_seamless", "ecmwf_ifs025", "icon_seamless_eps", "ecmwf_aifs025")


# ---------------------------------------------------------------------------
# Bucket label parsing
# ---------------------------------------------------------------------------

def _clean_bucket_label(raw: str | None) -> str | None:
    """
    Normalise Polymarket groupItemTitle to a canonical bucket string.

    Examples:
      "36-37°F"        → "36-37"
      "78°F or higher" → "78+"
      "67°F or below"  → "67-"
      "12°C"           → "12"
      "8"              → "8"
    """
    if not raw:
        return None
    raw = raw.strip()

    # "or higher / or above / +" including optional degree unit between number and phrase
    m = re.match(r"(-?\d+)\s*[°º]?[A-Z]?\s*(?:or higher|or above|\+)", raw, re.IGNORECASE)
    if m:
        return f"{m.group(1)}+"

    # "or below / or lower / and below"
    m = re.match(
        r"(-?\d+)\s*[°º]?[A-Z]?\s*(?:or below|or lower|and below|≤)", raw, re.IGNORECASE
    )
    if m:
        return f"{m.group(1)}-"

    # Range "36-37°F" or "-5--4°C"
    m = re.search(r"(\d)\s*[-–]\s*(-?\d)", raw)
    if m:
        split_idx = m.start() + 1
        left = raw[:split_idx].strip().rstrip("°ºFC ")
        right = raw[split_idx + 1:].strip().split("°")[0].split("º")[0].strip()
        left = re.sub(r"\s*[-–]\s*$", "", left)
        try:
            float(left)
            float(right)
            return f"{left}-{right}"
        except ValueError:
            pass

    # Plain number with optional unit: "12" or "8°C"
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*[°º]?[A-Z]?$", raw, re.IGNORECASE)
    if m:
        return m.group(1)

    return None


def _parse_bucket(label: str) -> tuple[float, float] | None:
    """
    Convert canonical bucket string to (low, high) numeric bounds.

    "36-37"  → (36, 37)
    "78+"    → (78, 999)    open upper tail
    "67-"    → (-999, 67)   "or below"
    "6"      → (5.5, 6.5)   single-degree Celsius (WU rounding window)
    """
    clean = label.strip()
    try:
        if clean.endswith("+"):
            return float(clean[:-1]), 999.0
        if clean.endswith("-"):
            return -999.0, float(clean[:-1])
        # Range: find dash separating two numbers (not a leading minus)
        m = re.search(r"(\d)-(-?\d)", clean)
        if m:
            idx = m.start() + 1
            return float(clean[:idx]), float(clean[idx + 1:])
        # Single integer (Celsius single-degree bucket)
        val = float(clean)
        return val - 0.5, val + 0.5
    except ValueError:
        return None


def _extract_winning_bucket(markets: list[dict]) -> str | None:
    """Return the canonical bucket label for the market that resolved YES."""
    for market in markets:
        prices_raw = market.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        if not prices:
            continue
        try:
            if float(str(prices[0])) >= 0.95:
                raw_label = market.get("groupItemTitle") or ""
                return _clean_bucket_label(raw_label)
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Forecast cache (persisted to disk to survive restarts + avoid rate limits)
# ---------------------------------------------------------------------------

_FORECAST_CACHE: dict[str, float | None] = {}

def _load_forecast_cache() -> None:
    global _FORECAST_CACHE
    if FORECAST_CACHE_PATH.exists():
        try:
            _FORECAST_CACHE = json.loads(FORECAST_CACHE_PATH.read_text())
        except (ValueError, json.JSONDecodeError):
            _FORECAST_CACHE = {}

def _save_forecast_cache() -> None:
    FORECAST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FORECAST_CACHE_PATH.write_text(json.dumps(_FORECAST_CACHE, indent=2))

def _cache_key(icao: str, model: str, date_iso: str) -> str:
    return f"{icao}|{model}|{date_iso}"


# ---------------------------------------------------------------------------
# Open-Meteo previous-runs forecast
# ---------------------------------------------------------------------------

async def _fetch_forecast(
    client: httpx.AsyncClient,
    icao: str,
    station: dict,
    model: str,
    target_date_iso: str,
    past_days: int,
) -> float | None:
    """
    Fetch the max temperature forecast that model issued on target_date - 1
    (previous_day1 variable) for target_date.

    Results are cached to disk to survive rate limits across runs.
    """
    ck = _cache_key(icao, model, target_date_iso)
    if ck in _FORECAST_CACHE:
        return _FORECAST_CACHE[ck]

    unit = "fahrenheit" if station.get("resolution_unit") == "F" else "celsius"
    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "hourly": "temperature_2m_previous_day1",
        "models": model,
        "temperature_unit": unit,
        "timezone": "auto",
        "past_days": past_days,
    }

    for attempt in range(3):
        try:
            resp = await client.get(ENSEMBLE_PREVIOUS_RUNS_API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                # Rate limited — wait and retry (only makes sense if user has time)
                wait = 15 * (attempt + 1)
                print(f"    [rate-limit] 429 on {model} for {icao}. Waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            payload = resp.json()
            break
        except (httpx.HTTPError, ValueError):
            _FORECAST_CACHE[ck] = None  # type: ignore[assignment]
            return None
    else:
        _FORECAST_CACHE[ck] = None  # type: ignore[assignment]
        return None

    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        _FORECAST_CACHE[ck] = None  # type: ignore[assignment]
        return None

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    values = hourly.get("temperature_2m_previous_day1", [])

    day_vals: list[float] = []
    for ts, val in zip(times, values):
        if val is None:
            continue
        try:
            if datetime.fromisoformat(str(ts)).date().isoformat() == target_date_iso:
                day_vals.append(float(val))
        except (ValueError, TypeError):
            continue

    result = max(day_vals) if day_vals else None
    _FORECAST_CACHE[ck] = result
    return result


# ---------------------------------------------------------------------------
# Polymarket resolved event fetch
# ---------------------------------------------------------------------------

async def _fetch_resolved_event(client: httpx.AsyncClient, slug: str) -> dict | None:
    """Return event dict if resolved, else None."""
    try:
        resp = await client.get(
            f"{GAMMA_API_URL}/events/slug/{slug}", timeout=15
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        event = resp.json()
    except (httpx.HTTPError, ValueError):
        return None

    markets = event.get("markets", [])
    if not markets:
        return None
    # Must have at least one closed market
    if not any(m.get("closed") for m in markets):
        return None
    return event


# ---------------------------------------------------------------------------
# Main calibration loop
# ---------------------------------------------------------------------------

async def calibrate(
    past_days: int,
    station_filter: str | None,
    verbose: bool,
) -> dict:
    today = datetime.now(UTC).date()
    results: dict[str, dict] = {}

    async with httpx.AsyncClient() as client:
        for icao, station in STATIONS.items():
            if station_filter and icao != station_filter:
                continue
            city_slug = station.get("city_slug")
            if not city_slug:
                continue

            unit = station.get("resolution_unit", "F")

            # per-model: list of {"date", "model_temp", "winning_bucket", "hit": bool, "error"?: str}
            per_model: dict[str, list[dict]] = {m: [] for m in MODELS}

            print(f"\n{'='*60}")
            print(f"Station: {icao} ({station['market_label']})  unit={unit}")

            for day_offset in range(2, past_days + 1):
                target_date = today - timedelta(days=day_offset)
                target_iso = target_date.isoformat()
                month = target_date.strftime("%B").lower()
                slug = (
                    f"highest-temperature-in-{city_slug}-on-"
                    f"{month}-{target_date.day}-{target_date.year}"
                )

                event = await _fetch_resolved_event(client, slug)
                if event is None:
                    continue

                winning_bucket = _extract_winning_bucket(event.get("markets", []))
                if winning_bucket is None:
                    if verbose:
                        print(f"  {target_iso}: could not identify winning bucket")
                    continue

                bounds = _parse_bucket(winning_bucket)
                if bounds is None:
                    if verbose:
                        print(f"  {target_iso}: unparseable bucket '{winning_bucket}'")
                    continue

                low, high = bounds

                # Fetch each model's previous-day-1 forecast
                for model in MODELS:
                    forecast_temp = await _fetch_forecast(
                        client, icao, station, model, target_iso, past_days + 5
                    )
                    if forecast_temp is None:
                        per_model[model].append({
                            "date": target_iso,
                            "winning_bucket": winning_bucket,
                            "error": "no_forecast",
                        })
                        continue

                    hit = low <= forecast_temp <= high
                    # Distance from bucket centre (how far off the model was)
                    bucket_centre = (
                        (low + min(high, low + 4)) / 2
                        if high == 999.0
                        else (low + high) / 2
                    )
                    error_deg = forecast_temp - bucket_centre

                    row = {
                        "date": target_iso,
                        "model_temp": round(forecast_temp, 2),
                        "winning_bucket": winning_bucket,
                        "bucket_low": low,
                        "bucket_high": high,
                        "hit": hit,
                        "error_deg": round(error_deg, 2),
                    }
                    per_model[model].append(row)

                    if verbose:
                        mark = "HIT ✓" if hit else f"miss ({error_deg:+.1f}°)"
                        print(
                            f"  {target_iso}  bucket={winning_bucket:<8}  "
                            f"{model:<22}  pred={forecast_temp:.1f}  {mark}"
                        )

            # Summarise per-model
            model_summaries: dict[str, dict] = {}
            for model, rows in per_model.items():
                valid = [r for r in rows if "error" not in r]
                if not valid:
                    model_summaries[model] = {"n_days": 0}
                    continue
                hits = [r for r in valid if r["hit"]]
                errors = [r["error_deg"] for r in valid]
                model_summaries[model] = {
                    "n_days": len(valid),
                    "n_hits": len(hits),
                    "hit_rate": round(len(hits) / len(valid), 3),
                    "mae_deg": round(sum(abs(e) for e in errors) / len(errors), 3),
                    "bias_deg": round(sum(errors) / len(errors), 3),  # + = warm bias, - = cold
                    "rmse_deg": round(
                        math.sqrt(sum(e ** 2 for e in errors) / len(errors)), 3
                    ),
                }

            results[icao] = {
                "unit": unit,
                "market_label": station["market_label"],
                "models": model_summaries,
                "raw": {m: per_model[m] for m in MODELS},
            }

    return results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_summary(results: dict) -> None:
    print("\n" + "=" * 90)
    print("MODEL ACCURACY vs POLYMARKET RESOLUTION — Ground Truth Calibration")
    print("=" * 90)
    print(
        f"{'Station':<8} {'Model':<22} {'Days':>5} {'HitRate':>8} "
        f"{'Hits':>6} {'MAE':>7} {'Bias':>7} {'RMSE':>7}"
    )
    print("-" * 90)

    # Collect all rows, sort by HitRate DESC
    all_rows = []
    for icao, data in results.items():
        unit = data.get("unit", "F")
        for model, m in data.get("models", {}).items():
            if m.get("n_days", 0) < 3:
                continue
            all_rows.append((icao, unit, model, m))

    all_rows.sort(key=lambda r: r[3].get("hit_rate", 0), reverse=True)

    for icao, unit, model, m in all_rows:
        print(
            f"{icao:<8} {model:<22} {m['n_days']:>5} "
            f"{m.get('hit_rate', 0):>8.1%} "
            f"{m.get('n_hits', 0):>6} "
            f"{m.get('mae_deg', 0):>7.2f}{unit} "
            f"{m.get('bias_deg', 0):>+7.2f}{unit} "
            f"{m.get('rmse_deg', 0):>7.2f}{unit}"
        )

    print()
    print("KEY:")
    print("  HitRate — % of days model's point forecast fell inside Polymarket winning bucket")
    print("  MAE     — mean absolute error from bucket centre (lower = more accurate)")
    print("  Bias    — systematic warm (+) or cold (-) offset from bucket centre")
    print("  RMSE    — root mean squared error from bucket centre")
    print()

    # Best model per station
    print("BEST MODEL PER STATION:")
    for icao, data in results.items():
        unit = data.get("unit", "F")
        models_with_data = [
            (model, m)
            for model, m in data.get("models", {}).items()
            if m.get("n_days", 0) >= 3
        ]
        if not models_with_data:
            continue
        best_model, best_m = max(models_with_data, key=lambda x: x[1].get("hit_rate", 0))
        print(
            f"  {icao:<6} → {best_model:<22}  "
            f"HitRate={best_m['hit_rate']:.0%}  "
            f"MAE={best_m['mae_deg']:.2f}{unit}  "
            f"Bias={best_m['bias_deg']:+.2f}{unit}"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate model accuracy against Polymarket resolved outcomes"
    )
    parser.add_argument("--past-days", type=int, default=60,
                        help="How many days back to score (default 60)")
    parser.add_argument("--station", type=str, default=None,
                        help="Single ICAO to run (e.g. EGLC)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every day's result line")
    args = parser.parse_args()

    print(
        f"Running model accuracy calibration: "
        f"past_days={args.past_days}  station={args.station or 'ALL'}"
    )

    _load_forecast_cache()
    cached_count = len([v for v in _FORECAST_CACHE.values() if v is not None])
    print(f"Loaded forecast cache: {len(_FORECAST_CACHE)} entries ({cached_count} with data)")

    results = await calibrate(args.past_days, args.station, args.verbose)

    _save_forecast_cache()
    print(f"Saved forecast cache: {len(_FORECAST_CACHE)} entries")
    _print_summary(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "past_days": args.past_days,
        "note": (
            "Hit rate = model point-forecast fell inside the Polymarket-resolved bucket. "
            "No market prices used. Pure model accuracy vs ground truth."
        ),
        "stations": {
            icao: {
                "unit": d["unit"],
                "market_label": d["market_label"],
                "models": d["models"],
            }
            for icao, d in results.items()
        },
    }
    with OUTPUT_PATH.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
