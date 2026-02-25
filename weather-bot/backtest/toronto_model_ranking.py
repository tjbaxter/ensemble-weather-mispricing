"""toronto_model_ranking.py

Two-phase Toronto model ranking:
  Phase 1: probe all ~51 Open-Meteo models on ONE date → weed out ones with no
            Toronto coverage or no previous_day1 support (~2 min)
  Phase 2: run only valid models across all 81 resolved dates → rank by MAE

Uses concurrent.futures for parallel requests (10 workers).

Usage:
    python backtest/toronto_model_ranking.py
    python backtest/toronto_model_ranking.py --top 8
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests

DATA_DIR   = Path(__file__).resolve().parent / "data"
CACHE_PATH = DATA_DIR / "toronto_prediction_cache.json"
OUT_JSON   = DATA_DIR / "toronto_model_ranking.json"
RESOLVED_JSON = DATA_DIR / "resolved_markets.json"
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

ALL_MODELS = [
    # GFS / NOAA
    "gfs_seamless", "gfs025", "gfs05",
    "gfs_graphcast025", "ncep_aigfs025",
    "ncep_nbm_conus", "cfs",
    # ECMWF
    "ecmwf_ifs025", "ecmwf_ifs04",
    "ecmwf_aifs025",
    # DWD ICON
    "icon_seamless", "icon_global", "icon_eu", "icon_d2",
    # Météo-France
    "meteofrance_seamless", "meteofrance_arpege_world",
    "meteofrance_arpege_europe", "meteofrance_arome_france",
    # Canadian GEM
    "gem_seamless", "gem_global", "gem_regional", "gem_hrdps_continental",
    # UK Met Office
    "ukmet_seamless", "ukmet_global_deterministic",
    # JMA
    "jma_seamless", "jma_gsm", "jma_msm",
    # DMI
    "dmi_seamless",
    # KNMI
    "knmi_seamless",
    # KMA
    "kma_gdps", "kma_seamless",
    # BOM
    "bom_access_global",
    # SMHI / Nordic
    "smhi_seamless", "metno_seamless", "metno_nordic",
    # ARPAE
    "arpae_cosmo_seamless", "arpae_cosmo_2i", "arpae_cosmo_5m",
]

LAT, LON   = 43.6772, -79.6306
TIMEZONE   = "America/Toronto"
MAX_WORKERS = 10

_cache: dict[str, float | None] = {}


def _load_cache() -> None:
    global _cache
    if CACHE_PATH.exists():
        try:
            _cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            _cache = {}
    print(f"Cache: {len(_cache)} entries loaded", flush=True)


def _save_cache() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache, indent=2))


def _fetch_one(model: str, target_date: str) -> float | None:
    """Fetch max(temperature_2m_previous_day1) for target_date. Thread-safe."""
    ck = f"{model}|{target_date}"
    if ck in _cache:
        return _cache[ck]

    today  = date.today()
    target = date.fromisoformat(target_date)
    days_back = (today - target).days + 2

    params: dict = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m_previous_day1",
        "models": model,
        "temperature_unit": "celsius",
        "timezone": TIMEZONE,
        "forecast_days": 1,
    }
    if days_back <= 92:
        params["past_days"] = days_back
    else:
        params["start_date"] = (target - timedelta(days=1)).isoformat()
        params["end_date"]   = (target + timedelta(days=1)).isoformat()
        del params["forecast_days"]

    for attempt in range(3):
        try:
            r = requests.get(PREV_RUNS_URL, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(20 * (attempt + 1))
                continue
            if r.status_code in (400, 422, 404):
                _cache[ck] = None
                return None
            r.raise_for_status()
            payload = r.json()
            break
        except Exception:
            if attempt == 2:
                _cache[ck] = None
                return None
            time.sleep(2 ** attempt)
    else:
        _cache[ck] = None
        return None

    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    hourly = payload.get("hourly", {})
    times  = hourly.get("time", [])
    tkey   = next((k for k in hourly if k.startswith("temperature_2m_previous_day1")), None)
    if tkey is None:
        _cache[ck] = None
        return None

    vals = [
        float(v) for ts, v in zip(times, hourly[tkey])
        if v is not None and str(ts)[:10] == target_date
    ]
    result = max(vals) if vals else None
    _cache[ck] = result
    return result


def probe_models(probe_date: str) -> list[str]:
    """Phase 1: find which models actually return data for Toronto on probe_date."""
    print(f"\n── Phase 1: probing {len(ALL_MODELS)} models on {probe_date} ──", flush=True)
    valid: list[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, m, probe_date): m for m in ALL_MODELS}
        for fut in as_completed(futures):
            model = futures[fut]
            val   = fut.result()
            status = f"{val:.1f}°C" if val is not None else "✗ no data"
            print(f"  {model:<40} {status}", flush=True)
            if val is not None:
                valid.append(model)

    print(f"\n✓ {len(valid)}/{len(ALL_MODELS)} models have Toronto coverage", flush=True)
    return valid


def score_models(valid_models: list[str], records: list[dict]) -> list[dict]:
    """Phase 2: fetch all valid models × all dates in parallel, then score."""
    print(f"\n── Phase 2: {len(valid_models)} valid models × {len(records)} dates ──", flush=True)

    # Build task list (skip if already in cache)
    tasks: list[tuple[str, str]] = [
        (m, r["target_date"])
        for r in records
        for m in valid_models
        if f"{m}|{r['target_date']}" not in _cache
    ]
    print(f"  {len(tasks)} API calls needed ({len(valid_models)*len(records) - len(tasks)} cached)", flush=True)

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, m, d): (m, d) for m, d in tasks}
        for fut in as_completed(futures):
            fut.result()  # result already stored in _cache
            completed += 1
            if completed % 50 == 0:
                pct = completed / len(tasks) * 100 if tasks else 100
                print(f"  … {completed}/{len(tasks)} ({pct:.0f}%)", flush=True)
                _save_cache()

    _save_cache()

    # Score each model
    results: list[dict] = []
    for model in valid_models:
        pairs: list[tuple[float, float]] = []
        for rec in records:
            actual = rec.get("resolved_temp")
            if actual is None:
                continue
            pred = _cache.get(f"{model}|{rec['target_date']}")
            if pred is not None:
                pairs.append((pred, float(actual)))

        n = len(pairs)
        if n < 10:
            continue

        errs    = [abs(p - a) for p, a in pairs]
        mae     = sum(errs) / n
        rmse    = (sum(e**2 for e in errs) / n) ** 0.5
        bias    = sum(p - a for p, a in pairs) / n
        w1      = sum(1 for e in errs if e <= 1.0) / n * 100
        # bucket accuracy: round(pred) == round(actual)  (1°C buckets for Toronto)
        buck    = sum(1 for p, a in pairs if round(p) == round(a)) / n * 100

        results.append({
            "model": model, "n": n,
            "mae": round(mae, 3), "rmse": round(rmse, 3),
            "bias": round(bias, 3),
            "within_1c_pct": round(w1, 1),
            "bucket_acc_pct": round(buck, 1),
        })

    return sorted(results, key=lambda x: x["mae"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _load_cache()

    records = json.loads(RESOLVED_JSON.read_text())
    records = sorted(records, key=lambda r: r["target_date"])
    print(f"Toronto markets: {len(records)}  ({records[0]['target_date']} → {records[-1]['target_date']})", flush=True)

    # Use a recent, well-archived date for probing
    probe_date = records[-10]["target_date"]
    valid_models = probe_models(probe_date)

    if not valid_models:
        print("No valid models found. Exiting.")
        return

    results = score_models(valid_models, records)

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'RK':<4} {'MODEL':<38} {'N':>4} {'MAE':>6} {'RMSE':>6} {'BIAS':>6} {'<=1°C':>6} {'BUCK%':>6}")
    print("-" * 90)
    for rank, r in enumerate(results, 1):
        flag = "  ← TOP" if rank <= args.top else ""
        print(f"{rank:<4} {r['model']:<38} {r['n']:>4} "
              f"{r['mae']:>6.3f} {r['rmse']:>6.3f} {r['bias']:>+6.3f} "
              f"{r['within_1c_pct']:>5.1f}% {r['bucket_acc_pct']:>5.1f}%{flag}", flush=True)
    print("=" * 90)

    print(f"\nTop {args.top} for Toronto (by MAE):")
    for r in results[:args.top]:
        print(f"  {r['model']:<38}  MAE={r['mae']:.3f}°C  bucket={r['bucket_acc_pct']:.1f}%")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
