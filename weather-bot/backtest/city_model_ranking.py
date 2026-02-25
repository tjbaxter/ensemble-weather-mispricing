"""city_model_ranking.py  —  reusable two-phase model ranking for any city

Phase 1: probe all ~44 Open-Meteo models on one date → weed out coverage gaps
Phase 2: parallel fetch across all resolved dates → score by MAE + bucket accuracy
Phase 3: exhaustive combo search (all subsets 1-8) → best ensemble

Usage:
    python backtest/city_model_ranking.py --city Seattle
    python backtest/city_model_ranking.py --city Toronto --max-size 6
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import requests
import time

DATA_DIR      = Path(__file__).resolve().parent / "data"
RESOLVED_JSON = DATA_DIR / "resolved_markets.json"
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

ALL_MODELS = [
    "gfs_seamless", "gfs025",
    "gfs_graphcast025", "ncep_aigfs025", "ncep_nbm_conus", "cfs",
    "ecmwf_ifs025", "ecmwf_ifs04", "ecmwf_aifs025",
    "icon_seamless", "icon_global", "icon_eu", "icon_d2",
    "meteofrance_seamless", "meteofrance_arpege_world",
    "meteofrance_arpege_europe", "meteofrance_arome_france",
    "gem_seamless", "gem_global", "gem_regional", "gem_hrdps_continental",
    "ukmet_seamless", "ukmet_global_deterministic",
    "jma_seamless", "jma_gsm", "jma_msm",
    "dmi_seamless",
    "knmi_seamless",
    "kma_gdps", "kma_seamless",
    "bom_access_global",
    "smhi_seamless", "metno_seamless", "metno_nordic",
    "arpae_cosmo_seamless", "arpae_cosmo_2i", "arpae_cosmo_5m",
]

MAX_WORKERS = 12


# ── City configs ──────────────────────────────────────────────────────────────
CITY_CONFIG = {
    "Seattle": {
        "lat": 47.4502, "lon": -122.3088,
        "timezone": "America/Los_Angeles",
        "unit": "F",           # "F" or "C"
        "bucket_style": "range_2f",   # 2°F even-odd pairs
    },
    "Toronto": {
        "lat": 43.6772, "lon": -79.6306,
        "timezone": "America/Toronto",
        "unit": "C",
        "bucket_style": "exact_1c",
    },
}


def _hround(x: float) -> int:
    return math.floor(x + 0.5)


def _bucket_match(pred: float, actual: float, bucket_style: str) -> bool:
    """True if pred lands in the same Polymarket bucket as actual."""
    p = _hround(pred)
    a = _hround(actual)
    if bucket_style == "exact_1c":
        return p == a
    if bucket_style == "range_2f":
        # Even-odd 2°F pairs: 40-41, 42-43, 44-45 …
        # Map each temp to its bucket floor: floor(x/2)*2
        return (p // 2) == (a // 2)
    return p == a


def _fetch_one(model: str, target_date: str, lat: float, lon: float,
               timezone: str, unit: str, cache: dict) -> float | None:
    """Fetch max(temperature_2m_previous_day1). Thread-safe; updates cache in place."""
    ck = f"{model}|{target_date}"
    if ck in cache:
        return cache[ck]

    today  = date.today()
    target = date.fromisoformat(target_date)
    days_back = (today - target).days + 2

    temp_unit = "fahrenheit" if unit == "F" else "celsius"
    params: dict = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m_previous_day1",
        "models": model,
        "temperature_unit": temp_unit,
        "timezone": timezone,
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
                cache[ck] = None
                return None
            r.raise_for_status()
            payload = r.json()
            break
        except Exception:
            if attempt == 2:
                cache[ck] = None
                return None
            time.sleep(2 ** attempt)
    else:
        cache[ck] = None
        return None

    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    hourly = payload.get("hourly", {})
    times  = hourly.get("time", [])
    tkey   = next((k for k in hourly if k.startswith("temperature_2m_previous_day1")), None)
    if tkey is None:
        cache[ck] = None
        return None

    vals = [
        float(v) for ts, v in zip(times, hourly[tkey])
        if v is not None and str(ts)[:10] == target_date
    ]
    result = max(vals) if vals else None
    cache[ck] = result
    return result


def probe_models(records, cfg, cache) -> list[str]:
    probe_date = records[-10]["target_date"]
    lat, lon   = cfg["lat"], cfg["lon"]
    tz, unit   = cfg["timezone"], cfg["unit"]
    print(f"\n── Phase 1: probing {len(ALL_MODELS)} models on {probe_date} ──", flush=True)
    valid: list[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, m, probe_date, lat, lon, tz, unit, cache): m
                   for m in ALL_MODELS}
        for fut in as_completed(futures):
            model = futures[fut]
            val   = fut.result()
            u     = cfg["unit"]
            status = f"{val:.1f}°{u}" if val is not None else "✗ no data"
            print(f"  {model:<40} {status}", flush=True)
            if val is not None:
                valid.append(model)

    print(f"\n✓ {len(valid)}/{len(ALL_MODELS)} models have coverage", flush=True)
    return valid


def fetch_all(records, valid_models, cfg, cache) -> None:
    lat, lon   = cfg["lat"], cfg["lon"]
    tz, unit   = cfg["timezone"], cfg["unit"]
    n_dates    = len(records)

    tasks = [
        (m, r["target_date"])
        for r in records
        for m in valid_models
        if f"{m}|{r['target_date']}" not in cache
    ]
    print(f"\n── Phase 2: {len(valid_models)} models × {n_dates} dates ──", flush=True)
    print(f"  {len(tasks)} API calls needed ({len(valid_models)*n_dates - len(tasks)} cached)", flush=True)

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, m, d, lat, lon, tz, unit, cache): (m, d)
                   for m, d in tasks}
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            if completed % 100 == 0:
                pct = completed / len(tasks) * 100 if tasks else 100
                print(f"  … {completed}/{len(tasks)} ({pct:.0f}%)", flush=True)


def score_models(records, valid_models, cfg, cache) -> list[dict]:
    bucket_style = cfg["bucket_style"]
    results = []
    for model in valid_models:
        pairs = []
        for rec in records:
            actual = rec.get("resolved_temp")
            if actual is None:
                continue
            pred = cache.get(f"{model}|{rec['target_date']}")
            if pred is not None:
                pairs.append((pred, float(actual)))

        n = len(pairs)
        if n < 10:
            continue
        errs   = [abs(p - a) for p, a in pairs]
        mae    = sum(errs) / n
        rmse   = (sum(e**2 for e in errs) / n) ** 0.5
        bias   = sum(p - a for p, a in pairs) / n
        w1     = sum(1 for e in errs if e <= 1.0) / n * 100
        buck   = sum(1 for p, a in pairs if _bucket_match(p, a, bucket_style)) / n * 100
        results.append({"model": model, "n": n, "mae": round(mae, 3),
                        "rmse": round(rmse, 3), "bias": round(bias, 3),
                        "within_1_pct": round(w1, 1), "bucket_acc": round(buck, 1)})
    return sorted(results, key=lambda x: x["mae"])


def ensemble_search(records, valid_models, cfg, cache, max_size=8) -> list[dict]:
    bucket_style = cfg["bucket_style"]
    dates   = [r["target_date"] for r in records]
    actuals = np.array([r["resolved_temp"] for r in records], dtype=float)
    n_dates = len(dates)

    preds = np.full((len(valid_models), n_dates), np.nan)
    for mi, model in enumerate(valid_models):
        for di, d in enumerate(dates):
            val = cache.get(f"{model}|{d}")
            if val is not None:
                preds[mi, di] = val

    total = sum(math.comb(len(valid_models), k) for k in range(1, max_size + 1))
    print(f"\n── Phase 3: exhaustive search — {total:,} combos ──", flush=True)

    results = []
    checked = 0
    for size in range(1, max_size + 1):
        for combo in combinations(range(len(valid_models)), size):
            checked += 1
            if checked % 100000 == 0:
                print(f"  … {checked:,}/{total:,} ({checked/total*100:.0f}%)", flush=True)

            subset   = preds[list(combo), :]
            with np.errstate(all="ignore"):
                avg  = np.nanmean(subset, axis=0)
            valid_mask = ~np.isnan(avg)
            n_valid    = int(valid_mask.sum())
            if n_valid < int(n_dates * 0.5):
                continue

            avg_v = avg[valid_mask]
            act_v = actuals[valid_mask]
            errs  = np.abs(avg_v - act_v)
            mae   = float(errs.mean())
            w1    = float((errs <= 1.0).mean() * 100)

            buck_hits = sum(
                _bucket_match(float(avg_v[i]), float(act_v[i]), bucket_style)
                for i in range(len(avg_v))
            )
            buck = buck_hits / n_valid * 100

            results.append({
                "models":        [valid_models[i] for i in combo],
                "size":          size,
                "n":             n_valid,
                "bucket_acc":    round(buck, 2),
                "mae":           round(mae, 4),
                "within_1_pct":  round(w1, 1),
            })

    results.sort(key=lambda x: (-x["bucket_acc"], x["mae"]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Seattle")
    parser.add_argument("--max-size", type=int, default=8)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    city = args.city
    cfg  = CITY_CONFIG.get(city)
    if cfg is None:
        print(f"Unknown city '{city}'. Available: {list(CITY_CONFIG.keys())}")
        return

    # Load resolved markets for this city
    all_records = json.loads(RESOLVED_JSON.read_text())
    records = sorted(
        [r for r in all_records if r["city"].lower() == city.lower()],
        key=lambda r: r["target_date"]
    )
    if not records:
        print(f"No resolved markets found for {city}")
        return
    print(f"\n{city}: {len(records)} resolved markets "
          f"({records[0]['target_date']} → {records[-1]['target_date']})", flush=True)

    cache_path = DATA_DIR / f"{city.lower().replace(' ','_')}_prediction_cache.json"
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            print(f"Cache: {len(cache)} entries loaded", flush=True)
        except Exception:
            pass

    def save_cache():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))

    # Phase 1
    valid_models = probe_models(records, cfg, cache)
    save_cache()

    # Phase 2
    fetch_all(records, valid_models, cfg, cache)
    save_cache()

    # Score singles
    singles = score_models(records, valid_models, cfg, cache)
    unit_lbl = f"°{cfg['unit']}"
    print(f"\n{'RK':<4} {'MODEL':<38} {'N':>4} {'MAE':>7} {'RMSE':>7} {'BIAS':>7} {'≤1':>6} {'BUCK%':>6}")
    print("-" * 85)
    for rank, r in enumerate(singles, 1):
        print(f"{rank:<4} {r['model']:<38} {r['n']:>4} {r['mae']:>6.3f}{unit_lbl} "
              f"{r['rmse']:>6.3f} {r['bias']:>+6.3f} {r['within_1_pct']:>5.1f}% {r['bucket_acc']:>5.1f}%")

    # Phase 3
    combos = ensemble_search(records, valid_models, cfg, cache, args.max_size)

    print(f"\n{'RK':<4} {'SIZE':<5} {'N':>4} {'BUCK%':>7} {'MAE':>7} {'≤1':>6}  MODELS")
    print("=" * 110)
    for rank, r in enumerate(combos[:args.top], 1):
        print(f"{rank:<4} {r['size']:<5} {r['n']:>4} {r['bucket_acc']:>6.1f}% "
              f"{r['mae']:>6.3f} {r['within_1_pct']:>5.1f}%  {' + '.join(r['models'])}")
    print("=" * 110)

    print(f"\nBest per size:")
    for size in range(1, args.max_size + 1):
        best = next((r for r in combos if r["size"] == size), None)
        if best:
            print(f"  Size {size}: {best['bucket_acc']:.1f}% bucket, MAE {best['mae']:.3f}{unit_lbl}"
                  f"  — {' + '.join(best['models'])}")

    single_best = max(singles, key=lambda x: x["bucket_acc"])
    ensemble_best = combos[0] if combos else None
    print(f"\nSingle best:    {single_best['bucket_acc']:.1f}% ({single_best['model']})")
    if ensemble_best:
        gain = ensemble_best["bucket_acc"] - single_best["bucket_acc"]
        print(f"Ensemble best:  {ensemble_best['bucket_acc']:.1f}% "
              f"(size {ensemble_best['size']}: {' + '.join(ensemble_best['models'])})")
        print(f"Gain:           +{gain:.1f} pp")

    # Save outputs
    out = DATA_DIR / f"{city.lower().replace(' ','_')}_model_ranking.json"
    out.write_text(json.dumps({"singles": singles, "ensembles": combos[:200]}, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
