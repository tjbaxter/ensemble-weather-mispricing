"""nyc_full_backtest.py

Full NYC backtest over all resolved markets (Jan 2025 → present).
Fetches all 20 valid models one DATE at a time (20 parallel workers per date),
saves after every date, then runs exhaustive ensemble search.

This avoids the bulk-parallel approach that triggers Open-Meteo rate limits.
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import requests

DATA_DIR      = Path(__file__).resolve().parent / "data"
CACHE_PATH    = DATA_DIR / "nyc_prediction_cache.json"
RESOLVED_JSON = DATA_DIR / "resolved_markets.json"
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OUT_JSON      = DATA_DIR / "nyc_model_ranking.json"

LAT, LON   = 40.7769, -73.874
TIMEZONE   = "America/New_York"
UNIT       = "F"

# The 20 models confirmed to cover NYC (from Phase 1 probe of Seattle/Toronto)
VALID_MODELS = [
    "icon_seamless",
    "gfs_seamless",
    "ecmwf_ifs025",
    "meteofrance_arpege_world",
    "meteofrance_seamless",
    "gem_seamless",
    "gem_global",
    "gem_regional",
    "gem_hrdps_continental",
    "jma_seamless",
    "jma_gsm",
    "dmi_seamless",
    "kma_seamless",
    "kma_gdps",
    "knmi_seamless",
    "metno_seamless",
    "ncep_aigfs025",
    "ncep_nbm_conus",
    "gfs_graphcast025",
    "icon_global",
]

_cache: dict[str, float | None] = {}


def _load_cache() -> None:
    global _cache
    if CACHE_PATH.exists():
        try:
            _cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            _cache = {}
    print(f"Cache: {len(_cache)} entries ({sum(1 for v in _cache.values() if v is not None)} with data)", flush=True)


def _save_cache() -> None:
    CACHE_PATH.write_text(json.dumps(_cache, indent=2))


def _fetch_model(model: str, target_date: str) -> tuple[str, float | None]:
    """Fetch one model for one date. Returns (model, value)."""
    ck = f"{model}|{target_date}"
    if ck in _cache:
        return model, _cache[ck]

    today  = date.today()
    target = date.fromisoformat(target_date)
    days_back = (today - target).days + 2

    params: dict = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m_previous_day1",
        "models": model,
        "temperature_unit": "fahrenheit",
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
            r = requests.get(PREV_RUNS_URL, params=params, timeout=25)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"\n  [429] {model} rate-limited, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            if r.status_code in (400, 422, 404):
                _cache[ck] = None
                return model, None
            r.raise_for_status()
            payload = r.json()
            break
        except Exception:
            if attempt == 2:
                _cache[ck] = None
                return model, None
            time.sleep(3 ** attempt)
    else:
        _cache[ck] = None
        return model, None

    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    hourly = payload.get("hourly", {})
    times  = hourly.get("time", [])
    tkey   = next((k for k in hourly if k.startswith("temperature_2m_previous_day1")), None)
    if tkey is None:
        _cache[ck] = None
        return model, None

    vals = [
        float(v) for ts, v in zip(times, hourly[tkey])
        if v is not None and str(ts)[:10] == target_date
    ]
    result = max(vals) if vals else None
    _cache[ck] = result
    return model, result


def _hround(x: float) -> int:
    return math.floor(x + 0.5)


def _bucket_match_2f(pred: float, actual: float) -> bool:
    """NYC 2°F even-odd bucket: 38-39, 40-41, 42-43 …"""
    return (_hround(pred) // 2) == (_hround(actual) // 2)


def fetch_all_dates(records: list[dict]) -> None:
    """Fetch all models for all dates, one date at a time with 20 parallel workers."""
    n = len(records)
    cached_count = sum(
        1 for r in records for m in VALID_MODELS
        if f"{m}|{r['target_date']}" in _cache
    )
    needed = n * len(VALID_MODELS) - cached_count
    print(f"\nFetching {needed} new calls ({n} dates × {len(VALID_MODELS)} models, {cached_count} cached)", flush=True)

    for i, rec in enumerate(records):
        tdate  = rec["target_date"]
        actual = rec.get("resolved_temp", "?")

        # Skip if all models already cached for this date
        missing = [m for m in VALID_MODELS if f"{m}|{tdate}" not in _cache]
        if not missing:
            continue

        results_this_date = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(_fetch_model, m, tdate): m for m in missing}
            for fut in as_completed(futures):
                model, val = fut.result()
                results_this_date[model] = val
                time.sleep(0.15)  # gentle inter-request delay

        _save_cache()

        filled = sum(1 for v in results_this_date.values() if v is not None)
        vals_str = " ".join(
            f"{results_this_date.get(m, '?'):.0f}" if results_this_date.get(m) is not None else "---"
            for m in ["ncep_nbm_conus", "gem_seamless", "icon_seamless", "ecmwf_ifs025", "gfs_seamless"]
        )
        print(f"[{i+1:3d}/{n}] {tdate}  actual={actual:>5}°F  top-5: {vals_str}  ({filled}/{len(missing)} models)", flush=True)

        time.sleep(1.0)  # 1s between dates to avoid rate limits


def score_models(records: list[dict]) -> list[dict]:
    results = []
    for model in VALID_MODELS:
        pairs = []
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
        errs = [abs(p - a) for p, a in pairs]
        results.append({
            "model":         model,
            "n":             n,
            "mae":           round(sum(errs) / n, 3),
            "rmse":          round((sum(e**2 for e in errs) / n) ** 0.5, 3),
            "bias":          round(sum(p - a for p, a in pairs) / n, 3),
            "within_1f_pct": round(sum(1 for e in errs if e <= 1.0) / n * 100, 1),
            "bucket_acc":    round(sum(1 for p, a in pairs if _bucket_match_2f(p, a)) / n * 100, 1),
        })
    return sorted(results, key=lambda x: x["mae"])


def ensemble_search(records: list[dict], valid_for_search: list[str], max_size: int = 6) -> list[dict]:
    dates   = [r["target_date"] for r in records]
    actuals = np.array([r["resolved_temp"] for r in records], dtype=float)
    n_dates = len(dates)

    preds = np.full((len(valid_for_search), n_dates), np.nan)
    for mi, model in enumerate(valid_for_search):
        for di, d in enumerate(dates):
            val = _cache.get(f"{model}|{d}")
            if val is not None:
                preds[mi, di] = val

    total = sum(math.comb(len(valid_for_search), k) for k in range(1, max_size + 1))
    print(f"\nEnsemble search: {total:,} combos (size 1–{max_size}) over {n_dates} dates …", flush=True)

    results = []
    for checked, size in enumerate(range(1, max_size + 1)):
        for combo in combinations(range(len(valid_for_search)), size):
            subset = preds[list(combo), :]
            with np.errstate(all="ignore"):
                avg = np.nanmean(subset, axis=0)
            valid_mask = ~np.isnan(avg)
            n_valid = int(valid_mask.sum())
            if n_valid < int(n_dates * 0.5):
                continue

            avg_v = avg[valid_mask]
            act_v = actuals[valid_mask]
            errs  = np.abs(avg_v - act_v)
            mae   = float(errs.mean())
            w1    = float((errs <= 1.0).mean() * 100)
            buck  = sum(_bucket_match_2f(float(avg_v[i]), float(act_v[i])) for i in range(n_valid)) / n_valid * 100

            results.append({
                "models":       [valid_for_search[i] for i in combo],
                "size":         size,
                "n":            n_valid,
                "bucket_acc":   round(buck, 2),
                "mae":          round(mae, 4),
                "within_1f_pct": round(w1, 1),
            })
        print(f"  Size {size} done", flush=True)

    return sorted(results, key=lambda x: (-x["bucket_acc"], x["mae"]))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _load_cache()

    records = sorted(json.loads(RESOLVED_JSON.read_text()), key=lambda r: r["target_date"])
    print(f"NYC markets: {len(records)}  ({records[0]['target_date']} → {records[-1]['target_date']})", flush=True)

    fetch_all_dates(records)

    singles = score_models(records)

    print(f"\n{'RK':<4} {'MODEL':<38} {'N':>5} {'MAE':>7} {'BIAS':>7} {'<=1°F':>6} {'BUCK%':>6}")
    print("-" * 80)
    for rank, r in enumerate(singles, 1):
        print(f"{rank:<4} {r['model']:<38} {r['n']:>5} {r['mae']:>6.3f}°F "
              f"{r['bias']:>+6.3f} {r['within_1f_pct']:>5.1f}% {r['bucket_acc']:>5.1f}%")

    # Only search models with good coverage (≥50% of dates)
    min_n = int(len(records) * 0.5)
    search_models = [r["model"] for r in singles if r["n"] >= min_n]
    print(f"\n{len(search_models)} models with ≥50% coverage for ensemble search")

    combos = ensemble_search(records, search_models, max_size=6)

    print(f"\n{'RK':<4} {'SZ':<3} {'N':>5} {'BUCK%':>7} {'MAE':>7} {'<=1°F':>6}  MODELS")
    print("=" * 115)
    for rank, r in enumerate(combos[:30], 1):
        print(f"{rank:<4} {r['size']:<3} {r['n']:>5} {r['bucket_acc']:>6.1f}% "
              f"{r['mae']:>6.3f} {r['within_1f_pct']:>5.1f}%  {' + '.join(r['models'])}")
    print("=" * 115)

    print(f"\nBest per size:")
    for sz in range(1, 7):
        best = next((r for r in combos if r["size"] == sz), None)
        if best:
            print(f"  Size {sz}: {best['bucket_acc']:.1f}% bucket  MAE {best['mae']:.3f}°F"
                  f"  → {' + '.join(best['models'])}")

    single_best = max(singles, key=lambda x: x["bucket_acc"])
    ens_best = combos[0] if combos else None
    print(f"\nSingle best:  {single_best['bucket_acc']:.1f}% ({single_best['model']}, N={single_best['n']})")
    if ens_best:
        gain = ens_best["bucket_acc"] - single_best["bucket_acc"]
        print(f"Ensemble best: {ens_best['bucket_acc']:.1f}% size-{ens_best['size']} → {' + '.join(ens_best['models'])}")
        print(f"Gain: +{gain:.1f} pp")

    OUT_JSON.write_text(json.dumps({"singles": singles, "ensembles": combos[:200]}, indent=2))
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
