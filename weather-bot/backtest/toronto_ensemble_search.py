"""toronto_ensemble_search.py

Exhaustive permutation search over all subsets (size 1–8) of the 20 valid
Toronto models. For each subset, averages predictions and computes:
  - Bucket accuracy  (round(avg) == round(actual))
  - MAE
  - ≤1°C rate
  - Coverage (% of 81 days with ≥1 model prediction available)

Uses numpy + itertools — no API calls, runs entirely from the local cache.

Usage:
    python backtest/toronto_ensemble_search.py
    python backtest/toronto_ensemble_search.py --max-size 5 --top 20
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

DATA_DIR      = Path(__file__).resolve().parent / "data"
CACHE_PATH    = DATA_DIR / "toronto_prediction_cache.json"
RESOLVED_JSON = DATA_DIR / "resolved_markets.json"
OUT_JSON      = DATA_DIR / "toronto_ensemble_search.json"

# The 20 models confirmed to have Toronto coverage (from Phase 1 probe)
VALID_MODELS = [
    "ncep_nbm_conus",
    "kma_gdps",
    "kma_seamless",
    "meteofrance_arpege_world",
    "meteofrance_seamless",
    "gem_global",
    "ncep_aigfs025",
    "gem_regional",
    "gem_seamless",
    "gem_hrdps_continental",
    "gfs_seamless",
    "jma_gsm",
    "jma_seamless",
    "gfs_graphcast025",
    "dmi_seamless",
    "knmi_seamless",
    "metno_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "icon_global",
]


def _hround(x: float) -> int:
    return math.floor(x + 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-size", type=int, default=8,
                        help="Max ensemble size to search (default 8)")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of top combos to print (default 30)")
    parser.add_argument("--min-coverage", type=float, default=50.0,
                        help="Skip combos covering <N%% of dates (default 50)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    cache: dict[str, float | None] = json.loads(CACHE_PATH.read_text())
    records = sorted(json.loads(RESOLVED_JSON.read_text()), key=lambda r: r["target_date"])

    actuals   = np.array([r["resolved_temp"] for r in records], dtype=float)
    dates     = [r["target_date"] for r in records]
    n_dates   = len(dates)

    # Build prediction matrix: shape (n_models, n_dates), NaN where missing
    preds = np.full((len(VALID_MODELS), n_dates), np.nan)
    for mi, model in enumerate(VALID_MODELS):
        for di, date in enumerate(dates):
            val = cache.get(f"{model}|{date}")
            if val is not None:
                preds[mi, di] = val

    print(f"Loaded {n_dates} dates, {len(VALID_MODELS)} models")
    print(f"Searching all subsets size 1–{args.max_size} …\n")

    # Count total combos
    total = sum(
        math.comb(len(VALID_MODELS), k)
        for k in range(1, args.max_size + 1)
    )
    print(f"Total combinations: {total:,}")

    results: list[dict] = []
    checked = 0

    for size in range(1, args.max_size + 1):
        for combo in combinations(range(len(VALID_MODELS)), size):
            checked += 1
            if checked % 50000 == 0:
                pct = checked / total * 100
                print(f"  … {checked:,}/{total:,} ({pct:.1f}%)", flush=True)

            subset = preds[list(combo), :]          # shape (size, n_dates)
            # Average ignoring NaN
            with np.errstate(all="ignore"):
                avg = np.nanmean(subset, axis=0)    # (n_dates,)

            valid_mask = ~np.isnan(avg)
            n_valid = int(valid_mask.sum())
            coverage = n_valid / n_dates * 100

            if coverage < args.min_coverage:
                continue

            avg_v  = avg[valid_mask]
            act_v  = actuals[valid_mask]

            # Bucket accuracy: round(avg) == round(actual)
            pred_rounded = np.array([_hround(x) for x in avg_v])
            act_rounded  = np.array([_hround(x) for x in act_v])
            buck_acc = float((pred_rounded == act_rounded).mean() * 100)

            errs = np.abs(avg_v - act_v)
            mae  = float(errs.mean())
            w1   = float((errs <= 1.0).mean() * 100)

            results.append({
                "models":        [VALID_MODELS[i] for i in combo],
                "size":          size,
                "n":             n_valid,
                "coverage_pct":  round(coverage, 1),
                "bucket_acc":    round(buck_acc, 2),
                "mae":           round(mae, 4),
                "within_1c_pct": round(w1, 1),
            })

    print(f"\nDone. Evaluated {len(results):,} combos with ≥{args.min_coverage:.0f}% coverage.\n")

    # Sort by bucket accuracy (primary), then MAE (tiebreak)
    results.sort(key=lambda x: (-x["bucket_acc"], x["mae"]))

    # ── Print top results ──────────────────────────────────────────────────────
    print("=" * 100)
    print(f"{'RK':<4} {'SIZE':<5} {'N':>4} {'BUCK%':>7} {'MAE':>6} {'≤1°C':>6}  MODELS")
    print("-" * 100)
    for rank, r in enumerate(results[:args.top], 1):
        model_str = " + ".join(r["models"])
        print(f"{rank:<4} {r['size']:<5} {r['n']:>4} {r['bucket_acc']:>6.1f}% "
              f"{r['mae']:>6.3f} {r['within_1c_pct']:>5.1f}%  {model_str}")
    print("=" * 100)

    # ── Stats by size ──────────────────────────────────────────────────────────
    print("\nBest per ensemble size:")
    for size in range(1, args.max_size + 1):
        size_res = [r for r in results if r["size"] == size]
        if not size_res:
            continue
        best = size_res[0]
        print(f"  Size {size}: {best['bucket_acc']:.1f}% bucket, MAE {best['mae']:.3f}°C  "
              f"— {' + '.join(best['models'])}")

    # Save
    OUT_JSON.write_text(json.dumps(results[:500], indent=2))
    print(f"\nTop 500 saved → {OUT_JSON}")

    # ── Baseline: single best model ───────────────────────────────────────────
    singles = [r for r in results if r["size"] == 1]
    print(f"\nSingle-model baseline: {singles[0]['bucket_acc']:.1f}% ({singles[0]['models'][0]})")
    print(f"Best ensemble found:   {results[0]['bucket_acc']:.1f}% "
          f"(size {results[0]['size']}: {' + '.join(results[0]['models'])})")

    improvement = results[0]["bucket_acc"] - singles[0]["bucket_acc"]
    print(f"Ensemble gain:         +{improvement:.1f} percentage points")


if __name__ == "__main__":
    main()
