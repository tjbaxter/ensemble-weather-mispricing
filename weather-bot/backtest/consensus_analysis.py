"""consensus_analysis.py

For any city: analyse how bucket accuracy varies with model consensus.
Consensus = fraction of ensemble models that land in the same bucket as the
ensemble mean prediction.

Usage:
    python backtest/consensus_analysis.py --city Ankara
    python backtest/consensus_analysis.py --city Toronto
"""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

# Best ensembles per city from model ranking backtests
CITY_ENSEMBLES: dict[str, list[str]] = {
    "Ankara":  ["icon_global", "meteofrance_arpege_world", "jma_seamless",
                "icon_seamless", "ecmwf_ifs025", "gfs_graphcast025"],
    "Toronto": ["ecmwf_ifs025", "gfs_graphcast025", "jma_seamless",
                "icon_seamless", "icon_global", "meteofrance_arpege_world"],
    "Seattle": ["ecmwf_ifs025", "gfs_graphcast025", "icon_global",
                "meteofrance_arpege_world", "jma_seamless", "icon_seamless"],
    "nyc":     ["ecmwf_ifs025", "gfs_graphcast025", "icon_global",
                "ncep_aigfs025", "jma_seamless", "gfs_seamless"],
}


def hround(x: float) -> int:
    return math.floor(x + 0.5)


def run(city: str) -> None:
    city_key = city.lower().replace(" ", "_")
    cache_path = DATA / f"{city_key}_prediction_cache.json"
    ranking_path = DATA / f"{city_key}_model_ranking.json"

    if not cache_path.exists():
        print(f"No prediction cache found for {city} at {cache_path}")
        return

    pred_cache: dict = json.loads(cache_path.read_text())
    resolved_all: list = json.loads((DATA / "resolved_markets.json").read_text())
    resolved = {r["target_date"]: r for r in resolved_all
                if r["city_slug"] == city.lower() or r["city"].lower() == city.lower()}

    if not resolved:
        print(f"No resolved markets for {city}")
        return

    # Load best ensemble from ranking file or use defaults
    ensemble = CITY_ENSEMBLES.get(city)
    if not ensemble and ranking_path.exists():
        ranking = json.loads(ranking_path.read_text())
        best = ranking.get("ensembles", [{}])[0]
        ensemble = best.get("models", [])
    if not ensemble:
        print(f"No ensemble defined for {city}")
        return

    print(f"\n{'='*70}")
    print(f"  CONSENSUS ANALYSIS — {city.upper()}")
    print(f"  Ensemble: {len(ensemble)} models")
    print(f"  Dates: {len(resolved)}  |  Models: {', '.join(ensemble[:3])} ...")
    print(f"{'='*70}")

    rows: list[dict] = []

    for date_str, rec in sorted(resolved.items()):
        actual = rec.get("resolved_temp")
        if actual is None:
            continue
        actual_bucket = hround(float(actual))

        preds = {m: pred_cache[f"{m}|{date_str}"]
                 for m in ensemble
                 if f"{m}|{date_str}" in pred_cache and pred_cache[f"{m}|{date_str}"] is not None}

        if len(preds) < 2:
            continue

        vals = list(preds.values())
        ens_mean = sum(vals) / len(vals)
        ens_bucket = hround(ens_mean)
        spread = max(vals) - min(vals)

        n_agree = sum(1 for v in vals if hround(v) == ens_bucket)
        n_total = len(vals)
        frac = n_agree / n_total
        correct = ens_bucket == actual_bucket

        rows.append({
            "date": date_str,
            "actual": actual_bucket,
            "ens_mean": round(ens_mean, 2),
            "ens_bucket": ens_bucket,
            "spread": round(spread, 2),
            "n_agree": n_agree,
            "n_total": n_total,
            "frac": round(frac, 3),
            "correct": correct,
            "individual_preds": {m: hround(v) for m, v in preds.items()},
        })

    unit = resolved[next(iter(resolved))].get("unit", "C")

    # Per-date table
    print(f"\n{'Date':<12} {'Actual':>7} {'Ens':>7} {'Spread':>7} {'Agree':>7} {'Frac':>6}  Result")
    print("-" * 65)
    for r in rows:
        print(f"{r['date']:<12} {r['actual']:>5}°{unit}  {r['ens_mean']:>6.1f}  "
              f"{r['spread']:>5.1f}°{unit}  {r['n_agree']}/{r['n_total']:>1}  "
              f"{r['frac']:>5.0%}  {'✅' if r['correct'] else '❌'}")

    # Overall
    overall_acc = sum(r["correct"] for r in rows) / len(rows) * 100
    print(f"\nOverall: {sum(r['correct'] for r in rows)}/{len(rows)} = {overall_acc:.1f}% bucket accuracy")

    # Threshold sweep
    print(f"\n{'CONSENSUS THRESHOLD ANALYSIS':}")
    print(f"{'Threshold':>12}  {'Bet days':>9}  {'Accuracy':>9}  {'Skip days':>10}  {'Skip%':>6}")
    print("─" * 55)
    for thresh in [1/6, 2/6, 3/6, 4/6, 5/6, 6/6]:
        eligible = [r for r in rows if r["frac"] >= thresh - 0.01]
        skipped = len(rows) - len(eligible)
        if eligible:
            acc = sum(r["correct"] for r in eligible) / len(eligible) * 100
            n_correct = sum(r["correct"] for r in eligible)
            frac_str = f">={thresh*100:.0f}%  ({round(thresh*len(ensemble))+0:.0f}/{len(ensemble)} agree)"
            print(f"  {frac_str:<22} {len(eligible):>7}d  {acc:>8.1f}%  {skipped:>9}d  {skipped/len(rows)*100:>5.0f}%")

    # Spread vs consensus combined
    print(f"\n{'SPREAD + CONSENSUS COMBINED':}")
    print(f"{'Condition':>30}  {'N':>4}  {'Accuracy':>9}")
    print("─" * 50)
    for spread_thresh in [1.0, 2.0, 3.0]:
        for cons_thresh in [0.50, 0.67, 0.83]:
            eligible = [r for r in rows
                        if r["spread"] <= spread_thresh and r["frac"] >= cons_thresh - 0.01]
            if len(eligible) >= 3:
                acc = sum(r["correct"] for r in eligible) / len(eligible) * 100
                label = f"spread≤{spread_thresh:.0f}° AND ≥{cons_thresh*100:.0f}% agree"
                print(f"  {label:<30}  {len(eligible):>4}  {acc:>8.1f}%")

    # Recommend signal
    print(f"\n{'RECOMMENDATION':}")
    best_combo = None
    best_acc = 0.0
    for spread_thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for cons_thresh in [0.50, 0.67, 0.83]:
            eligible = [r for r in rows
                        if r["spread"] <= spread_thresh and r["frac"] >= cons_thresh - 0.01]
            if len(eligible) >= 5:
                acc = sum(r["correct"] for r in eligible) / len(eligible) * 100
                if acc > best_acc:
                    best_acc = acc
                    best_combo = (spread_thresh, cons_thresh, len(eligible), eligible)

    if best_combo:
        sp, co, n, eligible = best_combo
        print(f"  Best filter: spread ≤ {sp:.1f}°{unit}  AND  ≥{co*100:.0f}% models agree")
        print(f"  Result:      {sum(r['correct'] for r in eligible)}/{n} = {best_acc:.1f}% accuracy")
        print(f"  Bet days:    {n}/{len(rows)} ({n/len(rows)*100:.0f}% of markets, skip {len(rows)-n})")
        print(f"  Implied p:   {best_acc/100:.2f}  → at 30¢ price, Kelly = {((best_acc/100)*(1/0.30-1) - (1-best_acc/100))/(1/0.30-1)*100:.1f}% bankroll")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Ankara")
    args = parser.parse_args()
    run(args.city)


if __name__ == "__main__":
    main()
