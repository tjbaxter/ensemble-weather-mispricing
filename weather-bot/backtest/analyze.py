"""analyze.py

Core backtest analysis. Computes per-model performance metrics against
Polymarket-resolved ground truth.

Metrics:
  - Bucket Accuracy: did round(model_pred) land in the correct bucket?
  - MAE: mean absolute error vs resolved temp
  - Bias: systematic warm/cold offset
  - Directional alpha: when model disagrees with ensemble, how often is it right?

Usage:
    python backtest/analyze.py
    python backtest/analyze.py --city Seoul
    python backtest/analyze.py --model ncep_aigfs025
    python backtest/analyze.py --min-markets 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "backtest" / "data"
IN_JSON  = DATA_DIR / "model_predictions.json"

MODELS = [
    "ncep_aigfs025",
    "gfs_graphcast025",
    "kma_gdps",
    "ecmwf_ifs025",
    "gem_global",
    "gfs_seamless",
    "icon_seamless",
]

ELITE = {"ncep_aigfs025", "gfs_graphcast025", "kma_gdps"}


# ── Bucket logic ──────────────────────────────────────────────────────────────

def _model_bucket(pred_temp: float, resolved_label: str) -> str:
    """Snap a predicted temperature to the canonical bucket label format.

    The resolved_label tells us whether this is a single degree, range, or
    boundary ("12+", "4-") market — we need to produce the same format.
    """
    # Determine bucket style from resolved label
    if resolved_label.endswith("+"):
        # "12+" means "12 or higher" — threshold is the numeric value
        threshold = float(resolved_label[:-1])
        return f"{int(threshold)}+" if pred_temp >= threshold else _round_to_label(pred_temp)
    if resolved_label.endswith("-"):
        threshold = float(resolved_label[:-1])
        return f"{int(threshold)}-" if pred_temp <= threshold else _round_to_label(pred_temp)
    # Single degree or range: just round to nearest int
    return _round_to_label(pred_temp)


def _round_to_label(temp: float) -> str:
    """Round temp to nearest integer, return as string (e.g. 4.7 → '5')."""
    return str(int(round(temp)))


def _is_hit(pred_temp: float, resolved_temp: float, bucket_label: str) -> bool:
    """Did the model predict the correct bucket?

    For boundary buckets ("12+", "4-") we check whether the prediction
    would land in that same boundary bucket, not just equal the exact value.
    """
    if bucket_label.endswith("+"):
        threshold = float(bucket_label[:-1])
        return pred_temp >= threshold
    if bucket_label.endswith("-"):
        threshold = float(bucket_label[:-1])
        return pred_temp <= threshold
    # Single degree: round and compare
    return round(pred_temp) == round(resolved_temp)


# ── Core statistics ───────────────────────────────────────────────────────────

def _model_stats(
    rows: list[dict],
    model: str,
) -> dict:
    hits = 0
    abs_errors: list[float] = []
    errors: list[float] = []
    valid = 0

    for row in rows:
        pred = row.get(f"pred_{model}")
        if pred is None:
            continue
        actual = row["resolved_temp"]
        label  = row["bucket_label"]
        valid += 1
        if _is_hit(pred, actual, label):
            hits += 1
        err = pred - actual
        errors.append(err)
        abs_errors.append(abs(err))

    if valid == 0:
        return {"model": model, "markets": 0, "accuracy": None, "mae": None, "bias": None}

    return {
        "model":    model,
        "markets":  valid,
        "hits":     hits,
        "accuracy": hits / valid,
        "mae":      sum(abs_errors) / len(abs_errors),
        "bias":     sum(errors) / len(errors),
    }


def _per_city_stats(rows: list[dict], model: str, min_markets: int = 2) -> list[dict]:
    from collections import defaultdict
    by_city: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_city[r["city_slug"]].append(r)

    out = []
    for city, city_rows in sorted(by_city.items()):
        s = _model_stats(city_rows, model)
        if s["markets"] >= min_markets:
            s["city"] = city
            out.append(s)
    return sorted(out, key=lambda x: -(x["accuracy"] or 0))


def _ensemble_stats(rows: list[dict], models: list[str]) -> dict:
    """Test a simple mean-ensemble of available models."""
    hits = 0
    abs_errors: list[float] = []
    errors: list[float] = []
    valid = 0

    for row in rows:
        preds = [row.get(f"pred_{m}") for m in models]
        preds = [p for p in preds if p is not None]
        if not preds:
            continue
        mean_pred = sum(preds) / len(preds)
        actual = row["resolved_temp"]
        label  = row["bucket_label"]
        valid += 1
        if _is_hit(mean_pred, actual, label):
            hits += 1
        err = mean_pred - actual
        errors.append(err)
        abs_errors.append(abs(err))

    if valid == 0:
        return {"model": "ensemble_avg", "markets": 0, "accuracy": None, "mae": None, "bias": None}

    return {
        "model":    "ensemble_avg",
        "markets":  valid,
        "hits":     hits,
        "accuracy": hits / valid,
        "mae":      sum(abs_errors) / len(abs_errors),
        "bias":     sum(errors) / len(errors),
    }


def _alpha_signal(rows: list[dict], primary: str = "ncep_aigfs025") -> dict:
    """When primary model disagrees with ensemble consensus, how often is primary right?

    Disagreement = primary's bucket differs from the plurality bucket of other models.
    """
    disagree_total = 0
    disagree_correct = 0

    for row in rows:
        pred_primary = row.get(f"pred_{primary}")
        if pred_primary is None:
            continue
        other_preds = [
            row.get(f"pred_{m}")
            for m in MODELS if m != primary
            if row.get(f"pred_{m}") is not None
        ]
        if not other_preds:
            continue

        ensemble_mean = sum(other_preds) / len(other_preds)
        # Disagreement: primary bucket != ensemble bucket
        primary_bucket = round(pred_primary)
        ensemble_bucket = round(ensemble_mean)

        if primary_bucket == ensemble_bucket:
            continue

        # They disagree — who was right?
        actual = row["resolved_temp"]
        label  = row["bucket_label"]
        disagree_total += 1

        primary_hit  = _is_hit(pred_primary,  actual, label)
        ensemble_hit = _is_hit(ensemble_mean, actual, label)

        if primary_hit:
            disagree_correct += 1

    if disagree_total == 0:
        return {"n": 0, "primary_right": None, "actionable": False}

    rate = disagree_correct / disagree_total
    return {
        "n":            disagree_total,
        "primary_right": rate,
        "actionable":   rate > 0.50 and disagree_total >= 10,
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def _fmt(v, pct: bool = False, dp: int = 1) -> str:
    if v is None:
        return "    —"
    if pct:
        return f"{v*100:6.1f}%"
    return f"{v:+{dp+4}.{dp}f}" if v < 0 or True else f"{v:{dp+3}.{dp}f}"


def _print_table(stats: list[dict]) -> None:
    col_w = 24
    print(f"\n{'Model':<{col_w}} {'Acc':>7}  {'MAE':>6}  {'Bias':>7}  {'Markets':>7}")
    print("─" * 60)
    for s in stats:
        if s["accuracy"] is None:
            print(f"  {s['model']:<{col_w-2}} {'  no data':>7}")
            continue
        acc  = f"{s['accuracy']*100:.1f}%"
        mae  = f"{s['mae']:.2f}°"
        bias = f"{s['bias']:+.2f}°"
        n    = str(s["markets"])
        marker = " ◀ PRIMARY" if s["model"] == "ncep_aigfs025" else ""
        print(f"  {s['model']:<{col_w-2}} {acc:>7}  {mae:>6}  {bias:>7}  {n:>7}{marker}")


def _print_city_table(city_stats: list[dict], model: str) -> None:
    print(f"\n  Per-city ({model}):")
    print(f"  {'City':<12} {'Acc':>7}  {'MAE':>6}  {'Bias':>7}  {'N':>4}")
    print("  " + "─" * 44)
    for s in city_stats:
        acc  = f"{s['accuracy']*100:.1f}%" if s["accuracy"] is not None else "—"
        mae  = f"{s['mae']:.2f}°" if s["mae"] is not None else "—"
        bias = f"{s['bias']:+.2f}°" if s["bias"] is not None else "—"
        n    = str(s["markets"])
        print(f"  {s['city']:<12} {acc:>7}  {mae:>6}  {bias:>7}  {n:>4}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_analysis(
    rows: list[dict],
    city_filter: str | None = None,
    model_filter: str | None = None,
    min_markets: int = 3,
) -> dict:
    if city_filter:
        rows = [r for r in rows if city_filter.lower() in (r["city_slug"], r["city"].lower())]

    models = [model_filter] if model_filter else MODELS

    # Work only on rows that have at least one prediction (skip unfetched cities)
    rows_with_preds = [
        r for r in rows
        if any(r.get(f"pred_{m}") is not None for m in MODELS)
    ]

    # Overall stats — computed on fetched rows only
    all_stats = [_model_stats(rows_with_preds, m) for m in models]

    # Ensemble
    usable = [m for m in models if any(r.get(f"pred_{m}") is not None for r in rows_with_preds)]
    ens_stats = _ensemble_stats(rows_with_preds, usable)
    elite_models = [m for m in ELITE if m in usable]
    ens_elite    = _ensemble_stats(rows_with_preds, elite_models)
    ens_elite["model"] = "ensemble_elite3"

    combined = sorted(
        all_stats + [ens_stats, ens_elite],
        key=lambda s: -(s["accuracy"] or 0),
    )

    # Alpha signal
    alpha = _alpha_signal(rows_with_preds, primary="ncep_aigfs025")

    # Date range from rows with predictions
    dates = sorted(r["target_date"] for r in rows_with_preds)
    period = f"{dates[0]} to {dates[-1]}" if dates else "?"

    # Print
    print("\n" + "=" * 65)
    print("  POLYMARKET WEATHER BACKTEST RESULTS")
    print("=" * 65)
    print(f"  Period:      {period}")
    print(f"  Fetched:     {len(rows_with_preds)} / {len(rows)} total markets have predictions")
    print(f"  (random-chance baseline on ~9 buckets ≈ 11%)")
    print("\nMODEL PERFORMANCE — Bucket Accuracy (next-day predictions):")
    _print_table(combined)

    # Per-city for primary model
    primary_city = _per_city_stats(rows_with_preds, "ncep_aigfs025", min_markets=min_markets)
    if primary_city:
        _print_city_table(primary_city, "ncep_aigfs025")

    # Alpha signal
    print(f"\n  ALPHA SIGNAL (ncep_aigfs025 vs ensemble disagreements):")
    if alpha["n"] == 0:
        print("    Not enough disagreements yet.")
    else:
        pct = alpha["primary_right"] * 100
        print(f"    Disagreements: {alpha['n']}")
        print(f"    Primary correct: {pct:.1f}%")
        if alpha["actionable"]:
            print("    ✅ TRADEABLE SIGNAL — primary consistently beats ensemble")
        else:
            print("    ⚠️  Signal not yet confirmed (need >50% accuracy, n≥10)")

    print("=" * 65)

    return {
        "period":        period,
        "n_markets":     len(rows_with_preds),
        "n_total_raw":   len(rows),
        "model_stats":   combined,
        "alpha_signal":  alpha,
        "city_stats":    primary_city,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",        default=None)
    parser.add_argument("--model",       default=None)
    parser.add_argument("--min-markets", type=int, default=3)
    args = parser.parse_args()

    if not IN_JSON.exists():
        print(f"ERROR: {IN_JSON} not found. Run fetch_model_predictions.py first.")
        sys.exit(1)

    rows = json.loads(IN_JSON.read_text())
    print(f"Loaded {len(rows)} records")

    results = run_analysis(
        rows,
        city_filter=args.city,
        model_filter=args.model,
        min_markets=args.min_markets,
    )

    # Save results
    out_path = DATA_DIR / "results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
