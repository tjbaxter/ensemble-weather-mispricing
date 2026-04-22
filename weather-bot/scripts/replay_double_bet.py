#!/usr/bin/env python3
"""Replay the V3.5 double-bet selection for a city/date.

Shows which two buckets Prime Alpha would select (center + adjacent),
the scoring rationale, and the combined package edge if market prices
are provided.

Usage:
    python3 scripts/replay_double_bet.py --city Ankara --date 2026-04-22
    python3 scripts/replay_double_bet.py --city London --date 2026-04-22 \
        --market-prices "13=0.20,14=0.35,15=0.25"
    python3 scripts/replay_double_bet.py --city Dallas --date 2026-04-22 --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.cities import STATIONS  # noqa: E402
from config.accuracy_cities import ACCURACY_CITIES  # noqa: E402
from strategy.prime_alpha import build_prime_alpha_plan  # noqa: E402
from strategy.execution import apply_execution_filter  # noqa: E402

MODEL_SNAPSHOT_LOG = ROOT / "data" / "model_snapshot_log.json"
ACCURACY_CACHE = ROOT / "data" / "accuracy_rows_cache.json"


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def load_snapshot_models(city: str, target_date: str) -> dict[str, float]:
    payload = json.loads(MODEL_SNAPSHOT_LOG.read_text(encoding="utf-8"))
    city_rows = payload.get(city, {})
    if not isinstance(city_rows, dict):
        raise KeyError(f"No snapshot rows for city={city}")
    entry = city_rows.get(target_date, {})
    preds = entry.get("preds", {}) if isinstance(entry, dict) else {}
    if not isinstance(preds, dict) or not preds:
        raise KeyError(f"No snapshot predictions for {city} {target_date}")
    return {str(k): float(v) for k, v in preds.items() if v is not None}


def find_station(city: str) -> tuple[str, str]:
    for icao, cfg in STATIONS.items():
        label = cfg.get("market_label", "")
        if label.lower() == city.lower() or icao.lower() == city.lower():
            unit = cfg.get("resolution_unit", "F")
            return icao, str(unit)
    raise KeyError(f"No station found for city={city}")


def get_bucket_labels(city: str, icao: str) -> list[str]:
    cfg = ACCURACY_CITIES.get(city, {})
    pm = cfg.get("polymarket", {})
    if pm:
        latest = max(pm.keys())
        entry = pm[latest]
        if isinstance(entry, (list, tuple)):
            label = entry[0]
            return _generate_buckets_from_sample(label, cfg)
    return []


def _generate_buckets_from_sample(label: str, cfg: dict) -> list[str]:
    style = cfg.get("bucket_style", "exact_1c")
    unit_suffix = "°C" if cfg.get("temperature_unit", "celsius") == "celsius" else "°F"
    if style == "range_2f":
        pm = cfg.get("polymarket", {})
        labels = set()
        for entry in pm.values():
            if isinstance(entry, (list, tuple)):
                labels.add(str(entry[0]))
        return sorted(labels, key=_bucket_sort_key)
    pm = cfg.get("polymarket", {})
    ints = set()
    for entry in pm.values():
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            val = entry[1]
            if isinstance(val, (int, float)) and val is not None:
                ints.add(int(val))
    if not ints:
        return []
    lo, hi = max(0, min(ints) - 3), max(ints) + 3
    return [f"{i}{unit_suffix}" for i in range(lo, hi + 1)]


def _bucket_sort_key(b: str) -> float:
    clean = b.replace("°F", "").replace("°C", "").replace("≥", "").replace("≤", "").strip()
    if "-" in clean:
        return float(clean.split("-")[0])
    try:
        return float(clean)
    except ValueError:
        return 0.0


def compute_display_temp(model_values: dict[str, float], unit: str) -> int | None:
    if not model_values:
        return None
    raw = mean(float(v) for v in model_values.values())
    if str(unit).upper() == "F":
        return _round_half_up(raw)
    temp_f = raw * 9.0 / 5.0 + 32.0
    rounded_f = _round_half_up(temp_f)
    return _round_half_up((rounded_f - 32.0) * 5.0 / 9.0)


def parse_market_prices(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    result: dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        bucket, price = pair.split("=", 1)
        result[bucket.strip()] = float(price.strip())
    return result


def recent_accuracy_summary(city: str, days: int = 7) -> list[dict]:
    if not ACCURACY_CACHE.exists():
        return []
    data = json.loads(ACCURACY_CACHE.read_text(encoding="utf-8"))
    rows = data.get(city, [])
    return rows[-days:] if rows else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", required=True, help="City name (e.g. Ankara, London, Dallas)")
    parser.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    parser.add_argument("--bucket-labels", help="Comma-separated buckets (auto-detected if omitted)")
    parser.add_argument("--prior-resolved-bucket", help="Override prior day's resolved bucket")
    parser.add_argument("--market-prices", help="bucket=price pairs for edge calc, e.g. '13=0.20,14=0.35'")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of formatted text")
    args = parser.parse_args()

    city = args.city
    target_date = args.date
    icao, unit = find_station(city)

    model_values = load_snapshot_models(city, target_date)
    predicted_display = compute_display_temp(model_values, unit)

    if args.bucket_labels:
        bucket_labels = [b.strip() for b in args.bucket_labels.split(",")]
    else:
        bucket_labels = get_bucket_labels(city, icao)
        if not bucket_labels:
            print(f"ERROR: Could not auto-detect buckets for {city}. Use --bucket-labels.")
            sys.exit(1)

    market_prices = parse_market_prices(args.market_prices)

    plan = build_prime_alpha_plan(
        city=city,
        station_icao=icao,
        target_date=target_date,
        bucket_labels=bucket_labels,
        current_model_values=model_values,
        predicted_display_temp=predicted_display,
        unit=unit,
        prior_resolved_bucket=args.prior_resolved_bucket,
    )

    exec_decision = None
    if market_prices and plan.selected_buckets:
        exec_decision = apply_execution_filter(
            candidate_buckets=plan.selected_buckets,
            bucket_market_prices=market_prices,
            bucket_model_probs=plan.bucket_probabilities,
            selection_layer=plan.selection_layer,
            regime_strength=plan.regime_strength,
        )

    if args.json:
        out = {
            "city": city,
            "station_icao": icao,
            "target_date": target_date,
            "unit": unit,
            "predicted_display_temp": predicted_display,
            "trust_source": plan.trust_source,
            "prior_resolved_bucket": plan.prior_resolved_bucket,
            "regime_strength": plan.regime_strength,
            "beta_used": plan.beta_used,
            "selected_buckets": plan.selected_buckets,
            "notes": plan.notes,
            "selection_layer": plan.selection_layer,
            "bucket_probabilities": plan.bucket_probabilities,
            "families": plan.families,
        }
        if exec_decision:
            out["execution_decision"] = exec_decision.to_dict()
        print(json.dumps(out, indent=2))
        return

    # ── Formatted output ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  DOUBLE BET REPLAY: {city} {target_date}")
    print(f"{'=' * 60}")
    print(f"  Station: {icao} | Unit: °{unit} | Display temp: {predicted_display}")
    print(f"  Prior resolved: {plan.prior_resolved_bucket or 'none'}")
    print(f"  Regime: {plan.regime_strength:.2f} | β: {plan.beta_used:.2f}")
    print()

    # Layer A summary
    print("── LAYER A: Model Scoring ──")
    print(f"  {'Family':22s} {'μ':>6s} {'σ':>5s} {'w':>6s} {'short':>6s} {'long':>6s}")
    for fam, info in plan.families.items():
        tag = " ×" if info.get("suppressed") else ""
        print(f"  {fam:22s} {info['forecast']:6.1f} {info['sigma']:5.2f} "
              f"{info['weight']:.3f} {info['short_skill']:.3f} {info['long_skill']:.3f}{tag}")
    print()

    # Layer B summary
    sel = plan.selection_layer
    print("── LAYER B: Selection ──")
    rw = sel.get("recent_winners", {})
    print(f"  Recent winners:  {rw if rw else 'none'}")
    print(f"  Winner center:   {sel.get('winner_center', '-')}")
    rs = sel.get("recent_support", {})
    print(f"  Recent support:  {rs if rs else 'none'}")
    print(f"  Working set:     {sel.get('working_set', [])}")
    print(f"  Working center:  {sel.get('working_center', '-')}")
    streaks = sel.get("streak_scores", {})
    if streaks:
        print(f"  Streaks:         {streaks}")
    print()

    # Double bet decision
    print("── DOUBLE BET DECISION ──")
    if not plan.selected_buckets:
        print("  ⊘ NO BET — working set empty (no model has recent signal)")
        for note in plan.notes:
            print(f"    → {note}")
    elif len(plan.selected_buckets) == 2:
        b1, b2 = plan.selected_buckets
        p1 = plan.bucket_probabilities.get(b1, 0)
        p2 = plan.bucket_probabilities.get(b2, 0)
        combined_p = min(0.95, p1 + p2)

        center = sel.get("center_bucket", "?")
        adj_info = sel.get("adjacent_bucket", {})
        adj_bucket = adj_info.get("bucket", "?") if isinstance(adj_info, dict) else "?"

        print(f"  CENTER:   {center:>8s}  P={p1:.1%}")
        print(f"  ADJACENT: {adj_bucket:>8s}  P={p2:.1%}")
        print(f"  COMBINED: P(either) = {combined_p:.1%}")

        if isinstance(adj_info, dict):
            print(f"  Adjacent reasoning: support={adj_info.get('model_support', 0)} models, "
                  f"lean={adj_info.get('lean_bonus', 0):.0f}, "
                  f"gauss={adj_info.get('gaussian_prob', 0):.3f}")

        runner = sel.get("adjacent_runner_up", {})
        if runner:
            print(f"  Runner-up: {runner.get('bucket', '?')} "
                  f"(support={runner.get('model_support', 0)}, "
                  f"gauss={runner.get('gaussian_prob', 0):.3f})")

        if market_prices:
            print()
            print("── PACKAGE EDGE ──")
            m1 = market_prices.get(b1, 0)
            m2 = market_prices.get(b2, 0)
            cost = m1 + m2
            edge = combined_p - cost
            print(f"  {b1}: model={p1:.1%} market={m1:.1%} edge={p1-m1:+.1%}")
            print(f"  {b2}: model={p2:.1%} market={m2:.1%} edge={p2-m2:+.1%}")
            print(f"  Package: model={combined_p:.1%} cost={cost:.1%} edge={edge:+.1%}")
            if edge > 0:
                print(f"  → TRADE (positive package edge)")
            else:
                print(f"  → NO TRADE (negative package edge)")
    else:
        print(f"  Selected: {plan.selected_buckets}")

    # Notes
    if plan.notes:
        print()
        print("── Notes ──")
        for note in plan.notes:
            print(f"  {note}")

    # Recent accuracy context
    recent = recent_accuracy_summary(city, days=7)
    if recent:
        print()
        print("── Recent Accuracy (last 7 days) ──")
        for row in recent:
            d = row.get("date", "?")
            ew = row.get("best_ens_d1_win")
            sym = "✓" if ew else ("✗" if ew is False else "?")
            pred = row.get("best_ens_d1")
            print(f"  {d}: ens={pred}  {sym}")

    print()


if __name__ == "__main__":
    main()
