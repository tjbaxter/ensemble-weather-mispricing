"""run_backtest.py

Orchestrator for the full historical backtesting pipeline.

Usage:
    python backtest/run_backtest.py                         # Full pipeline
    python backtest/run_backtest.py --fetch-only            # Just fetch data
    python backtest/run_backtest.py --analyze-only          # Skip fetch, just analyze
    python backtest/run_backtest.py --city Seoul            # Filter to one city
    python backtest/run_backtest.py --model ncep_aigfs025   # Focus on one model
    python backtest/run_backtest.py --since 2025-11-01      # Custom start date

Steps:
  1. fetch_polymarket_resolved  → data/resolved_markets.json
  2. fetch_model_predictions    → data/model_predictions.json
  3. analyze                    → terminal output + data/results.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import modules directly so we can call their functions
from backtest.fetch_polymarket_resolved import fetch_all_resolved, OUT_JSON as RESOLVED_JSON, OUT_DIR
from backtest.fetch_model_predictions   import fetch_all_predictions, _load_cache, _save_cache, IN_JSON as PRED_IN, OUT_JSON as PRED_OUT, DATA_DIR
from backtest.analyze                   import run_analysis

import json


def _save_resolved(records: list[dict]) -> None:
    import csv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESOLVED_JSON.open("w") as f:
        json.dump(records, f, indent=2)
    csv_path = OUT_DIR / "resolved_markets.csv"
    if records:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def _save_predictions(records_enriched: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with PRED_OUT.open("w") as f:
        json.dump(records_enriched, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket Weather Backtest Engine")
    parser.add_argument("--fetch-only",   action="store_true", help="Only fetch data, skip analysis")
    parser.add_argument("--analyze-only", action="store_true", help="Skip fetch, use existing data")
    parser.add_argument("--city",         default=None,        help="Filter to one city slug")
    parser.add_argument("--model",        default=None,        help="Focus on one model")
    parser.add_argument("--since",        default="2024-01-01", help="Earliest date YYYY-MM-DD")
    parser.add_argument("--min-markets",  type=int, default=3, help="Min markets per city for display")
    args = parser.parse_args()

    t0 = time.time()

    # ── STEP 1: Fetch resolved Polymarket markets ─────────────────────────────
    if not args.analyze_only:
        print("━" * 65)
        print("STEP 1/3: Fetching resolved Polymarket markets …")
        print("━" * 65)
        records = fetch_all_resolved(since_date=args.since, city_filter=args.city)
        _save_resolved(records)
        print(f"✓ {len(records)} resolved markets saved to {RESOLVED_JSON}\n")
    else:
        if not RESOLVED_JSON.exists():
            print(f"ERROR: {RESOLVED_JSON} not found. Run without --analyze-only first.")
            sys.exit(1)
        records = json.loads(RESOLVED_JSON.read_text())
        print(f"STEP 1 skipped — loaded {len(records)} existing records from {RESOLVED_JSON}\n")

    if not records:
        print("No resolved markets found. Exiting.")
        sys.exit(0)

    # ── STEP 2: Fetch model predictions (previous_day1) ───────────────────────
    if not args.analyze_only:
        print("━" * 65)
        print("STEP 2/3: Fetching previous_day1 model predictions …")
        print("         (NO data leakage — Day-1 forecast only)")
        print("━" * 65)
        _load_cache()
        predictions = fetch_all_predictions(
            records,
            model_filter=args.model,
            city_filter=args.city,
        )
        _save_cache()

        # Enrich records with predictions
        from backtest.fetch_model_predictions import MODELS
        enriched: list[dict] = []
        for rec in records:
            eid = rec["event_id"]
            preds = predictions.get(eid, {})
            row = dict(rec)
            for model in MODELS:
                row[f"pred_{model}"] = preds.get(model)
            enriched.append(row)

        _save_predictions(enriched)
        print(f"\n✓ Predictions saved to {PRED_OUT}\n")
    else:
        if not PRED_OUT.exists():
            print(f"ERROR: {PRED_OUT} not found. Run without --analyze-only first.")
            sys.exit(1)
        enriched = json.loads(PRED_OUT.read_text())
        print(f"STEP 2 skipped — loaded {len(enriched)} enriched records\n")

    if args.fetch_only:
        print("--fetch-only: skipping analysis.")
        sys.exit(0)

    # ── STEP 3: Analyze ───────────────────────────────────────────────────────
    print("━" * 65)
    print("STEP 3/3: Analyzing model performance …")
    print("━" * 65)
    results = run_analysis(
        enriched,
        city_filter=args.city,
        model_filter=args.model,
        min_markets=args.min_markets,
    )

    elapsed = time.time() - t0
    print(f"\nBacktest complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
