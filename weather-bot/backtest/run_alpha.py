"""run_alpha.py

Phase 2 orchestrator: fetch market prices + run financial alpha analysis.

Usage:
    python -m backtest.run_alpha                     # full pipeline
    python -m backtest.run_alpha --skip-fetch        # skip price fetching
    python -m backtest.run_alpha --city seoul        # single city
    python -m backtest.run_alpha --city nyc london   # multiple cities
    python -m backtest.run_alpha --skip-prices       # use model-median fallback
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.fetch_market_prices import fetch_all_prices
from backtest.analyze_alpha import run_alpha_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Financial Alpha Analysis")
    parser.add_argument("--skip-fetch",   action="store_true", help="Skip price fetching")
    parser.add_argument("--skip-prices",  action="store_true", help="Use model-median fallback (no CLOB)")
    parser.add_argument("--city",         nargs="*", default=None)
    parser.add_argument("--force-refetch",action="store_true", help="Re-fetch all prices")
    args = parser.parse_args()

    t0 = time.time()

    if not args.skip_fetch and not args.skip_prices:
        print("━" * 65)
        print("PHASE 2 — STEP 1/2: Fetching historical market prices (CLOB T-24h) …")
        print("━" * 65)
        city = args.city[0] if args.city and len(args.city) == 1 else None
        fetch_all_prices(city_filter=city, force_refetch=args.force_refetch)
        print()
    else:
        if args.skip_prices:
            print("Skipping price fetch — will use model-median as crowd proxy.\n")
        else:
            print("Skipping price fetch — using existing market_prices.json.\n")

    print("━" * 65)
    print("PHASE 2 — STEP 2/2: Financial alpha analysis …")
    print("━" * 65)
    run_alpha_analysis(
        city_filter=args.city,
        skip_prices=args.skip_prices,
    )

    print(f"\nPhase 2 complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
