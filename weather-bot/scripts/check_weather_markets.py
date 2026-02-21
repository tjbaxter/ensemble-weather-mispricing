"""Quick health check for active weather markets."""

from __future__ import annotations

import asyncio
import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow running script directly from weather-bot root.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.polymarket import PolymarketDataClient


async def main() -> None:
    parser = argparse.ArgumentParser(description="Check weather market discovery status.")
    parser.add_argument("--diagnostic", action="store_true", help="Enable verbose discovery diagnostics.")
    args = parser.parse_args()

    client = PolymarketDataClient(diagnostic=args.diagnostic)
    try:
        markets = await client.discover_weather_markets()
        print(f"discovered_markets={len(markets)}")
        print(f"discovery_stats={client.last_discovery_stats}")

        if not markets:
            print("tradable_candidates=0")
            print("reason=No active weather markets matched station/priority filters.")
            return

        hydrated = await client.hydrate_prices(markets)
        print(f"hydrated_markets={len(hydrated)}")

        station_counts = Counter(m["station_icao"] for m in markets)
        print("station_counts=")
        for station, count in sorted(station_counts.items()):
            print(f"  {station}: {count}")

        print("sample_candidates=")
        for market in hydrated[:5]:
            print(f"  - {market['station_icao']} | {market['date']} | {market['question']}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
