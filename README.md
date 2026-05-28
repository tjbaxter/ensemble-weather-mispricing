# ensemble-weather-mispricing

A deterministic pricing engine for short-term weather derivatives. Maps NOAA/GFS ensemble forecasts to discrete probability buckets to identify and exploit EV-positive mispricings against retail sentiment.

## What this does

Polymarket runs daily binary contracts on city high-temperature bands ("Will Chicago's high tomorrow land in 70–74°F?"). The market prices these via order flow. The fair value of each bucket is computable from numerical-weather-prediction ensembles, which publish a distribution over outcomes hours before settlement.

This system:

1. Pulls the latest NOAA/GFS ensemble forecasts for tracked cities.
2. Computes a per-bucket probability from the ensemble distribution with a calibration adjustment.
3. Compares against live Polymarket bucket prices, fee-adjusted.
4. Logs EV-positive opportunities and paper-trades them through a Kelly-sized sizer.
5. After settlement, reconciles outcomes against three independent sources (Polymarket resolution, weather.com observations, AviationWeather METAR) to catch resolution disputes.

## Architecture

```
weather-bot/
├── main.py                       # Live loop: forecast pull -> pricing -> signal log
├── scanner.py                    # Polymarket market discovery + filtering
├── dashboard.py                  # Streamlit live + historical dashboard
├── strategy/                     # Bucket-probability pricers and calibration
├── backtest/                     # Historical replays + morning-rate alpha study
├── execution/                    # Paper + live Polymarket order placement
├── monitoring/                   # Health + telemetry sidecars
├── scripts/verify_resolution_sources.py   # Three-source parity report
└── .env.example                  # Required env vars
```

## Setup

```bash
git clone https://github.com/tjbaxter/ensemble-weather-mispricing.git
cd ensemble-weather-mispricing
python -m venv .venv
source .venv/bin/activate
pip install -r weather-bot/requirements.txt
cp weather-bot/.env.example weather-bot/.env
# Fill in WU_OBS_KEY and forecast provider keys
./run_paper.sh
```

## Key design choices

- **Paper-first.** `PAPER_TRADING=true` by default. Live trading requires explicit `LIVE_TRADING=true` plus a funded Polygon wallet.
- **Deterministic pricing.** Same forecast input → same bucket probabilities. No model drift between runs.
- **Three-source resolution check.** Polymarket sometimes resolves slowly or contests outcomes. The parity report flags any disagreement between Polymarket's resolved bucket and the underlying weather.com / METAR data before capital is committed to similar future markets.
- **Kill switch.** `touch weather-bot/data/.kill_switch` halts live order placement immediately.

## Limitations

- Forecast skill drops sharply beyond ~5 days; this system targets ≤48h horizons.
- Polymarket liquidity on these buckets is thin — bankroll is intentionally capped.
- The pricing model assumes the ensemble distribution is well-calibrated; large-scale calibration adjustment is a continuous research item, not a fixed parameter.

## Status

Paper-trading actively. Live trading enabled for a small bankroll on hand-selected markets.
