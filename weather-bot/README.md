# Polymarket Weather Trading Bot

A conservative, paper-first Python bot for weather prediction markets on Polymarket. It is station-first (ICAO-based), models Weather Underground rounding behavior, caches fragile weather APIs, and only trades when disagreement clears strict risk and execution filters.

## Safety First

- Paper mode is the default; live mode is blocked in `main.py` until additional checks are added.
- Hard caps: per-trade size, daily exposure, spread filter, and max drawdown shutdown.
- Signals are ignored near resolution (`HOURS_BEFORE_RESOLUTION_CUTOFF`) where informed flow compresses edge.
- API keys are loaded from `.env`; no secrets are hardcoded.

## Strategy Summary

1. Discover active weather markets from Gamma.
2. Map each market to a known resolution station (skip unknown).
3. Hydrate bucket prices from CLOB and skip wide spreads.
4. Build station/date forecast probabilities with source-specific pipelines.
5. Compute edge and apply rounding-confidence-adjusted Kelly sizing.
6. Skip danger windows around METAR releases and log every decision.

## Resolution Source

Markets resolve from Weather Underground historical station pages (e.g., `KLGA` for NYC and `EGLC` for London). Forecasting logic is explicitly designed to predict what Weather Underground will display, not generic city forecasts.

## Architecture

```text
weather-bot/
├── config/
├── data/
├── strategy/
├── execution/
├── monitoring/
├── backtest/
├── main.py
├── requirements.txt
└── .env.example
```

## Quick Start

1. Create venv and install (Python 3 only):
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `python3 -m pip install --upgrade pip`
   - `python3 -m pip install -r requirements.txt`
2. Copy environment template:
   - `cp .env.example .env`
3. Run paper trader:
   - `python3 main.py`
4. Optional live-only dependency (later milestone):
   - `python3 -m pip install -r requirements-live.txt`

## Quick Market Check

Before running the main loop for long sessions, check if tradable weather markets are currently active:

- `python3 scripts/check_weather_markets.py`
- `python3 scripts/check_weather_markets.py --diagnostic`

You can also run one-shot runtime diagnostics:

- `python3 main.py --diagnostic`

Historical forecast calibration (no need to wait for live resolution):

- `python3 scripts/backtest_calibration.py --past-days 30`
- writes calibration metrics to `logs/calibration.json`

## Paper Trading Validation (Required)

Run paper mode for 3-5 days before enabling any live pathway:

- Confirm positive edge after spread/liquidity filters.
- Verify signal quality by city/date/bucket.
- Validate drawdown behavior under noisy days.
- Inspect logs: `logs/signals.csv` and `logs/trades.csv`.

## Performance Table (Fill After Run)

| Metric | Value |
|---|---|
| Total trades | TBD |
| Win rate | TBD |
| Total P&L | TBD |
| Sharpe ratio | TBD |
| Max drawdown | TBD |
| ROI | TBD |

## Equity Curve

Save a generated chart to `docs/equity_curve.png` after paper or historical runs.

## Example Trade Walkthrough

On a sample NYC market, the model estimated the `40-41` bucket at materially higher probability than Polymarket's implied price. The bot generated a high-edge `BUY_YES` signal, sized it via quarter-Kelly with a hard dollar cap, and logged the simulated fill in paper mode for later resolution and P&L attribution.

## Risk Management

- Fractional Kelly (`KELLY_FRACTION`) with strict cap (`MAX_POSITION_SIZE`).
- Daily deployment cap (`MAX_DAILY_EXPOSURE`).
- Drawdown kill switch (`MAX_DRAWDOWN_PCT`).
- Liquidity and spread quality gates.
- No trading in final 3 hours pre-resolution.
- Priority filter defaults to `HIGH` stations only.
- Practical order floor (`$5`) to avoid low-value micro orders.
- Trading pause in METAR danger window (`XX:53` to `XX:58` UTC).

## Seoul Case Study (Critical Lesson)

A real Seoul trade looked like a major edge using city-level forecast data, but the market resolved at Incheon Airport (`RKSI`) with a different microclimate and Celsius rounding behavior. Exiting at a loss was the correct risk decision. The bot now enforces station matching and rounding-confidence controls to avoid repeating this class of error.

## Notes

- Use only legally permitted infrastructure/jurisdictions.
- Keep wallet and API credentials isolated from identity-linked wallets.
- Treat live deployment as a separate milestone after sustained paper performance.
