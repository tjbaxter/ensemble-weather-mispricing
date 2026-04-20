"""Global bot configuration.

All values are intentionally conservative and paper-first by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# Trading parameters
# Alpha = model probability minus market implied probability.
# Lowered to 0.03 to capture cheap bucket opportunities (sub-20¢) where
# even a 3-5pp edge is strongly +EV due to leverage (6-7x return on win).
# Kelly sizing handles position sizing — a 3pp edge on a 15¢ bucket gets
# a tiny Kelly fraction, so no ruin risk from the lower threshold.
ALPHA_THRESHOLD = 0.03
MIN_FORECAST_CONFIDENCE = 0.15   # lowered to allow cheap bucket signals through
MAX_POSITION_SIZE = 25.00        # hard cap per position (Kelly output is always ≤ this)
MAX_DAILY_EXPOSURE = 50.00
MAX_POSITIONS_PER_MARKET = 1
# Global Kelly fraction — overridden per-city by STATIONS[icao]["kelly_fraction"]
# London uses 0.50 (half-Kelly, 75-day validated), others use 0.25 (quarter-Kelly)
KELLY_FRACTION = 0.25
# Bet size guardrails applied after Kelly formula
KELLY_MAX_BET_USD = 25.00   # absolute ceiling regardless of bankroll × Kelly
KELLY_MIN_BET_USD = 5.00    # floor: Kelly below this still fires at minimum viable size

# Risk limits
MAX_DRAWDOWN_PCT = 0.15
INITIAL_BANKROLL = 250.00

# Dynamic risk engine (production sizing)
# If enabled, per-bet and per-cycle deployment are capped as a percentage of
# current equity proxy (cash + cost-basis exposure), not fixed dollars.
# Defaults match legacy behaviour at the $250 initial bankroll:
#   10% per-position -> $25
#   20% daily budget -> $50
DYNAMIC_RISK_SIZING_ENABLED = True
EQUITY_MAX_POSITION_PCT = 0.10
EQUITY_DAILY_EXPOSURE_PCT = 0.20

# Quality multipliers for adaptive capital allocation
QUALITY_MULT_CONF_HIGH = 1.15
QUALITY_MULT_CONF_MEDIUM = 1.00
QUALITY_MULT_CONF_LOW = 0.80
QUALITY_D2_MULT = 0.95
QUALITY_D3_MULT = 0.90
QUALITY_RED_SPREAD_MULT = 0.85
QUALITY_EDGE_REF = 0.05      # edge level where full edge boost is reached
QUALITY_EDGE_SLOPE = 0.15    # additional multiplier at/above EDGE_REF
QUALITY_MIN_MULT = 0.60
QUALITY_MAX_MULT = 1.35

# Timing
SCAN_INTERVAL_SECONDS = 120
FORECAST_REFRESH_SECONDS = 1800
HOURS_BEFORE_RESOLUTION_CUTOFF = 3

# Temporal forecast confidence discount
# NWP models lose skill further out. We shrink forecast_prob toward 0 before
# edge calculation so the alpha threshold filters out weakly-priced D+2/D+3 bets.
# Values are multiplicative: D+2 forecast_prob *= 0.88, D+3 *= 0.80.
# Based on typical NWP skill decay: AROME/ECMWF ≈ 90-95% skill retention at D+2
# vs D+1, further degraded by ~8-10pp at D+3.
D2_P_WIN_DISCOUNT: float = float(os.getenv("D2_P_WIN_DISCOUNT", "0.88"))
D3_P_WIN_DISCOUNT: float = float(os.getenv("D3_P_WIN_DISCOUNT", "0.80"))

# D+2 / D+3 maximum YES entry price.
# At 2+ days out the forecast is uncertain enough that we only want exposure
# when the market is drastically underpricing something — i.e. the bucket is
# very cheap. If the market already prices it at 20¢+, the crowd has priced
# the uncertainty in and there's no exploitable mispricing worth the D+2 risk.
# Only enter D+2 markets where YES price ≤ 15¢. D+3 tightened further to 10¢.
D2_MAX_YES_ENTRY_PRICE: float = float(os.getenv("D2_MAX_YES_ENTRY_PRICE", "0.15"))
D3_MAX_YES_ENTRY_PRICE: float = float(os.getenv("D3_MAX_YES_ENTRY_PRICE", "0.10"))

# Event-driven model run scheduler
# Each tuple is (hour_utc, minute_utc, label).
# Times chosen so Open-Meteo has ingested the upstream data:
#   GFS 00Z init → NOMADS ready ~05:11 UTC → Open-Meteo ~+45 min → trigger 06:10
#   GFS 06Z init → NOMADS ready ~09:00 UTC → Open-Meteo ~+45 min → trigger 10:00
#   GFS 12Z+ECMWF 12Z → ready ~17:45–18:55 UTC → trigger 18:30
#   GFS 18Z init → NOMADS ready ~22:00 UTC → Open-Meteo ~+45 min → trigger 23:00
#   Midday market-discovery pass (no model data, catches new markets opened by Polymarket)
MODEL_RUN_TRIGGER_TIMES_UTC: list[tuple[int, int, str]] = [
    (6, 10, "GFS_00Z+ECMWF_00Z"),
    (10, 0, "GFS_06Z"),
    (13, 0, "MIDDAY_DISCOVERY"),
    (18, 30, "GFS_12Z+ECMWF_12Z"),
    (23, 0, "GFS_18Z"),
]
# During boost window after each trigger, scan this fast to catch repricing lag
MODEL_RUN_BOOST_SCAN_INTERVAL_SECONDS = 45
# How long to stay in fast-scan mode after a trigger fires (minutes)
MODEL_RUN_BOOST_WINDOW_MINUTES = 30
# Legacy flag — kept for backwards compat but superseded by event-driven scheduler
MODEL_RUN_BOOST_ENABLED = True

# Market quality controls
MIN_MARKET_LIQUIDITY = 50.0
MIN_MARKET_VOLUME = 50.0
MAX_BID_ASK_SPREAD = 0.10
MIN_ORDER_USD = 1.0
PRACTICAL_MIN_ORDER_USD = 5.0
FIXED_ORDER_USD = 5.0
FIXED_SIZE_BANKROLL_THRESHOLD = 2000.0
SOFT_PRICE_GUARDRAILS_ENABLED = True
SOFT_MIN_YES_PRICE = 0.03
SOFT_MAX_YES_PRICE = 0.85
SOFT_MIN_NO_PRICE = 0.15
SOFT_MAX_NO_PRICE = 0.97
SOFT_PRICE_EDGE_PENALTY = 0.02
HARD_MIN_YES_ENTRY_PRICE = 0.05
HARD_MAX_YES_ENTRY_PRICE = 0.45

# Station and market controls
STATION_PRIORITY_FILTER = {"HIGH", "MEDIUM", "LOW"}
NWS_CACHE_TTL_SECONDS = 900
DISCOVERY_MAX_PAGINATION_PAGES = 20
ENABLE_SEARCH_FALLBACK = False
CLOB_PREFILTER_MAX_HOURS_TO_RESOLUTION = 48
CLOB_PREFILTER_MIN_LIQUIDITY = 500.0
CLOB_PREFILTER_PRIORITY = {"HIGH", "MEDIUM", "LOW"}

# Forecast model controls
ENABLE_ENSEMBLE_FORECASTS = True

# ── TIER 1: Elite AI models ──────────────────────────────────────────────────
# Validated: all three predicted 14°C+ on Seoul Feb 21 (resolved 14-16°C).
# Traditional models (GFS seamless, ECMWF IFS) predicted only 7.5-11.9°C.
# These three share a structural advantage: neural network pattern recognition
# catches rapid synoptic shifts that physics-based NWP parameterisations miss.
ENSEMBLE_PRIMARY_MODEL = "ncep_aigfs025"       # day-before predicted 14.2°C ✅
ENSEMBLE_ADDITIONAL_MODELS = (
    "gfs_graphcast025",  # DeepMind GraphCast — predicted 14.1°C on day ✅
    "kma_gdps",          # Korea NWP — home turf advantage, predicted 14.0°C ✅

    # ── TIER 2: Secondary reference models ───────────────────────────────────
    # Got 12°C+ on Feb 21 (directionally right, magnitude short).
    # Kept to prevent AI echo-chamber — if all 3 elite models disagree with
    # these, ensemble std rises and may trigger the skip threshold.
    "gem_global",        # Canadian GEM — 12.1°C on Feb 21, 3.6°C on Feb 23
    "ecmwf_ifs025",      # Traditional ECMWF — 11.9°C on Feb 21 (close)

    # gfs_seamless excluded: only 7.5°C on Feb 21 — worst performer
    # ecmwf_aifs025 excluded — returns no data on previous-runs-api
)
ENSEMBLE_PREVIOUS_RUNS_API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ENSEMBLE_DAILY_API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
CALIBRATION_JSON_PATH = "logs/calibration.json"
MODEL_RANKINGS_JSON_PATH = "logs/model_rankings.json"
ENSEMBLE_BATCH_CACHE_TTL_SECONDS = 300
ENSEMBLE_CONFIDENCE_STD_HIGH = 1.5
ENSEMBLE_CONFIDENCE_STD_LOW = 3.0
ENSEMBLE_STD_SKIP_THRESHOLD = 5.0
ENSEMBLE_DISABLE_CLASSIC_CONFIDENCE_GATE = True

# High-Delta regime detection (AI vs traditional GFS divergence)
# When AI model (ncep_aigfs025) diverges from GFS baseline by >= this many degrees,
# the bot is in a "rapid synoptic shift" regime where AI models have demonstrated
# dramatically superior accuracy (e.g. Seoul Feb 21: AI=14.2°C, GFS=7.4°C).
HIGH_DELTA_THRESHOLD_DEG = 3.0
# Size multiplier applied to positions in high-delta regime (capped at MAX_POSITION_SIZE)
HIGH_DELTA_SIZE_MULTIPLIER = 2.0
# Mean shift applied to the Gaussian distribution centre during high-delta regime.
# AI models are trained with MSE loss which penalises extreme predictions, causing
# systematic cold bias during rapid warm events (and warm bias during rapid cold events).
# Shifting the mean by this amount before bucket probability calculation corrects for
# the AI's mean-reversion bias WITHOUT buying multiple buckets (one clean EV calculation).
# Source: 2025 ECMWF/arXiv research on GraphCast/Pangu cold bias during extremes.
# Start at 1.0°C empirical; update from calibration data once we have enough resolutions.
HIGH_DELTA_MEAN_SHIFT_DEG = 1.0

# Overround market filter
# If sum of all YES bucket prices > this threshold, the market is structurally
# broken (crowd has over-bid totals above 100%). BUY_YES is rejected; BUY_NO allowed.
OVERROUND_REJECT_YES_THRESHOLD = 1.15

# Laddering controls
ENABLE_LADDER_STRATEGY = False
LADDER_WIDTH = 3
LADDER_MAX_TOTAL_COST = 0.85
LADDER_MIN_EDGE = 0.08

# ── TOP2 shadow models (2A / 2B / 2C) ─────────────────────────────────────────
# Three parallel shadow strategies that never execute real trades but are logged
# and resolved daily so we can compare their virtual P&L against each other and
# the live strategy.  After enough data, graduate the best one to live execution.
#
#  TOP2_EQUAL  (2A) — always bets top-2 YES buckets by model prob, equal Kelly
#  TOP2_COND   (2B) — only bets 2 if model is split (2nd ≥ SPLIT_THRESHOLD × 1st)
#  TOP2_PROP   (2C) — always top-2 but secondary gets LOW Kelly (half the size)
#
# Rationale: model is empirically "one bucket off" when wrong, so buying both the
# favourite and runner-up should dramatically improve virtual win rate.  The three
# variants test whether always-2 or conditional-2, and equal vs proportional sizing,
# actually outperform the single-bucket live strategy net of the extra capital spent.
ENABLE_TOP2_SHADOWS = True

TOP2_SHADOW_MIN_PROB = 0.10         # second bucket must have ≥ 10% model probability to qualify
TOP2_SHADOW_SPLIT_THRESHOLD = 0.65  # 2B: second/first ratio must be ≥ this to count as "split"

# ── PURDEY_MK1 / CAVENDISH_MK1 shadow models ───────────────────────────────
# Two new hard-capped shadow strategies.
#   PURDEY_MK1    — strict top-2 by model probability, 60/40 split, NEVER 3+
#   CAVENDISH_MK1 — peak bucket + temperature flanks (one below, one above),
#                   50/25/25 split, NEVER 4+
# Both reuse ENABLE_TOP2_SHADOWS as their on/off toggle.
ENABLE_PURDEY_CAVENDISH = True

# ── City Accuracy Filter ────────────────────────────────────────────────────
# Skip cities where historical model ensemble accuracy is below this threshold.
# Based on best_ens_d1_win from accuracy_rows_cache.json.
# Wellington ~35%, Seoul ~42% → skip; Seattle ~55%, London ~56% → trade.
MIN_CITY_ACCURACY_THRESHOLD = 0.50

# Price scanner (Strategy 3) — continuous price monitoring between model runs
PRICE_SCAN_ENABLED        = True
PRICE_SCAN_INTERVAL       = 300      # seconds between price polls (5 min)
PRICE_SCAN_MIN_EV         = ALPHA_THRESHOLD  # min EV to trigger a dip trade
PRICE_SCAN_MAX_DAYS_AHEAD = 2        # skip D+3+ signals (too uncertain for dip-trading)

# Data-release danger windows
METAR_DANGER_PRE_MINUTE = 53
METAR_DANGER_POST_MINUTE = 58
SPECI_COOLDOWN_SECONDS = 180

# API endpoints
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
NOAA_API_URL = "https://api.weather.gov"
MET_OFFICE_API_URL = "https://data.hub.api.metoffice.gov.uk/sitespecific/v0/point"
ACCUWEATHER_SNAPSHOT_LOGGING_ENABLED = True

# Runtime mode
LIVE_TRADING = False
PAPER_TRADING = True


@dataclass(frozen=True)
class PolymarketCredentials:
    """Credentials needed for authenticated CLOB operations."""

    api_key: str
    secret: str
    passphrase: str
    private_key: str
    wallet_address: str


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


SETTLEMENT_WATCHER_POLL_SECONDS = max(5, int(_env_float("SETTLEMENT_WATCHER_POLL_SECONDS", 10.0)))
SETTLEMENT_WATCHER_OFFICIAL_REFRESH_SECONDS = max(
    30,
    int(_env_float("SETTLEMENT_WATCHER_OFFICIAL_REFRESH_SECONDS", 60.0)),
)
SETTLEMENT_WATCHER_FINAL_PRICE_THRESHOLD = _env_float("SETTLEMENT_WATCHER_FINAL_PRICE_THRESHOLD", 0.995)
SETTLEMENT_WATCHER_SPLIT_PRICE_TOLERANCE = _env_float("SETTLEMENT_WATCHER_SPLIT_PRICE_TOLERANCE", 0.05)


def load_runtime_overrides() -> dict[str, float | bool]:
    """Optionally override key runtime config values via environment variables."""
    return {
        "LIVE_TRADING": _env_bool("LIVE_TRADING", LIVE_TRADING),
        "PAPER_TRADING": _env_bool("PAPER_TRADING", PAPER_TRADING),
        "INITIAL_BANKROLL": _env_float("INITIAL_BANKROLL", INITIAL_BANKROLL),
    }


def load_station_priority_filter() -> set[str]:
    raw = os.getenv("STATION_PRIORITY_FILTER", "")
    if not raw.strip():
        return set(STATION_PRIORITY_FILTER)
    parsed = {part.strip().upper() for part in raw.split(",") if part.strip()}
    return parsed or set(STATION_PRIORITY_FILTER)


def load_clob_prefilter_priority() -> set[str]:
    raw = os.getenv("CLOB_PREFILTER_PRIORITY", "")
    if not raw.strip():
        return set(CLOB_PREFILTER_PRIORITY)
    parsed = {part.strip().upper() for part in raw.split(",") if part.strip()}
    return parsed or set(CLOB_PREFILTER_PRIORITY)


def load_polymarket_credentials() -> Optional[PolymarketCredentials]:
    """Load credentials from environment.

    Returns None if any required variable is missing.
    """
    api_key = os.getenv("POLY_API_KEY")
    secret = os.getenv("POLY_SECRET")
    passphrase = os.getenv("POLY_PASSPHRASE")
    private_key = os.getenv("PRIVATE_KEY")
    wallet_address = os.getenv("WALLET_ADDRESS")
    values = [api_key, secret, passphrase, private_key, wallet_address]
    if any(v is None or not v.strip() for v in values):
        return None
    return PolymarketCredentials(
        api_key=api_key.strip(),
        secret=secret.strip(),
        passphrase=passphrase.strip(),
        private_key=private_key.strip(),
        wallet_address=wallet_address.strip(),
    )
