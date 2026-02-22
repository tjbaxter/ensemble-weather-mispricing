"""Global bot configuration.

All values are intentionally conservative and paper-first by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# Trading parameters
ALPHA_THRESHOLD = 0.08
MIN_FORECAST_CONFIDENCE = 0.30
MAX_POSITION_SIZE = 3.00
MAX_DAILY_EXPOSURE = 50.00
MAX_POSITIONS_PER_MARKET = 1
KELLY_FRACTION = 0.25

# Risk limits
MAX_DRAWDOWN_PCT = 0.15
INITIAL_BANKROLL = 300.00

# Timing
SCAN_INTERVAL_SECONDS = 120
FORECAST_REFRESH_SECONDS = 1800
HOURS_BEFORE_RESOLUTION_CUTOFF = 3

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
