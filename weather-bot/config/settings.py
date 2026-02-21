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
MODEL_RUN_BOOST_ENABLED = True
MODEL_RUN_BOOST_SCAN_INTERVAL_SECONDS = 30
MODEL_RUN_BOOST_WINDOW_MINUTES = 20

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
ENSEMBLE_PRIMARY_MODEL = "gfs_seamless"
ENSEMBLE_ADDITIONAL_MODELS = ("ecmwf_ifs025", "icon_seamless_eps")
ENSEMBLE_PREVIOUS_RUNS_API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ENSEMBLE_DAILY_API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
CALIBRATION_JSON_PATH = "logs/calibration.json"
MODEL_RANKINGS_JSON_PATH = "logs/model_rankings.json"
ENSEMBLE_BATCH_CACHE_TTL_SECONDS = 300
ENSEMBLE_CONFIDENCE_STD_HIGH = 1.5
ENSEMBLE_CONFIDENCE_STD_LOW = 3.0
ENSEMBLE_STD_SKIP_THRESHOLD = 5.0
ENSEMBLE_DISABLE_CLASSIC_CONFIDENCE_GATE = True

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
