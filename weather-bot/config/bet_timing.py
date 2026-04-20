"""Per-city betting window configuration.

Defines when Prime Alpha should place bets for each city, based on:
1. When the daily max temperature is typically "locked in" (past sunset + temp dropping)
2. When fresh model data is available (18Z for Europe, 00Z for Americas)
3. Optimal timing to capture edge before market repricing

Two clusters:
- European cities: Bet same evening (22:00-23:45 UTC) after 18Z models
- Americas cities: Bet overnight (02:00-07:00 UTC) after 00Z models
"""

from __future__ import annotations

from typing import Any

# ── Betting Window Configurations ────────────────────────────────────────────────
# earliest_bet_utc: First UTC hour we can place bets (inclusive)
# latest_bet_utc: Last UTC hour we should place bets (inclusive)
# If latest < earliest, window crosses midnight
# preferred_model_run: Which model run gives freshest forecasts for this window

BETTING_WINDOWS: dict[str, dict[str, Any]] = {
    # ── European Cluster ─────────────────────────────────────────────────────────
    # Bet same evening after 18Z models land (~20:30-21:00 UTC)
    # Max temp locked by sunset (18:00-20:00 local) + 2°C drop
    "London": {
        "earliest_bet_utc": 22.5,   # 22:30 UTC (after 18Z models settle)
        "latest_bet_utc": 23.75,    # 23:45 UTC
        "preferred_model_run": "18Z",
        "fallback_model_run": "12Z",
        "cluster": "europe",
    },
    "Ankara": {
        "earliest_bet_utc": 22.0,   # Earlier sunset in Turkey
        "latest_bet_utc": 23.5,
        "preferred_model_run": "18Z",
        "fallback_model_run": "12Z",
        "cluster": "europe",
    },
    "Paris": {
        "earliest_bet_utc": 22.5,
        "latest_bet_utc": 23.75,
        "preferred_model_run": "18Z",
        "fallback_model_run": "12Z",
        "cluster": "europe",
    },

    # ── Americas Cluster ─────────────────────────────────────────────────────────
    # Bet overnight after 00Z models land (~02:30-03:00 UTC)
    # Max temp locked by late evening local time
    "NYC": {
        "earliest_bet_utc": 2.0,    # 02:00 UTC = 10pm ET previous day
        "latest_bet_utc": 4.0,
        "preferred_model_run": "00Z",
        "fallback_model_run": "18Z",
        "cluster": "americas",
    },
    "Chicago": {
        "earliest_bet_utc": 3.0,    # 03:00 UTC = 10pm CT previous day
        "latest_bet_utc": 5.0,
        "preferred_model_run": "00Z",
        "fallback_model_run": "18Z",
        "cluster": "americas",
    },
    "Dallas": {
        "earliest_bet_utc": 3.0,
        "latest_bet_utc": 5.5,
        "preferred_model_run": "00Z",
        "fallback_model_run": "18Z",
        "cluster": "americas",
    },
    "Atlanta": {
        "earliest_bet_utc": 2.0,
        "latest_bet_utc": 4.0,
        "preferred_model_run": "00Z",
        "fallback_model_run": "18Z",
        "cluster": "americas",
    },
    "Miami": {
        "earliest_bet_utc": 2.0,
        "latest_bet_utc": 4.0,
        "preferred_model_run": "00Z",
        "fallback_model_run": "18Z",
        "cluster": "americas",
    },
    "Seattle": {
        "earliest_bet_utc": 5.0,    # 05:00 UTC = 10pm PT previous day
        "latest_bet_utc": 7.0,
        "preferred_model_run": "00Z",
        "fallback_model_run": "18Z",
        "cluster": "americas",
    },
    "Toronto": {
        "earliest_bet_utc": 2.0,
        "latest_bet_utc": 4.0,
        "preferred_model_run": "00Z",
        "fallback_model_run": "18Z",
        "cluster": "americas",
    },

    # ── South America (crosses midnight UTC) ─────────────────────────────────────
    "Buenos Aires": {
        "earliest_bet_utc": 23.5,   # 23:30 UTC = 8:30pm local
        "latest_bet_utc": 1.5,      # 01:30 UTC next day = 10:30pm local
        "preferred_model_run": "18Z",
        "fallback_model_run": "12Z",
        "cluster": "south_america",
    },

    # ── Asia-Pacific (not yet active, placeholder) ───────────────────────────────
    "Seoul": {
        "earliest_bet_utc": 12.0,   # 12:00 UTC = 9pm KST
        "latest_bet_utc": 14.0,
        "preferred_model_run": "06Z",
        "fallback_model_run": "00Z",
        "cluster": "asia",
    },
    "Wellington": {
        "earliest_bet_utc": 8.0,    # 08:00 UTC = 8pm NZST
        "latest_bet_utc": 10.0,
        "preferred_model_run": "00Z",
        "fallback_model_run": "18Z",
        "cluster": "oceania",
    },
}


# ── Early Resolution Detection Parameters ────────────────────────────────────────
# These control when we consider a city's daily max "locked in"

RESOLUTION_PARAMS: dict[str, dict[str, Any]] = {
    # European cities - earlier sunset, earlier lock
    "London": {
        "min_local_hour_for_lock": 17,
        "sunset_local_hour": 20,
        "temp_drop_threshold_c": 2.0,
    },
    "Ankara": {
        "min_local_hour_for_lock": 16,
        "sunset_local_hour": 19,
        "temp_drop_threshold_c": 2.0,
    },
    "Paris": {
        "min_local_hour_for_lock": 17,
        "sunset_local_hour": 20,
        "temp_drop_threshold_c": 2.0,
    },

    # Americas cities
    "NYC": {
        "min_local_hour_for_lock": 17,
        "sunset_local_hour": 19,
        "temp_drop_threshold_f": 3.5,
    },
    "Chicago": {
        "min_local_hour_for_lock": 17,
        "sunset_local_hour": 19,
        "temp_drop_threshold_f": 3.5,
    },
    "Dallas": {
        "min_local_hour_for_lock": 17,
        "sunset_local_hour": 20,
        "temp_drop_threshold_f": 3.5,
    },
    "Atlanta": {
        "min_local_hour_for_lock": 17,
        "sunset_local_hour": 20,
        "temp_drop_threshold_f": 3.5,
    },
    "Miami": {
        "min_local_hour_for_lock": 17,
        "sunset_local_hour": 19,
        "temp_drop_threshold_f": 3.5,
    },
    "Seattle": {
        "min_local_hour_for_lock": 17,
        "sunset_local_hour": 20,
        "temp_drop_threshold_f": 3.5,
    },
    "Toronto": {
        "min_local_hour_for_lock": 17,
        "sunset_local_hour": 20,
        "temp_drop_threshold_c": 2.0,
    },
    "Buenos Aires": {
        "min_local_hour_for_lock": 16,
        "sunset_local_hour": 18,
        "temp_drop_threshold_c": 2.0,
    },
    "Seoul": {
        "min_local_hour_for_lock": 17,
        "sunset_local_hour": 19,
        "temp_drop_threshold_c": 2.0,
    },
    "Wellington": {
        "min_local_hour_for_lock": 16,
        "sunset_local_hour": 18,
        "temp_drop_threshold_c": 2.0,
    },
}


def get_betting_window(city: str) -> dict[str, Any] | None:
    """Get betting window config for a city."""
    return BETTING_WINDOWS.get(city)


def get_resolution_params(city: str) -> dict[str, Any] | None:
    """Get early resolution detection params for a city."""
    return RESOLUTION_PARAMS.get(city)
