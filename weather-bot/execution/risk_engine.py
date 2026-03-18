"""Dynamic risk controls for position sizing and daily deployment.

Applies quality-aware multipliers and equity-percentage caps while preserving
minimum viable order size so strategies continue to execute.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import (
    DYNAMIC_RISK_SIZING_ENABLED,
    EQUITY_DAILY_EXPOSURE_PCT,
    EQUITY_MAX_POSITION_PCT,
    KELLY_MIN_BET_USD,
    QUALITY_D2_MULT,
    QUALITY_D3_MULT,
    QUALITY_EDGE_REF,
    QUALITY_EDGE_SLOPE,
    QUALITY_MAX_MULT,
    QUALITY_MIN_MULT,
    QUALITY_MULT_CONF_HIGH,
    QUALITY_MULT_CONF_LOW,
    QUALITY_MULT_CONF_MEDIUM,
    QUALITY_RED_SPREAD_MULT,
)


@dataclass(frozen=True)
class RiskDecision:
    size_usd: float
    daily_budget_usd: float
    position_cap_usd: float
    quality_mult: float
    skipped: bool
    reason: str = ""


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _confidence_mult(rounding_confidence: str) -> float:
    conf = (rounding_confidence or "MEDIUM").upper()
    if conf == "HIGH":
        return QUALITY_MULT_CONF_HIGH
    if conf == "LOW":
        return QUALITY_MULT_CONF_LOW
    return QUALITY_MULT_CONF_MEDIUM


def _quality_multiplier(signal: dict) -> float:
    mult = _confidence_mult(str(signal.get("rounding_confidence", "MEDIUM")))

    # Penalize longer-horizon forecasts.
    days_ahead = int(signal.get("days_ahead", 1) or 1)
    if days_ahead >= 3:
        mult *= QUALITY_D3_MULT
    elif days_ahead == 2:
        mult *= QUALITY_D2_MULT

    # Deterministic spread color captures model disagreement.
    if str(signal.get("spread_colour", "")).upper() == "RED":
        mult *= QUALITY_RED_SPREAD_MULT

    # Reward stronger edge; weak edges remain closer to baseline.
    edge = max(0.0, float(signal.get("edge", 0.0) or 0.0))
    edge_scale = min(edge / max(QUALITY_EDGE_REF, 1e-9), 2.0)
    mult *= 1.0 + QUALITY_EDGE_SLOPE * edge_scale

    return _clamp(mult, QUALITY_MIN_MULT, QUALITY_MAX_MULT)


def _equity_proxy(cash_usd: float, active_exposure_usd: float) -> float:
    # Conservative mark uses cost basis for open positions.
    return max(float(cash_usd or 0.0) + float(active_exposure_usd or 0.0), 0.0)


def apply_risk_controls(
    *,
    requested_size_usd: float,
    signal: dict,
    cash_usd: float,
    active_exposure_usd: float,
    deployed_today_usd: float,
) -> RiskDecision:
    """Return risk-adjusted size and skip reason (if any)."""
    req = max(0.0, float(requested_size_usd or 0.0))
    if req <= 0:
        return RiskDecision(0.0, 0.0, 0.0, 1.0, True, "non_positive_size")

    if not DYNAMIC_RISK_SIZING_ENABLED:
        return RiskDecision(round(req, 2), 0.0, 0.0, 1.0, False, "")

    equity = _equity_proxy(cash_usd, active_exposure_usd)
    if equity <= 0:
        return RiskDecision(0.0, 0.0, 0.0, 1.0, True, "zero_equity")

    daily_budget = max(KELLY_MIN_BET_USD, equity * EQUITY_DAILY_EXPOSURE_PCT)
    position_cap = max(KELLY_MIN_BET_USD, equity * EQUITY_MAX_POSITION_PCT)
    quality_mult = _quality_multiplier(signal)

    target = req * quality_mult
    target = min(target, position_cap)
    target = round(max(target, KELLY_MIN_BET_USD), 2)

    remaining = daily_budget - float(deployed_today_usd or 0.0)
    if remaining < KELLY_MIN_BET_USD:
        return RiskDecision(0.0, round(daily_budget, 2), round(position_cap, 2), quality_mult, True, "daily_budget_exhausted")

    allowed = min(target, remaining)
    if allowed < KELLY_MIN_BET_USD:
        return RiskDecision(0.0, round(daily_budget, 2), round(position_cap, 2), quality_mult, True, "below_min_bet")

    return RiskDecision(round(allowed, 2), round(daily_budget, 2), round(position_cap, 2), quality_mult, False, "")

