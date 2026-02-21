"""Kelly sizing with additional safety constraints."""

from __future__ import annotations


def kelly_size(
    market_price: float,
    win_prob: float,
    bankroll: float,
    edge: float,
    kelly_fraction: float = 0.25,
    max_position: float = 3.0,
    rounding_confidence: str = "HIGH",
) -> float:
    """Return USD size using fractional Kelly and hard caps."""
    if edge <= 0 or bankroll <= 0:
        return 0.0
    if market_price <= 0.01 or market_price >= 0.99:
        return 0.0
    if not (0.0 <= win_prob <= 1.0):
        return 0.0

    b = (1.0 - market_price) / market_price
    q = 1.0 - win_prob
    full_kelly = (win_prob * b - q) / b
    if full_kelly <= 0:
        return 0.0

    confidence_multiplier = 1.0 if rounding_confidence.upper() == "HIGH" else 0.25
    dollar_amount = full_kelly * kelly_fraction * confidence_multiplier * bankroll
    dollar_amount = min(dollar_amount, max_position)
    return round(max(dollar_amount, 0.0), 2)
