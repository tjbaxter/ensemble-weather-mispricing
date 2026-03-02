"""Kelly sizing with confidence-tier scaling and hard caps.

Confidence tiers (relative to the per-city kelly_fraction base):
  HIGH   — multi-model consensus or live METAR confirmation → 2× base fraction
  MEDIUM — single model run, standard forecast signal       → 1× base fraction
  LOW    — weak signal / distant date                       → 0.5× base fraction

With the default base of 0.25 (quarter Kelly) this maps to:
  HIGH   → 0.50× full Kelly (half Kelly)
  MEDIUM → 0.25× full Kelly (quarter Kelly)
  LOW    → 0.125× full Kelly (eighth Kelly)

Per-city overrides (e.g., London = 0.50) scale all three tiers uniformly:
  London HIGH → 1.0× full Kelly  (validated over 75 days)
  London MED  → 0.5× full Kelly
  London LOW  → 0.25× full Kelly
"""

from __future__ import annotations

# Scale applied to the per-city kelly_fraction for each confidence tier.
_CONFIDENCE_SCALE: dict[str, float] = {
    "HIGH":   2.0,
    "MEDIUM": 1.0,
    "LOW":    0.5,
}


def kelly_size(
    market_price: float,
    win_prob: float,
    bankroll: float,
    edge: float,
    kelly_fraction: float = 0.25,
    max_position: float = 25.0,
    rounding_confidence: str = "MEDIUM",
) -> float:
    """Return USD size using fractional Kelly and hard caps.

    Args:
        market_price:      Current best ask (cost to buy one $1 YES share).
        win_prob:          Model's estimated probability of winning.
        bankroll:          Current cash available.
        edge:              model_prob - market_price (pre-validated as > 0).
        kelly_fraction:    Per-city base Kelly fraction (default 0.25 = quarter-Kelly).
        max_position:      Hard upper cap in USD.
        rounding_confidence: 'HIGH' | 'MEDIUM' | 'LOW' — scales the fraction.
    """
    if edge <= 0 or bankroll <= 0:
        return 0.0
    if market_price <= 0.01 or market_price >= 0.99:
        return 0.0
    if not (0.0 <= win_prob <= 1.0):
        return 0.0

    b = (1.0 - market_price) / market_price   # net payout per $1 risked
    q = 1.0 - win_prob
    full_kelly = (win_prob * b - q) / b
    if full_kelly <= 0:
        return 0.0

    scale = _CONFIDENCE_SCALE.get(rounding_confidence.upper(), 1.0)
    effective_fraction = kelly_fraction * scale
    dollar_amount = full_kelly * effective_fraction * bankroll
    dollar_amount = min(dollar_amount, max_position)
    return round(max(dollar_amount, 0.0), 2)
