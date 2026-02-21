"""Edge computations for binary temperature-bucket markets."""

from __future__ import annotations

from config.settings import ALPHA_THRESHOLD, MIN_FORECAST_CONFIDENCE


def calculate_edge(
    forecast_prob: float,
    market_prob: float,
    min_forecast_confidence: float | None = None,
) -> tuple[str, float, float]:
    """Return action, edge, and win probability for Kelly sizing."""
    if not (0.0 <= forecast_prob <= 1.0 and 0.0 <= market_prob <= 1.0):
        return ("NO_TRADE", 0.0, 0.0)

    yes_edge = forecast_prob - market_prob
    no_edge = market_prob - forecast_prob
    no_forecast = 1.0 - forecast_prob

    conf_floor = MIN_FORECAST_CONFIDENCE if min_forecast_confidence is None else min_forecast_confidence

    if yes_edge > ALPHA_THRESHOLD and forecast_prob > conf_floor:
        return ("BUY_YES", yes_edge, forecast_prob)
    if no_edge > ALPHA_THRESHOLD and no_forecast > conf_floor:
        return ("BUY_NO", no_edge, no_forecast)
    return ("NO_TRADE", 0.0, 0.0)
