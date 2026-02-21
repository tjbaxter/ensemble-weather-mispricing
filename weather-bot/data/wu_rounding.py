"""Weather Underground display rounding simulation."""

from __future__ import annotations


def predict_wu_display_fahrenheit(temp_f: float) -> int:
    return int(round(temp_f))


def predict_wu_display_celsius(temp_c_tenths: float) -> int:
    temp_f = temp_c_tenths * 9.0 / 5.0 + 32.0
    temp_f_rounded = round(temp_f)
    temp_c_back = (temp_f_rounded - 32.0) * 5.0 / 9.0
    return int(round(temp_c_back))


def is_boundary_temperature(temp: float, unit: str, margin: float = 0.3) -> bool:
    """Return true when near a risky bucket boundary."""
    if unit.upper() == "F":
        # Degree-F markets are generally coarser and less fragile than C.
        return abs(temp - round(temp)) >= 0.5
    fractional = abs(temp - round(temp))
    return fractional >= (0.5 - margin)
