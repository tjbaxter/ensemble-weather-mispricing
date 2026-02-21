"""METAR parsing helpers, including T-group extraction."""

from __future__ import annotations

import re


T_GROUP_PATTERN = re.compile(r"\bT(\d)(\d{3})(\d)(\d{3})\b")


def extract_t_group_temperature_c(raw_metar: str) -> float | None:
    """Return temperature in C from T-group if present."""
    if not raw_metar:
        return None
    match = T_GROUP_PATTERN.search(raw_metar)
    if not match:
        return None
    sign = -1.0 if match.group(1) == "1" else 1.0
    value = int(match.group(2)) / 10.0
    return sign * value


def parse_metar_temp_c(observation: dict) -> tuple[float | None, str]:
    """Return best available temp in C and confidence.

    Confidence is `high` when T-group precision exists, else `low`.
    """
    raw = str(observation.get("rawOb", "") or "")
    t_group_temp = extract_t_group_temperature_c(raw)
    if t_group_temp is not None:
        return t_group_temp, "high"

    temp = observation.get("temp")
    if temp is None:
        return None, "low"
    try:
        return float(temp), "low"
    except (TypeError, ValueError):
        return None, "low"
