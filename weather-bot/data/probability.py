"""Probability utilities for ensemble weather forecasts."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    from scipy.stats import gaussian_kde
except ImportError:  # pragma: no cover - optional runtime dependency fallback.
    gaussian_kde = None


def parse_bucket_bounds(bucket: str) -> tuple[float, float]:
    """Convert bucket label to half-open numeric interval."""
    clean = bucket.replace("°F", "").replace("°C", "").strip()
    if clean.endswith("+"):
        low = float(clean[:-1].strip())
        return low, float("inf")
    if "-" in clean:
        left, right = clean.split("-", 1)
        low = float(left.strip())
        high = float(right.strip()) + 1.0
        return low, high
    value = float(clean)
    return value, value + 1.0


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _gaussian_bucket_probs(ensemble_temps: Iterable[float], bucket_labels: list[str]) -> dict[str, float]:
    vals = np.array(list(ensemble_temps), dtype=float)
    if vals.size == 0:
        return {label: 0.0 for label in bucket_labels}
    mu = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=0))
    sigma = max(sigma, 0.8)
    probs: dict[str, float] = {}
    total = 0.0
    for label in bucket_labels:
        low, high = parse_bucket_bounds(label)
        if math.isinf(high):
            p = 1.0 - _normal_cdf(low, mu, sigma)
        else:
            p = _normal_cdf(high, mu, sigma) - _normal_cdf(low, mu, sigma)
        p = max(0.0, min(1.0, p))
        probs[label] = p
        total += p
    if total <= 0:
        return {label: 1.0 / len(bucket_labels) for label in bucket_labels} if bucket_labels else {}
    return {label: probs[label] / total for label in bucket_labels}


def ensemble_to_bucket_probs(
    ensemble_temps: list[float],
    bucket_labels: list[str],
    bandwidth_adjust: float = 1.0,
) -> dict[str, float]:
    """Map ensemble members to bucket probabilities with KDE."""
    if not ensemble_temps or not bucket_labels:
        return {label: 0.0 for label in bucket_labels}

    # Fallback when SciPy isn't available.
    if gaussian_kde is None:
        return _gaussian_bucket_probs(ensemble_temps, bucket_labels)

    temps = np.array(ensemble_temps, dtype=float)
    if np.unique(temps).size < 3:
        temps = temps + np.random.normal(0.0, 0.1, size=temps.shape[0])

    try:
        kde = gaussian_kde(temps, bw_method=bandwidth_adjust)
    except Exception:
        return _gaussian_bucket_probs(ensemble_temps, bucket_labels)

    probs: dict[str, float] = {}
    total = 0.0
    low_guard = float(np.min(temps) - 20.0)
    high_guard = float(np.max(temps) + 20.0)

    for label in bucket_labels:
        low, high = parse_bucket_bounds(label)
        if math.isinf(high):
            prob = float(kde.integrate_box_1d(low, high_guard))
        else:
            prob = float(kde.integrate_box_1d(low, high))
        prob = max(0.0, prob)
        probs[label] = prob
        total += prob

    if total <= 0:
        return {label: 1.0 / len(bucket_labels) for label in bucket_labels}
    return {label: probs[label] / total for label in bucket_labels}
