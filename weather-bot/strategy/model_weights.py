"""Recency-weighted model accuracy scoring for MK2 / ACE strategies.

Tracks individual forecast model accuracy and computes weights that shift
bucket probability distributions toward models that have been accurate
recently.  Models that nailed yesterday's temperature get a large boost;
models with good week-long track records get a moderate boost; unknown
models get a default weight of 1.0.

Data sources (priority order):
  1. data/model_accuracy_log.json  — accumulated prediction + actual pairs
  2. logs/calibration.json         — 45-day historical stats (baseline)
  3. Default weight 1.0
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

_log = logging.getLogger("weather-bot.model_weights")

_LOG_PATH = Path("data/model_accuracy_log.json")
_CALIBRATION_PATH = Path("logs/calibration.json")

# Map station-specific model keys → calibration.json keys (baseline fallback)
_CALIB_MAP: dict[str, str] = {
    "gfs_graphcast025": "gfs_seamless",
    "gfs_seamless": "gfs_seamless",
    "ecmwf_ifs025": "ecmwf_ifs025",
    "icon_seamless": "icon_seamless_eps",
    "icon_global": "icon_seamless_eps",
    "icon_seamless_eps": "icon_seamless_eps",
}


# ── Accuracy log I/O ────────────────────────────────────────────────────


def _load_log() -> dict[str, Any]:
    if _LOG_PATH.exists():
        try:
            return json.loads(_LOG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_log(log: dict[str, Any]) -> None:
    try:
        _LOG_PATH.write_text(json.dumps(log, indent=2, default=str))
    except OSError as exc:
        _log.warning("Failed to save model accuracy log: %s", exc)


def log_predictions(
    station_icao: str,
    target_date: str,
    model_values: dict[str, float],
) -> None:
    """Record what each model predicted for a given station + date."""
    if not model_values:
        return
    log = _load_log()
    key = f"{station_icao}/{target_date}"
    entry = log.get(key, {})
    entry["preds"] = {k: round(float(v), 1) for k, v in model_values.items()}
    entry.setdefault("actual", None)
    log[key] = entry
    _save_log(log)


def log_actual_temperature(
    station_icao: str,
    target_date: str,
    actual_temp: float,
) -> None:
    """Record the observed temperature for a station + date."""
    log = _load_log()
    key = f"{station_icao}/{target_date}"
    entry = log.get(key, {})
    entry["actual"] = round(float(actual_temp), 1)
    log[key] = entry
    _save_log(log)


# ── Weight computation ──────────────────────────────────────────────────


def compute_weights(
    station_icao: str,
    available_models: list[str],
    lookback_days: int = 7,
) -> dict[str, float]:
    """Return {model_name: weight} — higher means more trusted.

    Weight factors:
      • Yesterday within 1°F  → +3.0
      • Yesterday within 2°F  → +2.0
      • Yesterday within 3°F  → +0.5
      • Yesterday > 4°F off   → -0.5
      • Recent proportion within 2°F → up to +2.0
      • Recent mean error penalty → up to -1.5
      • Calibration baseline (gfs/ecmwf/icon only) → ±0.5
    """
    today = date.today()
    cutoff = today - timedelta(days=lookback_days)
    yesterday = today - timedelta(days=1)

    log = _load_log()

    # Gather per-model errors from the log
    model_errors: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for key, entry in log.items():
        parts = key.split("/", 1)
        if len(parts) != 2 or parts[0] != station_icao:
            continue
        try:
            target_d = date.fromisoformat(parts[1])
        except (ValueError, TypeError):
            continue
        if target_d < cutoff or target_d >= today:
            continue
        actual = entry.get("actual")
        preds = entry.get("preds", {})
        if actual is None:
            continue
        actual_f = float(actual)
        for model_name, pred_val in preds.items():
            try:
                model_errors[model_name].append(
                    (target_d, abs(float(pred_val) - actual_f))
                )
            except (ValueError, TypeError):
                continue

    # Load calibration baseline
    calib_mae: dict[str, float] = {}
    if _CALIBRATION_PATH.exists():
        try:
            cdata = json.loads(_CALIBRATION_PATH.read_text())
            station_cal = cdata.get("stations", cdata).get(station_icao, {})
            for cmodel, stats in station_cal.items():
                if isinstance(stats, dict) and "mean_abs_error" in stats:
                    calib_mae[cmodel] = float(stats["mean_abs_error"])
        except (json.JSONDecodeError, OSError, AttributeError):
            pass

    weights: dict[str, float] = {}

    for model in available_models:
        w = 1.0

        if model in model_errors:
            errors = model_errors[model]
            n = len(errors)
            mean_err = sum(e for _, e in errors) / n
            n_within_2 = sum(1 for _, e in errors if e <= 2.5)

            # Yesterday performance
            yest = [e for d, e in errors if d == yesterday]
            if yest:
                best = min(yest)
                if best <= 1.0:
                    w += 3.0
                elif best <= 2.0:
                    w += 2.0
                elif best <= 3.0:
                    w += 0.5
                else:
                    w -= 0.5

            # Week proportion within 2°F
            w += (n_within_2 / n) * 2.0

            # Mean error penalty
            w -= min(mean_err / 3.0, 1.5)

        else:
            # No log data — try calibration baseline
            calib_key = _CALIB_MAP.get(model)
            if calib_key and calib_key in calib_mae:
                mae = calib_mae[calib_key]
                if mae <= 2.0:
                    w += 0.5
                elif mae > 4.0:
                    w -= 0.5

        weights[model] = max(w, 0.1)

    _log.info(
        "Model weights %s: %s",
        station_icao,
        " ".join(f"{m}={w:.1f}" for m, w in sorted(weights.items(), key=lambda x: -x[1])),
    )
    return weights


# ── Weighted bucket probabilities ───────────────────────────────────────


def weighted_bucket_probs(
    model_preds: dict[str, float],
    weights: dict[str, float],
    bucket_labels: list[str],
) -> dict[str, float]:
    """Compute bucket probabilities from accuracy-weighted model predictions.

    Creates synthetic ensemble members by repeating each model's prediction
    proportional to its weight, then runs them through the existing KDE-based
    probability engine.  Higher-weighted models dominate the distribution.
    """
    from data.probability import ensemble_to_bucket_probs

    if not model_preds or not bucket_labels:
        return {}

    samples: list[float] = []
    for model, temp in model_preds.items():
        w = weights.get(model, 1.0)
        n_copies = max(1, round(w * 10))
        samples.extend([float(temp)] * n_copies)

    if not samples:
        return {}

    return ensemble_to_bucket_probs(samples, bucket_labels)


def models_in_bucket(
    model_preds: dict[str, float],
    bucket: str,
) -> int:
    """Count how many individual models predict a temp in this bucket's range."""
    from data.probability import parse_bucket_bounds

    low, high = parse_bucket_bounds(bucket)
    return sum(1 for t in model_preds.values() if low <= float(t) < high)
