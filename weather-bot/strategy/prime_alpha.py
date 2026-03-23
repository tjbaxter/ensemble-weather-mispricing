"""Deterministic PRIME_ALPHA range selection.

PRIME_ALPHA is designed to be replayable and auditable:
- score each model's accuracy over a rolling 7-day window
- models above the minimum hit-rate threshold are trusted
- build one contiguous bucket window from trusted models' range
- drop lone-wolf edge buckets with no corroboration
- cap the window width to avoid spraying extra buckets
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from data.probability import parse_bucket_bounds

_ROOT = Path(__file__).resolve().parents[1]
_ACCURACY_CACHE_PATH = _ROOT / "data" / "accuracy_rows_cache.json"
_MODEL_ACCURACY_LOG_PATH = _ROOT / "data" / "model_accuracy_log.json"
_MODEL_SNAPSHOT_LOG_PATH = _ROOT / "data" / "model_snapshot_log.json"
_POLYMARKET_CACHE_PATH = _ROOT / "data" / "polymarket_cache.json"
_LOG = logging.getLogger("weather-bot.prime_alpha")

PRIME_ALPHA_MAX_BUCKETS = 3
PRIME_ALPHA_MIN_TRUSTED_SOURCES = 2
PRIME_ALPHA_FALLBACK_TOP_MODELS = 3
PRIME_ALPHA_MIN_EDGE_CORROBORATION = 2
PRIME_ALPHA_TRUST_WINDOW_DAYS = 7
PRIME_ALPHA_MIN_HIT_RATE = 0.25
PRIME_ALPHA_FLAGSHIP_KEY = "flagship_ensemble"


@dataclass
class PrimeAlphaPlan:
    city: str
    station_icao: str
    target_date: str
    prior_date: str
    prior_resolved_bucket: str | None
    trust_source: str
    fallback_used: str | None
    trusted_models: list[str]
    fallback_models: list[str]
    trusted_flagship: bool
    trust_scores: dict[str, float]
    current_display_by_source: dict[str, int]
    bucket_support: dict[str, float]
    all_model_bucket_counts: dict[str, int]
    initial_selected_buckets: list[str]
    selected_buckets: list[str]
    range_low: int | None
    range_high: int | None
    notes: list[str]

    def to_strategy_context(self) -> dict[str, Any]:
        return {
            "selection_path": "prime_alpha",
            "prior_date": self.prior_date,
            "prior_resolved_bucket": self.prior_resolved_bucket,
            "trust_source": self.trust_source,
            "fallback_used": self.fallback_used,
            "trusted_models": self.trusted_models,
            "fallback_models": self.fallback_models,
            "trusted_flagship": self.trusted_flagship,
            "trust_scores": self.trust_scores,
            "current_display_by_source": self.current_display_by_source,
            "bucket_support": self.bucket_support,
            "all_model_bucket_counts": self.all_model_bucket_counts,
            "initial_selected_buckets": self.initial_selected_buckets,
            "selected_buckets": self.selected_buckets,
            "range_low": self.range_low,
            "range_high": self.range_high,
            "notes": self.notes,
        }


def build_prime_alpha_plan(
    *,
    city: str,
    station_icao: str,
    target_date: str,
    bucket_labels: list[str],
    current_model_values: dict[str, float],
    predicted_display_temp: float | int | None,
    unit: str,
    model_weights: dict[str, float] | None = None,
    trust_overrides: dict[str, bool | None] | None = None,
    prior_resolved_bucket: str | None = None,
) -> PrimeAlphaPlan:
    """Build the deterministic PRIME_ALPHA bucket plan for one city-date."""
    snapshot_preds = _load_snapshot_log_preds(city, target_date)
    if snapshot_preds:
        merged = dict(snapshot_preds)
        merged.update(current_model_values)
        current_model_values = merged

    ordered_buckets = sorted(dict.fromkeys(bucket_labels), key=_bucket_sort_key)
    prior_date = (date.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    notes: list[str] = []

    trust_scores, trust_source, prior_bucket = _resolve_rolling_trust(
        city=city,
        station_icao=station_icao,
        target_date=target_date,
        current_model_values=current_model_values,
        unit=unit,
        trust_overrides=trust_overrides,
        prior_resolved_bucket=prior_resolved_bucket,
        window_days=PRIME_ALPHA_TRUST_WINDOW_DAYS,
        min_hit_rate=PRIME_ALPHA_MIN_HIT_RATE,
    )
    if prior_bucket:
        notes.append(f"prior_bucket={prior_bucket}")

    trusted_models = sorted(
        model for model, score in trust_scores.items()
        if model != PRIME_ALPHA_FLAGSHIP_KEY and score >= PRIME_ALPHA_MIN_HIT_RATE
    )
    flagship_score = trust_scores.get(PRIME_ALPHA_FLAGSHIP_KEY, 0.0)
    trusted_flagship = flagship_score >= PRIME_ALPHA_MIN_HIT_RATE

    current_display_by_source: dict[str, int] = {}
    for model in trusted_models:
        current_display_by_source[model] = _market_display_temp(current_model_values[model], unit)
    if trusted_flagship and predicted_display_temp is not None:
        current_display_by_source[PRIME_ALPHA_FLAGSHIP_KEY] = _market_display_temp(
            predicted_display_temp,
            unit,
        )

    fallback_models: list[str] = []
    fallback_used: str | None = None

    if len(current_display_by_source) < PRIME_ALPHA_MIN_TRUSTED_SOURCES:
        if model_weights is None:
            from strategy.model_weights import compute_weights

            model_weights = compute_weights(station_icao, list(current_model_values.keys()))
        ranked_models = [
            model
            for model, _weight in sorted(
                model_weights.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )
            if model in current_model_values
        ]
        for model in ranked_models:
            if model in current_display_by_source:
                continue
            fallback_models.append(model)
            current_display_by_source[model] = _market_display_temp(current_model_values[model], unit)
            if len(current_display_by_source) >= max(PRIME_ALPHA_MIN_TRUSTED_SOURCES, PRIME_ALPHA_FALLBACK_TOP_MODELS):
                break
        if fallback_models:
            fallback_used = "top_weighted_models"
            notes.append(f"fallback_models={','.join(fallback_models)}")

    if not current_display_by_source and predicted_display_temp is not None:
        current_display_by_source[PRIME_ALPHA_FLAGSHIP_KEY] = _market_display_temp(
            predicted_display_temp,
            unit,
        )
        fallback_used = fallback_used or "flagship_only"
        notes.append("fallback_flagship_only")

    displays = list(current_display_by_source.values())
    range_low = min(displays) if displays else None
    range_high = max(displays) if displays else None

    bucket_support = _bucket_support_map(
        ordered_buckets,
        current_display_by_source=current_display_by_source,
        model_weights=model_weights or {},
        trust_scores=trust_scores,
    )

    initial_selected_buckets: list[str] = []
    if range_low is not None and range_high is not None:
        initial_selected_buckets = [
            bucket
            for bucket in ordered_buckets
            if _bucket_overlaps_display_range(bucket, range_low, range_high)
        ]

    all_displays = {
        model: _market_display_temp(val, unit)
        for model, val in current_model_values.items()
    }
    if predicted_display_temp is not None:
        all_displays[PRIME_ALPHA_FLAGSHIP_KEY] = _market_display_temp(
            predicted_display_temp, unit,
        )
    all_model_bucket_counts: dict[str, int] = {}
    for bucket in initial_selected_buckets:
        all_model_bucket_counts[bucket] = sum(
            1 for d in all_displays.values() if _bucket_contains_display(bucket, d)
        )

    selected_buckets = _drop_uncorroborated_edges(
        initial_selected_buckets,
        current_display_by_source=all_displays,
        min_sources=PRIME_ALPHA_MIN_EDGE_CORROBORATION,
        notes=notes,
    )
    if len(selected_buckets) > PRIME_ALPHA_MAX_BUCKETS:
        trimmed = _choose_dense_bucket_window(
            ordered_buckets=ordered_buckets,
            selected_buckets=selected_buckets,
            bucket_support=bucket_support,
            flagship_bucket=_display_bucket_for_source(
                ordered_buckets,
                current_display_by_source.get(PRIME_ALPHA_FLAGSHIP_KEY),
            ),
            max_buckets=PRIME_ALPHA_MAX_BUCKETS,
        )
        if trimmed != selected_buckets:
            notes.append(
                f"trimmed_window={','.join(selected_buckets)}->"
                f"{','.join(trimmed)}"
            )
        selected_buckets = trimmed

    return PrimeAlphaPlan(
        city=city,
        station_icao=station_icao,
        target_date=target_date,
        prior_date=prior_date,
        prior_resolved_bucket=prior_bucket,
        trust_source=trust_source,
        fallback_used=fallback_used,
        trusted_models=trusted_models,
        fallback_models=fallback_models,
        trusted_flagship=trusted_flagship,
        trust_scores={k: round(v, 3) for k, v in trust_scores.items() if v > 0},
        current_display_by_source=current_display_by_source,
        bucket_support=bucket_support,
        all_model_bucket_counts={b: all_model_bucket_counts.get(b, 0) for b in selected_buckets},
        initial_selected_buckets=initial_selected_buckets,
        selected_buckets=selected_buckets,
        range_low=range_low,
        range_high=range_high,
        notes=notes,
    )


def _resolve_rolling_trust(
    *,
    city: str,
    station_icao: str,
    target_date: str,
    current_model_values: dict[str, float],
    unit: str,
    trust_overrides: dict[str, bool | None] | None,
    prior_resolved_bucket: str | None,
    window_days: int,
    min_hit_rate: float,
) -> tuple[dict[str, float], str, str | None]:
    """Compute rolling hit-rate scores over the last N resolved days.

    Returns (scores, source_tag, prior_bucket) where scores maps each
    model key → float hit rate (0.0–1.0).  Models with no data get 0.0.
    """
    if trust_overrides:
        scores: dict[str, float] = {}
        for k, v in trust_overrides.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                scores[k] = float(v)
            elif v is True:
                scores[k] = 1.0
            else:
                scores[k] = 0.0
        return scores, "override", prior_resolved_bucket

    target = date.fromisoformat(target_date)
    prior_bucket: str | None = prior_resolved_bucket

    all_models = list(current_model_values.keys()) + [PRIME_ALPHA_FLAGSHIP_KEY]
    hits: dict[str, int] = {m: 0 for m in all_models}
    attempts: dict[str, int] = {m: 0 for m in all_models}
    source_tag = "none"
    days_scored = 0

    for offset in range(1, window_days + 1):
        day = (target - timedelta(days=offset)).isoformat()

        cache_row = _load_accuracy_cache_row(city, day)
        if cache_row:
            resolved_label = str(cache_row.get("resolved") or "")
            if not resolved_label:
                continue
            if offset == 1 and not prior_bucket:
                prior_bucket = resolved_label
            source_tag = "rolling_accuracy_cache"
            days_scored += 1
            for model in current_model_values:
                hit_key = f"{model}_d1_win"
                win = cache_row.get(hit_key)
                if win is not None:
                    attempts[model] += 1
                    if win is True:
                        hits[model] += 1
            ens_win = cache_row.get("best_ens_d1_win")
            if ens_win is not None:
                attempts[PRIME_ALPHA_FLAGSHIP_KEY] += 1
                if ens_win is True:
                    hits[PRIME_ALPHA_FLAGSHIP_KEY] += 1
            continue

        snapshot_preds = _load_snapshot_log_preds(city, day)
        pm_bucket = _load_polymarket_resolved_bucket(city, day)
        resolved = pm_bucket or (
            _load_prior_day_winner_bucket(station_icao=station_icao, prior_date=day)
        )
        if not resolved:
            model_log = _load_model_accuracy_entry(station_icao, day)
            actual = model_log.get("actual") if isinstance(model_log, dict) else None
            if actual is not None:
                actual_disp = _market_display_temp(actual, unit)
                resolved = str(actual_disp)
        if not resolved:
            continue

        if offset == 1 and not prior_bucket:
            prior_bucket = resolved

        preds_for_day = snapshot_preds or (
            (_load_model_accuracy_entry(station_icao, day).get("preds", {}) or {})
        )
        if not preds_for_day:
            continue

        source_tag = source_tag if "cache" in source_tag else "rolling_snapshot+resolved"
        days_scored += 1
        for model in current_model_values:
            pred = preds_for_day.get(model)
            if pred is None:
                continue
            attempts[model] += 1
            if _bucket_contains_display(resolved, _market_display_temp(float(pred), unit)):
                hits[model] += 1
        ens_hit = _infer_flagship_hit_from_preds(resolved, preds_for_day, unit)
        if ens_hit is not None:
            attempts[PRIME_ALPHA_FLAGSHIP_KEY] += 1
            if ens_hit:
                hits[PRIME_ALPHA_FLAGSHIP_KEY] += 1

    scores: dict[str, float] = {}
    for model in all_models:
        if attempts[model] > 0:
            scores[model] = hits[model] / attempts[model]
        else:
            scores[model] = 0.0

    if days_scored == 0:
        source_tag = "none"

    return scores, source_tag, prior_bucket


def _load_accuracy_cache_row(city: str, row_date: str) -> dict[str, Any] | None:
    cache = _load_json_dict(_ACCURACY_CACHE_PATH)
    if not cache:
        return None
    rows = cache.get(city)
    if not isinstance(rows, list):
        for key, value in cache.items():
            if str(key).strip().lower() == city.strip().lower() and isinstance(value, list):
                rows = value
                break
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and str(row.get("date", "")) == row_date:
            return row
    return None


def _load_model_accuracy_entry(station_icao: str, row_date: str) -> dict[str, Any]:
    log = _load_json_dict(_MODEL_ACCURACY_LOG_PATH)
    entry = log.get(f"{station_icao}/{row_date}")
    return entry if isinstance(entry, dict) else {}


def _load_prior_day_winner_bucket(*, station_icao: str, prior_date: str) -> str | None:
    logs_dir = _ROOT / "logs"
    if not logs_dir.exists():
        return None
    for csv_path in sorted(logs_dir.glob("resolved*.csv")):
        try:
            with csv_path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("station_icao") != station_icao:
                        continue
                    if row.get("target_date") != prior_date:
                        continue
                    if row.get("outcome") != "WIN":
                        continue
                    bucket = str(row.get("bucket", "") or "").strip()
                    if bucket:
                        return bucket
        except OSError:
            continue
    return None


def _load_snapshot_log_preds(city: str, target_date: str) -> dict[str, float]:
    """Load all model predictions from model_snapshot_log.json for city/date.

    The snapshot log typically covers more models than model_accuracy_log.json,
    making it a better source for trust resolution across all city models.
    """
    log = _load_json_dict(_MODEL_SNAPSHOT_LOG_PATH)
    city_data = log.get(city)
    if not isinstance(city_data, dict):
        return {}
    entry = city_data.get(target_date)
    if not isinstance(entry, dict):
        return {}
    preds = entry.get("preds", {})
    if not isinstance(preds, dict):
        return {}
    return {str(k): float(v) for k, v in preds.items() if v is not None}


def _load_polymarket_resolved_bucket(city: str, date_str: str) -> str | None:
    """Load the Polymarket-resolved bucket from polymarket_cache.json."""
    cache = _load_json_dict(_POLYMARKET_CACHE_PATH)
    city_data = cache.get(city)
    if not isinstance(city_data, dict):
        return None
    entry = city_data.get(date_str)
    if not isinstance(entry, (list, tuple)) or len(entry) < 1:
        return None
    label = str(entry[0])
    clean = label.replace("°F", "").replace("°C", "").strip()
    return clean if clean else None


def _infer_flagship_hit_from_preds(
    prior_bucket: str,
    preds: dict[str, Any],
    unit: str,
) -> bool | None:
    values = [float(value) for value in preds.values() if value is not None]
    if not values:
        return None
    ensemble_value = sum(values) / len(values)
    return _bucket_contains_display(prior_bucket, _market_display_temp(ensemble_value, unit))


def _infer_flagship_display_hit(
    *,
    actual_display: int,
    preds: dict[str, Any],
    unit: str,
) -> bool | None:
    values = [float(value) for value in preds.values() if value is not None]
    if not values:
        return None
    ensemble_value = sum(values) / len(values)
    return _market_display_temp(ensemble_value, unit) == actual_display


def _bucket_support_map(
    ordered_buckets: list[str],
    *,
    current_display_by_source: dict[str, int],
    model_weights: dict[str, float],
    trust_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    support: dict[str, float] = {}
    ts = trust_scores or {}
    for bucket in ordered_buckets:
        total = 0.0
        for source, display in current_display_by_source.items():
            if not _bucket_contains_display(bucket, display):
                continue
            base = 1.25 if source == PRIME_ALPHA_FLAGSHIP_KEY else float(
                model_weights.get(source, 1.0) or 1.0
            )
            accuracy_mult = ts.get(source, 1.0) if ts else 1.0
            total += base * max(accuracy_mult, 0.1)
        if total > 0:
            support[bucket] = round(total, 3)
    return support


def _display_bucket_for_source(
    ordered_buckets: list[str],
    display: int | None,
) -> str | None:
    if display is None:
        return None
    for bucket in ordered_buckets:
        if _bucket_contains_display(bucket, display):
            return bucket
    return None


def _choose_dense_bucket_window(
    *,
    ordered_buckets: list[str],
    selected_buckets: list[str],
    bucket_support: dict[str, float],
    flagship_bucket: str | None,
    max_buckets: int,
) -> list[str]:
    if len(selected_buckets) <= max_buckets:
        return selected_buckets
    try:
        left = ordered_buckets.index(selected_buckets[0])
        right = ordered_buckets.index(selected_buckets[-1])
    except ValueError:
        return selected_buckets[:max_buckets]

    best_window = selected_buckets[:max_buckets]
    best_key: tuple[float, int, int, float] | None = None

    for start in range(left, right - max_buckets + 2):
        window = ordered_buckets[start : start + max_buckets]
        if len(window) < max_buckets:
            continue
        support_sum = sum(float(bucket_support.get(bucket, 0.0)) for bucket in window)
        coverage = sum(1 for bucket in selected_buckets if bucket in window)
        contains_flagship = 1 if flagship_bucket and flagship_bucket in window else 0
        middle_index = start + (len(window) - 1) / 2.0
        distance_from_center = abs(((left + right) / 2.0) - middle_index)
        key = (support_sum, coverage, contains_flagship, -distance_from_center)
        if best_key is None or key > best_key:
            best_key = key
            best_window = window
    return best_window


def _drop_uncorroborated_edges(
    selected_buckets: list[str],
    *,
    current_display_by_source: dict[str, int],
    min_sources: int,
    notes: list[str],
) -> list[str]:
    """Remove edge buckets that only have a single model's support.

    If a bucket at the low or high end of the range is predicted by just one
    model while every other model clusters elsewhere, it's a lone-wolf outlier.
    Dropping it tightens the range and avoids wasting a bet.  Interior buckets
    are never removed (they bridge two corroborated edges).

    The filter only activates when there are enough total sources to make
    "lone wolf" meaningful (>= 2 * min_sources).  With a small model set
    (e.g. 4 trusted models), one model per bucket is normal, not an outlier.
    """
    if len(selected_buckets) <= 1:
        return list(selected_buckets)

    total_sources = len(current_display_by_source)
    if total_sources < min_sources * 2:
        return list(selected_buckets)

    def _count_sources(bucket: str) -> int:
        return sum(
            1 for display in current_display_by_source.values()
            if _bucket_contains_display(bucket, display)
        )

    result = list(selected_buckets)

    while len(result) > 1 and _count_sources(result[-1]) < min_sources:
        notes.append(f"dropped_high={result[-1]}(lone_wolf)")
        result.pop()

    while len(result) > 1 and _count_sources(result[0]) < min_sources:
        notes.append(f"dropped_low={result[0]}(lone_wolf)")
        result.pop(0)

    return result


def _bucket_contains_display(bucket: str, display_temp: int) -> bool:
    low, high = parse_bucket_bounds(bucket)
    return low <= float(display_temp) < high


def _bucket_overlaps_display_range(bucket: str, low_display: int, high_display: int) -> bool:
    bucket_low, bucket_high = parse_bucket_bounds(bucket)
    return bucket_low <= float(high_display) and bucket_high > float(low_display)


def _bucket_sort_key(bucket: str) -> float:
    low, _high = parse_bucket_bounds(bucket)
    return low


def _market_display_temp(raw_temp: float | int, unit: str) -> int:
    value = float(raw_temp)
    if str(unit).upper() == "F":
        return _round_half_up(value)
    temp_f = value * 9.0 / 5.0 + 32.0
    temp_f_rounded = _round_half_up(temp_f)
    temp_c_back = (temp_f_rounded - 32.0) * 5.0 / 9.0
    return _round_half_up(temp_c_back)


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception as exc:  # pragma: no cover - defensive file IO
        _LOG.debug("Failed to load %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}
