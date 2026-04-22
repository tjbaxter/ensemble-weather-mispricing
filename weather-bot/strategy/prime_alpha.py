"""PRIME_ALPHA V3 — Two-layer architecture: scoring + deterministic bucket selector.

Layer A (scoring): family grouping, continuous error scoring, short/long skill
    split (days 1-3 / 4-7), regime-adaptive weighting.
Layer B (selection): recent-winner set, recent-support set, working set,
    robust center, outlier-winner control, center + anchor + bridge rules.

The Gaussian mixture is kept for diagnostics only. The final bucket selection
comes from Layer B, not the mixture.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings import PRIME_ALPHA_ALLOW_GAUSSIAN_FALLBACK
from data.probability import parse_bucket_bounds

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ACCURACY_LOG_PATH = _ROOT / "data" / "model_accuracy_log.json"
_MODEL_SNAPSHOT_LOG_PATH = _ROOT / "data" / "model_snapshot_log.json"
_POLYMARKET_CACHE_PATH = _ROOT / "data" / "polymarket_cache.json"
_COMMERCIAL_FORECAST_LOG_PATH = _ROOT / "data" / "commercial_forecast_log.json"
_ACCURACY_CACHE_PATH = _ROOT / "data" / "accuracy_rows_cache.json"
_LOG = logging.getLogger("weather-bot.prime_alpha")

# ── V3 Constants ────────────────────────────────────────────────────────────────

TRUST_WINDOW_DAYS = 7

TAU_FAHRENHEIT = 3.0
TAU_CELSIUS = 1.5

SHORT_WINDOW_DAYS = 3
SHORT_DECAY = 0.6

LONG_START_OFFSET = 4
LONG_END_OFFSET = 7
LONG_DECAY = 0.8

REGIME_SHIFT_THRESHOLD_F = 8.0
REGIME_SHIFT_THRESHOLD_C = 4.5

ALPHA_LONG = 1.0
BETA_MIN = 1.0
BETA_MAX = 3.0

REGIME_SUPPRESS_MIN = 0.6
SHORT_SKILL_BAD_THRESHOLD = 0.20
LONG_SKILL_BAD_THRESHOLD = 0.25
MIN_FAMILY_WEIGHT = 0.01

SHRINK_TO_EQUAL = 0.25

SIGMA_FLOOR_F = 1.0
SIGMA_FLOOR_C = 0.5
SIGMA_DEFAULT_F = 3.0
SIGMA_DEFAULT_C = 1.5
SIGMA_MAE_CORRECTION = 0.798

SUPPORT_MIN_HITS = 2
SUPPORT_LOOKBACK_START = 2
SUPPORT_LOOKBACK_END = 4
SUPPORT_DISTANCE_F = 2.5
SUPPORT_DISTANCE_C = 1.5

OUTLIER_DISTANCE_F = 2.0
OUTLIER_DISTANCE_C = 1.0
OUTLIER_LONG_SKILL_MAX = 0.35
UPPER_SUPPORT_MIN_FAMILIES = 2

COMMERCIAL_PROMOTE_LOOKBACK = 5
COMMERCIAL_PROMOTE_MIN_HITS = 3
COMMERCIAL_PROMOTE_MAX_WEIGHT = 0.10

PRIME_ALPHA_MAX_BUCKETS = 2
MIN_BUCKET_PROB = 0.05
FALLBACK_MAX_DISTANCE_F = 2.0
FALLBACK_MAX_DISTANCE_C = 1.0

STREAK_MIN_LENGTH = 2
BIMODAL_GAP_C = 2.0
BIMODAL_GAP_F = 3.0

# ── Provisional Prior-Day Resolution ────────────────────────────────────────────
# These thresholds control when we can determine the prior day's result BEFORE
# official Polymarket resolution. Two detection methods:
#   1. Temperature physics: temp dropped from max + evening time = max locked
#   2. Market consensus: any bucket at 98%+ = market knows the answer
#
# Settings made aggressive to enable D+1 betting during optimal windows.
PROVISIONAL_EARLY_LOCAL_HOUR = 16      # Start checking at 4pm local (was 17)
PROVISIONAL_LATE_LOCAL_HOUR = 19       # Relaxed mode at 7pm local (was 20)
PROVISIONAL_COOLOFF_C = 1.5            # 1.5°C drop from max (was 2.0)
PROVISIONAL_COOLOFF_F = 2.5            # 2.5°F drop from max (was 3.5)
PROVISIONAL_MAX_STABLE_MINUTES = 20    # Max stable for 20 min (was 60)
PROVISIONAL_MARKET_CONFIDENCE = 0.98   # 98% market consensus (was 98.5%)
PROVISIONAL_RUNNERUP_MAX = 0.02        # Runner-up max 2% (was 1.5%)
PROVISIONAL_MIN_POLLS = 2              # Need 2 readings (was 3)
PROVISIONAL_POLL_SPACING_SEC = 300     # 5 min between polls (was 10 min)

# Fast-path: if ANY bucket hits this threshold, skip all other checks
# 95% is high enough to be confident, low enough to trigger reliably
MARKET_CONSENSUS_INSTANT_THRESHOLD = 0.95

# ── Model Family Definitions ───────────────────────────────────────────────────
# Only merge models that are empirically near-identical (duplicate signals).
# Genuinely different submodels (e.g. UKMO local 2km vs global 10km, MF AROME
# vs ARPEGE) remain as singletons so they can compete independently.

MODEL_FAMILIES: dict[str, list[str]] = {
    "GFS": ["gfs_seamless", "gfs_hrrr"],
    "GEM": ["gem_seamless", "gem_regional"],
    "MF_AROME": ["meteofrance_arome_france", "meteofrance_arome_france_hd"],
}

_MODEL_TO_FAMILY: dict[str, str] = {}
for _fam, _members in MODEL_FAMILIES.items():
    for _m in _members:
        _MODEL_TO_FAMILY[_m] = _fam

FLAGSHIP_MEMBERS = ["icon_seamless", "gem_seamless", "gem_regional", "gfs_seamless"]
PRIME_ALPHA_FLAGSHIP_KEY = "flagship_ensemble"
COMMERCIAL_WU_KEY = "weather_com"
COMMERCIAL_ACCU_KEY = "accuweather"
_DIAGNOSTIC_ONLY_KEYS = {PRIME_ALPHA_FLAGSHIP_KEY, COMMERCIAL_WU_KEY, COMMERCIAL_ACCU_KEY}

_CITY_NAME_ALIASES: dict[str, list[str]] = {
    "NYC": ["New York", "NYC"],
    "New York": ["New York", "NYC"],
}


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
    families: dict[str, dict[str, Any]] = field(default_factory=dict)
    diagnostics_only: dict[str, dict[str, Any]] = field(default_factory=dict)
    regime_strength: float = 0.0
    beta_used: float = 1.0
    bucket_probabilities: dict[str, float] = field(default_factory=dict)
    selection_layer: dict[str, Any] = field(default_factory=dict)

    def to_strategy_context(self) -> dict[str, Any]:
        return {
            "selection_path": "prime_alpha_v3",
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
            "families": self.families,
            "diagnostics_only": self.diagnostics_only,
            "regime_strength": self.regime_strength,
            "beta_used": self.beta_used,
            "bucket_probabilities": self.bucket_probabilities,
            "selection_layer": self.selection_layer,
        }


# ── Math helpers ────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_cdf_at(value: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if value >= mu else 0.0
    return _norm_cdf((value - mu) / sigma)


def _bucket_probability(lower: float, upper: float,
                        means: list[float], sigmas: list[float],
                        weights: list[float]) -> float:
    total = 0.0
    for mu, sigma, w in zip(means, sigmas, weights):
        total += w * (_norm_cdf_at(upper, mu, sigma) - _norm_cdf_at(lower, mu, sigma))
    return total


def _weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return sum(values) / len(values)
    cumulative = 0.0
    for val, w in pairs:
        cumulative += w
        if cumulative >= total / 2.0:
            return val
    return pairs[-1][0]


# ── Family grouping ─────────────────────────────────────────────────────────────

def _group_into_families(model_values: dict[str, float]) -> dict[str, dict[str, Any]]:
    families: dict[str, list[tuple[str, float]]] = {}
    for model, value in model_values.items():
        fam = _MODEL_TO_FAMILY.get(model, model)
        families.setdefault(fam, []).append((model, value))
    result: dict[str, dict[str, Any]] = {}
    for fam, members in families.items():
        values = [v for _, v in members]
        result[fam] = {
            "members": [m for m, _ in members],
            "forecast": sum(values) / len(values),
        }
    return result


def _resolved_midpoint(bucket_label: str) -> float | None:
    clean = (bucket_label.replace("\u00b0F", "").replace("\u00b0C", "")
             .replace("\u2265", "").replace("\u2264", "")
             .replace(">=", "").replace("<=", "").strip())
    if clean.endswith("+"):
        try:
            return float(clean[:-1].strip())
        except ValueError:
            pass
    if "-" in clean:
        parts = clean.split("-", 1)
        try:
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
        except ValueError:
            pass
    try:
        return float(clean)
    except ValueError:
        return None


def _family_hits_bucket(family_forecast: float, bucket_label: str, unit: str) -> bool:
    display = _market_display_temp(family_forecast, unit)
    label = (bucket_label.replace("\u00b0F", "").replace("\u00b0C", "")
             .replace("°F", "").replace("°C", "").strip())
    if label.startswith("\u2265") or label.startswith(">="):
        try:
            return display >= float(label.lstrip("\u2265>=").strip())
        except ValueError:
            return False
    if label.startswith("\u2264") or label.startswith("<="):
        try:
            return display <= float(label.lstrip("\u2264<=").strip())
        except ValueError:
            return False
    try:
        lo, hi = parse_bucket_bounds(label)
    except (ValueError, TypeError):
        return False
    return lo <= display < hi


def _forecast_to_bucket(forecast: float, ordered_buckets: list[str], unit: str) -> str | None:
    display = _market_display_temp(forecast, unit)
    for bucket in ordered_buckets:
        lo, hi = parse_bucket_bounds(bucket)
        if lo <= display < hi:
            return bucket
    return None


def _bucket_index(bucket: str, ordered_buckets: list[str]) -> int:
    try:
        return ordered_buckets.index(bucket)
    except ValueError:
        return -1


# ── Core V3 ─────────────────────────────────────────────────────────────────────

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
    snapshot_preds = _load_snapshot_log_preds(city, target_date)
    if snapshot_preds:
        merged = dict(snapshot_preds)
        merged.update(current_model_values)
        current_model_values = merged

    ordered_buckets = sorted(dict.fromkeys(bucket_labels), key=_bucket_sort_key)
    prior_date = (date.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    notes: list[str] = []
    is_f = unit.upper() == "F"
    tau = TAU_FAHRENHEIT if is_f else TAU_CELSIUS
    sigma_floor = SIGMA_FLOOR_F if is_f else SIGMA_FLOOR_C
    sigma_default = SIGMA_DEFAULT_F if is_f else SIGMA_DEFAULT_C

    # ── Diagnostic-only sources ─────────────────────────────────────────────
    diag_forecasts: dict[str, float] = {}
    flagship_vals = [current_model_values[m] for m in FLAGSHIP_MEMBERS
                     if m in current_model_values]
    if flagship_vals:
        diag_forecasts[PRIME_ALPHA_FLAGSHIP_KEY] = sum(flagship_vals) / len(flagship_vals)
    commercial = _load_commercial_forecast(city, target_date)
    diag_forecasts.update(commercial)

    # ── Core families (current day) ─────────────────────────────────────────
    all_family_info = _group_into_families(current_model_values)
    core_families = {f: info for f, info in all_family_info.items()
                     if f not in _DIAGNOSTIC_ONLY_KEYS}

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER A — Scoring
    # ══════════════════════════════════════════════════════════════════════════

    target_d = date.fromisoformat(target_date)
    resolved_history: list[dict[str, Any]] = []
    prior_bucket_label: str | None = prior_resolved_bucket

    for offset in range(1, TRUST_WINDOW_DAYS + 1):
        day = (target_d - timedelta(days=offset)).isoformat()
        resolved_label = _load_polymarket_resolved_bucket(city, day)
        if not resolved_label:
            resolved_label = _load_prior_day_winner_bucket(
                station_icao=station_icao, prior_date=day)
        if not resolved_label:
            entry = _load_model_accuracy_entry(station_icao, day)
            actual = entry.get("actual") if isinstance(entry, dict) else None
            if actual is not None:
                resolved_label = str(_market_display_temp(float(actual), unit))
        if not resolved_label and offset == 1 and prior_resolved_bucket:
            resolved_label = prior_resolved_bucket
        if not resolved_label:
            continue
        midpoint = _resolved_midpoint(resolved_label)
        if midpoint is None:
            continue
        if offset == 1 and not prior_bucket_label:
            prior_bucket_label = resolved_label

        preds = _load_snapshot_log_preds(city, day)
        if not preds:
            preds = (_load_model_accuracy_entry(station_icao, day)
                     .get("preds", {}) or {})

        resolved_history.append({
            "date": day, "offset": offset,
            "resolved_label": resolved_label, "midpoint": midpoint,
            "preds": preds,
        })

    if prior_bucket_label:
        notes.append(f"prior_bucket={prior_bucket_label}")
    notes.append(f"days_scored={len(resolved_history)}")

    # ── Regime strength ─────────────────────────────────────────────────────
    sorted_hist = sorted(resolved_history, key=lambda x: x["date"])
    midpoints = [h["midpoint"] for h in sorted_hist]
    regime_strength = 0.0
    if len(midpoints) >= 2:
        yesterday_mid = midpoints[-1]
        trailing = midpoints[-4:-1] if len(midpoints) >= 4 else midpoints[:-1]
        trailing_avg = sum(trailing) / len(trailing)
        shift = abs(yesterday_mid - trailing_avg)
        threshold = REGIME_SHIFT_THRESHOLD_F if is_f else REGIME_SHIFT_THRESHOLD_C
        regime_strength = min(1.0, shift / threshold)

    beta = BETA_MIN + regime_strength * (BETA_MAX - BETA_MIN)
    notes.append(f"regime={regime_strength:.2f},beta={beta:.2f}")

    # ── Per-family day scores ───────────────────────────────────────────────
    family_day_scores: dict[str, list[tuple[int, float]]] = {
        f: [] for f in core_families}
    family_errors: dict[str, list[float]] = {f: [] for f in core_families}
    family_hist_forecasts: dict[str, dict[int, float]] = {
        f: {} for f in core_families}

    for hist in resolved_history:
        offset = hist["offset"]
        midpoint = hist["midpoint"]
        preds = hist["preds"]
        for fam, finfo in core_families.items():
            member_preds = [float(preds[m]) for m in finfo["members"]
                           if m in preds and preds[m] is not None]
            if not member_preds:
                continue
            fam_forecast = sum(member_preds) / len(member_preds)
            error = abs(fam_forecast - midpoint)
            family_day_scores[fam].append((offset, math.exp(-error / tau)))
            family_errors[fam].append(error)
            family_hist_forecasts[fam][offset] = fam_forecast

    # ── Short skill (1-3), Long skill (4-7) ─────────────────────────────────
    family_short: dict[str, float] = {}
    family_long: dict[str, float] = {}
    family_raw: dict[str, float] = {}
    family_sigmas: dict[str, float] = {}

    for fam in core_families:
        scores = family_day_scores[fam]
        errors = family_errors[fam]

        s_sum, s_w = 0.0, 0.0
        for offset, score in scores:
            if 1 <= offset <= SHORT_WINDOW_DAYS:
                w = SHORT_DECAY ** (offset - 1)
                s_sum += score * w
                s_w += w
        short_skill = s_sum / s_w if s_w > 0 else 0.0

        l_sum, l_w = 0.0, 0.0
        for offset, score in scores:
            if LONG_START_OFFSET <= offset <= LONG_END_OFFSET:
                w = LONG_DECAY ** (offset - LONG_START_OFFSET)
                l_sum += score * w
                l_w += w
        if l_w > 0:
            long_skill = l_sum / l_w
        elif s_w > 0:
            long_skill = short_skill
        else:
            long_skill = 0.0

        family_short[fam] = short_skill
        family_long[fam] = long_skill
        family_raw[fam] = ALPHA_LONG * long_skill + beta * short_skill

        mae = sum(errors) / len(errors) if errors else (
            sigma_default * SIGMA_MAE_CORRECTION)
        family_sigmas[fam] = max(mae / SIGMA_MAE_CORRECTION, sigma_floor)

    # ── Soft suppression ────────────────────────────────────────────────────
    suppressed: list[str] = []
    if regime_strength >= REGIME_SUPPRESS_MIN:
        for fam in core_families:
            if (family_short[fam] <= SHORT_SKILL_BAD_THRESHOLD
                    and family_long[fam] <= LONG_SKILL_BAD_THRESHOLD):
                family_raw[fam] = 0.0
                suppressed.append(fam)
    if suppressed:
        notes.append(f"suppressed={','.join(suppressed)}")

    # ── Softmax + shrinkage → w_final ───────────────────────────────────────
    active = [f for f in core_families
              if f not in suppressed and family_raw.get(f, 0) > 0]
    if not active:
        active = list(core_families.keys())

    exp_w = {f: math.exp(family_raw[f]) for f in active}
    total_exp = sum(exp_w.values()) or 1.0
    n_active = len(active) or 1
    eq_w = 1.0 / n_active

    family_weights: dict[str, float] = {}
    for f in active:
        family_weights[f] = ((1.0 - SHRINK_TO_EQUAL) * (exp_w[f] / total_exp)
                             + SHRINK_TO_EQUAL * eq_w)
    for f in suppressed:
        family_weights[f] = MIN_FAMILY_WEIGHT

    total_w = sum(family_weights.values()) or 1.0
    for f in family_weights:
        family_weights[f] /= total_w

    # ── Diagnostic Gaussian bucket probabilities ────────────────────────────
    f_names = [f for f in family_weights if family_weights[f] > 0]
    f_means = [core_families[f]["forecast"] for f in f_names]
    f_sigs = [family_sigmas.get(f, sigma_default) for f in f_names]
    f_wts = [family_weights[f] for f in f_names]

    bucket_probs: dict[str, float] = {}
    for bucket in ordered_buckets:
        lo, hi = parse_bucket_bounds(bucket)
        if math.isinf(hi):
            hi = lo + 100.0
        if math.isinf(lo):
            lo = hi - 100.0
        bucket_probs[bucket] = round(
            _bucket_probability(lo, hi, f_means, f_sigs, f_wts), 6)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER B — Deterministic Bucket Selection
    # ══════════════════════════════════════════════════════════════════════════

    sel_info: dict[str, Any] = {}

    # ── Recent winners: families that hit yesterday's resolved bucket ───────
    day1_hist = next((h for h in resolved_history if h["offset"] == 1), None)
    recent_winners: list[str] = []
    if day1_hist:
        resolved_bucket = day1_hist["resolved_label"]
        for fam in core_families:
            hist_fc = family_hist_forecasts[fam].get(1)
            if hist_fc is not None and _family_hits_bucket(
                    hist_fc, resolved_bucket, unit):
                recent_winners.append(fam)

    sel_info["recent_winners"] = {
        f: round(core_families[f]["forecast"], 1) for f in recent_winners}

    # ── Streak scoring from canonical accuracy cache ─────────────────────
    accuracy_rows = _load_accuracy_cache(city)
    acc_by_date: dict[str, dict] = {
        r["date"]: r for r in accuracy_rows if "date" in r
    }

    streak_scores: dict[str, int] = {}
    streak_quality: dict[str, float] = {}
    for fam in core_families:
        streak = 0
        quality = 0.0
        members = core_families[fam]["members"]
        for hist in sorted(resolved_history, key=lambda h: h["offset"]):
            cache_row = acc_by_date.get(hist["date"])
            if cache_row:
                fam_hit = any(
                    cache_row.get(f"{m}_d1_win") is True for m in members
                )
            else:
                hist_fc = family_hist_forecasts[fam].get(hist["offset"])
                if hist_fc is None:
                    break
                fam_hit = _family_hits_bucket(
                    hist_fc, hist["resolved_label"], unit)

            if fam_hit:
                streak += 1
                quality += _streak_day_quality(hist["resolved_label"])
            else:
                break
        streak_scores[fam] = streak
        streak_quality[fam] = round(quality, 2)

    max_streak = max(streak_scores.values()) if streak_scores else 0

    if max_streak >= STREAK_MIN_LENGTH:
        streak_top_tier = [f for f, s in streak_scores.items()
                          if s == max_streak]
        if len(streak_top_tier) >= 2:
            strong_streak = streak_top_tier
        else:
            strong_streak = [f for f, s in streak_scores.items()
                             if s >= max(STREAK_MIN_LENGTH, max_streak - 1)]
    else:
        strong_streak = []

    sel_info["streak_scores"] = streak_scores
    sel_info["streak_quality"] = streak_quality
    sel_info["strong_streak"] = strong_streak
    sel_info["max_streak"] = max_streak

    # ── Winner center ───────────────────────────────────────────────────────
    winner_forecasts = [core_families[f]["forecast"] for f in recent_winners]
    winner_skills = [family_short.get(f, 0) for f in recent_winners]
    winner_center = (_weighted_median(winner_forecasts, winner_skills)
                     if recent_winners else None)
    sel_info["winner_center"] = (round(winner_center, 1)
                                 if winner_center is not None else None)

    # ── Recent support: non-winners with strong recent streak + near cluster
    support_distance = SUPPORT_DISTANCE_F if is_f else SUPPORT_DISTANCE_C
    recent_support: list[str] = []
    if winner_center is not None:
        for fam in core_families:
            if fam in recent_winners or fam in suppressed:
                continue
            hits = 0
            for hist in resolved_history:
                off = hist["offset"]
                if not (SUPPORT_LOOKBACK_START <= off <= SUPPORT_LOOKBACK_END):
                    continue
                hist_fc = family_hist_forecasts[fam].get(off)
                if hist_fc is None:
                    continue
                if _family_hits_bucket(hist_fc, hist["resolved_label"], unit):
                    hits += 1
            if hits < SUPPORT_MIN_HITS:
                continue
            fc_now = core_families[fam]["forecast"]
            if abs(fc_now - winner_center) <= support_distance:
                recent_support.append(fam)

    sel_info["recent_support"] = {
        f: round(core_families[f]["forecast"], 1) for f in recent_support}

    # ── Working set ─────────────────────────────────────────────────────────
    if strong_streak:
        working_set = list(dict.fromkeys(strong_streak))
    else:
        working_set = list(dict.fromkeys(recent_winners + recent_support))
    sel_info["working_set"] = working_set

    # ── Commercial promotion (city-specific, earned, capped) ────────────
    promoted_commercial: dict[str, float] = {}
    for comm_key in (COMMERCIAL_WU_KEY, COMMERCIAL_ACCU_KEY):
        comm_forecast = diag_forecasts.get(comm_key)
        if comm_forecast is None:
            continue
        hits = 0
        checked = 0
        for hist in resolved_history:
            if hist["offset"] > COMMERCIAL_PROMOTE_LOOKBACK:
                continue
            checked += 1
            hist_comm = _load_commercial_forecast(city, hist["date"])
            hist_val = hist_comm.get(comm_key)
            if hist_val is not None and _family_hits_bucket(
                    hist_val, hist["resolved_label"], unit):
                hits += 1
        if checked > 0 and hits >= COMMERCIAL_PROMOTE_MIN_HITS:
            promoted_commercial[comm_key] = comm_forecast
            notes.append(f"promoted={comm_key}({hits}/{checked})")

    sel_info["promoted_commercial"] = {
        k: round(v, 1) for k, v in promoted_commercial.items()}

    # ── Working center ──────────────────────────────────────────────────────
    ws_forecasts = [core_families[f]["forecast"] for f in working_set]
    ws_weights = [family_weights.get(f, 0) for f in working_set]
    for comm_fc in promoted_commercial.values():
        ws_forecasts.append(comm_fc)
        ws_weights.append(COMMERCIAL_PROMOTE_MAX_WEIGHT)
    working_center = (_weighted_median(ws_forecasts, ws_weights)
                      if ws_forecasts else None)
    sel_info["working_center"] = (round(working_center, 1)
                                  if working_center is not None else None)

    # ── Outlier-winner control ──────────────────────────────────────────────
    outlier_dist = OUTLIER_DISTANCE_F if is_f else OUTLIER_DISTANCE_C
    high_outlier_winners: list[str] = []
    if working_center is not None:
        for fam in recent_winners:
            fc = core_families[fam]["forecast"]
            if (fc - working_center) <= outlier_dist:
                continue
            if family_long.get(fam, 0) > OUTLIER_LONG_SKILL_MAX:
                continue
            others_in_zone = [
                f2 for f2 in working_set if f2 != fam
                and core_families[f2]["forecast"] > working_center + outlier_dist]
            if others_in_zone:
                continue
            high_outlier_winners.append(fam)

    sel_info["high_outlier_winners"] = high_outlier_winners

    # ── Bimodal detection on streak leaders ───────────────────────────────
    bimodal_selected = False
    if strong_streak and len(strong_streak) >= 2:
        display_by_fam = {
            f: _market_display_temp(core_families[f]["forecast"], unit)
            for f in strong_streak
        }
        unique_displays = sorted(set(display_by_fam.values()))

        if len(unique_displays) >= 2:
            bimodal_gap = BIMODAL_GAP_F if is_f else BIMODAL_GAP_C
            best_gap = 0
            best_split = -1
            for i in range(len(unique_displays) - 1):
                gap = unique_displays[i + 1] - unique_displays[i]
                if gap > best_gap:
                    best_gap = gap
                    best_split = i

            if best_gap >= bimodal_gap and best_split >= 0:
                lower_temps = set(unique_displays[: best_split + 1])
                upper_temps = set(unique_displays[best_split + 1 :])
                lower_fams = [
                    f for f in strong_streak if display_by_fam[f] in lower_temps
                ]
                upper_fams = [
                    f for f in strong_streak if display_by_fam[f] in upper_temps
                ]

                def _best_display(fams):
                    counts: dict[int, int] = {}
                    for f in fams:
                        d = display_by_fam[f]
                        counts[d] = counts.get(d, 0) + 1
                    return max(counts, key=counts.get)

                lower_display = _best_display(lower_fams)
                upper_display = _best_display(upper_fams)
                lower_bucket = _forecast_to_bucket(
                    float(lower_display), ordered_buckets, unit
                )
                upper_bucket = _forecast_to_bucket(
                    float(upper_display), ordered_buckets, unit
                )

                if (
                    lower_bucket
                    and upper_bucket
                    and lower_bucket != upper_bucket
                ):
                    bimodal_selected = True
                    notes.append(
                        f"bimodal_streak="
                        f"{lower_display}({len(lower_fams)})"
                        f"_vs_{upper_display}({len(upper_fams)})"
                        f"_gap={best_gap:.0f}"
                    )
                    sel_info["bimodal"] = {
                        "lower": {
                            "display": lower_display,
                            "bucket": lower_bucket,
                            "families": lower_fams,
                        },
                        "upper": {
                            "display": upper_display,
                            "bucket": upper_bucket,
                            "families": upper_fams,
                        },
                        "gap": best_gap,
                    }

    # ── Deterministic bucket selection ──────────────────────────────────────
    # V3.5: Always exactly 2 buckets — center + best adjacent.
    # Single-bucket prediction is borderline unprofitable; the temperature
    # almost always lands in one of two neighbouring buckets.  By covering
    # both we capture the physical uncertainty while keeping package cost low.
    selected: list[str] = []

    if bimodal_selected:
        selected = [
            sel_info["bimodal"]["lower"]["bucket"],
            sel_info["bimodal"]["upper"]["bucket"],
        ]
    elif working_center is not None and working_set:
        center_bucket = _forecast_to_bucket(
            working_center, ordered_buckets, unit)
        sel_info["center_bucket"] = center_bucket

        if center_bucket:
            selected.append(center_bucket)
            center_idx = _bucket_index(center_bucket, ordered_buckets)

            # Pick the best adjacent bucket (immediately above or below).
            adj_lower = (ordered_buckets[center_idx - 1]
                         if center_idx > 0 else None)
            adj_upper = (ordered_buckets[center_idx + 1]
                         if center_idx < len(ordered_buckets) - 1 else None)

            def _adjacent_score(bucket: str) -> tuple[int, float, float]:
                support = sum(
                    1 for f in working_set
                    if _forecast_to_bucket(
                        core_families[f]["forecast"],
                        ordered_buckets, unit) == bucket
                )
                center_mid = _resolved_midpoint(center_bucket)
                adj_mid = _resolved_midpoint(bucket)
                lean = 0.0
                if (center_mid is not None and adj_mid is not None
                        and working_center is not None):
                    if adj_mid > center_mid and working_center > center_mid:
                        lean = 1.0
                    elif adj_mid < center_mid and working_center < center_mid:
                        lean = 1.0
                prob = bucket_probs.get(bucket, 0.0)
                return (support, lean, prob)

            adj_candidates = [(b, _adjacent_score(b))
                              for b in (adj_lower, adj_upper) if b is not None]
            if adj_candidates:
                best_adj, adj_sc = max(adj_candidates, key=lambda x: x[1])
                selected.append(best_adj)
                sel_info["adjacent_bucket"] = {
                    "bucket": best_adj,
                    "model_support": adj_sc[0],
                    "lean_bonus": adj_sc[1],
                    "gaussian_prob": round(adj_sc[2], 4),
                }
                if len(adj_candidates) == 2:
                    other = [c for c in adj_candidates if c[0] != best_adj][0]
                    sel_info["adjacent_runner_up"] = {
                        "bucket": other[0],
                        "model_support": other[1][0],
                        "lean_bonus": other[1][1],
                        "gaussian_prob": round(other[1][2], 4),
                    }
                notes.append(f"double_bet={center_bucket}+{best_adj}")

    if not selected:
        if not working_set and not strong_streak:
            notes.append("layer_a_empty_no_fallback")
        else:
            notes.append("pass_no_center_bucket")

    selected = sorted(selected, key=_bucket_sort_key)
    sel_info["selected"] = selected

    # ── Diagnostics ─────────────────────────────────────────────────────────
    fam_diag: dict[str, dict[str, Any]] = {}
    for f in f_names:
        fam_diag[f] = {
            "members": core_families[f]["members"],
            "forecast": round(core_families[f]["forecast"], 2),
            "sigma": round(family_sigmas.get(f, sigma_default), 3),
            "weight": round(family_weights.get(f, 0), 4),
            "short_skill": round(family_short.get(f, 0), 4),
            "long_skill": round(family_long.get(f, 0), 4),
            "raw_score": round(family_raw.get(f, 0), 4),
            "suppressed": f in suppressed,
        }

    diag_only: dict[str, dict[str, Any]] = {}
    for key, val in diag_forecasts.items():
        diag_only[key] = {"forecast": round(val, 2)}

    # Legacy-compatible fields
    trusted_models_list, trust_compat, display_by = [], {}, {}
    for f in f_names:
        for m in core_families[f]["members"]:
            trusted_models_list.append(m)
            trust_compat[m] = round(family_raw.get(f, 0), 3)
            if m in current_model_values:
                display_by[m] = _market_display_temp(
                    current_model_values[m], unit)
    displays = list(display_by.values())

    bucket_support_c = {b: round(p, 4) for b, p in bucket_probs.items()
                        if p > 0.001}
    bucket_counts_c: dict[str, int] = {}
    for b in selected:
        lo, hi = parse_bucket_bounds(b)
        bucket_counts_c[b] = max(
            sum(1 for mu in f_means if lo <= mu < hi), 1)

    ranked_all = sorted(bucket_probs.items(), key=lambda x: -x[1])

    return PrimeAlphaPlan(
        city=city, station_icao=station_icao, target_date=target_date,
        prior_date=prior_date, prior_resolved_bucket=prior_bucket_label,
        trust_source="v3.5_double_bet" if resolved_history else "v3.5_no_history",
        fallback_used=None, trusted_models=trusted_models_list,
        fallback_models=[], trusted_flagship=False,
        trust_scores=trust_compat, current_display_by_source=display_by,
        bucket_support=bucket_support_c,
        all_model_bucket_counts=bucket_counts_c,
        initial_selected_buckets=[b for b, _ in ranked_all[:10]],
        selected_buckets=selected,
        range_low=min(displays) if displays else None,
        range_high=max(displays) if displays else None,
        notes=notes, families=fam_diag, diagnostics_only=diag_only,
        regime_strength=regime_strength, beta_used=beta,
        bucket_probabilities={b: round(p, 4) for b, p in bucket_probs.items()
                              if p > 0.005},
        selection_layer=sel_info,
    )


# ── Data Loaders ────────────────────────────────────────────────────────────────

def _load_snapshot_log_preds(city: str, target_date: str) -> dict[str, float]:
    log = _load_json_dict(_MODEL_SNAPSHOT_LOG_PATH)
    names = _CITY_NAME_ALIASES.get(city, [city])
    city_data: dict | None = None
    for n in names:
        c = log.get(n)
        if isinstance(c, dict):
            city_data = c
            break
    if city_data is None:
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
    cache = _load_json_dict(_POLYMARKET_CACHE_PATH)
    city_data = cache.get(city)
    if not isinstance(city_data, dict):
        return None
    entry = city_data.get(date_str)
    if not isinstance(entry, (list, tuple)) or len(entry) < 1:
        return None
    label = str(entry[0])
    clean = (label.replace("\u00b0F", "").replace("\u00b0C", "")
             .replace("°F", "").replace("°C", "").strip())
    return clean if clean else None


def _load_model_accuracy_entry(
        station_icao: str, row_date: str) -> dict[str, Any]:
    log = _load_json_dict(_MODEL_ACCURACY_LOG_PATH)
    entry = log.get(f"{station_icao}/{row_date}")
    return entry if isinstance(entry, dict) else {}


def _load_prior_day_winner_bucket(
        *, station_icao: str, prior_date: str) -> str | None:
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


def _load_commercial_forecast(
        city: str, target_date: str) -> dict[str, float]:
    log = _load_json_dict(_COMMERCIAL_FORECAST_LOG_PATH)
    entry = log.get(city, {}).get(target_date, {})
    result: dict[str, float] = {}
    wu = entry.get("wu")
    if wu is not None:
        result[COMMERCIAL_WU_KEY] = float(wu)
    accu = entry.get("accu")
    if accu is not None:
        result[COMMERCIAL_ACCU_KEY] = float(accu)
    return result


# ── Provisional Prior-Day Resolution ────────────────────────────────────────────
# Ephemeral in-memory state: tracks when a running-max bucket was first observed
# unchanged for each (city, prior_date) pair. Lost on process restart. NEVER
# written to disk or to the official Polymarket cache.

_PROVISIONAL_POLL_STATE: dict[tuple[str, str], dict[str, Any]] = {}


def _resolve_official_timestamp(
    city: str, station_icao: str, prior_date: str,
) -> tuple[str | None, str]:
    """Return (ISO timestamp, source) for when prior-day resolution became available.

    Sources:
      ``resolved_csv``           — real Polymarket resolution event time
      ``settlement_snapshot``    — real settlement event time
      ``local_midnight_fallback``— conservative synthetic bound (local end-of-day)

    Uses the actual resolution event time from resolved.csv (``resolved_at``
    column) or settlement_snapshot — NOT the polymarket_cache file mtime.

    Fallback: local end-of-day for the station's timezone, so the gate never
    opens before the city's calendar day has actually finished. For a western
    city like NYC in EDT (UTC-4), midnight local Mar 29 = 04:00 UTC Mar 29,
    which is 4 hours later than the naive UTC-midnight fallback.
    """
    from config.cities import STATIONS as _STATIONS

    # 1. resolved.csv — actual Polymarket resolution event timestamp
    logs_dir = _ROOT / "logs"
    if logs_dir.exists():
        for csv_path in sorted(logs_dir.glob("resolved*.csv")):
            try:
                with csv_path.open(encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        if row.get("station_icao") != station_icao:
                            continue
                        if row.get("target_date") != prior_date:
                            continue
                        ts = (
                            row.get("resolved_at")
                            or row.get("resolved_at_utc")
                            or row.get("timestamp_utc")
                        )
                        if ts:
                            return str(ts).strip(), "resolved_csv"
            except OSError:
                continue

    # 2. settlement_snapshot.json
    settlement_path = _ROOT / "data" / "settlement_snapshot.json"
    if settlement_path.exists():
        try:
            snap = json.loads(settlement_path.read_text(encoding="utf-8"))
            if isinstance(snap, dict):
                for entry in snap.values():
                    if not isinstance(entry, dict):
                        continue
                    if (entry.get("city") == city or entry.get("station_icao") == station_icao) \
                            and entry.get("date") == prior_date:
                        ts = (
                            entry.get("resolved_at")
                            or entry.get("resolved_at_utc")
                            or entry.get("timestamp_utc")
                        )
                        if ts:
                            return str(ts).strip(), "settlement_snapshot"
        except Exception:
            pass

    # 3. Fallback: local end-of-day for the station's timezone.
    #    Conservative synthetic bound — the gate cannot open until the
    #    city's calendar day is actually over. Never uses naked UTC midnight.
    try:
        import zoneinfo
        tz_name = _STATIONS.get(station_icao, {}).get("timezone")
        if tz_name:
            local_tz = zoneinfo.ZoneInfo(tz_name)
        else:
            utc_offset_hours = _STATIONS.get(station_icao, {}).get(
                "station_standard_offset_hours",
            )
            if utc_offset_hours is not None:
                next_day = date.fromisoformat(prior_date) + timedelta(days=1)
                local_midnight_utc = datetime(
                    next_day.year, next_day.month, next_day.day,
                    tzinfo=UTC,
                ) - timedelta(hours=int(utc_offset_hours))
                return local_midnight_utc.isoformat(), "local_midnight_fallback"
            else:
                return None, "none"

        next_day = date.fromisoformat(prior_date) + timedelta(days=1)
        local_midnight = datetime(
            next_day.year, next_day.month, next_day.day,
            tzinfo=local_tz,
        )
        return local_midnight.astimezone(UTC).isoformat(), "local_midnight_fallback"
    except Exception:
        return None, "none"


def get_effective_prior_resolved_bucket(
    *,
    city: str,
    station_icao: str,
    prior_date: str,
    prior_day_market_prices: dict[str, float] | None = None,
    replay_override: str | None = None,
) -> dict[str, Any]:
    """Determine the prior day's resolved bucket using a precedence cascade.

    Precedence:
      1. Explicit replay override (replay CLI only)
      2. Official Polymarket cache / resolved.csv / model_accuracy_log
      3. Source-confirmed provisional (WU obs + market confirmation, in-memory)
      4. None

    NEVER writes to the official cache. Provisional results are ephemeral.
    """
    from config.cities import STATIONS
    from datetime import datetime as _dt, timezone as _tz

    result: dict[str, Any] = {
        "bucket": None,
        "mode": "none",
        "confidence": 0.0,
        "poll_count": 0,
        "wu_running_max": None,
        "wu_latest_temp": None,
        "signal_available_at_utc": None,
        "prior_signal_timestamp_source": "none",
    }

    # 1. Replay override takes absolute precedence
    if replay_override:
        result["bucket"] = replay_override
        result["mode"] = "replay_override"
        result["confidence"] = 1.0
        result["signal_available_at_utc"] = _dt.now(UTC).isoformat()
        result["prior_signal_timestamp_source"] = "replay_override"
        return result

    # 2. Official cache
    official = _load_polymarket_resolved_bucket(city, prior_date)
    if not official:
        official = _load_prior_day_winner_bucket(
            station_icao=station_icao, prior_date=prior_date)
    if not official:
        entry = _load_model_accuracy_entry(station_icao, prior_date)
        actual = entry.get("actual") if isinstance(entry, dict) else None
        if actual is not None:
            unit = str(STATIONS.get(station_icao, {}).get("resolution_unit", "F"))
            official = str(_market_display_temp(float(actual), unit))
    if official:
        _off_ts, _off_src = _resolve_official_timestamp(
            city, station_icao, prior_date
        )
        result["bucket"] = official
        result["mode"] = "official"
        result["confidence"] = 1.0
        result["signal_available_at_utc"] = _off_ts
        result["prior_signal_timestamp_source"] = _off_src
        return result

    # 2.5 FAST-PATH: Market consensus alone (98%+ on any bucket)
    # If the market has decided, we know the answer - no need for WU/stability
    if prior_day_market_prices:
        sorted_prices = sorted(
            prior_day_market_prices.items(), key=lambda kv: kv[1], reverse=True
        )
        if sorted_prices:
            dominant_bucket, dominant_price = sorted_prices[0]
            if dominant_price >= MARKET_CONSENSUS_INSTANT_THRESHOLD:
                result["bucket"] = str(dominant_bucket)
                result["mode"] = "market_consensus"
                result["confidence"] = round(dominant_price, 4)
                result["signal_available_at_utc"] = _dt.now(_tz.utc).isoformat()
                result["prior_signal_timestamp_source"] = "market_consensus_instant"
                return result

    # 3. Source-confirmed provisional (WU + market confirmation)
    station_cfg = STATIONS.get(station_icao, {})
    tz_name = station_cfg.get("timezone")
    unit = str(station_cfg.get("resolution_unit", "F"))

    if not tz_name:
        return result

    try:
        import zoneinfo
        local_tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        return result

    local_now = _dt.now(local_tz)
    local_hour = local_now.hour

    if local_hour < PROVISIONAL_EARLY_LOCAL_HOUR:
        return result

    # Fetch WU observations
    try:
        from data.wu_observations import fetch_wu_observed_max
    except ImportError:
        return result

    wu_data = fetch_wu_observed_max(station_icao, prior_date)
    if not wu_data:
        return result

    running_max = wu_data.get("running_max")
    latest_temp = wu_data.get("latest_temp")
    wu_unit = wu_data.get("unit", "F")

    if running_max is None or latest_temp is None:
        return result

    result["wu_running_max"] = running_max
    result["wu_latest_temp"] = latest_temp

    cooloff = PROVISIONAL_COOLOFF_F if wu_unit == "F" else PROVISIONAL_COOLOFF_C
    if latest_temp > running_max - cooloff:
        return result

    wu_max_bucket = str(_market_display_temp(running_max, wu_unit))

    # Track running-max stability
    state_key = (city, prior_date)
    now_mono = _dt.now(_tz.utc).timestamp()
    state = _PROVISIONAL_POLL_STATE.get(state_key)

    if state is None or state.get("bucket") != wu_max_bucket:
        _PROVISIONAL_POLL_STATE[state_key] = {
            "bucket": wu_max_bucket,
            "first_seen_at": now_mono,
            "last_seen_at": now_mono,
            "poll_count": 1,
        }
        state = _PROVISIONAL_POLL_STATE[state_key]
    else:
        state["last_seen_at"] = now_mono
        state["poll_count"] += 1

    stable_minutes = (state["last_seen_at"] - state["first_seen_at"]) / 60.0
    if stable_minutes < PROVISIONAL_MAX_STABLE_MINUTES:
        result["poll_count"] = state["poll_count"]
        return result

    # Market confirmation
    if prior_day_market_prices:
        sorted_prices = sorted(
            prior_day_market_prices.items(), key=lambda kv: kv[1], reverse=True
        )
        if len(sorted_prices) >= 2:
            dominant_bucket, dominant_price = sorted_prices[0]
            _, runnerup_price = sorted_prices[1]
        elif len(sorted_prices) == 1:
            dominant_bucket, dominant_price = sorted_prices[0]
            runnerup_price = 0.0
        else:
            return result

        market_confirms = (
            dominant_price >= PROVISIONAL_MARKET_CONFIDENCE
            and runnerup_price <= PROVISIONAL_RUNNERUP_MAX
            and str(dominant_bucket) == wu_max_bucket
        )
    else:
        market_confirms = False

    if local_hour >= PROVISIONAL_LATE_LOCAL_HOUR:
        # After 20:00: source stability + market dominant match is enough
        if prior_day_market_prices:
            sorted_prices = sorted(
                prior_day_market_prices.items(), key=lambda kv: kv[1], reverse=True
            )
            if sorted_prices and str(sorted_prices[0][0]) == wu_max_bucket:
                _prov_ts = _persist_provisional_timestamp(state_key, state)
                result["bucket"] = wu_max_bucket
                result["mode"] = "provisional_source_confirmed"
                result["confidence"] = round(
                    min(sorted_prices[0][1], 0.99) if sorted_prices else 0.95, 4
                )
                result["poll_count"] = state["poll_count"]
                result["signal_available_at_utc"] = _prov_ts
                result["prior_signal_timestamp_source"] = "provisional_first_seen"
                return result
        # Without market prices we cannot confirm the WU observation against
        # actual Polymarket data.  Return without a signal so the gate blocks
        # execution (prevents the Seoul-type bug where a provisional signal
        # was created for a date with no Polymarket market).
        result["poll_count"] = state["poll_count"]
        return result

    # Before 20:00: require full market confirmation + poll stability
    if not market_confirms:
        result["poll_count"] = state["poll_count"]
        return result

    if state["poll_count"] < PROVISIONAL_MIN_POLLS:
        result["poll_count"] = state["poll_count"]
        return result

    poll_span = state["last_seen_at"] - state["first_seen_at"]
    min_span = PROVISIONAL_POLL_SPACING_SEC * (PROVISIONAL_MIN_POLLS - 1)
    if poll_span < min_span:
        result["poll_count"] = state["poll_count"]
        return result

    _prov_ts = _persist_provisional_timestamp(state_key, state)
    result["bucket"] = wu_max_bucket
    result["mode"] = "provisional_source_confirmed"
    result["confidence"] = round(dominant_price, 4)
    result["poll_count"] = state["poll_count"]
    result["signal_available_at_utc"] = _prov_ts
    result["prior_signal_timestamp_source"] = "provisional_first_seen"
    return result


def _persist_provisional_timestamp(
    state_key: tuple[str, str],
    state: dict[str, Any],
) -> str:
    """Return the first time the provisional gate passed for this city/date.

    Persists in _PROVISIONAL_POLL_STATE so subsequent calls reuse the same
    timestamp instead of regenerating datetime.now() each time.
    """
    existing = state.get("first_provisional_available_at_utc")
    if existing:
        return existing
    ts = datetime.now(UTC).isoformat()
    state["first_provisional_available_at_utc"] = ts
    return ts


def _bucket_sort_key(bucket: str) -> float:
    lo, _ = parse_bucket_bounds(bucket)
    return lo


def _market_display_temp(raw_temp: float | int, unit: str) -> int:
    return _round_half_up(float(raw_temp))


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = (json.loads(path.read_text(encoding="utf-8"))
                   if path.exists() else {})
    except Exception as exc:
        _LOG.debug("Failed to load %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_accuracy_cache(city: str) -> list[dict[str, Any]]:
    """Load canonical accuracy rows from the dashboard cache."""
    try:
        if _ACCURACY_CACHE_PATH.exists():
            raw = json.loads(
                _ACCURACY_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw.get(city, [])
    except Exception as exc:
        _LOG.debug("Failed to load accuracy cache: %s", exc)
    return []


def _streak_day_quality(resolved_label: str) -> float:
    """Score how informative a resolved bucket label is.

    Exact single-degree → 1.0, narrow range → 0.7,
    coarse threshold (>=X, <=X, X+) → 0.3.
    """
    clean = resolved_label.strip()
    if (clean.startswith("≥") or clean.startswith(">=")
            or clean.startswith("≤") or clean.startswith("<=")
            or clean.endswith("+")):
        return 0.3
    if "-" in clean:
        return 0.7
    return 1.0
