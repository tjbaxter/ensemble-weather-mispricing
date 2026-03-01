"""Conviction signal scorer.

Implements Tom's manual betting framework as a quantitative score:

1. Consensus gate      — need N+ models agreeing on the same bucket
2. Hot-hand weights    — models correct recently get more vote; losing streaks discounted
3. VIP model alignment — Weather.com + AccuWeather + key ensemble members
4. Spread signal       — GREEN spread adds confidence

Returns a score 0-1. Only emit CONVICTION signal when score >= MIN_CONVICTION_SCORE.

Tunable parameters live at the top of this file — adjust after a week of data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_ACCURACY_CACHE  = ROOT / "data" / "accuracy_rows_cache.json"
_COMMERCIAL_LOG  = ROOT / "data" / "commercial_forecast_log.json"

# ── Tunable parameters ───────────────────────────────────────────────────────
HOT_HAND_WINDOW       = 7     # days of history to look back for streak detection

# Streak-to-weight lookup.
# The PRIMARY driver is the consecutive streak ending on the most recent day
# (i.e. was the model right on day-1, day-2, day-3 ... right up to our event?).
# Longer streaks get compounding weight; models on a cold streak get discounted.
_STREAK_WEIGHTS = {
     5: 2.50,   # on fire: right 5 days in a row right up to today
     4: 2.10,
     3: 1.75,
     2: 1.40,   # right the last 2 days — meaningful signal
     1: 1.15,   # right only yesterday — small bonus
     0: 1.00,   # no data
    -1: 0.70,
    -2: 0.42,
    -3: 0.25,
    -4: 0.18,
}

MIN_MODELS_FOR_SCORE  = 3     # need at least this many models with data
CONSENSUS_SOFT_MIN    = 4     # below this, consensus score starts dropping off
CONSENSUS_HARD_MIN    = 3     # below this, conviction score is capped at 0.25

MIN_CONVICTION_SCORE  = 0.38  # threshold to emit CONVICTION signal (tunable)

# VIP model keys: their agreement is weighted extra heavily in the score.
# These are models Tom specifically calls out as high-signal.
_VIP_MODEL_KEYS: set[str] = {
    "ukmo_seamless",
    "ukmo_uk_deterministic_2km",
    "meteofrance_arome_france",
    "meteofrance_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "dmi_seamless",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hround(x: float) -> int:
    """Round half-up (matches dashboard resolution logic)."""
    return math.floor(x + 0.5)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


# ── Hot-hand ─────────────────────────────────────────────────────────────────

def _streak_weight(streak: int) -> float:
    """Convert consecutive streak to vote multiplier using the lookup table.
    Streaks beyond the table's range clamp to the nearest boundary value."""
    if streak >= 5:
        return _STREAK_WEIGHTS[5]
    if streak <= -4:
        return _STREAK_WEIGHTS[-4]
    return _STREAK_WEIGHTS.get(streak, 1.0)


def compute_hot_hand(city: str, model_keys: list[str]) -> dict[str, dict]:
    """
    Returns {model_key: {"accuracy": float|None, "streak": int, "weight": float}}

    streak > 0  = model was correct on each of the last N consecutive days
                  ending with the most recent resolved day (the day before our event)
    streak < 0  = model was wrong on each of the last N consecutive days
    streak == 0 = no data

    Weight is primarily driven by streak length (see _STREAK_WEIGHTS).
    Accuracy over the full window is a 30% secondary modifier so a model
    that's been on a streak but was flaky before gets a slight haircut.
    """
    cache = _load_json(_ACCURACY_CACHE)
    rows  = cache.get(city, [])

    # Take last HOT_HAND_WINDOW resolved rows, most recent first
    resolved = sorted(
        [r for r in rows if isinstance(r, dict) and r.get("date")],
        key=lambda r: r["date"],
        reverse=True,
    )[:HOT_HAND_WINDOW]

    result: dict[str, dict] = {}
    for mk in model_keys:
        win_key  = f"{mk}_d1_win"
        outcomes = [r[win_key] for r in resolved if r.get(win_key) is not None]

        if not outcomes:
            result[mk] = {"accuracy": None, "streak": 0, "weight": 1.0}
            continue

        # Rolling accuracy over the full window (secondary signal)
        accuracy = sum(bool(o) for o in outcomes) / len(outcomes)

        # Streak: consecutive same result starting from the MOST RECENT day.
        # This is the "was the model right right up to our event?" question.
        streak = 0
        first  = bool(outcomes[0])
        for o in outcomes:
            if bool(o) == first:
                streak += 1 if first else -1
            else:
                break

        # Primary weight from streak table
        primary = _streak_weight(streak)

        # Secondary accuracy modifier: scale ±20% around primary
        # accuracy=1.0 → +20%, accuracy=0.5 → neutral, accuracy=0.0 → -20%
        acc_modifier = 1.0 + 0.20 * (accuracy - 0.5) / 0.5

        # Blend: 70% streak, 30% accuracy-modified
        weight = 0.70 * primary + 0.30 * (primary * acc_modifier)
        weight = max(0.10, min(3.0, weight))

        result[mk] = {
            "accuracy":   round(accuracy, 3),
            "streak":     streak,
            "weight":     round(weight, 3),
            "streak_w":   round(primary, 3),
        }

    return result


# ── Commercial forecasts ──────────────────────────────────────────────────────

def get_commercial_temps(city: str, target_date: str) -> tuple[float | None, float | None]:
    """
    Return (accu_temp, wu_temp) for the given city + target_date from the
    commercial_forecast_log.json.  Returns (None, None) if not available.
    """
    log = _load_json(_COMMERCIAL_LOG)
    city_log = log.get(city, {})
    entry = city_log.get(target_date, {})
    accu = entry.get("accu")
    wu   = entry.get("wu")
    return (float(accu) if accu is not None else None,
            float(wu)   if wu   is not None else None)


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_conviction(
    center_temp: float,
    model_values: dict[str, float],   # {model_key: predicted_temp}
    spread_colour: str,               # "GREEN" | "RED" | "UNKNOWN"
    wu_temp: float | None,            # Weather.com (from forecast bundle)
    accu_temp: float | None,          # AccuWeather (from commercial log)
    hot_hand: dict[str, dict],        # from compute_hot_hand()
) -> tuple[float, dict]:
    """
    Compute conviction score (0-1) and a breakdown dict for logging.

    Score components (weights add to 1.0):
      40% — hot-hand weighted consensus (which fraction of model *votes* agree,
             where recent winners cast heavier votes)
      25% — raw consensus ratio (plain count agreement)
      20% — VIP commercial alignment (Weather.com + AccuWeather)
      15% — spread signal (GREEN vs RED)

    All components are then gated by the absolute model-count consensus.
    Below CONSENSUS_HARD_MIN agreeing models the score is capped.
    """
    if not model_values or len(model_values) < MIN_MODELS_FOR_SCORE:
        return 0.0, {"reason": "insufficient_models"}

    center_bucket = _hround(center_temp)

    # ── 1. Raw consensus ─────────────────────────────────────────────────────
    total_models = len(model_values)
    agreeing_models = {mk: t for mk, t in model_values.items() if _hround(t) == center_bucket}
    n_agree  = len(agreeing_models)
    consensus_ratio = n_agree / total_models

    # Hard gate: if fewer than CONSENSUS_HARD_MIN models agree, cap score
    if n_agree < CONSENSUS_HARD_MIN:
        return (
            0.10,
            {
                "score": 0.10, "reason": "consensus_hard_gate",
                "n_agree": n_agree, "total_models": total_models,
                "center_bucket": center_bucket,
            },
        )

    # Soft gate: 0→1 ramp between HARD_MIN and SOFT_MIN
    if n_agree < CONSENSUS_SOFT_MIN:
        consensus_gate = (n_agree - CONSENSUS_HARD_MIN) / (CONSENSUS_SOFT_MIN - CONSENSUS_HARD_MIN)
    else:
        consensus_gate = 1.0

    # ── 2. Hot-hand weighted consensus ───────────────────────────────────────
    total_weight    = sum(hot_hand.get(mk, {}).get("weight", 1.0) for mk in model_values)
    agreeing_weight = sum(
        hot_hand.get(mk, {}).get("weight", 1.0) for mk in agreeing_models
    )
    hh_consensus = agreeing_weight / total_weight if total_weight else consensus_ratio

    # Extra boost if VIP models specifically agree
    vip_agreeing  = sum(1 for mk in agreeing_models if mk in _VIP_MODEL_KEYS)
    vip_total     = sum(1 for mk in model_values if mk in _VIP_MODEL_KEYS)
    vip_model_ratio = vip_agreeing / vip_total if vip_total else 0.5

    # Blend hot-hand consensus with VIP model bonus
    hh_component = 0.7 * hh_consensus + 0.3 * vip_model_ratio

    # ── 3. Commercial VIP alignment ──────────────────────────────────────────
    commercial_signals = []
    if wu_temp is not None:
        commercial_signals.append(_hround(wu_temp) == center_bucket)
    if accu_temp is not None:
        commercial_signals.append(_hround(accu_temp) == center_bucket)
    vip_commercial_ratio = (
        sum(commercial_signals) / len(commercial_signals)
        if commercial_signals else 0.5   # neutral if neither is available
    )

    # ── 4. Spread signal ─────────────────────────────────────────────────────
    spread_component = {"GREEN": 1.0, "RED": 0.25, "UNKNOWN": 0.55}.get(spread_colour, 0.55)

    # ── Combined raw score ───────────────────────────────────────────────────
    raw_score = (
        0.40 * hh_component
        + 0.25 * consensus_ratio
        + 0.20 * vip_commercial_ratio
        + 0.15 * spread_component
    )

    # Apply consensus gate
    score = max(0.0, min(1.0, raw_score * consensus_gate))

    # Hot-hand details for top agreeing models (for logging)
    top_hot = sorted(
        [(mk, hot_hand.get(mk, {}).get("streak", 0), hot_hand.get(mk, {}).get("accuracy"))
         for mk in agreeing_models],
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]

    breakdown = {
        "score":                round(score, 3),
        "center_bucket":        center_bucket,
        "n_agree":              n_agree,
        "total_models":         total_models,
        "consensus_ratio":      round(consensus_ratio, 3),
        "hh_consensus":         round(hh_consensus, 3),
        "vip_model_ratio":      round(vip_model_ratio, 3),
        "vip_commercial_ratio": round(vip_commercial_ratio, 3),
        "spread_colour":        spread_colour,
        "spread_component":     round(spread_component, 3),
        "consensus_gate":       round(consensus_gate, 3),
        "wu_agrees":            (_hround(wu_temp) == center_bucket) if wu_temp is not None else None,
        "accu_agrees":          (_hround(accu_temp) == center_bucket) if accu_temp is not None else None,
        "top_hot_models":       [(mk, streak, acc) for mk, streak, acc in top_hot],
    }

    return score, breakdown
