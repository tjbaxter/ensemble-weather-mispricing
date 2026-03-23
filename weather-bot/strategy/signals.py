"""Signal generation for weather bucket markets."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import UTC, date as _date, datetime

from config.cities import STATIONS
from config.settings import (
    ALPHA_THRESHOLD,
    D2_MAX_YES_ENTRY_PRICE,
    D2_P_WIN_DISCOUNT,
    D3_MAX_YES_ENTRY_PRICE,
    D3_P_WIN_DISCOUNT,
    ENABLE_LADDER_STRATEGY,
    ENSEMBLE_DISABLE_CLASSIC_CONFIDENCE_GATE,
    ENSEMBLE_STD_SKIP_THRESHOLD,
    ENABLE_TOP2_SHADOWS,
    FIXED_ORDER_USD,
    HIGH_DELTA_SIZE_MULTIPLIER,
    KELLY_MAX_BET_USD,
    KELLY_MIN_BET_USD,
    DYNAMIC_RISK_SIZING_ENABLED,
    EQUITY_MAX_POSITION_PCT,
    TOP2_SHADOW_MIN_PROB,
    TOP2_SHADOW_SPLIT_THRESHOLD,
    HIGH_DELTA_THRESHOLD_DEG,
    HOURS_BEFORE_RESOLUTION_CUTOFF,
    KELLY_FRACTION,
    LADDER_MAX_TOTAL_COST,
    LADDER_MIN_EDGE,
    LADDER_WIDTH,
    HARD_MAX_YES_ENTRY_PRICE,
    HARD_MIN_YES_ENTRY_PRICE,
    METAR_DANGER_POST_MINUTE,
    METAR_DANGER_PRE_MINUTE,
    MIN_FORECAST_CONFIDENCE,
    MIN_ORDER_USD,
    OVERROUND_REJECT_YES_THRESHOLD,
    SOFT_MAX_NO_PRICE,
    SOFT_MAX_YES_PRICE,
    SOFT_MIN_NO_PRICE,
    SOFT_MIN_YES_PRICE,
    SOFT_PRICE_EDGE_PENALTY,
    SOFT_PRICE_GUARDRAILS_ENABLED,
    PRACTICAL_MIN_ORDER_USD,
)
from strategy.edge_calculator import calculate_edge
from strategy.kelly import kelly_size
from strategy.ladder import create_ladder
from strategy.conviction import (
    MIN_CONVICTION_SCORE,
    compute_hot_hand,
    get_commercial_temps,
    score_conviction,
)
from monitoring.deep_observability import get_deep_observability


@dataclass
class Signal:
    market_id: str
    token_id: str
    side: str
    edge: float
    forecast_prob: float
    market_prob: float
    size_usd: float
    city: str
    station_icao: str
    date: str
    bucket: str
    rounding_confidence: str
    predicted_display_temp: float | None
    # Observability fields — populated from forecast bundle
    spread_colour: str = "UNKNOWN"   # GREEN or RED
    det_spread: float = 0.0          # raw model spread in °C/°F
    model_values_json: str = "{}"    # JSON: {model: temp, ...}
    ev_per_bet: float = 0.0          # expected value in USD
    kelly_fraction_used: float = 0.0 # Kelly fraction applied
    # Timing fields — for early-entry calibration analysis
    days_ahead: int = 1              # 1=D+1, 2=D+2, 3=D+3
    hours_to_resolution: float = 0.0 # fractional hours until market closes
    temporal_discount: float = 1.0   # discount applied to forecast_prob (< 1 for D+2/D+3)
    # Strategy tag — used for A/B comparison
    # LADDER    = one of several adjacent buckets placed by the ladder strategy
    # CONVICTION = shadow signal: single highest-probability bucket, full Kelly, never executed
    # SINGLE    = non-ladder directional pick
    strategy: str = "SINGLE"
    decision_id: str = ""
    market_scan_id: str = ""
    snapshot_id: str = ""
    prob_calc_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _kelly_position_cap(bankroll: float) -> float:
    """Per-bet cap used inside Kelly sizing before execution-time overlays."""
    if DYNAMIC_RISK_SIZING_ENABLED:
        return max(KELLY_MIN_BET_USD, float(bankroll) * EQUITY_MAX_POSITION_PCT)
    return KELLY_MAX_BET_USD


_DEEP_OBS = get_deep_observability()
_OBS_MARKET_SCAN_ID = ""
_OBS_MODE = "paper"


def set_signal_observability_context(
    market_scan_id: str | None,
    mode: str | None = None,
) -> None:
    global _OBS_MARKET_SCAN_ID, _OBS_MODE
    _OBS_MARKET_SCAN_ID = market_scan_id or ""
    if mode:
        _OBS_MODE = mode


def _signal_observability_fields(forecast_bundle: dict | None) -> dict[str, str]:
    bundle = forecast_bundle or {}
    return {
        "market_scan_id": _OBS_MARKET_SCAN_ID,
        "snapshot_id": str(bundle.get("__snapshot_id", "") or ""),
        "prob_calc_id": str(bundle.get("__prob_calc_id", "") or ""),
    }


def _log_signal_eval(
    *,
    strategy: str,
    city: str,
    station_icao: str,
    target_date: str,
    bucket: str,
    side: str,
    forecast_prob: float,
    market_prob: float,
    edge: float,
    size_usd: float,
    decision: str,
    rejection_reason: str | None,
    gate_results: dict[str, dict],
    forecast_bundle: dict,
    strategy_context: dict,
) -> str:
    return _DEEP_OBS.log_signal_eval(
        {
            "decision_id": "",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "strategy": strategy,
            "city": city,
            "station_icao": station_icao,
            "target_date": target_date,
            "bucket": bucket,
            "side": side,
            "forecast_prob": forecast_prob,
            "market_prob": market_prob,
            "edge": edge,
            "size_usd": size_usd,
            "decision": decision,
            "rejection_reason": rejection_reason,
            "snapshot_id": forecast_bundle.get("__snapshot_id", ""),
            "prob_calc_id": forecast_bundle.get("__prob_calc_id", ""),
            "market_scan_id": _OBS_MARKET_SCAN_ID,
            "gate_results": gate_results,
            "strategy_context": strategy_context,
        },
        mode=_OBS_MODE,
    )


def calculate_hours_to_resolution(end_date_iso: str) -> float:
    end_dt = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
    now = datetime.now(UTC)
    delta = end_dt - now
    return max(delta.total_seconds() / 3600.0, 0.0)


def generate_signals(
    markets: list[dict],
    forecasts: dict[str, dict[str, dict]],
    bankroll: float,
) -> list[Signal]:
    """Generate candidate trades sorted by descending edge."""
    signals: list[Signal] = []

    if _in_metar_danger_window(datetime.now(UTC)):
        return []

    for market in markets:
        station_icao = market["station_icao"]
        city = market["city"]
        date = market["date"]
        end_date_iso = market["end_date_iso"]

        if calculate_hours_to_resolution(end_date_iso) < HOURS_BEFORE_RESOLUTION_CUTOFF:
            continue

        forecast_bundle = forecasts.get(station_icao, {}).get(date)
        if not forecast_bundle:
            continue
        forecast = forecast_bundle.get("probs", {})
        rounding_confidence = forecast_bundle.get("rounding_confidence", "LOW")
        predicted_display_temp = forecast_bundle.get("predicted_display_temp")
        ensemble_std = float(forecast_bundle.get("ensemble_std", 0.0) or 0.0)
        ensemble_skip = bool(forecast_bundle.get("ensemble_skip", False) or ensemble_std > ENSEMBLE_STD_SKIP_THRESHOLD)
        min_confidence = 0.0 if (ENSEMBLE_DISABLE_CLASSIC_CONFIDENCE_GATE and "ensemble_std" in forecast_bundle) else None

        # Observability: extract spread signal and per-model predictions
        det_spread = float(forecast_bundle.get("det_spread", ensemble_std * 2.0) or 0.0)
        det_spread_colour = str(forecast_bundle.get("det_spread_colour", "UNKNOWN"))
        det_model_values: dict = forecast_bundle.get("det_model_values") or {}
        city_kelly = float(STATIONS.get(station_icao, {}).get("kelly_fraction", KELLY_FRACTION))

        # Temporal discount: D+2 and D+3 markets get a haircut on forecast_prob.
        # This shrinks the apparent edge so only strongly mispriced D+2/D+3 buckets
        # pass the alpha threshold — early entries must be genuinely cheap.
        hours_to_resolution = calculate_hours_to_resolution(end_date_iso)
        try:
            days_ahead = (_date.fromisoformat(date) - _date.today()).days
        except ValueError:
            days_ahead = 1
        if days_ahead >= 3:
            temporal_discount = D3_P_WIN_DISCOUNT
        elif days_ahead >= 2:
            temporal_discount = D2_P_WIN_DISCOUNT
        else:
            temporal_discount = 1.0

        if ensemble_skip:
            continue

        # --- D+2 / D+3 price cap ---
        # At 2+ days out the forecast uncertainty is high enough that we only
        # enter when the market is dramatically underpricing a bucket (very cheap).
        # If the crowd already has a bucket at 20¢+ it has priced in the uncertainty;
        # there's no exploitable edge worth the extra forecast risk.
        if days_ahead >= 3:
            _d_max = D3_MAX_YES_ENTRY_PRICE
        elif days_ahead >= 2:
            _d_max = D2_MAX_YES_ENTRY_PRICE
        else:
            _d_max = None  # D+1 — use normal HARD_MAX_YES_ENTRY_PRICE

        # --- Overround filter ---
        # If the crowd has collectively bid all YES bucket prices above 115%,
        # the market is structurally overpriced for YES. Block BUY_YES entirely.
        bucket_yes_sum = sum(info["price"] for info in market["buckets"].values())
        market_overround = bucket_yes_sum > OVERROUND_REJECT_YES_THRESHOLD

        # --- High-Delta regime detection ---
        # Compare AI model vs crowd's weather app (WU = IBM GRAF).
        # WU crowd temp is the exact forecast retail traders see on wunderground.com.
        # Falls back to GFS+ECMWF blend if WU fetch failed.
        ai_temp = forecast_bundle.get("primary_model_temp")
        wu_crowd_temp = forecast_bundle.get("wu_crowd_temp")
        baseline_temp = wu_crowd_temp if wu_crowd_temp is not None else forecast_bundle.get("baseline_model_temp")
        high_delta = (
            ai_temp is not None
            and baseline_temp is not None
            and abs(ai_temp - baseline_temp) >= HIGH_DELTA_THRESHOLD_DEG
        )

        if ENABLE_LADDER_STRATEGY and forecast:
            center_bucket = max(forecast.items(), key=lambda kv: kv[1])[0]
            market_prices = {bucket: info["price"] for bucket, info in market["buckets"].items()}
            ladder = create_ladder(
                ensemble_probs=forecast,
                market_prices=market_prices,
                center_bucket=center_bucket,
                width=LADDER_WIDTH,
                max_total_cost=LADDER_MAX_TOTAL_COST,
                min_edge=LADDER_MIN_EDGE,
            )
            if ladder:
                ladder_size = _compute_size(
                    bankroll=bankroll,
                    market_prob=min(0.99, max(0.01, sum(item["price"] for item in ladder))),
                    win_prob=min(1.0, max(0.0, sum(item["model_prob"] for item in ladder))),
                    edge=max(0.0, ladder[0]["ladder_edge"]),
                    rounding_confidence=rounding_confidence,
                    station_icao=station_icao,
                )
                if ladder_size >= max(MIN_ORDER_USD, PRACTICAL_MIN_ORDER_USD):
                    each_size = round(ladder_size / len(ladder), 2)
                    total_cost = 0.0
                    for item in ladder:
                        if item["price"] < HARD_MIN_YES_ENTRY_PRICE or item["price"] > HARD_MAX_YES_ENTRY_PRICE:
                            continue
                        bucket = item["bucket"]
                        token_info = market["buckets"][bucket]
                        signals.append(
                            Signal(
                                market_id=market["condition_id"],
                                token_id=token_info["yes_token_id"],
                                side="BUY_YES",
                                edge=item["ladder_edge"],
                                forecast_prob=item["model_prob"],
                                market_prob=item["price"],
                                size_usd=each_size,
                                city=city,
                                station_icao=station_icao,
                                date=date,
                                bucket=bucket,
                                rounding_confidence=rounding_confidence,
                                predicted_display_temp=predicted_display_temp,
                                spread_colour=det_spread_colour,
                                det_spread=round(det_spread, 3),
                                model_values_json=json.dumps(
                                    {k: round(v, 2) for k, v in det_model_values.items()},
                                    separators=(",", ":"),
                                ),
                                kelly_fraction_used=city_kelly,
                                days_ahead=days_ahead,
                                hours_to_resolution=round(hours_to_resolution, 1),
                                temporal_discount=temporal_discount,
                                strategy="LADDER",
                                **_signal_observability_fields(forecast_bundle),
                            )
                        )
                        total_cost += each_size

                    # Shadow CONVICTION signal: single best bucket, full ladder budget.
                    # Scored by Tom's conviction framework before being logged.
                    # Never executed — A/B comparison only.
                    if center_bucket in market["buckets"]:
                        center_info = market["buckets"][center_bucket]
                        center_item = next((i for i in ladder if i["bucket"] == center_bucket), ladder[0])
                        conviction_size = round(total_cost or ladder_size, 2)

                        # ── Conviction scoring ────────────────────────────────
                        hot_hand   = compute_hot_hand(city, list(det_model_values.keys()))
                        accu_temp, comm_wu_temp = get_commercial_temps(city, date)
                        # Prefer forecast-bundle Weather.com (live); fall back to logged
                        wu_for_score = wu_crowd_temp if wu_crowd_temp is not None else comm_wu_temp

                        conv_score, conv_breakdown = score_conviction(
                            center_temp=float(predicted_display_temp or center_item["model_prob"]),
                            model_values=det_model_values,
                            spread_colour=det_spread_colour,
                            wu_temp=wu_for_score,
                            accu_temp=accu_temp,
                            hot_hand=hot_hand,
                        )

                        # Skip CONVICTION signal if score is too low — not a high-conviction bet
                        if conv_score < MIN_CONVICTION_SCORE:
                            continue

                        ev_conviction = (
                            center_item["model_prob"]
                            * (conviction_size / max(center_item["price"], 0.001))
                            * (1.0 - center_item["price"])
                            - (1.0 - center_item["model_prob"]) * conviction_size
                        )

                        # Embed score in model_values_json for dashboard / resolver visibility
                        conv_meta = {k: round(v, 2) for k, v in det_model_values.items()}
                        conv_meta["__conviction_score"] = round(conv_score, 3)
                        conv_meta["__n_agree"] = conv_breakdown.get("n_agree", 0)
                        conv_meta["__wu_agrees"] = int(bool(conv_breakdown.get("wu_agrees")))
                        conv_meta["__accu_agrees"] = int(bool(conv_breakdown.get("accu_agrees")))
                        conv_meta["__spread"] = det_spread_colour

                        signals.append(
                            Signal(
                                market_id=market["condition_id"],
                                token_id=center_info["yes_token_id"],
                                side="BUY_YES",
                                edge=center_item["ladder_edge"],
                                forecast_prob=center_item["model_prob"],
                                market_prob=center_item["price"],
                                size_usd=conviction_size,
                                city=city,
                                station_icao=station_icao,
                                date=date,
                                bucket=center_bucket,
                                rounding_confidence=rounding_confidence,
                                predicted_display_temp=predicted_display_temp,
                                spread_colour=det_spread_colour,
                                det_spread=round(det_spread, 3),
                                model_values_json=json.dumps(
                                    conv_meta, separators=(",", ":"),
                                ),
                                ev_per_bet=round(ev_conviction, 3),
                                kelly_fraction_used=city_kelly,
                                days_ahead=days_ahead,
                                hours_to_resolution=round(hours_to_resolution, 1),
                                temporal_discount=temporal_discount,
                                strategy="CONVICTION",
                                **_signal_observability_fields(forecast_bundle),
                            )
                        )
                    continue

        for bucket, token_info in market["buckets"].items():
            raw_forecast_prob = forecast.get(bucket, 0.0)
            # Apply temporal discount: D+2/D+3 signals must clear a higher hurdle
            forecast_prob = raw_forecast_prob * temporal_discount
            market_prob = token_info["price"]
            action, edge, win_prob = calculate_edge(
                forecast_prob,
                market_prob,
                min_forecast_confidence=min_confidence,
            )

            if action == "NO_TRADE":
                _log_signal_eval(
                    strategy="SINGLE",
                    city=city,
                    station_icao=station_icao,
                    target_date=date,
                    bucket=bucket,
                    side="BUY_YES",
                    forecast_prob=float(forecast_prob),
                    market_prob=float(market_prob),
                    edge=float(edge),
                    size_usd=0.0,
                    decision="REJECT",
                    rejection_reason="no_trade",
                    gate_results={"edge_or_confidence": {"passed": False}},
                    forecast_bundle=forecast_bundle,
                    strategy_context={"selection_path": "single"},
                )
                continue

            # Overround guard: structurally broken market, reject BUY_YES
            if action == "BUY_YES" and market_overround:
                _log_signal_eval(
                    strategy="SINGLE",
                    city=city,
                    station_icao=station_icao,
                    target_date=date,
                    bucket=bucket,
                    side=action,
                    forecast_prob=float(forecast_prob),
                    market_prob=float(market_prob),
                    edge=float(edge),
                    size_usd=0.0,
                    decision="REJECT",
                    rejection_reason="market_overround",
                    gate_results={"overround_guard": {"passed": False, "sum_yes_prices": bucket_yes_sum}},
                    forecast_bundle=forecast_bundle,
                    strategy_context={"selection_path": "single"},
                )
                continue

            if action == "BUY_YES":
                if market_prob < HARD_MIN_YES_ENTRY_PRICE:
                    _log_signal_eval(
                        strategy="SINGLE",
                        city=city,
                        station_icao=station_icao,
                        target_date=date,
                        bucket=bucket,
                        side=action,
                        forecast_prob=float(forecast_prob),
                        market_prob=float(market_prob),
                        edge=float(edge),
                        size_usd=0.0,
                        decision="REJECT",
                        rejection_reason="price_below_floor",
                        gate_results={"price_floor": {"passed": False, "min": HARD_MIN_YES_ENTRY_PRICE}},
                        forecast_bundle=forecast_bundle,
                        strategy_context={"selection_path": "single"},
                    )
                    continue
                # D+2/D+3: only enter if price is absurdly cheap
                if _d_max is not None and market_prob > _d_max:
                    _log_signal_eval(
                        strategy="SINGLE",
                        city=city,
                        station_icao=station_icao,
                        target_date=date,
                        bucket=bucket,
                        side=action,
                        forecast_prob=float(forecast_prob),
                        market_prob=float(market_prob),
                        edge=float(edge),
                        size_usd=0.0,
                        decision="REJECT",
                        rejection_reason="dplus_price_ceiling",
                        gate_results={"dplus_price_ceiling": {"passed": False, "max": _d_max}},
                        forecast_bundle=forecast_bundle,
                        strategy_context={"selection_path": "single", "days_ahead": days_ahead},
                    )
                    continue
                if market_prob > HARD_MAX_YES_ENTRY_PRICE:
                    _log_signal_eval(
                        strategy="SINGLE",
                        city=city,
                        station_icao=station_icao,
                        target_date=date,
                        bucket=bucket,
                        side=action,
                        forecast_prob=float(forecast_prob),
                        market_prob=float(market_prob),
                        edge=float(edge),
                        size_usd=0.0,
                        decision="REJECT",
                        rejection_reason="hard_price_ceiling",
                        gate_results={"price_ceiling": {"passed": False, "max": HARD_MAX_YES_ENTRY_PRICE}},
                        forecast_bundle=forecast_bundle,
                        strategy_context={"selection_path": "single"},
                    )
                    continue

            # NO bet filter: only take NO bets where the market is genuinely
            # mispricing the YES side.  Buying NO at 0.70+ (YES < 0.30) is
            # penny-collecting: risking $5 to make cents.  We only want NO bets
            # where the YES price is > 0.35 AND our model gives YES < 20%
            # probability — i.e. the market is overconfident in something we
            # think is a clear miss.
            if action == "BUY_NO":
                yes_price = market_prob           # market_prob is always the YES price
                model_yes_prob = forecast_prob    # our model's probability of YES
                if yes_price <= 0.35:
                    continue   # market is already skeptical — no edge buying NO
                if model_yes_prob >= 0.20:
                    continue   # model says meaningful YES chance — skip NO side

            effective_edge, _guardrail_penalized = _effective_edge_with_soft_guardrails(action, market_prob, edge)
            if effective_edge <= ALPHA_THRESHOLD:
                _log_signal_eval(
                    strategy="SINGLE",
                    city=city,
                    station_icao=station_icao,
                    target_date=date,
                    bucket=bucket,
                    side=action,
                    forecast_prob=float(forecast_prob),
                    market_prob=float(market_prob),
                    edge=float(effective_edge),
                    size_usd=0.0,
                    decision="REJECT",
                    rejection_reason="edge_below_alpha",
                    gate_results={"alpha_threshold": {"passed": False, "threshold": ALPHA_THRESHOLD}},
                    forecast_bundle=forecast_bundle,
                    strategy_context={"selection_path": "single", "soft_guardrail_penalized": bool(_guardrail_penalized)},
                )
                continue

            size = _compute_size(
                bankroll=bankroll,
                market_prob=market_prob if action == "BUY_YES" else (1.0 - market_prob),
                win_prob=win_prob,
                edge=effective_edge,
                rounding_confidence=rounding_confidence,
                high_delta=high_delta,
                station_icao=station_icao,
            )
            if size < max(MIN_ORDER_USD, PRACTICAL_MIN_ORDER_USD):
                _log_signal_eval(
                    strategy="SINGLE",
                    city=city,
                    station_icao=station_icao,
                    target_date=date,
                    bucket=bucket,
                    side=action,
                    forecast_prob=float(forecast_prob),
                    market_prob=float(market_prob),
                    edge=float(effective_edge),
                    size_usd=float(size),
                    decision="REJECT",
                    rejection_reason="size_below_min",
                    gate_results={"size_min": {"passed": False, "min": max(MIN_ORDER_USD, PRACTICAL_MIN_ORDER_USD)}},
                    forecast_bundle=forecast_bundle,
                    strategy_context={"selection_path": "single"},
                )
                continue

            token_id = token_info["yes_token_id"] if action == "BUY_YES" else token_info["no_token_id"]
            entry_price = market_prob if action == "BUY_YES" else (1.0 - market_prob)
            # EV = p_win * profit_if_correct - (1-p_win) * cost
            # Shares = size / entry_price; profit if win = shares * (1 - entry_price)
            ev = win_prob * (size / entry_price) * (1.0 - entry_price) - (1.0 - win_prob) * size
            sig_obj = Signal(
                    market_id=market["condition_id"],
                    token_id=token_id,
                    side=action,
                    edge=effective_edge,
                    forecast_prob=forecast_prob,
                    market_prob=market_prob,
                    size_usd=size,
                    city=city,
                    station_icao=station_icao,
                    date=date,
                    bucket=bucket,
                    rounding_confidence=rounding_confidence,
                    predicted_display_temp=predicted_display_temp,
                    spread_colour=det_spread_colour,
                    det_spread=round(det_spread, 3),
                    model_values_json=json.dumps(
                        {k: round(v, 2) for k, v in det_model_values.items()},
                        separators=(",", ":"),
                    ),
                    ev_per_bet=round(ev, 3),
                    kelly_fraction_used=city_kelly,
                    days_ahead=days_ahead,
                    hours_to_resolution=round(hours_to_resolution, 1),
                    temporal_discount=temporal_discount,
                    strategy="SINGLE",
                    **_signal_observability_fields(forecast_bundle),
                )
            sig_obj.decision_id = _log_signal_eval(
                strategy="SINGLE",
                city=city,
                station_icao=station_icao,
                target_date=date,
                bucket=bucket,
                side=action,
                forecast_prob=float(forecast_prob),
                market_prob=float(market_prob),
                edge=float(effective_edge),
                size_usd=float(size),
                decision="TRADE",
                rejection_reason=None,
                gate_results={
                    "alpha_threshold": {"passed": True, "threshold": ALPHA_THRESHOLD},
                    "price_guard": {"passed": True},
                },
                forecast_bundle=forecast_bundle,
                strategy_context={"selection_path": "single", "high_delta": bool(high_delta)},
            )
            signals.append(sig_obj)

    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals


def _in_metar_danger_window(now_utc: datetime) -> bool:
    minute = now_utc.minute
    return METAR_DANGER_PRE_MINUTE <= minute < METAR_DANGER_POST_MINUTE


def _compute_size(
    bankroll: float,
    market_prob: float,
    win_prob: float,
    edge: float,
    rounding_confidence: str,
    high_delta: bool = False,
    station_icao: str = "",
) -> float:
    """Return Kelly-sized bet in USD, with floor and ceiling applied.

    Kelly runs from day one regardless of bankroll size — the math is
    identical whether we have $300 or $30 000.  Flat-fee fallbacks mask
    calibration by spending the same on weak and strong signals alike.

    The three tiers in kelly_size() (HIGH / MEDIUM / LOW) scale the
    effective fraction, so the confidence level already does the heavy
    lifting of de-risking uncertain signals.
    """
    city_kelly = float(STATIONS.get(station_icao, {}).get("kelly_fraction", KELLY_FRACTION))

    size = kelly_size(
        market_price=market_prob,
        win_prob=win_prob,
        bankroll=bankroll,
        edge=edge,
        kelly_fraction=city_kelly,
        max_position=_kelly_position_cap(bankroll),
        rounding_confidence=rounding_confidence,
    )
    if high_delta:
        size = min(size * HIGH_DELTA_SIZE_MULTIPLIER, _kelly_position_cap(bankroll))

    # Minimum viable trade — Kelly output below this floor still fires but at
    # minimum size.  Signals too small even at minimum are filtered upstream.
    return max(size, KELLY_MIN_BET_USD) if size > 0 else 0.0


def summarize_top_missed_edges(
    markets: list[dict],
    forecasts: dict[str, dict[str, dict]],
    bankroll: float,
    limit: int = 3,
) -> str:
    """Return a compact summary of best skipped opportunities."""
    if _in_metar_danger_window(datetime.now(UTC)):
        return "metar_danger_window"

    misses: list[dict] = []
    reason_counts: dict[str, int] = {}

    for market in markets:
        station_icao = market["station_icao"]
        date = market["date"]
        end_date_iso = market["end_date_iso"]

        if calculate_hours_to_resolution(end_date_iso) < HOURS_BEFORE_RESOLUTION_CUTOFF:
            reason_counts["resolution_cutoff"] = reason_counts.get("resolution_cutoff", 0) + 1
            continue

        forecast_bundle = forecasts.get(station_icao, {}).get(date)
        if not forecast_bundle:
            reason_counts["missing_forecast"] = reason_counts.get("missing_forecast", 0) + 1
            continue
        forecast = forecast_bundle.get("probs", {})
        rounding_confidence = forecast_bundle.get("rounding_confidence", "LOW")
        ensemble_std = float(forecast_bundle.get("ensemble_std", 0.0) or 0.0)
        ensemble_skip = bool(forecast_bundle.get("ensemble_skip", False) or ensemble_std > ENSEMBLE_STD_SKIP_THRESHOLD)
        min_confidence = 0.0 if (ENSEMBLE_DISABLE_CLASSIC_CONFIDENCE_GATE and "ensemble_std" in forecast_bundle) else None

        if ensemble_skip:
            reason_counts["ensemble_std_too_high"] = reason_counts.get("ensemble_std_too_high", 0) + 1
            continue

        for bucket, token_info in market["buckets"].items():
            forecast_prob = forecast.get(bucket, 0.0)
            market_prob = token_info["price"]
            raw_edge = abs(forecast_prob - market_prob)

            action, edge, win_prob = calculate_edge(
                forecast_prob,
                market_prob,
                min_forecast_confidence=min_confidence,
            )
            if action == "NO_TRADE":
                if raw_edge <= ALPHA_THRESHOLD:
                    reason = "edge_below_threshold"
                else:
                    reason = "confidence_below_threshold"
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                misses.append(
                    {
                        "edge": raw_edge,
                        "market": market,
                        "bucket": bucket,
                        "reason": reason,
                        "forecast_prob": forecast_prob,
                        "market_prob": market_prob,
                    }
                )
                continue

            if action == "BUY_YES" and market_prob < HARD_MIN_YES_ENTRY_PRICE:
                reason_counts["price_floor"] = reason_counts.get("price_floor", 0) + 1
                misses.append(
                    {
                        "edge": raw_edge,
                        "market": market,
                        "bucket": bucket,
                        "reason": "price_floor",
                        "forecast_prob": forecast_prob,
                        "market_prob": market_prob,
                    }
                )
                continue

            if action == "BUY_YES" and market_prob > HARD_MAX_YES_ENTRY_PRICE:
                reason_counts["price_ceiling"] = reason_counts.get("price_ceiling", 0) + 1
                misses.append(
                    {
                        "edge": raw_edge,
                        "market": market,
                        "bucket": bucket,
                        "reason": "price_ceiling",
                        "forecast_prob": forecast_prob,
                        "market_prob": market_prob,
                    }
                )
                continue

            # NO bet filter: skip penny-collecting NO bets on clear extremes.
            # Only take NO when market is overconfident (YES > 0.35) AND model
            # thinks YES probability is genuinely low (< 20%).
            if action == "BUY_NO":
                if market_prob <= 0.35 or forecast_prob >= 0.20:
                    reason_counts["no_bet_filtered"] = reason_counts.get("no_bet_filtered", 0) + 1
                    continue

            effective_edge, guardrail_penalized = _effective_edge_with_soft_guardrails(action, market_prob, edge)
            if effective_edge <= ALPHA_THRESHOLD:
                reason = "soft_price_guardrail" if guardrail_penalized else "edge_below_threshold"
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                misses.append(
                    {
                        "edge": raw_edge,
                        "market": market,
                        "bucket": bucket,
                        "reason": reason,
                        "forecast_prob": forecast_prob,
                        "market_prob": market_prob,
                    }
                )
                continue

            trade_price = market_prob if action == "BUY_YES" else (1.0 - market_prob)
            size = kelly_size(
                market_price=trade_price,
                win_prob=win_prob,
                bankroll=bankroll,
                edge=effective_edge,
                kelly_fraction=KELLY_FRACTION,
                max_position=_kelly_position_cap(bankroll),
                rounding_confidence=rounding_confidence,
            )
            size = max(size, KELLY_MIN_BET_USD) if size > 0 else 0.0

            if size < max(MIN_ORDER_USD, PRACTICAL_MIN_ORDER_USD):
                reason_counts["size_below_min_order"] = reason_counts.get("size_below_min_order", 0) + 1
                misses.append(
                    {
                        "edge": raw_edge,
                        "market": market,
                        "bucket": bucket,
                        "reason": "size_below_min_order",
                        "forecast_prob": forecast_prob,
                        "market_prob": market_prob,
                    }
                )

    if not misses and not reason_counts:
        return "none"

    misses.sort(key=lambda m: m["edge"], reverse=True)
    top = misses[: max(0, limit)]
    top_bits = [
        (
            f"{item['market']['station_icao']}:{item['bucket']}:"
            f"edge={item['edge']:.3f}:reason={item['reason']}:"
            f"fp={item['forecast_prob']:.3f}:mp={item['market_prob']:.3f}"
        )
        for item in top
    ]
    counts_part = ",".join(f"{k}={v}" for k, v in sorted(reason_counts.items()))
    top_part = ";".join(top_bits) if top_bits else "none"
    return f"reasons[{counts_part}] top[{top_part}] conf_min={MIN_FORECAST_CONFIDENCE:.2f} edge_min={ALPHA_THRESHOLD:.2f}"


def generate_top2_shadow_signals(
    markets: list[dict],
    forecasts: dict[str, dict[str, dict]],
    bankroll: float,
) -> list["Signal"]:
    """Generate TOP2_EQUAL / TOP2_COND / TOP2_PROP shadow signals.

    These are *never executed* — logged to signals.csv with action='conviction_signal'
    and resolved daily so we can compare three dual-bucket sizing strategies against
    the live single-bucket approach.

    Variant definitions
    -------------------
    TOP2_EQUAL  (2A)  Always buy the top-2 YES buckets by model probability.
                      Both legs get MEDIUM Kelly sizing (equal capital on each).

    TOP2_COND   (2B)  Buy top-2 only when the model is genuinely split
                      (second_prob ≥ TOP2_SHADOW_SPLIT_THRESHOLD × first_prob).
                      If there's a clear favourite, acts like a single-bucket pick.
                      Both legs MEDIUM Kelly when it does go dual.

    TOP2_PROP   (2C)  Always top-2 but with proportional sizing:
                      primary → MEDIUM Kelly, secondary → LOW Kelly (~half size).
                      Tests whether saving capital on the weaker leg beats equal sizing.

    Rationale: empirically the model is almost always within one bucket when wrong.
    Buying both the favourite and runner-up should raise the virtual win rate
    significantly.  Running all three variants simultaneously finds the best
    risk/return tradeoff without committing real capital.
    """
    if not ENABLE_TOP2_SHADOWS:
        return []
    if _in_metar_danger_window(datetime.now(UTC)):
        return []

    shadows: list[Signal] = []

    for market in markets:
        station_icao  = market["station_icao"]
        city          = market["city"]
        date          = market["date"]
        end_date_iso  = market["end_date_iso"]

        if calculate_hours_to_resolution(end_date_iso) < HOURS_BEFORE_RESOLUTION_CUTOFF:
            continue

        forecast_bundle = forecasts.get(station_icao, {}).get(date)
        if not forecast_bundle:
            continue

        forecast           = forecast_bundle.get("probs", {})
        rounding_conf      = forecast_bundle.get("rounding_confidence", "LOW")
        pred_display_temp  = forecast_bundle.get("predicted_display_temp")
        ensemble_std       = float(forecast_bundle.get("ensemble_std", 0.0) or 0.0)
        ensemble_skip      = bool(
            forecast_bundle.get("ensemble_skip", False)
            or ensemble_std > ENSEMBLE_STD_SKIP_THRESHOLD
        )
        if ensemble_skip:
            continue

        det_spread        = float(forecast_bundle.get("det_spread", ensemble_std * 2.0) or 0.0)
        det_spread_colour = str(forecast_bundle.get("det_spread_colour", "UNKNOWN"))
        det_model_values  = forecast_bundle.get("det_model_values") or {}
        city_kelly        = float(STATIONS.get(station_icao, {}).get("kelly_fraction", KELLY_FRACTION))

        hours_to_res = calculate_hours_to_resolution(end_date_iso)
        try:
            days_ahead = (_date.fromisoformat(date) - _date.today()).days
        except ValueError:
            days_ahead = 1

        if days_ahead >= 3:
            temporal_discount = D3_P_WIN_DISCOUNT
            _d_max = D3_MAX_YES_ENTRY_PRICE
        elif days_ahead >= 2:
            temporal_discount = D2_P_WIN_DISCOUNT
            _d_max = D2_MAX_YES_ENTRY_PRICE
        else:
            temporal_discount = 1.0
            _d_max = HARD_MAX_YES_ENTRY_PRICE

        # Collect BUY_YES candidates: positive edge, price guards, minimum model prob
        candidates: list[dict] = []
        for bucket, token_info in market["buckets"].items():
            model_prob  = forecast.get(bucket, 0.0) * temporal_discount
            market_prob = token_info["price"]
            if model_prob < TOP2_SHADOW_MIN_PROB:
                _log_signal_eval(
                    strategy="TOP2_SHADOW",
                    city=city,
                    station_icao=station_icao,
                    target_date=date,
                    bucket=bucket,
                    side="BUY_YES",
                    forecast_prob=float(model_prob),
                    market_prob=float(market_prob),
                    edge=float(model_prob - market_prob),
                    size_usd=0.0,
                    decision="REJECT",
                    rejection_reason="model_prob_below_min",
                    gate_results={"model_prob_min": {"passed": False, "threshold": TOP2_SHADOW_MIN_PROB}},
                    forecast_bundle=forecast_bundle,
                    strategy_context={"selection_path": "top2_shadow"},
                )
                continue
            if model_prob <= market_prob:               # no positive edge
                _log_signal_eval(
                    strategy="TOP2_SHADOW",
                    city=city,
                    station_icao=station_icao,
                    target_date=date,
                    bucket=bucket,
                    side="BUY_YES",
                    forecast_prob=float(model_prob),
                    market_prob=float(market_prob),
                    edge=float(model_prob - market_prob),
                    size_usd=0.0,
                    decision="REJECT",
                    rejection_reason="non_positive_edge",
                    gate_results={"edge_positive": {"passed": False}},
                    forecast_bundle=forecast_bundle,
                    strategy_context={"selection_path": "top2_shadow"},
                )
                continue
            if market_prob < HARD_MIN_YES_ENTRY_PRICE or market_prob > _d_max:
                _log_signal_eval(
                    strategy="TOP2_SHADOW",
                    city=city,
                    station_icao=station_icao,
                    target_date=date,
                    bucket=bucket,
                    side="BUY_YES",
                    forecast_prob=float(model_prob),
                    market_prob=float(market_prob),
                    edge=float(model_prob - market_prob),
                    size_usd=0.0,
                    decision="REJECT",
                    rejection_reason="price_guard_failed",
                    gate_results={"price_guard": {"passed": False, "min": HARD_MIN_YES_ENTRY_PRICE, "max": _d_max}},
                    forecast_bundle=forecast_bundle,
                    strategy_context={"selection_path": "top2_shadow"},
                )
                continue
            candidates.append({
                "bucket":      bucket,
                "token_id":    token_info["yes_token_id"],
                "model_prob":  model_prob,
                "market_prob": market_prob,
                "edge":        model_prob - market_prob,
            })

        candidates.sort(key=lambda c: c["model_prob"], reverse=True)
        if not candidates:
            continue

        top1 = candidates[0]
        top2 = candidates[1] if len(candidates) >= 2 else None

        mv_json = json.dumps({k: round(v, 2) for k, v in det_model_values.items()}, separators=(",", ":"))

        # Compute ONE total budget for this city using top1 as the representative bucket.
        # All three shadow strategies share this budget and split it differently —
        # that's what makes them genuinely different from each other.
        _base_sz = kelly_size(
            market_price=top1["market_prob"],
            win_prob=top1["model_prob"],
            bankroll=bankroll,
            edge=top1["edge"],
            kelly_fraction=city_kelly,
            max_position=_kelly_position_cap(bankroll),
            rounding_confidence="MEDIUM",
        )
        _base_sz = max(_base_sz, KELLY_MIN_BET_USD) if _base_sz > 0 else KELLY_MIN_BET_USD

        def _make(cand: dict, strategy: str, size_usd: float) -> Signal:
            sz = round(max(size_usd, KELLY_MIN_BET_USD), 2)
            ev = (
                cand["model_prob"] * (sz / cand["market_prob"]) * (1.0 - cand["market_prob"])
                - (1.0 - cand["model_prob"]) * sz
            )
            sig = Signal(
                market_id=market["condition_id"],
                token_id=cand["token_id"],
                side="BUY_YES",
                edge=cand["edge"],
                forecast_prob=cand["model_prob"],
                market_prob=cand["market_prob"],
                size_usd=sz,
                city=city,
                station_icao=station_icao,
                date=date,
                bucket=cand["bucket"],
                rounding_confidence=rounding_conf,
                predicted_display_temp=pred_display_temp,
                spread_colour=det_spread_colour,
                det_spread=round(det_spread, 3),
                model_values_json=mv_json,
                ev_per_bet=round(ev, 3),
                kelly_fraction_used=city_kelly,
                days_ahead=days_ahead,
                hours_to_resolution=round(hours_to_res, 1),
                temporal_discount=temporal_discount,
                strategy=strategy,
                **_signal_observability_fields(forecast_bundle),
            )
            sig.decision_id = _log_signal_eval(
                strategy=strategy,
                city=city,
                station_icao=station_icao,
                target_date=date,
                bucket=cand["bucket"],
                side="BUY_YES",
                forecast_prob=float(cand["model_prob"]),
                market_prob=float(cand["market_prob"]),
                edge=float(cand["edge"]),
                size_usd=float(sz),
                decision="TRADE",
                rejection_reason=None,
                gate_results={"top2_shadow_candidate": {"passed": True}},
                forecast_bundle=forecast_bundle,
                strategy_context={"selection_path": "top2_shadow"},
            )
            return sig

        # ── 2A — TOP2_EQUAL: total budget split exactly 50/50 ───────────────────
        # Both legs get half the budget. Identical capital on favourite and runner-up.
        half = _base_sz / 2.0
        shadows.append(_make(top1, "TOP2_EQUAL", half))
        if top2:
            shadows.append(_make(top2, "TOP2_EQUAL", half))

        # ── 2B — TOP2_COND: total budget split 65/35, or all-in on top1 if tight ─
        # Only goes dual when the model is genuinely split (runner-up ≥ threshold × top).
        # When not split the full budget rides on the favourite — a stronger single bet.
        is_split = (
            top2 is not None
            and (top2["model_prob"] / top1["model_prob"]) >= TOP2_SHADOW_SPLIT_THRESHOLD
        )
        if is_split and top2:
            shadows.append(_make(top1, "TOP2_COND", _base_sz * 0.65))
            shadows.append(_make(top2, "TOP2_COND", _base_sz * 0.35))
        else:
            shadows.append(_make(top1, "TOP2_COND", _base_sz))

        # ── 2C — TOP2_PROP: total budget split proportional to model probability ──
        # If model says 41% vs 19%, sizes are ~68% vs ~32% of the budget.
        if top2:
            p1, p2 = top1["model_prob"], top2["model_prob"]
            total_p = p1 + p2
            shadows.append(_make(top1, "TOP2_PROP", _base_sz * (p1 / total_p)))
            shadows.append(_make(top2, "TOP2_PROP", _base_sz * (p2 / total_p)))
        else:
            shadows.append(_make(top1, "TOP2_PROP", _base_sz))

    return shadows


def _bucket_lower_bound(bucket: str) -> float:
    """Return the lower temperature bound of a bucket label for sorting."""
    b = str(bucket).strip()
    if b.endswith("+"):
        try:
            return float(b[:-1])
        except ValueError:
            return 999.0
    parts = b.split("-")
    if len(parts) == 2:
        try:
            return float(parts[0])
        except ValueError:
            pass
    return 999.0


def generate_purdey_cavendish_signals(
    markets: list[dict],
    forecasts: dict[str, dict[str, dict]],
    bankroll: float,
) -> list["Signal"]:
    """Generate PURDEY_MK1 and CAVENDISH_MK1 shadow signals.

    KEY INSIGHT: Polymarket creates separate binary markets per temperature
    bucket, so each ``market`` in the list has exactly 1 bucket.  We must
    group by (station, date) first to reconstruct the full bucket range,
    then apply multi-bucket selection logic over the aggregated set.

    PURDEY_MK1 — Hard cap of 2 bets per city-date.
        Bets on the top-1 bucket (60 % of budget) and top-2 bucket (40 %)
        by model probability.  If the runner-up doesn't meet
        TOP2_SHADOW_MIN_PROB, only 1 bet is placed.

    CAVENDISH_MK1 — Hard cap of 3 bets per city-date.
        Bets on the peak bucket (50 %) and its immediate temperature
        neighbours: one cooler (25 %) and one warmer (25 %).
        Flanks must earn their place: ≥5% ensemble probability AND
        at least 1 individual model predicting a temp in that bucket.
    """
    from config.settings import ENABLE_TOP2_SHADOWS
    from strategy.model_weights import models_in_bucket
    _log = logging.getLogger("weather-bot.signals")

    if not ENABLE_TOP2_SHADOWS:
        return []
    if _in_metar_danger_window(datetime.now(UTC)):
        return []

    # ── Step 1: group individual binary markets by (station, date) ──────
    from collections import defaultdict
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for market in markets:
        key = (market["station_icao"], market["date"])
        grouped[key].append(market)

    signals: list[Signal] = []

    for (station_icao, date_str), group_markets in grouped.items():
        ref = group_markets[0]
        city = ref["city"]
        end_date_iso = ref["end_date_iso"]

        if calculate_hours_to_resolution(end_date_iso) < HOURS_BEFORE_RESOLUTION_CUTOFF:
            continue

        forecast_bundle = forecasts.get(station_icao, {}).get(date_str)
        if not forecast_bundle:
            continue

        forecast = forecast_bundle.get("probs", {})
        rounding_conf = forecast_bundle.get("rounding_confidence", "LOW")
        pred_display = forecast_bundle.get("predicted_display_temp")
        ensemble_std = float(forecast_bundle.get("ensemble_std", 0.0) or 0.0)
        if bool(forecast_bundle.get("ensemble_skip", False) or ensemble_std > ENSEMBLE_STD_SKIP_THRESHOLD):
            continue

        det_spread = float(forecast_bundle.get("det_spread", ensemble_std * 2.0) or 0.0)
        det_spread_colour = str(forecast_bundle.get("det_spread_colour", "UNKNOWN"))
        det_model_values = forecast_bundle.get("det_model_values") or {}
        city_kelly = float(STATIONS.get(station_icao, {}).get("kelly_fraction", KELLY_FRACTION))
        mv_json = json.dumps({k: round(v, 2) for k, v in det_model_values.items()}, separators=(",", ":"))

        hours_to_res = calculate_hours_to_resolution(end_date_iso)
        try:
            days_ahead = (_date.fromisoformat(date_str) - _date.today()).days
        except ValueError:
            days_ahead = 1

        if days_ahead >= 3:
            temporal_discount = D3_P_WIN_DISCOUNT
            _d_max = D3_MAX_YES_ENTRY_PRICE
        elif days_ahead >= 2:
            temporal_discount = D2_P_WIN_DISCOUNT
            _d_max = D2_MAX_YES_ENTRY_PRICE
        else:
            temporal_discount = 1.0
            _d_max = HARD_MAX_YES_ENTRY_PRICE

        # ── Step 2: aggregate ALL buckets for this city-date from all binary markets ──
        all_buckets: dict[str, dict] = {}
        bucket_to_condition: dict[str, str] = {}
        for mkt in group_markets:
            for bucket, info in mkt.get("buckets", {}).items():
                all_buckets[bucket] = info
                bucket_to_condition[bucket] = mkt["condition_id"]

        sorted_buckets = sorted(all_buckets.keys(), key=_bucket_lower_bound)

        # ── Step 3: build candidate list (positive edge, price guards) ──
        candidates: list[dict] = []
        for bucket in sorted_buckets:
            token_info = all_buckets[bucket]
            model_prob = forecast.get(bucket, 0.0) * temporal_discount
            market_prob = token_info["price"]
            if model_prob < TOP2_SHADOW_MIN_PROB:
                _log_signal_eval(
                    strategy="PURDEY_CAVENDISH",
                    city=city,
                    station_icao=station_icao,
                    target_date=date_str,
                    bucket=bucket,
                    side="BUY_YES",
                    forecast_prob=float(model_prob),
                    market_prob=float(market_prob),
                    edge=float(model_prob - market_prob),
                    size_usd=0.0,
                    decision="REJECT",
                    rejection_reason="model_prob_below_min",
                    gate_results={"model_prob_min": {"passed": False, "threshold": TOP2_SHADOW_MIN_PROB}},
                    forecast_bundle=forecast_bundle,
                    strategy_context={"selection_path": "purdey_cavendish"},
                )
                continue
            if model_prob <= market_prob:
                _log_signal_eval(
                    strategy="PURDEY_CAVENDISH",
                    city=city,
                    station_icao=station_icao,
                    target_date=date_str,
                    bucket=bucket,
                    side="BUY_YES",
                    forecast_prob=float(model_prob),
                    market_prob=float(market_prob),
                    edge=float(model_prob - market_prob),
                    size_usd=0.0,
                    decision="REJECT",
                    rejection_reason="non_positive_edge",
                    gate_results={"edge_positive": {"passed": False}},
                    forecast_bundle=forecast_bundle,
                    strategy_context={"selection_path": "purdey_cavendish"},
                )
                continue
            if market_prob < HARD_MIN_YES_ENTRY_PRICE or market_prob > _d_max:
                _log_signal_eval(
                    strategy="PURDEY_CAVENDISH",
                    city=city,
                    station_icao=station_icao,
                    target_date=date_str,
                    bucket=bucket,
                    side="BUY_YES",
                    forecast_prob=float(model_prob),
                    market_prob=float(market_prob),
                    edge=float(model_prob - market_prob),
                    size_usd=0.0,
                    decision="REJECT",
                    rejection_reason="price_guard_failed",
                    gate_results={"price_guard": {"passed": False, "min": HARD_MIN_YES_ENTRY_PRICE, "max": _d_max}},
                    forecast_bundle=forecast_bundle,
                    strategy_context={"selection_path": "purdey_cavendish"},
                )
                continue
            candidates.append({
                "bucket": bucket,
                "token_id": token_info["yes_token_id"],
                "condition_id": bucket_to_condition[bucket],
                "model_prob": model_prob,
                "market_prob": market_prob,
                "edge": model_prob - market_prob,
            })

        if not candidates:
            continue

        candidates.sort(key=lambda c: c["model_prob"], reverse=True)
        top1 = candidates[0]
        top2 = candidates[1] if len(candidates) >= 2 else None

        def _make_pc(cand: dict, strategy: str, size_usd: float) -> Signal:
            sz = round(max(size_usd, KELLY_MIN_BET_USD), 2)
            ev = (
                cand["model_prob"] * (sz / cand["market_prob"]) * (1.0 - cand["market_prob"])
                - (1.0 - cand["model_prob"]) * sz
            )
            sig = Signal(
                market_id=cand["condition_id"],
                token_id=cand["token_id"],
                side="BUY_YES",
                edge=cand["edge"],
                forecast_prob=cand["model_prob"],
                market_prob=cand["market_prob"],
                size_usd=sz,
                city=city,
                station_icao=station_icao,
                date=date_str,
                bucket=cand["bucket"],
                rounding_confidence=rounding_conf,
                predicted_display_temp=pred_display,
                spread_colour=det_spread_colour,
                det_spread=round(det_spread, 3),
                model_values_json=mv_json,
                ev_per_bet=round(ev, 3),
                kelly_fraction_used=city_kelly,
                days_ahead=days_ahead,
                hours_to_resolution=round(hours_to_res, 1),
                temporal_discount=temporal_discount,
                strategy=strategy,
                **_signal_observability_fields(forecast_bundle),
            )
            sig.decision_id = _log_signal_eval(
                strategy=strategy,
                city=city,
                station_icao=station_icao,
                target_date=date_str,
                bucket=cand["bucket"],
                side="BUY_YES",
                forecast_prob=float(cand["model_prob"]),
                market_prob=float(cand["market_prob"]),
                edge=float(cand["edge"]),
                size_usd=float(sz),
                decision="TRADE",
                rejection_reason=None,
                gate_results={"purdey_cavendish_candidate": {"passed": True}},
                forecast_bundle=forecast_bundle,
                strategy_context={"selection_path": "purdey_cavendish"},
            )
            return sig

        _base_sz = kelly_size(
            market_price=top1["market_prob"],
            win_prob=top1["model_prob"],
            bankroll=bankroll,
            edge=top1["edge"],
            kelly_fraction=city_kelly,
            max_position=_kelly_position_cap(bankroll),
            rounding_confidence="MEDIUM",
        )
        _base_sz = max(_base_sz, KELLY_MIN_BET_USD) if _base_sz > 0 else KELLY_MIN_BET_USD

        # ── PURDEY_MK1: top-2 by model probability, hard cap 2, 60/40 split ────
        signals.append(_make_pc(top1, "PURDEY_MK1", _base_sz * 0.60))
        if top2 is not None:
            signals.append(_make_pc(top2, "PURDEY_MK1", _base_sz * 0.40))

        # ── CAVENDISH_MK1: peak + adjacent flanks, hard cap 3, 50/25/25 split ─
        peak_bucket = top1["bucket"]
        try:
            peak_idx = sorted_buckets.index(peak_bucket)
        except ValueError:
            peak_idx = -1

        signals.append(_make_pc(top1, "CAVENDISH_MK1", _base_sz * 0.50))

        # Flanks must earn their place: ≥5% ensemble probability AND
        # ≥1 individual model predicting a temp in that bucket.
        _CAVI_FLANK_MIN_PROB = 0.05
        if peak_idx >= 0:
            # Flank below (one cooler bucket in temperature order)
            if peak_idx > 0:
                flank_below_key = sorted_buckets[peak_idx - 1]
                flank_info = all_buckets[flank_below_key]
                flank_price = float(flank_info.get("price", 0) or 0)
                flank_token = flank_info.get("yes_token_id", "")
                if flank_token and HARD_MIN_YES_ENTRY_PRICE <= flank_price <= _d_max:
                    flank_model = forecast.get(flank_below_key, 0.0) * temporal_discount
                    fn_models = models_in_bucket(det_model_values, flank_below_key)
                    if flank_model >= _CAVI_FLANK_MIN_PROB and fn_models >= 1:
                        signals.append(_make_pc(
                            {"bucket": flank_below_key, "token_id": flank_token,
                             "condition_id": bucket_to_condition[flank_below_key],
                             "model_prob": flank_model,
                             "market_prob": flank_price, "edge": max(flank_model - flank_price, 0.0)},
                            "CAVENDISH_MK1", _base_sz * 0.25,
                        ))
                    else:
                        _log.info(
                            f"CAVENDISH_MK1 skipping lower flank {flank_below_key} for {city} {date_str}: "
                            f"prob={flank_model:.3f} n_models={fn_models}"
                        )

            # Flank above (one warmer bucket in temperature order)
            if peak_idx < len(sorted_buckets) - 1:
                flank_above_key = sorted_buckets[peak_idx + 1]
                flank_info = all_buckets[flank_above_key]
                flank_price = float(flank_info.get("price", 0) or 0)
                flank_token = flank_info.get("yes_token_id", "")
                if flank_token and HARD_MIN_YES_ENTRY_PRICE <= flank_price <= _d_max:
                    flank_model = forecast.get(flank_above_key, 0.0) * temporal_discount
                    fn_models = models_in_bucket(det_model_values, flank_above_key)
                    if flank_model >= _CAVI_FLANK_MIN_PROB and fn_models >= 1:
                        signals.append(_make_pc(
                            {"bucket": flank_above_key, "token_id": flank_token,
                             "condition_id": bucket_to_condition[flank_above_key],
                             "model_prob": flank_model,
                             "market_prob": flank_price, "edge": max(flank_model - flank_price, 0.0)},
                            "CAVENDISH_MK1", _base_sz * 0.25,
                        ))
                    else:
                        _log.info(
                            f"CAVENDISH_MK1 skipping upper flank {flank_above_key} for {city} {date_str}: "
                            f"prob={flank_model:.3f} n_models={fn_models}"
                        )

        _log.info(
            f"PC signals for {city} {date_str}: "
            f"buckets={len(sorted_buckets)} candidates={len(candidates)} "
            f"PURDEY={1 + (1 if top2 else 0)} "
            f"CAVENDISH peak={peak_bucket}[{peak_idx}] of {sorted_buckets}"
        )

    n_purdey    = sum(1 for s in signals if s.strategy == "PURDEY_MK1")
    n_cavendish = sum(1 for s in signals if s.strategy == "CAVENDISH_MK1")
    _log.info(f"PURDEY_MK1={n_purdey} | CAVENDISH_MK1={n_cavendish} signals")
    return signals


def generate_mk2_ace_signals(
    markets: list[dict],
    forecasts: dict[str, dict[str, dict]],
    bankroll: float,
) -> list["Signal"]:
    """Generate PURDEY_MK2, CAVENDISH_MK3, TRUE_ALPHA, PRIME_ALPHA, and PROPS_KELLY signals.

    All use recency-weighted model accuracy. Models that nailed yesterday's
    temperature get higher weight, shifting the distribution toward their
    predictions.  Every strategy requires ≥1 model supporting each bet —
    zero tolerance for 0-model bets at any price.

    PURDEY_MK2 — Hard cap of 2 bets per city-date.
        Top-2 buckets by recency-weighted model probability.

    CAVENDISH_MK3 — Hard cap of 3 bets per city-date.
        Peak + adjacent flanks, but each flank must earn its place:
        ≥5% weighted probability AND ≥1 model predicting that bucket.

    TRUE_ALPHA — Hard cap of 3 bets per city-date.
        The strictest model. Peak always. 2nd bet needs ≥10% weighted prob
        AND ≥1 model. 3rd bet needs ≥10% AND (≥2 models OR price ≤9¢ + ≥1
        model). Not restricted to adjacent flanks — picks best supported
        buckets anywhere. Proportional Kelly sizing.

    PRIME_ALPHA — Hard cap of 3 bets per city-date.
        Deterministic contiguous range built from today's temperatures of the
        models that hit the settled market yesterday, with the flagship
        ensemble included only when it also hit yesterday.

    PROPS_KELLY — Hard cap of 3 bets per city-date.
        Peak + earned adjacent flanks (same gates as MK3) with proportional
        Kelly sizing so the peak gets the largest share.
    """
    from collections import defaultdict
    from strategy.model_weights import (
        compute_weights,
        models_in_bucket,
        weighted_bucket_probs,
    )
    from strategy.prime_alpha import build_prime_alpha_plan

    _log = logging.getLogger("weather-bot.signals")

    if not ENABLE_TOP2_SHADOWS:
        return []
    if _in_metar_danger_window(datetime.now(UTC)):
        return []

    # Group binary markets by (station, date)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for market in markets:
        grouped[(market["station_icao"], market["date"])].append(market)

    signals: list[Signal] = []

    for (station_icao, date_str), group_markets in grouped.items():
        ref = group_markets[0]
        city = ref["city"]
        end_date_iso = ref["end_date_iso"]

        if calculate_hours_to_resolution(end_date_iso) < HOURS_BEFORE_RESOLUTION_CUTOFF:
            continue

        forecast_bundle = forecasts.get(station_icao, {}).get(date_str)
        if not forecast_bundle:
            continue

        det_model_values = forecast_bundle.get("det_model_values") or {}
        if not det_model_values:
            continue

        ensemble_std = float(forecast_bundle.get("ensemble_std", 0.0) or 0.0)
        if bool(forecast_bundle.get("ensemble_skip", False) or ensemble_std > ENSEMBLE_STD_SKIP_THRESHOLD):
            continue

        rounding_conf = forecast_bundle.get("rounding_confidence", "LOW")
        pred_display = forecast_bundle.get("predicted_display_temp")
        det_spread = float(forecast_bundle.get("det_spread", ensemble_std * 2.0) or 0.0)
        det_spread_colour = str(forecast_bundle.get("det_spread_colour", "UNKNOWN"))
        city_kelly = float(STATIONS.get(station_icao, {}).get("kelly_fraction", KELLY_FRACTION))
        mv_json = json.dumps(
            {k: round(v, 2) for k, v in det_model_values.items()},
            separators=(",", ":"),
        )

        hours_to_res = calculate_hours_to_resolution(end_date_iso)
        try:
            days_ahead = (_date.fromisoformat(date_str) - _date.today()).days
        except ValueError:
            days_ahead = 1

        if days_ahead >= 3:
            temporal_discount = D3_P_WIN_DISCOUNT
            _d_max = D3_MAX_YES_ENTRY_PRICE
        elif days_ahead >= 2:
            temporal_discount = D2_P_WIN_DISCOUNT
            _d_max = D2_MAX_YES_ENTRY_PRICE
        else:
            temporal_discount = 1.0
            _d_max = HARD_MAX_YES_ENTRY_PRICE

        # Aggregate all buckets for this city-date
        all_buckets: dict[str, dict] = {}
        bucket_to_condition: dict[str, str] = {}
        for mkt in group_markets:
            for bucket, info in mkt.get("buckets", {}).items():
                all_buckets[bucket] = info
                bucket_to_condition[bucket] = mkt["condition_id"]

        sorted_buckets = sorted(all_buckets.keys(), key=_bucket_lower_bound)
        if not sorted_buckets:
            continue

        # Compute recency-weighted model accuracy
        model_weights = compute_weights(station_icao, list(det_model_values.keys()))

        # WEIGHTED bucket probabilities (the key differentiator from MK1)
        w_probs = weighted_bucket_probs(det_model_values, model_weights, sorted_buckets)

        # ── Build ranked buckets: purely by WEIGHTED PROBABILITY ─────────
        # No edge requirement.  No old MIN_PROB filter.  The recency-
        # weighted model analysis is the SOLE driver of bucket selection.
        ranked: list[dict] = []
        for bucket in sorted_buckets:
            token_info = all_buckets[bucket]
            wp = w_probs.get(bucket, 0.0) * temporal_discount
            mkt_price = float(token_info.get("price", 0) or 0)
            token_id = token_info.get("yes_token_id", "")
            if not token_id or mkt_price < HARD_MIN_YES_ENTRY_PRICE or mkt_price > _d_max:
                _log_signal_eval(
                    strategy="MK2_ACE",
                    city=city,
                    station_icao=station_icao,
                    target_date=date_str,
                    bucket=bucket,
                    side="BUY_YES",
                    forecast_prob=float(wp),
                    market_prob=float(mkt_price),
                    edge=float(wp - mkt_price),
                    size_usd=0.0,
                    decision="REJECT",
                    rejection_reason="token_or_price_guard_failed",
                    gate_results={"token_and_price_guard": {"passed": False}},
                    forecast_bundle=forecast_bundle,
                    strategy_context={"selection_path": "mk2_ace"},
                )
                continue
            if wp < 0.01:
                _log_signal_eval(
                    strategy="MK2_ACE",
                    city=city,
                    station_icao=station_icao,
                    target_date=date_str,
                    bucket=bucket,
                    side="BUY_YES",
                    forecast_prob=float(wp),
                    market_prob=float(mkt_price),
                    edge=float(wp - mkt_price),
                    size_usd=0.0,
                    decision="REJECT",
                    rejection_reason="weighted_prob_below_min",
                    gate_results={"weighted_prob_min": {"passed": False, "threshold": 0.01}},
                    forecast_bundle=forecast_bundle,
                    strategy_context={"selection_path": "mk2_ace"},
                )
                continue
            ranked.append({
                "bucket": bucket,
                "token_id": token_id,
                "condition_id": bucket_to_condition[bucket],
                "model_prob": wp,
                "market_prob": mkt_price,
                "edge": wp - mkt_price,
                "n_models": models_in_bucket(det_model_values, bucket),
            })

        if not ranked:
            continue

        ranked.sort(key=lambda c: c["model_prob"], reverse=True)
        cand_by_bucket = {cand["bucket"]: cand for cand in ranked}
        top1 = ranked[0]
        top2 = ranked[1] if len(ranked) >= 2 else None

        def _kelly_for(cand: dict) -> float:
            """Independent Kelly size for a single bet."""
            wp_ = cand["model_prob"]
            mp_ = cand["market_prob"]
            edge_ = wp_ - mp_
            nm = cand.get("n_models", 0)
            conf = "HIGH" if nm >= 3 else ("MEDIUM" if nm >= 2 else "LOW")
            if edge_ > 0:
                return max(
                    kelly_size(
                        market_price=mp_,
                        win_prob=wp_,
                        bankroll=bankroll,
                        edge=edge_,
                        kelly_fraction=city_kelly,
                        max_position=_kelly_position_cap(bankroll),
                        rounding_confidence=conf,
                    ),
                    KELLY_MIN_BET_USD,
                )
            return KELLY_MIN_BET_USD

        def _make_sig(
            cand: dict,
            strategy: str,
            _override_sz: float | None = None,
            _strategy_context: dict | None = None,
        ) -> Signal:
            sz = _override_sz if _override_sz is not None else _kelly_for(cand)
            mp = max(cand["market_prob"], 0.01)
            wp_ = cand["model_prob"]
            ev = wp_ * (sz / mp) * (1.0 - mp) - (1.0 - wp_) * sz
            sig = Signal(
                market_id=cand["condition_id"],
                token_id=cand["token_id"],
                side="BUY_YES",
                edge=cand["edge"],
                forecast_prob=wp_,
                market_prob=cand["market_prob"],
                size_usd=sz,
                city=city,
                station_icao=station_icao,
                date=date_str,
                bucket=cand["bucket"],
                rounding_confidence=rounding_conf,
                predicted_display_temp=pred_display,
                spread_colour=det_spread_colour,
                det_spread=round(det_spread, 3),
                model_values_json=mv_json,
                ev_per_bet=round(ev, 3),
                kelly_fraction_used=city_kelly,
                days_ahead=days_ahead,
                hours_to_resolution=round(hours_to_res, 1),
                temporal_discount=temporal_discount,
                strategy=strategy,
                **_signal_observability_fields(forecast_bundle),
            )
            strategy_context_payload = {
                "selection_path": "mk2_ace",
                "n_models": cand.get("n_models", None),
            }
            if _strategy_context:
                strategy_context_payload.update(_strategy_context)
            sig.decision_id = _log_signal_eval(
                strategy=strategy,
                city=city,
                station_icao=station_icao,
                target_date=date_str,
                bucket=cand["bucket"],
                side="BUY_YES",
                forecast_prob=float(wp_),
                market_prob=float(cand["market_prob"]),
                edge=float(cand["edge"]),
                size_usd=float(sz),
                decision="TRADE",
                rejection_reason=None,
                gate_results={"mk2_candidate": {"passed": True}},
                forecast_bundle=forecast_bundle,
                strategy_context=strategy_context_payload,
            )
            return sig

        peak_bucket = top1["bucket"]

        # ── PURDEY_MK2: top-2 by weighted probability ───────────────────
        signals.append(_make_sig(top1, "PURDEY_MK2"))
        if top2 is not None:
            signals.append(_make_sig(top2, "PURDEY_MK2"))

        # ── peak_idx for CAVENDISH_MK3, TRUE_ALPHA, PROPS_KELLY ─────────
        try:
            peak_idx = sorted_buckets.index(peak_bucket)
        except ValueError:
            peak_idx = -1

        # ── CAVENDISH_MK3: peak + EARNED flanks only ────────────────────
        # Identical philosophy to MK2 but flanks must earn their place:
        #   • weighted probability ≥ 5%  (real model signal, not noise)
        #   • at least 1 model actually predicts a temp in that bucket
        #   • passes the same _d_max price cap as main bets
        # This prevents blindly betting on a flank just because it's adjacent
        # when no model is actually pointing there.
        CAVENDISH_MK3_FLANK_MIN_WP = 0.05
        signals.append(_make_sig(top1, "CAVENDISH_MK3"))

        if peak_idx >= 0:
            for offset in (-1, +1):
                idx = peak_idx + offset
                if idx < 0 or idx >= len(sorted_buckets):
                    continue
                fk = sorted_buckets[idx]
                fi = all_buckets[fk]
                fp = float(fi.get("price", 0) or 0)
                ft = fi.get("yes_token_id", "")
                if not ft or fp < HARD_MIN_YES_ENTRY_PRICE or fp > _d_max:
                    continue
                fwp = w_probs.get(fk, 0.0) * temporal_discount
                fn_models = models_in_bucket(det_model_values, fk)
                # Hard gate: flank must have real model support
                if fwp < CAVENDISH_MK3_FLANK_MIN_WP or fn_models < 1:
                    _log.info(
                        f"CAVENDISH_MK3 skipping flank {fk} for {city} {date_str}: "
                        f"w_prob={fwp:.3f} n_models={fn_models} (below threshold)"
                    )
                    continue
                signals.append(_make_sig(
                    {"bucket": fk, "token_id": ft,
                     "condition_id": bucket_to_condition[fk],
                     "model_prob": fwp,
                     "market_prob": fp,
                     "edge": fwp - fp,
                     "n_models": fn_models},
                    "CAVENDISH_MK3",
                ))

        _log.info(
            f"CAVENDISH_MK3 {city} {date_str}: weighted_peak={peak_bucket}"
        )

        # ── TRUE_ALPHA: strictest model, zero tolerance for 0-model bets ─
        # Peak always. 2nd: ≥10% weighted prob + ≥1 model.
        # 3rd: ≥10% + (≥2 models OR price ≤9¢ + ≥1 model).
        # Not restricted to adjacent flanks. Proportional Kelly sizing.
        TRUE_ALPHA_MIN_WP         = 0.10
        TRUE_ALPHA_3RD_MIN_MODELS = 2
        TRUE_ALPHA_VALUE_PRICE    = 0.09

        ta_bets: list[dict] = [top1]
        for cand in ranked[1:]:
            if len(ta_bets) >= 3:
                break
            wp = cand["model_prob"]
            nm = cand["n_models"]
            mp = cand["market_prob"]
            if wp < TRUE_ALPHA_MIN_WP:
                continue
            if len(ta_bets) == 1:
                if nm >= 1:
                    ta_bets.append(cand)
                else:
                    _log.info(f"TRUE_ALPHA skip 2nd {cand['bucket']} {city} {date_str}: 0 models (w={wp:.3f})")
            elif len(ta_bets) == 2:
                if (nm >= TRUE_ALPHA_3RD_MIN_MODELS) or (mp <= TRUE_ALPHA_VALUE_PRICE and nm >= 1):
                    ta_bets.append(cand)
                    _log.info(f"TRUE_ALPHA 3rd {cand['bucket']} {city} {date_str}: models={nm} price={mp:.3f}")
                else:
                    _log.info(f"TRUE_ALPHA skip 3rd {cand['bucket']} {city} {date_str}: models={nm} price={mp:.3f}")

        ta_total_wp = sum(b["model_prob"] for b in ta_bets)
        for bet in ta_bets:
            prop  = bet["model_prob"] / ta_total_wp if ta_total_wp > 0 else 1.0 / len(ta_bets)
            raw_k = _kelly_for(bet)
            sz    = round(max(raw_k * prop * len(ta_bets), KELLY_MIN_BET_USD), 2)
            signals.append(_make_sig(bet, "TRUE_ALPHA", _override_sz=sz))
        _log.info(
            f"TRUE_ALPHA {city} {date_str}: {len(ta_bets)} bets → "
            + " | ".join(f"{b['bucket']}(w={b['model_prob']:.2f},nm={b['n_models']})" for b in ta_bets)
        )

        prime_plan = build_prime_alpha_plan(
            city=city,
            station_icao=station_icao,
            target_date=date_str,
            bucket_labels=sorted_buckets,
            current_model_values=det_model_values,
            predicted_display_temp=pred_display,
            unit=str(STATIONS.get(station_icao, {}).get("resolution_unit", "F")),
            model_weights=model_weights,
        )
        prime_context = prime_plan.to_strategy_context()
        prime_cands: list[dict] = []
        for bucket in prime_plan.selected_buckets:
            cand = cand_by_bucket.get(bucket)
            if cand is None:
                _log_signal_eval(
                    strategy="PRIME_ALPHA",
                    city=city,
                    station_icao=station_icao,
                    target_date=date_str,
                    bucket=bucket,
                    side="BUY_YES",
                    forecast_prob=0.0,
                    market_prob=0.0,
                    edge=0.0,
                    size_usd=0.0,
                    decision="REJECT",
                    rejection_reason="prime_alpha_selected_bucket_not_viable",
                    gate_results={"prime_alpha_selected_bucket": {"passed": False}},
                    forecast_bundle=forecast_bundle,
                    strategy_context=prime_context,
                )
                continue
            prime_cands.append(cand)

        if prime_cands:
            combined_model_prob = min(0.95, sum(c["model_prob"] for c in prime_cands))
            combined_market_cost = sum(c["market_prob"] for c in prime_cands)
            combined_edge = combined_model_prob - combined_market_cost
            prime_context["paired_bet"] = {
                "combined_model_prob": round(combined_model_prob, 4),
                "combined_market_cost": round(combined_market_cost, 4),
                "combined_edge": round(combined_edge, 4),
            }

            if combined_edge > 0:
                nm_total = sum(c.get("n_models", 1) for c in prime_cands)
                conf = "HIGH" if nm_total >= 5 else ("MEDIUM" if nm_total >= 3 else "LOW")
                total_kelly = max(
                    kelly_size(
                        market_price=combined_market_cost,
                        win_prob=combined_model_prob,
                        bankroll=bankroll,
                        edge=combined_edge,
                        kelly_fraction=city_kelly,
                        max_position=_kelly_position_cap(bankroll),
                        rounding_confidence=conf,
                    ),
                    KELLY_MIN_BET_USD,
                )
                support_scores = {
                    c["bucket"]: max(prime_plan.all_model_bucket_counts.get(c["bucket"], 1), 1)
                    for c in prime_cands
                }
                total_support = sum(support_scores.values()) or 1.0

                for cand in prime_cands:
                    weight = support_scores[cand["bucket"]] / total_support
                    sz = round(max(total_kelly * weight, KELLY_MIN_BET_USD), 2)
                    signals.append(
                        _make_sig(
                            cand,
                            "PRIME_ALPHA",
                            _override_sz=sz,
                            _strategy_context=prime_context,
                        )
                    )
                _log.info(
                    f"PRIME_ALPHA {city} {date_str}: paired bet "
                    f"P(any)={combined_model_prob:.1%} cost={combined_market_cost:.1%} "
                    f"edge={combined_edge:+.1%} total=${total_kelly:.2f} → "
                    + " | ".join(
                        f"{c['bucket']}(${round(max(total_kelly * (support_scores[c['bucket']]/total_support), KELLY_MIN_BET_USD), 2)})"
                        for c in prime_cands
                    )
                )
            else:
                _log.info(
                    f"PRIME_ALPHA {city} {date_str}: no paired edge "
                    f"P(any)={combined_model_prob:.1%} cost={combined_market_cost:.1%} "
                    f"edge={combined_edge:+.1%}"
                )
        else:
            _log.info(
                f"PRIME_ALPHA {city} {date_str}: no viable buckets "
                f"(selected={prime_plan.selected_buckets})"
            )

        # ── PROPS_KELLY: peak + earned flanks, proportional Kelly sizing ─
        PROPS_KELLY_FLANK_MIN_WP = 0.05
        pk_bets: list[dict] = [top1]
        if peak_idx >= 0:
            for off in (-1, +1):
                idx = peak_idx + off
                if idx < 0 or idx >= len(sorted_buckets):
                    continue
                fk = sorted_buckets[idx]
                fi = all_buckets[fk]
                fp = float(fi.get("price", 0) or 0)
                ft = fi.get("yes_token_id", "")
                if not ft or fp < HARD_MIN_YES_ENTRY_PRICE or fp > _d_max:
                    continue
                fwp = w_probs.get(fk, 0.0) * temporal_discount
                fn_models = models_in_bucket(det_model_values, fk)
                if fwp < PROPS_KELLY_FLANK_MIN_WP or fn_models < 1:
                    continue
                pk_bets.append({
                    "bucket": fk, "token_id": ft,
                    "condition_id": bucket_to_condition[fk],
                    "model_prob": fwp, "market_prob": fp,
                    "edge": fwp - fp, "n_models": fn_models,
                })
        total_wp = sum(b["model_prob"] for b in pk_bets)
        for bet in pk_bets:
            prop = bet["model_prob"] / total_wp if total_wp > 0 else 1.0 / len(pk_bets)
            raw_kelly = _kelly_for(bet)
            sz = round(max(raw_kelly * prop * len(pk_bets), KELLY_MIN_BET_USD), 2)
            signals.append(_make_sig(bet, "PROPS_KELLY", _override_sz=sz))

    n_p2 = sum(1 for s in signals if s.strategy == "PURDEY_MK2")
    n_c3 = sum(1 for s in signals if s.strategy == "CAVENDISH_MK3")
    n_ta = sum(1 for s in signals if s.strategy == "TRUE_ALPHA")
    n_pa = sum(1 for s in signals if s.strategy == "PRIME_ALPHA")
    n_pk = sum(1 for s in signals if s.strategy == "PROPS_KELLY")
    _log.info(
        f"PURDEY_MK2={n_p2} | CAVENDISH_MK3={n_c3} | "
        f"TRUE_ALPHA={n_ta} | PRIME_ALPHA={n_pa} | PROPS_KELLY={n_pk} signals"
    )
    return signals


def _effective_edge_with_soft_guardrails(action: str, yes_price: float, edge: float) -> tuple[float, bool]:
    if not SOFT_PRICE_GUARDRAILS_ENABLED:
        return edge, False

    penalized = False
    if action == "BUY_YES":
        penalized = yes_price < SOFT_MIN_YES_PRICE or yes_price > SOFT_MAX_YES_PRICE
    elif action == "BUY_NO":
        no_price = 1.0 - yes_price
        penalized = no_price < SOFT_MIN_NO_PRICE or no_price > SOFT_MAX_NO_PRICE

    if not penalized:
        return edge, False
    return max(0.0, edge - SOFT_PRICE_EDGE_PENALTY), True
