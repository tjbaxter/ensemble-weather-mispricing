"""Signal generation for weather bucket markets."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import UTC, datetime

from config.settings import (
    ALPHA_THRESHOLD,
    ENABLE_LADDER_STRATEGY,
    ENSEMBLE_DISABLE_CLASSIC_CONFIDENCE_GATE,
    ENSEMBLE_STD_SKIP_THRESHOLD,
    FIXED_ORDER_USD,
    FIXED_SIZE_BANKROLL_THRESHOLD,
    HOURS_BEFORE_RESOLUTION_CUTOFF,
    KELLY_FRACTION,
    LADDER_MAX_TOTAL_COST,
    LADDER_MIN_EDGE,
    LADDER_WIDTH,
    MAX_POSITION_SIZE,
    METAR_DANGER_POST_MINUTE,
    METAR_DANGER_PRE_MINUTE,
    MIN_FORECAST_CONFIDENCE,
    MIN_ORDER_USD,
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

    def to_dict(self) -> dict:
        return asdict(self)


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

        if ensemble_skip:
            continue

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
                )
                if ladder_size >= max(MIN_ORDER_USD, PRACTICAL_MIN_ORDER_USD):
                    each_size = round(ladder_size / len(ladder), 2)
                    for item in ladder:
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
                            )
                        )
                    continue

        for bucket, token_info in market["buckets"].items():
            forecast_prob = forecast.get(bucket, 0.0)
            market_prob = token_info["price"]
            action, edge, win_prob = calculate_edge(
                forecast_prob,
                market_prob,
                min_forecast_confidence=min_confidence,
            )

            if action == "NO_TRADE":
                continue

            effective_edge, _guardrail_penalized = _effective_edge_with_soft_guardrails(action, market_prob, edge)
            if effective_edge <= ALPHA_THRESHOLD:
                continue

            size = _compute_size(
                bankroll=bankroll,
                market_prob=market_prob if action == "BUY_YES" else (1.0 - market_prob),
                win_prob=win_prob,
                edge=effective_edge,
                rounding_confidence=rounding_confidence,
            )
            if size < max(MIN_ORDER_USD, PRACTICAL_MIN_ORDER_USD):
                continue

            token_id = token_info["yes_token_id"] if action == "BUY_YES" else token_info["no_token_id"]
            signals.append(
                Signal(
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
                )
            )

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
) -> float:
    if bankroll <= FIXED_SIZE_BANKROLL_THRESHOLD:
        return FIXED_ORDER_USD
    return kelly_size(
        market_price=market_prob,
        win_prob=win_prob,
        bankroll=bankroll,
        edge=edge,
        kelly_fraction=KELLY_FRACTION,
        max_position=MAX_POSITION_SIZE,
        rounding_confidence=rounding_confidence,
    )


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

            if bankroll <= FIXED_SIZE_BANKROLL_THRESHOLD:
                size = FIXED_ORDER_USD
            else:
                trade_price = market_prob if action == "BUY_YES" else (1.0 - market_prob)
                size = kelly_size(
                    market_price=trade_price,
                    win_prob=win_prob,
                    bankroll=bankroll,
                    edge=effective_edge,
                    kelly_fraction=KELLY_FRACTION,
                    max_position=MAX_POSITION_SIZE,
                    rounding_confidence=rounding_confidence,
                )

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
