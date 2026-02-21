"""Temperature ladder trade construction."""

from __future__ import annotations

from data.probability import parse_bucket_bounds


def create_ladder(
    ensemble_probs: dict[str, float],
    market_prices: dict[str, float],
    center_bucket: str,
    width: int = 3,
    max_total_cost: float = 0.85,
    min_edge: float = 0.08,
) -> list[dict] | None:
    """Construct a ladder around center bucket when expected value is positive."""
    if center_bucket not in market_prices or width < 1:
        return None

    ordered = sorted(market_prices.keys(), key=lambda b: parse_bucket_bounds(b)[0])
    center_idx = ordered.index(center_bucket)
    half = width // 2
    start = max(0, center_idx - half)
    end = min(len(ordered), center_idx + half + 1)
    ladder_buckets = ordered[start:end]
    if not ladder_buckets:
        return None

    total_cost = sum(float(market_prices.get(b, 0.0) or 0.0) for b in ladder_buckets)
    total_prob = sum(float(ensemble_probs.get(b, 0.0) or 0.0) for b in ladder_buckets)
    edge = total_prob - total_cost

    if total_cost >= max_total_cost or edge < min_edge:
        return None

    out: list[dict] = []
    for bucket in ladder_buckets:
        out.append(
            {
                "bucket": bucket,
                "side": "BUY_YES",
                "price": float(market_prices.get(bucket, 0.0) or 0.0),
                "model_prob": float(ensemble_probs.get(bucket, 0.0) or 0.0),
                "ladder_total_cost": total_cost,
                "ladder_total_prob": total_prob,
                "ladder_edge": edge,
            }
        )
    return out
