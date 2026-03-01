"""Temperature ladder trade construction.

Rules (Tom's design):
 - The ladder flanks the model's favourite bucket — never more than 3 YES bets.
 - If the model is clearly committed to one bucket: buy center ± 1 (3 buckets).
 - If the model is torn between two adjacent buckets (2nd prob ≥ TORN_RATIO of
   the 1st): buy just those two adjacent buckets (2 buckets).
 - Hard cap: max 3 buckets, max 3 YES bets.
"""

from __future__ import annotations

from data.probability import parse_bucket_bounds

# Model is "torn" when the 2nd-highest adjacent bucket has ≥ this fraction of
# the top bucket's probability.  Tune between 0.60 (loose) and 0.75 (strict).
TORN_RATIO = 0.65


def _adjacent(b1: str, b2: str, ordered: list[str]) -> bool:
    """Return True when the two buckets sit next to each other in bucket order."""
    try:
        return abs(ordered.index(b1) - ordered.index(b2)) == 1
    except ValueError:
        return False


def create_ladder(
    ensemble_probs: dict[str, float],
    market_prices: dict[str, float],
    center_bucket: str,
    width: int = 3,           # kept for API compatibility — ignored internally
    max_total_cost: float = 0.85,
    min_edge: float = 0.08,
) -> list[dict] | None:
    """Construct a ladder around the model's favourite bucket.

    Returns a list of {bucket, side, price, model_prob, ladder_total_cost,
    ladder_total_prob, ladder_edge} dicts, or None when no trade fires.
    """
    if center_bucket not in market_prices:
        return None

    # Buckets sorted by lower-bound temperature
    ordered = sorted(market_prices.keys(), key=lambda b: parse_bucket_bounds(b)[0])
    center_idx = ordered.index(center_bucket)

    # Sort all buckets by model probability (desc) to detect the torn case
    by_prob = sorted(
        [(b, ensemble_probs.get(b, 0.0)) for b in ordered if b in market_prices],
        key=lambda x: x[1],
        reverse=True,
    )
    top_bucket,  top_prob  = by_prob[0]
    sec_bucket,  sec_prob  = by_prob[1] if len(by_prob) > 1 else (None, 0.0)

    # ── Choose ladder buckets ─────────────────────────────────────────────────
    torn = (
        sec_bucket is not None
        and sec_prob > 0
        and (sec_prob / top_prob) >= TORN_RATIO
        and _adjacent(top_bucket, sec_bucket, ordered)
    )

    if torn:
        # Model genuinely uncertain between two adjacent buckets → bet both only
        ladder_buckets = sorted([top_bucket, sec_bucket], key=lambda b: ordered.index(b))
    else:
        # Model has a clear favourite → flank it with one bucket each side
        start = max(0, center_idx - 1)
        end   = min(len(ordered), center_idx + 2)   # center ± 1 → 3 buckets
        ladder_buckets = ordered[start:end]

    # Hard cap: never more than 3 YES buckets
    if len(ladder_buckets) > 3:
        # Keep center + nearest neighbours
        if center_bucket in ladder_buckets:
            ci = ladder_buckets.index(center_bucket)
            ladder_buckets = ladder_buckets[max(0, ci - 1): ci + 2]
        else:
            ladder_buckets = ladder_buckets[:3]

    if not ladder_buckets:
        return None

    total_cost = sum(float(market_prices.get(b, 0.0) or 0.0) for b in ladder_buckets)
    total_prob = sum(float(ensemble_probs.get(b, 0.0) or 0.0) for b in ladder_buckets)
    edge = total_prob - total_cost

    if total_cost >= max_total_cost or edge < min_edge:
        return None

    return [
        {
            "bucket":             bucket,
            "side":               "BUY_YES",
            "price":              float(market_prices.get(bucket, 0.0) or 0.0),
            "model_prob":         float(ensemble_probs.get(bucket, 0.0) or 0.0),
            "ladder_total_cost":  total_cost,
            "ladder_total_prob":  total_prob,
            "ladder_edge":        edge,
        }
        for bucket in ladder_buckets
    ]
