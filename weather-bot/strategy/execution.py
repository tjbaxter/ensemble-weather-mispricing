"""Prime Alpha V3.5 Execution Layer.

Separates MODEL SELECTION (which buckets are live candidates) from
TRADE EXECUTION (which candidates are actually worth buying at current prices).

V3.5: Layer B always outputs exactly 2 candidate buckets (center + adjacent).
This module passes both through — the combined package edge check in
signals.py decides whether the pair is worth trading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_LOG = logging.getLogger("weather-bot.execution")

# ── Execution constants ─────────────────────────────────────────────────────────

EXEC_MAX_BUCKETS_DEFAULT = 2
EXEC_ALLOW_THIRD_BUCKET = False

THIRD_BUCKET_MIN_EDGE = 0.05
THIRD_BUCKET_MAX_PRICE = 0.18
THIRD_BUCKET_MIN_REGIME = 0.35

SECOND_BUCKET_MIN_EDGE = -1.0  # V3.5: always pass both buckets; package edge decides
MAIN_BUCKET_MIN_EDGE = 0.00

CENTER_PRIORITY_BONUS = 0.02

# Bridge package retention: if the full 3-bucket bridge package edge is within
# this tolerance of the best 2-bucket package, keep all 3.
BRIDGE_PACKAGE_TOLERANCE = 0.01


@dataclass
class ExecutionDecision:
    """Result of the execution filter applied to model candidate buckets."""

    candidate_buckets: list[str]
    execution_buckets: list[str] = field(default_factory=list)
    main_bucket: str | None = None
    second_bucket: str | None = None
    third_bucket: str | None = None
    reason_main: str = ""
    reason_second: str = ""
    reason_third: str = ""
    per_bucket_edge: dict[str, float] = field(default_factory=dict)
    total_package_cost: float = 0.0
    total_package_edge: float = 0.0
    bridge_structure_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_buckets": self.candidate_buckets,
            "execution_buckets": self.execution_buckets,
            "main_bucket": self.main_bucket,
            "second_bucket": self.second_bucket,
            "third_bucket": self.third_bucket,
            "reason_main": self.reason_main,
            "reason_second": self.reason_second,
            "reason_third": self.reason_third,
            "per_bucket_edge": self.per_bucket_edge,
            "total_package_cost": round(self.total_package_cost, 4),
            "total_package_edge": round(self.total_package_edge, 4),
            "bridge_structure_detected": self.bridge_structure_detected,
        }


def _parse_bucket_low(bucket: str) -> float:
    """Extract the lower bound of a bucket label for ordering."""
    clean = bucket.replace("°F", "").replace("°C", "").replace("≥", "").replace("≤", "").strip()
    if "-" in clean:
        return float(clean.split("-")[0])
    try:
        return float(clean)
    except ValueError:
        return 0.0


def _detect_bridge_structure(
    candidate_buckets: list[str],
    selection_layer: dict[str, Any],
) -> bool:
    """Detect a genuine bridge structure using two methods.

    Method A: selection_layer explicitly recorded a bridge bucket.
    Method B: derive from ordered candidate indices -- three consecutive
              bucket positions with the middle one bridging a gap.
    """
    if selection_layer.get("bridge"):
        return True

    if len(candidate_buckets) < 3:
        return False

    sorted_buckets = sorted(candidate_buckets, key=_parse_bucket_low)
    lows = [_parse_bucket_low(b) for b in sorted_buckets]

    for i in range(len(lows) - 2):
        gap_01 = lows[i + 1] - lows[i]
        gap_12 = lows[i + 2] - lows[i + 1]
        if gap_01 == gap_12 and gap_01 > 0:
            return True

    return False


def _package_edge(
    buckets: list[str],
    prices: dict[str, float],
    probs: dict[str, float],
) -> float:
    model_p = min(0.95, sum(probs.get(b, 0.0) for b in buckets))
    cost = sum(prices.get(b, 0.0) for b in buckets)
    return model_p - cost


def apply_execution_filter(
    *,
    candidate_buckets: list[str],
    bucket_market_prices: dict[str, float],
    bucket_model_probs: dict[str, float],
    selection_layer: dict[str, Any],
    regime_strength: float,
) -> ExecutionDecision:
    """Filter model candidate buckets down to execution buckets.

    For genuine bridge structures, evaluates PACKAGE EV rather than
    independent bucket EV. For non-bridge cases, selects 1-2 buckets
    by individual edge.
    """
    decision = ExecutionDecision(candidate_buckets=list(candidate_buckets))

    if not candidate_buckets:
        return decision

    edges: dict[str, float] = {}
    for b in candidate_buckets:
        mp = bucket_model_probs.get(b, 0.0)
        mkt = bucket_market_prices.get(b, 0.0)
        edges[b] = round(mp - mkt, 6)
    decision.per_bucket_edge = edges

    bridge = _detect_bridge_structure(candidate_buckets, selection_layer)
    decision.bridge_structure_detected = bridge

    center_bucket = selection_layer.get("center_bucket")

    # ── Bridge package path ─────────────────────────────────────────────────
    # For genuine bridge structures with 3 candidates, evaluate as a coherent
    # package: anchor + bridge + center. Do NOT let a tiny standalone negative
    # edge on the anchor veto a structurally selected bucket.
    if bridge and len(candidate_buckets) == 3:
        anchor_info = selection_layer.get("lower_anchor")
        bridge_bucket = selection_layer.get("bridge")
        anchor_bucket = (
            anchor_info.get("bucket")
            if isinstance(anchor_info, dict) else None
        )

        has_structural_roles = (
            center_bucket and anchor_bucket and bridge_bucket
            and center_bucket in candidate_buckets
            and anchor_bucket in candidate_buckets
            and bridge_bucket in candidate_buckets
        )

        if has_structural_roles:
            full_pkg = [anchor_bucket, bridge_bucket, center_bucket]
            pkg_center_bridge = [center_bucket, bridge_bucket]
            pkg_anchor_bridge = [anchor_bucket, bridge_bucket]
            pkg_center_only = [center_bucket]

            full_ev = _package_edge(full_pkg, bucket_market_prices, bucket_model_probs)
            cb_ev = _package_edge(pkg_center_bridge, bucket_market_prices, bucket_model_probs)
            ab_ev = _package_edge(pkg_anchor_bridge, bucket_market_prices, bucket_model_probs)
            best_2_ev = max(cb_ev, ab_ev)

            if full_ev >= best_2_ev - BRIDGE_PACKAGE_TOLERANCE:
                decision.execution_buckets = list(full_pkg)
                decision.main_bucket = center_bucket
                decision.second_bucket = anchor_bucket
                decision.third_bucket = bridge_bucket
                decision.reason_main = "bridge_package_center"
                decision.reason_second = "bridge_package_anchor"
                decision.reason_third = (
                    f"bridge_package_retained:"
                    f"full_ev={full_ev:+.4f},best_2={best_2_ev:+.4f},"
                    f"gap={best_2_ev - full_ev:.4f}<={BRIDGE_PACKAGE_TOLERANCE}"
                )
                _finalise(decision, bucket_market_prices, bucket_model_probs)
                return decision
            else:
                _LOG.info(
                    "Bridge package rejected: full_ev=%+.4f best_2=%+.4f "
                    "gap=%.4f > tolerance=%.4f",
                    full_ev, best_2_ev, best_2_ev - full_ev,
                    BRIDGE_PACKAGE_TOLERANCE,
                )

    # ── Non-bridge path (or bridge package rejected) ────────────────────────

    # Step 1: Pick main bucket
    effective_edges: dict[str, float] = {}
    for b in candidate_buckets:
        bonus = CENTER_PRIORITY_BONUS if (b == center_bucket) else 0.0
        effective_edges[b] = edges.get(b, 0.0) + bonus

    best_main = max(candidate_buckets, key=lambda b: effective_edges[b])
    raw_edge_best = edges.get(best_main, 0.0)

    if raw_edge_best < MAIN_BUCKET_MIN_EDGE and not any(
        edges.get(b, 0.0) >= MAIN_BUCKET_MIN_EDGE for b in candidate_buckets
    ):
        decision.reason_main = "no_candidate_clears_min_edge"
        _finalise(decision, bucket_market_prices, bucket_model_probs)
        return decision

    decision.main_bucket = best_main
    decision.execution_buckets.append(best_main)

    if best_main == center_bucket:
        if edges.get(best_main, 0.0) >= max(
            (edges.get(b, 0.0) for b in candidate_buckets if b != best_main),
            default=-999,
        ):
            decision.reason_main = "center_best_edge"
        else:
            decision.reason_main = "center_structural_priority"
    else:
        decision.reason_main = "better_ev_override"

    # Step 2: Pick second bucket
    remaining = [b for b in candidate_buckets if b != best_main]
    if remaining:
        best_second = max(remaining, key=lambda b: edges.get(b, 0.0))
        second_edge = edges.get(best_second, 0.0)

        if second_edge >= SECOND_BUCKET_MIN_EDGE:
            close_candidates = [
                b for b in remaining
                if abs(edges.get(b, 0.0) - second_edge) <= 0.02
            ]
            if len(close_candidates) > 1:
                main_low = _parse_bucket_low(best_main)
                best_second = min(
                    close_candidates,
                    key=lambda b: abs(_parse_bucket_low(b) - main_low),
                )
                decision.reason_second = "best_edge_adjacent_tiebreak"
            else:
                decision.reason_second = "best_edge"

            decision.second_bucket = best_second
            decision.execution_buckets.append(best_second)
        else:
            decision.reason_second = (
                f"skipped_edge={second_edge:.4f}<{SECOND_BUCKET_MIN_EDGE}"
            )

    # Step 3: Third bucket (non-bridge path -- off by default)
    remaining_after_second = [
        b for b in candidate_buckets if b not in decision.execution_buckets
    ]
    if remaining_after_second:
        third_candidate = remaining_after_second[0]
        third_edge = edges.get(third_candidate, 0.0)
        third_price = bucket_market_prices.get(third_candidate, 1.0)

        if not EXEC_ALLOW_THIRD_BUCKET:
            decision.reason_third = "off_by_default"
        elif third_edge < THIRD_BUCKET_MIN_EDGE:
            decision.reason_third = (
                f"skipped_edge={third_edge:.4f}<{THIRD_BUCKET_MIN_EDGE}"
            )
        elif third_price > THIRD_BUCKET_MAX_PRICE:
            decision.reason_third = (
                f"skipped_price={third_price:.4f}>{THIRD_BUCKET_MAX_PRICE}"
            )
        elif not bridge and regime_strength < THIRD_BUCKET_MIN_REGIME:
            decision.reason_third = (
                f"skipped_no_bridge_regime={regime_strength:.2f}"
                f"<{THIRD_BUCKET_MIN_REGIME}"
            )
        else:
            decision.third_bucket = third_candidate
            decision.execution_buckets.append(third_candidate)
            reasons = []
            if bridge:
                reasons.append("bridge_structure")
            if regime_strength >= THIRD_BUCKET_MIN_REGIME:
                reasons.append(f"regime={regime_strength:.2f}")
            decision.reason_third = "accepted:" + "+".join(reasons)
    else:
        decision.reason_third = "no_remaining_candidates"

    _finalise(decision, bucket_market_prices, bucket_model_probs)
    return decision


def _finalise(
    decision: ExecutionDecision,
    prices: dict[str, float],
    probs: dict[str, float],
) -> None:
    """Compute total package cost and edge for the execution set."""
    cost = sum(prices.get(b, 0.0) for b in decision.execution_buckets)
    model_p = min(0.95, sum(probs.get(b, 0.0) for b in decision.execution_buckets))
    decision.total_package_cost = cost
    decision.total_package_edge = round(model_p - cost, 6)
