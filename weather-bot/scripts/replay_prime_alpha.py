#!/usr/bin/env python3
"""Replay PRIME_ALPHA selection for a historical or fixture-driven scenario."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.cities import STATIONS  # noqa: E402
from strategy.prime_alpha import build_prime_alpha_plan  # noqa: E402
from strategy.execution import apply_execution_filter  # noqa: E402

MODEL_SNAPSHOT_LOG = ROOT / "data" / "model_snapshot_log.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", help="Replay from a JSON fixture file")
    parser.add_argument("--city", help="City key used in model_snapshot_log.json")
    parser.add_argument("--station-icao", help="Station ICAO, e.g. KATL")
    parser.add_argument("--date", help="Target date YYYY-MM-DD")
    parser.add_argument(
        "--bucket-labels",
        help="Comma-separated bucket labels for the market, e.g. 72-73,74-75,76-77,78-79",
    )
    parser.add_argument(
        "--expected-buckets",
        help="Optional comma-separated expected PRIME_ALPHA buckets to assert",
    )
    parser.add_argument(
        "--prior-resolved-bucket",
        help="Override the prior day's Polymarket-resolved bucket label (e.g. 82-83)",
    )
    parser.add_argument(
        "--market-prices",
        help=(
            "Comma-separated bucket=price pairs for execution layer testing, "
            "e.g. '11=0.45,12=0.30,13=0.15'"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")
    return parser


def load_fixture(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Fixture must be a JSON object.")
    return payload


def load_snapshot_models(city: str, target_date: str) -> dict[str, float]:
    payload = json.loads(MODEL_SNAPSHOT_LOG.read_text(encoding="utf-8"))
    city_rows = payload.get(city, {})
    if not isinstance(city_rows, dict):
        raise KeyError(f"No snapshot rows for city={city}")
    entry = city_rows.get(target_date, {})
    preds = entry.get("preds", {}) if isinstance(entry, dict) else {}
    if not isinstance(preds, dict) or not preds:
        raise KeyError(f"No snapshot predictions for {city} {target_date}")
    return {str(key): float(value) for key, value in preds.items() if value is not None}


def compute_display_temp_from_models(model_values: dict[str, float], unit: str) -> int | None:
    if not model_values:
        return None
    raw_mean = mean(float(value) for value in model_values.values())
    if str(unit).upper() == "F":
        return _round_half_up(raw_mean)
    temp_f = raw_mean * 9.0 / 5.0 + 32.0
    temp_f_rounded = _round_half_up(temp_f)
    temp_c_back = (temp_f_rounded - 32.0) * 5.0 / 9.0
    return _round_half_up(temp_c_back)


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_market_prices(raw: str | None) -> dict[str, float]:
    """Parse 'bucket=price,bucket=price' into a dict."""
    if not raw:
        return {}
    result: dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        bucket, price = pair.split("=", 1)
        result[bucket.strip()] = float(price.strip())
    return result


def build_input_payload(args: argparse.Namespace) -> tuple[dict[str, Any], list[str] | None]:
    if args.fixture:
        payload = load_fixture(args.fixture)
        expected = payload.get("expected_buckets")
        if isinstance(expected, list):
            expected_buckets = [str(item) for item in expected]
        else:
            expected_buckets = parse_csv_list(args.expected_buckets) or None
        return payload, expected_buckets

    if not all([args.city, args.station_icao, args.date, args.bucket_labels]):
        raise SystemExit(
            "--fixture or all of --city, --station-icao, --date, --bucket-labels are required"
        )

    station_icao = str(args.station_icao)
    unit = str(STATIONS.get(station_icao, {}).get("resolution_unit", "F"))
    model_values = load_snapshot_models(str(args.city), str(args.date))
    predicted_display = compute_display_temp_from_models(model_values, unit)
    payload = {
        "city": args.city,
        "station_icao": station_icao,
        "target_date": args.date,
        "bucket_labels": parse_csv_list(args.bucket_labels),
        "current_model_values": model_values,
        "predicted_display_temp": predicted_display,
    }
    if args.prior_resolved_bucket:
        payload["prior_resolved_bucket"] = args.prior_resolved_bucket
    expected_buckets = parse_csv_list(args.expected_buckets) or None
    return payload, expected_buckets


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload, expected_buckets = build_input_payload(args)

    city = str(payload["city"])
    station_icao = str(payload["station_icao"])
    target_date = str(payload["target_date"])
    bucket_labels = [str(item) for item in payload.get("bucket_labels", [])]
    current_model_values = {
        str(key): float(value)
        for key, value in dict(payload.get("current_model_values", {})).items()
        if value is not None
    }
    unit = str(STATIONS.get(station_icao, {}).get("resolution_unit", "F"))
    predicted_display_temp = payload.get("predicted_display_temp")
    trust_overrides = payload.get("trust_overrides")
    prior_resolved_bucket = payload.get("prior_resolved_bucket")
    market_prices = parse_market_prices(args.market_prices)

    plan = build_prime_alpha_plan(
        city=city,
        station_icao=station_icao,
        target_date=target_date,
        bucket_labels=bucket_labels,
        current_model_values=current_model_values,
        predicted_display_temp=predicted_display_temp,
        unit=unit,
        trust_overrides=trust_overrides if isinstance(trust_overrides, dict) else None,
        prior_resolved_bucket=str(prior_resolved_bucket) if prior_resolved_bucket else None,
    )

    # Determine prior-day resolution mode from what actually happened
    if prior_resolved_bucket:
        prior_mode = "replay_override"
    elif plan.prior_resolved_bucket:
        prior_mode = "official"
    else:
        prior_mode = "none"

    # Run execution layer if market prices provided
    exec_decision = None
    if market_prices and plan.selected_buckets:
        exec_decision = apply_execution_filter(
            candidate_buckets=plan.selected_buckets,
            bucket_market_prices=market_prices,
            bucket_model_probs=plan.bucket_probabilities,
            selection_layer=plan.selection_layer,
            regime_strength=plan.regime_strength,
        )

    summary: dict[str, Any] = {
        "city": city,
        "station_icao": station_icao,
        "target_date": target_date,
        "predicted_display_temp": predicted_display_temp,
        "trust_source": plan.trust_source,
        "prior_resolved_bucket": plan.prior_resolved_bucket,
        "prior_resolution_mode": prior_mode,
        "trusted_models": plan.trusted_models,
        "trust_scores": plan.trust_scores,
        "all_model_bucket_counts": plan.all_model_bucket_counts,
        "current_display_by_source": plan.current_display_by_source,
        "initial_selected_buckets": plan.initial_selected_buckets,
        "selected_buckets": plan.selected_buckets,
        "notes": plan.notes,
        "families": plan.families,
        "diagnostics_only": plan.diagnostics_only,
        "beta_used": plan.beta_used,
        "regime_strength": plan.regime_strength,
        "bucket_probabilities": plan.bucket_probabilities,
        "selection_layer": plan.selection_layer,
    }

    if exec_decision:
        summary["execution_decision"] = exec_decision.to_dict()

    true_alpha_buckets = payload.get("current_true_alpha_buckets")
    if isinstance(true_alpha_buckets, list):
        summary["current_true_alpha_buckets"] = [str(item) for item in true_alpha_buckets]

    if expected_buckets:
        summary["expected_buckets"] = expected_buckets
        summary["matches_expected"] = plan.selected_buckets == expected_buckets

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"City/date: {city} {target_date} ({station_icao})")
        print(f"Trust source: {plan.trust_source}")
        print(f"Prior resolved bucket: {plan.prior_resolved_bucket}")
        print(f"Prior-day resolution: {prior_mode}")
        print(f"Regime strength: {plan.regime_strength:.2f} | \u03b2(R): {plan.beta_used:.2f}")
        print()

        # ── Scoring layer ──
        print("\u2550\u2550 LAYER A: Scoring \u2550\u2550")
        mu_hdr = "\u03bc"
        sigma_hdr = "\u03c3"
        print(f"  {'Family':20s}  {mu_hdr:>6s}  {sigma_hdr:>5s}  {'w':>6s}  {'long':>6s}  {'short':>6s}  {'raw':>6s}  Members")
        for fam, info in plan.families.items():
            tag = " [SUPPRESSED]" if info.get("suppressed") else ""
            print(
                f"  {fam:20s}  {info['forecast']:6.1f}  "
                f"{info['sigma']:5.2f}  {info['weight']:.3f}  "
                f"{info['long_skill']:.3f}  {info['short_skill']:.3f}  "
                f"{info['raw_score']:.3f}  "
                f"{info['members']}{tag}"
            )
        if plan.diagnostics_only:
            print()
            print("\u2550\u2550 Diagnostic Only (not in mixture) \u2550\u2550")
            for key, info in plan.diagnostics_only.items():
                print(f"  {key:20s}  \u03bc={info['forecast']:6.1f}")

        # ── Selection layer ──
        sel = plan.selection_layer
        print()
        print("\u2550\u2550 LAYER B: Deterministic Selection \u2550\u2550")
        rw = sel.get("recent_winners", {})
        print(f"  Recent winners:  {rw if rw else 'none'}")
        print(f"  Winner center:   {sel.get('winner_center', '-')}")
        rs = sel.get("recent_support", {})
        print(f"  Recent support:  {rs if rs else 'none'}")
        print(f"  Working set:     {sel.get('working_set', [])}")
        print(f"  Working center:  {sel.get('working_center', '-')}")
        pc = sel.get("promoted_commercial", {})
        if pc:
            print(f"  Promoted comm:   {pc}")
        ho = sel.get("high_outlier_winners", [])
        if ho:
            print(f"  High outliers:   {ho}")
        cb = sel.get("center_bucket")
        if cb:
            print(f"  Center bucket:   {cb}")
        la = sel.get("lower_anchor")
        if la:
            print(f"  Lower anchor:    {la['bucket']} (via {la['family']})")
        br = sel.get("bridge")
        if br:
            print(f"  Bridge bucket:   {br}")
        ua = sel.get("upper_anchor")
        if ua:
            print(f"  Upper anchor:    {ua['bucket']} (via {ua['families']})")

        # ── Gaussian diagnostics ──
        print()
        print("\u2550\u2550 Gaussian Diagnostics (NOT used for selection) \u2550\u2550")
        for b, p in sorted(plan.bucket_probabilities.items(), key=lambda x: -x[1]):
            marker = " \u25c0" if b in plan.selected_buckets else ""
            print(f"  {b:10s}  P={p:.4f}  ({p*100:.1f}%){marker}")

        print()
        print(f"Model candidate buckets: {', '.join(plan.selected_buckets) or 'none'}")

        # ── Execution layer ──
        if exec_decision:
            print()
            print("\u2550\u2550 EXECUTION LAYER \u2550\u2550")
            print(f"  {'Bucket':10s}  {'Model P':>8s}  {'Market':>8s}  {'Edge':>8s}  {'Buy?':>5s}")
            for b in exec_decision.candidate_buckets:
                mp = plan.bucket_probabilities.get(b, 0.0)
                mkt = market_prices.get(b, 0.0)
                edge = exec_decision.per_bucket_edge.get(b, 0.0)
                bought = "YES" if b in exec_decision.execution_buckets else "no"
                print(f"  {b:10s}  {mp:8.4f}  {mkt:8.4f}  {edge:+8.4f}  {bought:>5s}")
            print()
            print(f"  Main bucket:     {exec_decision.main_bucket or '-'} ({exec_decision.reason_main})")
            print(f"  Second bucket:   {exec_decision.second_bucket or '-'} ({exec_decision.reason_second})")
            print(f"  Third bucket:    {exec_decision.third_bucket or '-'} ({exec_decision.reason_third})")
            print(f"  Bridge detected: {'yes' if exec_decision.bridge_structure_detected else 'no'}")
            print(f"  Package cost:    {exec_decision.total_package_cost:.4f}")
            print(f"  Package edge:    {exec_decision.total_package_edge:+.4f}")
            print()
            print(f"Execution buckets to buy: {', '.join(exec_decision.execution_buckets) or 'none'}")
        else:
            if not market_prices:
                print("  (no --market-prices provided, execution layer skipped)")

        if "current_true_alpha_buckets" in summary:
            print(
                "Current TRUE_ALPHA buckets: "
                + ", ".join(summary["current_true_alpha_buckets"])
            )
        if expected_buckets:
            status = "MATCH" if summary["matches_expected"] else "MISMATCH"
            print(f"Expected buckets: {', '.join(expected_buckets)} -> {status}")
        if plan.notes:
            print(f"Notes: {' | '.join(plan.notes)}")

    if expected_buckets and plan.selected_buckets != expected_buckets:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
