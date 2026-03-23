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

    summary: dict[str, Any] = {
        "city": city,
        "station_icao": station_icao,
        "target_date": target_date,
        "predicted_display_temp": predicted_display_temp,
        "trust_source": plan.trust_source,
        "prior_resolved_bucket": plan.prior_resolved_bucket,
        "trusted_models": plan.trusted_models,
        "fallback_models": plan.fallback_models,
        "trusted_flagship": plan.trusted_flagship,
        "current_display_by_source": plan.current_display_by_source,
        "initial_selected_buckets": plan.initial_selected_buckets,
        "selected_buckets": plan.selected_buckets,
        "notes": plan.notes,
    }

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
        print(f"Trusted models: {', '.join(plan.trusted_models) or 'none'}")
        print(f"Trusted flagship: {'yes' if plan.trusted_flagship else 'no'}")
        print(f"Current displays: {json.dumps(plan.current_display_by_source, sort_keys=True)}")
        print(f"Initial range buckets: {', '.join(plan.initial_selected_buckets) or 'none'}")
        print(f"Selected PRIME_ALPHA buckets: {', '.join(plan.selected_buckets) or 'none'}")
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
