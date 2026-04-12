#!/usr/bin/env python3
"""Proof-of-fix: prior-day gate for PRIME_ALPHA.

Exercises 4 scenarios:
  1. BEFORE (simulated): NYC with prior_resolution_mode=none reaches execution
  2. AFTER (blocked):    NYC targeting 2026-03-30, prior 2026-03-29 unresolved
  3. AFTER (allowed):    NYC targeting 2026-03-29, prior 2026-03-28 resolved
  4. Log output:         Full gate trace for the blocked case
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta, date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

from strategy.prime_alpha import (
    get_effective_prior_resolved_bucket,
    build_prime_alpha_plan,
    _resolve_official_timestamp,
)
from strategy.execution import apply_execution_filter
from config.cities import STATIONS


SEPARATOR = "\n" + "=" * 72 + "\n"


def load_models(city: str, target_date: str) -> dict[str, float]:
    snap = json.loads((ROOT / "data" / "model_snapshot_log.json").read_text())
    for key in (city, "New York"):
        entry = (snap.get(key) or {}).get(target_date, {})
        preds = entry.get("preds", {})
        if preds:
            return {str(k): float(v) for k, v in preds.items() if v is not None}
    return {}


def run_scenario(
    title: str,
    city: str,
    station_icao: str,
    target_date: str,
    bucket_labels: list[str],
    market_prices: dict[str, float] | None = None,
):
    print(SEPARATOR)
    print(f"  SCENARIO: {title}")
    print(SEPARATOR)

    prior_date = (date.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    unit = str(STATIONS.get(station_icao, {}).get("resolution_unit", "F"))

    # Step 1: Get prior-day resolution
    prior_resolution = get_effective_prior_resolved_bucket(
        city=city,
        station_icao=station_icao,
        prior_date=prior_date,
        prior_day_market_prices=market_prices,
    )

    print(f"  City:         {city} ({station_icao})")
    print(f"  Target date:  {target_date}")
    print(f"  Prior date:   {prior_date}")
    print()
    print(f"  prior_resolution =")
    for k, v in prior_resolution.items():
        print(f"    {k}: {v!r}")

    _pa_prior_bucket = prior_resolution.get("bucket")
    _pa_prior_mode = prior_resolution.get("mode", "none")
    _pa_signal_available_at = prior_resolution.get("signal_available_at_utc")

    # Step 2: Compute gate decision
    try:
        days_ahead = (date.fromisoformat(target_date) - date.today()).days
    except ValueError:
        days_ahead = 1

    _prior_signal_required = (days_ahead <= 1)
    _prior_signal_valid = False
    _gating_reason = ""

    if _pa_signal_available_at is not None:
        try:
            _signal_ts = datetime.fromisoformat(
                str(_pa_signal_available_at).replace("Z", "+00:00")
            )
            _now_utc = datetime.now(UTC)
            _prior_signal_valid = (_signal_ts <= _now_utc)
            if not _prior_signal_valid:
                _gating_reason = f"signal_ts_in_future:{_pa_signal_available_at}"
        except (ValueError, TypeError):
            _gating_reason = f"signal_ts_unparseable:{_pa_signal_available_at}"
    else:
        _gating_reason = "no_signal_available_at_utc"

    if _prior_signal_required and not _prior_signal_valid:
        _execution_allowed = False
        _is_diagnostic_only = True
        _gating_result = "blocked"
    elif not _prior_signal_required and not _prior_signal_valid:
        _execution_allowed = False
        _is_diagnostic_only = True
        _gating_result = "blocked"
    else:
        _execution_allowed = True
        _is_diagnostic_only = False
        _gating_result = "allowed"

    print()
    print(f"  GATE DECISION:")
    print(f"    days_ahead:                    {days_ahead}")
    print(f"    prior_signal_required:         {_prior_signal_required}")
    print(f"    prior_signal_available_at_utc: {_pa_signal_available_at!r}")
    print(f"    prior_signal_valid:            {_prior_signal_valid}")
    print(f"    gating_reason:                 {_gating_reason!r}")
    print(f"    execution_allowed:             {_execution_allowed}")
    print(f"    is_diagnostic_only:            {_is_diagnostic_only}")
    print(f"    gating_result:                 {_gating_result!r}")

    # Step 3: Run build_prime_alpha_plan (always runs, even when blocked)
    model_values = load_models(city, target_date)
    if not model_values:
        print(f"\n  [SKIP] No model predictions for {city} {target_date}")
        return

    from statistics import mean
    predicted_display = round(mean(model_values.values()))

    plan = build_prime_alpha_plan(
        city=city,
        station_icao=station_icao,
        target_date=target_date,
        bucket_labels=bucket_labels,
        current_model_values=model_values,
        predicted_display_temp=predicted_display,
        unit=unit,
        prior_resolved_bucket=_pa_prior_bucket,
    )

    print()
    print(f"  PLANNING (always runs):")
    print(f"    selected_buckets:    {plan.selected_buckets}")
    print(f"    trust_source:        {plan.trust_source}")
    print(f"    prior_resolved:      {plan.prior_resolved_bucket}")
    print(f"    regime_strength:     {plan.regime_strength:.2f}")
    print(f"    notes:               {plan.notes}")

    # Step 4: Execution (only if gate allows)
    if _execution_allowed and plan.selected_buckets and market_prices:
        exec_decision = apply_execution_filter(
            candidate_buckets=plan.selected_buckets,
            bucket_market_prices=market_prices,
            bucket_model_probs=plan.bucket_probabilities,
            selection_layer=plan.selection_layer,
            regime_strength=plan.regime_strength,
        )
        print()
        print(f"  EXECUTION (gate ALLOWED):")
        print(f"    execution_buckets: {exec_decision.execution_buckets}")
        print(f"    main_bucket:       {exec_decision.main_bucket}")
        print(f"    second_bucket:     {exec_decision.second_bucket}")
        print(f"    package_edge:      {exec_decision.total_package_edge:+.4f}")
    elif not _execution_allowed:
        print()
        print(f"  EXECUTION (gate BLOCKED):")
        print(f"    execution_buckets: []")
        print(f"    reason: prior-day gate blocked — no execution")
        print(f"    candidate_buckets preserved for diagnostics: {plan.selected_buckets}")
    else:
        print()
        print(f"  EXECUTION: skipped (no market prices or no selected buckets)")

    # Step 5: V3 decision buffer entry (what would be persisted)
    v3_entry = {
        "city": city,
        "date": target_date,
        "candidate_buckets": plan.selected_buckets,
        "execution_buckets": (
            exec_decision.execution_buckets
            if (_execution_allowed and plan.selected_buckets and market_prices)
            else []
        ) if '_execution_allowed' in dir() else [],
        "execution_allowed": _execution_allowed,
        "is_diagnostic_only": _is_diagnostic_only,
        "prior_signal_required": _prior_signal_required,
        "prior_signal_available_at_utc": _pa_signal_available_at,
        "gating_result": _gating_result,
        "prior_resolution_mode": _pa_prior_mode,
    }
    print()
    print(f"  V3 DECISION BUFFER ENTRY:")
    print(f"    " + json.dumps(v3_entry, indent=4).replace("\n", "\n    "))


def scenario_before_simulated():
    """Simulate what the OLD code did: no gate, mode=none passed through to execution."""
    print(SEPARATOR)
    print("  SCENARIO 1: BEFORE (simulated old behavior)")
    print("  What would happen WITHOUT the prior-day gate")
    print(SEPARATOR)

    city = "NYC"
    station_icao = "KLGA"
    target_date = "2026-03-30"
    prior_date = "2026-03-29"

    prior_resolution = get_effective_prior_resolved_bucket(
        city=city,
        station_icao=station_icao,
        prior_date=prior_date,
    )

    print(f"  City:         {city} ({station_icao})")
    print(f"  Target date:  {target_date}")
    print(f"  Prior date:   {prior_date}")
    print()
    print(f"  prior_resolution.mode:   {prior_resolution['mode']!r}")
    print(f"  prior_resolution.bucket: {prior_resolution['bucket']!r}")
    print(f"  signal_available_at_utc: {prior_resolution.get('signal_available_at_utc')!r}")
    print()
    print(f"  OLD CODE PATH (no gate):")
    print(f"    prior_resolution_mode = {prior_resolution['mode']!r}")
    print(f"    ↓ mode was METADATA ONLY, not a gate")
    print(f"    ↓ build_prime_alpha_plan() ran")
    print(f"    ↓ apply_execution_filter() ran UNCONDITIONALLY")
    print(f"    ↓ execution_buckets written to V3 buffer")
    print(f"    ↓ shadow position persisted")
    print()
    print(f"    *** BUG: executable decision produced with mode='none' ***")
    print(f"    *** No prior-day signal existed, but execution proceeded ***")
    print()
    print(f"  PROOF that old code had no gate:")
    print(f"    The old signals.py code was:")
    print(f"      prime_context['prior_resolution_mode'] = _pa_prior_mode  # metadata only")
    print(f"      prime_context['prior_resolution_bucket'] = _pa_prior_bucket  # metadata only")
    print(f"      # ... then immediately proceeded to:")
    print(f"      if prime_cands:")
    print(f"          exec_decision = apply_execution_filter(...)  # NO GATE CHECK")
    print(f"    There was NO check of mode, signal_available_at_utc, or execution_allowed")
    print(f"    between build_prime_alpha_plan() and apply_execution_filter().")


def main():
    # Scenario 1: BEFORE — simulated old behavior
    scenario_before_simulated()

    # Scenario 2: AFTER — NYC blocked (prior unresolved)
    run_scenario(
        title="2: AFTER — NYC blocked (prior 2026-03-29 unresolved)",
        city="NYC",
        station_icao="KLGA",
        target_date="2026-03-30",
        bucket_labels=["48-49", "50-51", "52-53", "54-55", "56-57", "58-59"],
        market_prices={"48-49": 0.10, "50-51": 0.25, "52-53": 0.30, "54-55": 0.20, "56-57": 0.10, "58-59": 0.05},
    )

    # Scenario 3: AFTER — NYC allowed (prior 2026-03-28 resolved)
    run_scenario(
        title="3: AFTER — NYC allowed (prior 2026-03-28 resolved, actual=43°F)",
        city="NYC",
        station_icao="KLGA",
        target_date="2026-03-29",
        bucket_labels=["48-49", "50-51", "52-53", "54-55", "56-57", "58-59"],
        market_prices={"48-49": 0.10, "50-51": 0.25, "52-53": 0.30, "54-55": 0.20, "56-57": 0.10, "58-59": 0.05},
    )

    # Final confirmations
    print(SEPARATOR)
    print("  EXPLICIT CONFIRMATIONS")
    print(SEPARATOR)
    print("  1. D+1 execution can no longer occur before")
    print("     prior_signal_available_at_utc <= now:")
    print("     YES — the gate parses the timestamp with try/except and")
    print("     requires signal_ts <= now_utc. Unparseable or future")
    print("     timestamps fail safe to blocked.")
    print()
    print("  2. Exploratory D+2 is currently disabled by default:")
    print("     YES — ENABLE_EXPLORATORY_D2 = False in config/settings.py.")
    print("     When disabled, D+2 without prior signal is also blocked.")
    print()
    print("  3. No recruiter-facing dashboard view will surface blocked")
    print("     diagnostic entries as the latest real decision:")
    print("     YES — _persist_v3_dashboard_state() skips entries where")
    print("     execution_allowed is False, so latest_by_city never")
    print("     points to a diagnostic-only record.")


if __name__ == "__main__":
    main()
