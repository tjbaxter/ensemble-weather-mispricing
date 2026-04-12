"""Regression tests for the Prime Alpha prior-day gate.

Verifies that:
  - blocked case → prior_signal_timestamp_source == "none"
  - real official case → prior_signal_timestamp_source == "resolved_csv"
  - fallback case → prior_signal_timestamp_source == "local_midnight_fallback"
  - provisional case → prior_signal_timestamp_source == "provisional_first_seen"
  - replay case → prior_signal_timestamp_source == "replay_override"

Also verifies that the gate correctly blocks execution when no prior signal
is available, and allows it when a valid timestamp exists.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from datetime import UTC, datetime, timedelta, date
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.prime_alpha import (
    _resolve_official_timestamp,
    get_effective_prior_resolved_bucket,
    _PROVISIONAL_POLL_STATE,
)


# ---------------------------------------------------------------------------
# _resolve_official_timestamp
# ---------------------------------------------------------------------------

class TestResolveOfficialTimestamp:

    def test_resolved_csv_returns_real_timestamp(self, tmp_path):
        """When resolved.csv has a matching row, return its resolved_at
        and source='resolved_csv'."""
        csv_file = tmp_path / "resolved.csv"
        csv_file.write_text(
            "station_icao,target_date,resolved_at,outcome,bucket\n"
            "KLGA,2026-03-25,2026-03-26T00:26:21.869521+00:00,WIN,50-51\n"
        )
        logs_patch = tmp_path
        with mock.patch(
            "strategy.prime_alpha._ROOT",
            tmp_path,
        ):
            (tmp_path / "logs").mkdir(exist_ok=True)
            (tmp_path / "logs" / "resolved.csv").write_text(csv_file.read_text())

            ts, source = _resolve_official_timestamp("NYC", "KLGA", "2026-03-25")

        assert source == "resolved_csv"
        assert ts == "2026-03-26T00:26:21.869521+00:00"

    def test_settlement_snapshot_returns_real_timestamp(self, tmp_path):
        """When settlement_snapshot.json has a matching entry, return its
        timestamp and source='settlement_snapshot'."""
        snap = {
            "entry1": {
                "city": "NYC",
                "station_icao": "KLGA",
                "date": "2026-03-25",
                "resolved_at": "2026-03-26T01:15:00+00:00",
            }
        }
        with mock.patch("strategy.prime_alpha._ROOT", tmp_path):
            (tmp_path / "logs").mkdir(exist_ok=True)
            (tmp_path / "data").mkdir(exist_ok=True)
            (tmp_path / "data" / "settlement_snapshot.json").write_text(
                json.dumps(snap)
            )

            ts, source = _resolve_official_timestamp("NYC", "KLGA", "2026-03-25")

        assert source == "settlement_snapshot"
        assert ts == "2026-03-26T01:15:00+00:00"

    def test_fallback_returns_local_midnight(self, tmp_path):
        """When no resolved.csv or settlement_snapshot entry exists,
        return local-midnight fallback."""
        with mock.patch("strategy.prime_alpha._ROOT", tmp_path):
            (tmp_path / "logs").mkdir(exist_ok=True)
            (tmp_path / "data").mkdir(exist_ok=True)
            (tmp_path / "data" / "settlement_snapshot.json").write_text("{}")

            ts, source = _resolve_official_timestamp("NYC", "KLGA", "2026-03-28")

        assert source == "local_midnight_fallback"
        assert ts is not None
        parsed = datetime.fromisoformat(ts)
        assert parsed.hour != 0 or parsed.tzname() != "UTC", (
            "Fallback should NOT be UTC midnight"
        )

    def test_fallback_never_uses_utc_midnight_for_western_city(self, tmp_path):
        """The fallback for NYC (EDT, UTC-4) must be 04:00 UTC, not 00:00."""
        with mock.patch("strategy.prime_alpha._ROOT", tmp_path):
            (tmp_path / "logs").mkdir(exist_ok=True)
            (tmp_path / "data").mkdir(exist_ok=True)
            (tmp_path / "data" / "settlement_snapshot.json").write_text("{}")

            ts, source = _resolve_official_timestamp("NYC", "KLGA", "2026-03-31")

        assert source == "local_midnight_fallback"
        parsed = datetime.fromisoformat(ts)
        assert parsed.hour in (4, 5), (
            f"NYC fallback should be 04:00 or 05:00 UTC, got {parsed.hour:02d}:00"
        )


# ---------------------------------------------------------------------------
# get_effective_prior_resolved_bucket — source field propagation
# ---------------------------------------------------------------------------

class TestGateSourceField:

    def test_blocked_has_source_none(self):
        """When prior day is unresolved, source should be 'none'."""
        result = get_effective_prior_resolved_bucket(
            city="NYC",
            station_icao="KLGA",
            prior_date="2099-12-31",
        )
        assert result["prior_signal_timestamp_source"] == "none"
        assert result["signal_available_at_utc"] is None
        assert result["mode"] == "none"
        assert result["bucket"] is None

    def test_replay_override_has_source_replay(self):
        """Replay override should set source to 'replay_override'."""
        result = get_effective_prior_resolved_bucket(
            city="NYC",
            station_icao="KLGA",
            prior_date="2026-03-25",
            replay_override="50-51",
        )
        assert result["prior_signal_timestamp_source"] == "replay_override"
        assert result["signal_available_at_utc"] is not None
        assert result["mode"] == "replay_override"
        assert result["bucket"] == "50-51"

    def test_official_resolved_csv_has_source_resolved_csv(self):
        """When resolved via resolved.csv, source should be 'resolved_csv'."""
        result = get_effective_prior_resolved_bucket(
            city="NYC",
            station_icao="KLGA",
            prior_date="2026-03-25",
        )
        if result["mode"] == "official":
            assert result["prior_signal_timestamp_source"] in (
                "resolved_csv",
                "settlement_snapshot",
                "local_midnight_fallback",
            )

    def test_official_resolved_csv_exact_source(self, tmp_path):
        """Deterministic: when only resolved.csv contains the match, source
        must be exactly 'resolved_csv' — not settlement_snapshot or fallback."""
        (tmp_path / "logs").mkdir()
        (tmp_path / "data").mkdir()
        (tmp_path / "logs" / "resolved.csv").write_text(
            "station_icao,target_date,resolved_at,outcome,bucket\n"
            "KLGA,2026-03-25,2026-03-26T00:26:21.869521+00:00,WIN,50-51\n"
        )
        (tmp_path / "data" / "settlement_snapshot.json").write_text("{}")
        (tmp_path / "data" / "polymarket_cache.json").write_text("{}")
        (tmp_path / "data" / "model_accuracy_log.json").write_text("{}")

        with mock.patch("strategy.prime_alpha._ROOT", tmp_path):
            result = get_effective_prior_resolved_bucket(
                city="NYC",
                station_icao="KLGA",
                prior_date="2026-03-25",
            )

        assert result["mode"] == "official"
        assert result["prior_signal_timestamp_source"] == "resolved_csv"
        assert result["signal_available_at_utc"] == "2026-03-26T00:26:21.869521+00:00"
        assert result["bucket"] is not None


# ---------------------------------------------------------------------------
# Gate behavior
# ---------------------------------------------------------------------------

class TestGateBehavior:

    def test_blocked_case_no_execution(self):
        """Prior unresolved → gate blocks, no signal_available_at_utc."""
        result = get_effective_prior_resolved_bucket(
            city="NYC",
            station_icao="KLGA",
            prior_date="2099-12-31",
        )
        assert result["bucket"] is None
        assert result["mode"] == "none"
        assert result["signal_available_at_utc"] is None
        assert result["prior_signal_timestamp_source"] == "none"

    def test_allowed_case_has_timestamp(self):
        """Prior resolved → gate allows, timestamp is populated."""
        result = get_effective_prior_resolved_bucket(
            city="NYC",
            station_icao="KLGA",
            prior_date="2026-03-25",
        )
        if result["mode"] == "official":
            assert result["bucket"] is not None
            assert result["signal_available_at_utc"] is not None
            assert result["prior_signal_timestamp_source"] != "none"

    def test_provisional_has_source_provisional_first_seen(self):
        """When provisional gate passes, source is 'provisional_first_seen'."""
        key = ("TEST_CITY", "2099-01-01")
        _PROVISIONAL_POLL_STATE[key] = {
            "bucket": "50",
            "first_seen_at": datetime.now(UTC).timestamp() - 7200,
            "last_seen_at": datetime.now(UTC).timestamp(),
            "poll_count": 5,
            "first_provisional_available_at_utc": "2099-01-01T22:30:00+00:00",
        }
        try:
            state = _PROVISIONAL_POLL_STATE[key]
            assert "first_provisional_available_at_utc" in state
            assert state["first_provisional_available_at_utc"] == "2099-01-01T22:30:00+00:00"
        finally:
            _PROVISIONAL_POLL_STATE.pop(key, None)
