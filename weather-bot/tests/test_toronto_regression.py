"""Regression tests for the Toronto Apr 1 bug and related fixes.

Covers:
  - Toronto bimodal selection: must produce ['10', '13'] with dashboard values
  - Broken parser scenario: only '14+' visible → PASS, not a live trade
  - Celsius display rounding: 10.3→10, not 11 (no F-roundtrip)
  - Gaussian fallback guard: distant tail buckets blocked
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.prime_alpha import (
    _market_display_temp,
    _round_half_up,
    build_prime_alpha_plan,
)

TORONTO_DASHBOARD_VALUES = {
    "ncep_nbm_conus": 10.3,
    "kma_gdps": 12.9,
    "meteofrance_arpege_world": 13.6,
    "gem_global": 12.5,
    "dmi_seamless": 13.3,
    "ncep_aigfs025": 9.9,
    "gem_regional": 14.4,
    "gfs_seamless": 10.0,
    "ecmwf_ifs025": 13.3,
}

FULL_CELSIUS_BUCKETS = [
    "<=4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14+",
]

# Historical predictions per model per date (from bot's snapshot log)
_HIST_PREDS = {
    "2026-03-31": {
        "ncep_nbm_conus": 13.3, "kma_gdps": 14.6,
        "meteofrance_arpege_world": 16.1, "gem_global": 18.9,
        "dmi_seamless": 17.0, "ncep_aigfs025": 17.1,
        "gem_regional": 19.7, "gfs_seamless": 11.0, "ecmwf_ifs025": 17.2,
    },
    "2026-03-30": {
        "ncep_nbm_conus": 15.5, "kma_gdps": 17.5,
        "meteofrance_arpege_world": 16.1, "gem_global": 16.7,
        "dmi_seamless": 18.8, "ncep_aigfs025": 18.2,
        "gem_regional": 18.0, "gfs_seamless": 19.0, "ecmwf_ifs025": 18.7,
    },
    "2026-03-29": {
        "ncep_nbm_conus": 9.9, "kma_gdps": 10.3,
        "meteofrance_arpege_world": 9.1, "gem_global": 8.8,
        "dmi_seamless": 9.9, "ncep_aigfs025": 12.3,
        "gem_regional": 9.3, "gfs_seamless": 11.5, "ecmwf_ifs025": 10.0,
    },
    "2026-03-28": {
        "ncep_nbm_conus": 1.3, "kma_gdps": 1.5,
        "meteofrance_arpege_world": 4.0, "gem_global": 1.3,
        "dmi_seamless": 1.1, "ncep_aigfs025": 0.6,
        "gem_regional": 2.0, "gfs_seamless": -0.1, "ecmwf_ifs025": 0.8,
    },
    "2026-03-27": {
        "ncep_nbm_conus": 2.6, "kma_gdps": 1.6,
        "meteofrance_arpege_world": 2.8, "gem_global": 1.7,
        "dmi_seamless": 1.1, "ncep_aigfs025": 1.8,
        "gem_regional": 0.9, "gfs_seamless": 2.4, "ecmwf_ifs025": 1.0,
    },
    "2026-03-26": {
        "ncep_nbm_conus": 11.4, "kma_gdps": 11.0,
        "meteofrance_arpege_world": 10.2, "gem_global": 11.2,
        "dmi_seamless": 14.8, "ncep_aigfs025": 12.3,
        "gem_regional": 14.1, "gfs_seamless": 13.9, "ecmwf_ifs025": 14.2,
    },
    "2026-03-25": {
        "ncep_nbm_conus": 5.1, "kma_gdps": 2.6,
        "meteofrance_arpege_world": 6.1, "gem_global": 4.3,
        "dmi_seamless": 4.2, "ncep_aigfs025": 7.2,
        "gem_regional": 4.5, "gfs_seamless": 5.0, "ecmwf_ifs025": 6.4,
    },
}

_RESOLVED_LABELS = {
    "2026-03-31": "\u22658",
    "2026-03-30": "\u226515",
    "2026-03-29": "\u226510",
    "2026-03-28": "1",
    "2026-03-27": "0",
    "2026-03-26": "10",
    "2026-03-25": "6",
}

ALL_MODELS = [
    "ncep_nbm_conus", "kma_gdps", "meteofrance_arpege_world",
    "gem_global", "dmi_seamless", "ncep_aigfs025",
    "gem_regional", "gfs_seamless", "ecmwf_ifs025",
]


def _build_accuracy_cache():
    """Build accuracy cache rows matching the dashboard's canonical hit/miss."""
    rows = []
    for dt, label in sorted(_RESOLVED_LABELS.items()):
        preds = _HIST_PREDS.get(dt, {})
        clean = label.replace("\u2265", ">=").replace("\u2264", "<=")

        if clean.startswith(">="):
            threshold = int(clean[2:])
            is_plus = True
            res_int = threshold
        elif clean.startswith("<="):
            threshold = int(clean[2:])
            is_plus = None
            res_int = threshold
        else:
            res_int = int(clean)
            is_plus = False

        row = {"date": dt, "resolved": label}
        for m in ALL_MODELS:
            val = preds.get(m)
            if val is None:
                continue
            p = _round_half_up(val)
            if is_plus is True:
                win = p >= res_int
            elif is_plus is None:
                win = p <= res_int
            else:
                win = p == res_int
            row[f"{m}_d1"] = val
            row[f"{m}_d1_win"] = win
        rows.append(row)
    return rows


def _mock_resolved_bucket(_city, day):
    return _RESOLVED_LABELS.get(day)


def _mock_prior_winner(station_icao, prior_date):
    return _RESOLVED_LABELS.get(prior_date)


def _mock_accuracy_entry(station_icao, row_date):
    preds = _HIST_PREDS.get(row_date, {})
    return {"preds": preds, "actual": None}


def _mock_snapshot_preds(_city, day):
    return _HIST_PREDS.get(day, {})


def _mock_commercial(_city, _day=None):
    return {}


def _mock_accuracy_cache(city):
    if city == "Toronto":
        return _build_accuracy_cache()
    return []


@pytest.fixture(autouse=True)
def _patch_data_loaders():
    """Mock all external data sources so tests are self-contained."""
    with (
        mock.patch(
            "strategy.prime_alpha._load_polymarket_resolved_bucket",
            side_effect=_mock_resolved_bucket,
        ),
        mock.patch(
            "strategy.prime_alpha._load_prior_day_winner_bucket",
            side_effect=_mock_prior_winner,
        ),
        mock.patch(
            "strategy.prime_alpha._load_model_accuracy_entry",
            side_effect=_mock_accuracy_entry,
        ),
        mock.patch(
            "strategy.prime_alpha._load_snapshot_log_preds",
            side_effect=_mock_snapshot_preds,
        ),
        mock.patch(
            "strategy.prime_alpha._load_commercial_forecast",
            side_effect=_mock_commercial,
        ),
        mock.patch(
            "strategy.prime_alpha._load_accuracy_cache",
            side_effect=_mock_accuracy_cache,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Celsius display rounding
# ---------------------------------------------------------------------------


class TestCelsiusRounding:
    """Celsius display must use direct rounding, matching the dashboard."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (10.3, 10),
            (9.9, 10),
            (10.0, 10),
            (13.3, 13),
            (12.5, 13),
            (13.6, 14),
            (0.4, 0),
            (0.5, 1),
            (-0.1, 0),
            (-0.5, 0),
            (-0.6, -1),
        ],
    )
    def test_celsius_direct_rounding(self, raw, expected):
        assert _market_display_temp(raw, "C") == expected

    def test_nbm_10_3_is_display_10(self):
        """The exact case that was broken: NBM 10.3°C must display as 10."""
        assert _market_display_temp(10.3, "C") == 10

    def test_fahrenheit_still_works(self):
        assert _market_display_temp(73.6, "F") == 74
        assert _market_display_temp(73.4, "F") == 73
        assert _market_display_temp(73.5, "F") == 74


# ---------------------------------------------------------------------------
# Toronto bimodal selection
# ---------------------------------------------------------------------------


class TestTorontoBimodal:
    """Toronto Apr 1 with full Celsius ladder and dashboard values."""

    def test_selects_10_and_13(self):
        plan = build_prime_alpha_plan(
            city="Toronto",
            station_icao="CYYZ",
            target_date="2026-04-01",
            bucket_labels=FULL_CELSIUS_BUCKETS,
            current_model_values=TORONTO_DASHBOARD_VALUES,
            predicted_display_temp=10,
            unit="C",
            prior_resolved_bucket="\u22658",
        )
        assert sorted(plan.selected_buckets) == ["10", "13"]

    def test_streak_scores_match_dashboard(self):
        plan = build_prime_alpha_plan(
            city="Toronto",
            station_icao="CYYZ",
            target_date="2026-04-01",
            bucket_labels=FULL_CELSIUS_BUCKETS,
            current_model_values=TORONTO_DASHBOARD_VALUES,
            predicted_display_temp=10,
            unit="C",
            prior_resolved_bucket="\u22658",
        )
        sl = plan.selection_layer
        scores = sl["streak_scores"]

        assert scores["dmi_seamless"] == 4
        assert scores["ecmwf_ifs025"] == 4
        assert scores["ncep_aigfs025"] == 4
        assert scores["ncep_nbm_conus"] == 4

        assert scores["kma_gdps"] < 4
        assert scores["meteofrance_arpege_world"] < 4

    def test_bimodal_clusters_are_10_and_13(self):
        plan = build_prime_alpha_plan(
            city="Toronto",
            station_icao="CYYZ",
            target_date="2026-04-01",
            bucket_labels=FULL_CELSIUS_BUCKETS,
            current_model_values=TORONTO_DASHBOARD_VALUES,
            predicted_display_temp=10,
            unit="C",
            prior_resolved_bucket="\u22658",
        )
        bm = plan.selection_layer.get("bimodal")
        assert bm is not None, "bimodal detection should fire"
        assert bm["lower"]["bucket"] == "10"
        assert bm["upper"]["bucket"] == "13"
        assert bm["gap"] >= 2.0

    def test_strong_streak_is_top_tier_only(self):
        plan = build_prime_alpha_plan(
            city="Toronto",
            station_icao="CYYZ",
            target_date="2026-04-01",
            bucket_labels=FULL_CELSIUS_BUCKETS,
            current_model_values=TORONTO_DASHBOARD_VALUES,
            predicted_display_temp=10,
            unit="C",
            prior_resolved_bucket="\u22658",
        )
        sl = plan.selection_layer
        strong = set(sl["strong_streak"])
        assert strong == {
            "dmi_seamless", "ecmwf_ifs025",
            "ncep_aigfs025", "ncep_nbm_conus",
        }


# ---------------------------------------------------------------------------
# Broken parser scenario: only 14+ visible
# ---------------------------------------------------------------------------


class TestBrokenParserPass:
    """If only '14+' is visible (simulating the old parser bug),
    the Gaussian guard must produce PASS, not a live trade."""

    def test_single_14plus_bucket_results_in_pass(self):
        plan = build_prime_alpha_plan(
            city="Toronto",
            station_icao="CYYZ",
            target_date="2026-04-01",
            bucket_labels=["14+"],
            current_model_values=TORONTO_DASHBOARD_VALUES,
            predicted_display_temp=10,
            unit="C",
            prior_resolved_bucket="\u22658",
        )
        has_pass = any("pass" in n.lower() for n in plan.notes)
        has_skip = any("gaussian_skip" in n for n in plan.notes)
        assert has_pass or has_skip or plan.selected_buckets == [], (
            f"Expected PASS or empty selection with only '14+' bucket, "
            f"got selected={plan.selected_buckets} notes={plan.notes}"
        )
        if plan.selected_buckets:
            assert "14+" not in plan.selected_buckets, (
                "14+ should NOT be selected when working center is ~10-13"
            )


# ---------------------------------------------------------------------------
# Streak quality scoring
# ---------------------------------------------------------------------------


class TestStreakQuality:

    def test_exact_bucket_higher_quality_than_threshold(self):
        plan = build_prime_alpha_plan(
            city="Toronto",
            station_icao="CYYZ",
            target_date="2026-04-01",
            bucket_labels=FULL_CELSIUS_BUCKETS,
            current_model_values=TORONTO_DASHBOARD_VALUES,
            predicted_display_temp=10,
            unit="C",
            prior_resolved_bucket="\u22658",
        )
        quality = plan.selection_layer["streak_quality"]
        for fam in ["dmi_seamless", "ncep_nbm_conus"]:
            assert quality[fam] > 0, f"{fam} should have positive quality"
            assert quality[fam] < 4 * 1.0, (
                "quality should be < 4.0 because coarse days score < 1.0"
            )
