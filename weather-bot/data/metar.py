"""METAR data client for resolution-day temperature tracking.

Polls aviationweather.gov every 60 s for all configured stations in a single
HTTP request, parses temperature with T-group precision where available,
converts to the market's resolution unit (°F for US, °C for international),
and maintains a persisted daily-high record that survives bot restarts.

Resolution-unit rounding follows NWS CLI convention:
    floor(x + 0.5)  — i.e. "round half up", works correctly for negatives.

T-group priority:
    METAR remarks section "T01420108" gives 0.1 °C precision → HIGH confidence.
    Fallback: AWC-parsed `temp` field (may include tenths) → MEDIUM confidence.
    Last resort: integer °C from main body → LOW confidence (±1 °F conversion error).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytz

# ── Path bootstrap ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.cities import STATIONS
from data.metar_parser import extract_t_group_temperature_c

log = logging.getLogger("weather-bot.metar")

# ── Constants ─────────────────────────────────────────────────────────────────
AWC_URL       = "https://aviationweather.gov/api/data/metar"
AWC_UA        = "PolymarketWeatherBot/1.0 (contact: weather-bot)"
AWC_TIMEOUT   = 15.0  # seconds

DAILY_HIGHS_PATH: Path = _HERE / "metar_daily_highs.json"

# Boundary guard distances by confidence level.
# LOW confidence (integer °C → °F fallback) can be off by ±1 °F.
# HIGH confidence (T-group): still guard ±0 but treat exact bucket-low as boundary.
_GUARD_BY_CONFIDENCE = {"HIGH": 0, "MEDIUM": 0, "LOW": 1}

# ── Temperature utilities ─────────────────────────────────────────────────────

def round_half_up(x: float) -> int:
    """NWS/WU rounding: floor(x + 0.5).  Handles negatives correctly.

    round_half_up(69.5)  == 70
    round_half_up(69.4)  == 69
    round_half_up(-1.5)  == -1   (rounds toward positive infinity)
    round_half_up(-1.6)  == -2
    """
    return math.floor(x + 0.5)


def celsius_to_resolution(temp_c: float, unit: str) -> int:
    """Convert °C to the market's resolution unit as a whole-degree integer."""
    if unit == "C":
        return round_half_up(temp_c)
    # °F markets
    f_exact = temp_c * 9.0 / 5.0 + 32.0
    return round_half_up(f_exact)


# ── METAR temperature parsing ─────────────────────────────────────────────────
# Main-body temp/dew pattern: "14/11" or "M02/M08" (M = minus).
# Anchored to word boundary / whitespace so we don't match altimeter digits.
_BODY_TEMP_RE = re.compile(
    r"(?:^|\s)(M?\d{1,3})/(M?\d{1,3})(?:\s|$)",
)

def _parse_main_body_temp_c(raw_ob: str) -> float | None:
    """Parse integer °C from the main METAR body (TTT/DDD field)."""
    m = _BODY_TEMP_RE.search(raw_ob)
    if not m:
        return None
    temp_str = m.group(1)
    if temp_str.startswith("M"):
        return float(-int(temp_str[1:]))
    return float(int(temp_str))


@dataclass
class ParsedTemp:
    temp_c: float                # temperature in °C (tenths when T-group available)
    temp_resolution: int         # whole degrees in market unit (°F or °C)
    unit: str                    # "F" or "C"
    source: str                  # "T-group" | "awc_field" | "main_body"
    confidence: str              # "HIGH" | "MEDIUM" | "LOW"
    raw_segment: str = ""        # debug: matched segment


def parse_metar_temp(raw_ob: str, awc_temp: float | None, unit: str) -> ParsedTemp | None:
    """Parse temperature from a METAR observation.

    Priority:
      1. T-group in remarks (0.1 °C, HIGH confidence)
      2. AWC `temp` field when it includes tenths (MEDIUM — AWC may have already
         done T-group parsing; tenths present means it did)
      3. Main body integer °C (LOW)

    Args:
        raw_ob:   Raw METAR string.
        awc_temp: Numeric temp from AWC JSON `temp` field (may be None or integer).
        unit:     "F" for US °F markets, "C" for international °C markets.
    """
    # ── 1. T-group ────────────────────────────────────────────────────────────
    t_group_c = extract_t_group_temperature_c(raw_ob or "")
    if t_group_c is not None:
        return ParsedTemp(
            temp_c=t_group_c,
            temp_resolution=celsius_to_resolution(t_group_c, unit),
            unit=unit,
            source="T-group",
            confidence="HIGH",
            raw_segment=f"{t_group_c:.1f}°C from T-group",
        )

    # ── 2. AWC field with tenths ───────────────────────────────────────────────
    if awc_temp is not None:
        awc_float = float(awc_temp)
        has_tenths = abs(awc_float - round(awc_float)) > 0.05
        if has_tenths:
            return ParsedTemp(
                temp_c=awc_float,
                temp_resolution=celsius_to_resolution(awc_float, unit),
                unit=unit,
                source="awc_field",
                confidence="MEDIUM",
                raw_segment=f"{awc_float:.1f}°C from AWC field",
            )

    # ── 3. Main body integer °C ───────────────────────────────────────────────
    body_c = _parse_main_body_temp_c(raw_ob or "")
    if body_c is None and awc_temp is not None:
        body_c = float(round(awc_temp))   # AWC integer fallback

    if body_c is not None:
        # For °C markets: integer °C IS the resolution value → HIGH confidence.
        # For °F markets: integer °C → °F conversion can be off by ±1 °F → LOW.
        conf = "HIGH" if unit == "C" else "LOW"
        return ParsedTemp(
            temp_c=body_c,
            temp_resolution=celsius_to_resolution(body_c, unit),
            unit=unit,
            source="main_body",
            confidence=conf,
            raw_segment=f"{body_c:.0f}°C from main body",
        )

    return None   # could not parse


# ── Bucket boundary utilities ─────────────────────────────────────────────────

def _parse_bucket_bounds(bucket: str) -> tuple[float, float]:
    """Return half-open [low, high) numeric interval for a bucket label.

    Handles: "58-59", "14+", "≤-12", "≥6", "12", "-4".
    """
    clean = (
        bucket
        .replace("°F", "").replace("°C", "")
        .replace("≤", "").replace("≥", "")
        .strip()
    )

    # Open-high: "14+"
    if clean.endswith("+"):
        return float(clean[:-1].strip()), float("inf")

    # Range: "58-59" or "-4--2" etc.
    # Find the LAST occurrence of "-" that separates two numbers.
    # Strategy: try splitting from right on "-".
    if "-" in clean:
        parts = clean.rsplit("-", 1)
        if len(parts) == 2 and parts[1].strip():
            try:
                high_val = float(parts[1].strip())
                # Reconstruct left side
                low_str = clean[: len(clean) - len(parts[1]) - 1].strip()
                if low_str == "" or low_str == "-":
                    raise ValueError
                low_val = float(low_str)
                return low_val, high_val + 1.0
            except ValueError:
                pass

    # Single value
    try:
        v = float(clean)
        return v, v + 1.0
    except ValueError:
        return float("nan"), float("nan")


def find_winning_bucket(temp_res: int, bucket_labels: list[str]) -> str | None:
    """Return the bucket label that contains temp_res, or None."""
    for label in bucket_labels:
        low, high = _parse_bucket_bounds(label)
        if math.isnan(low):
            continue
        if low <= temp_res < high:
            return label
    return None


def check_boundary(temp_res: int, bucket_labels: list[str], confidence: str) -> str:
    """Return 'CLEAR' or 'NO_TRADE_NEAR_BOUNDARY'.

    A boundary case is when the computed whole-unit high is within ±guard of
    any bucket-low transition, meaning a ±1 measurement error could put us in
    the wrong bucket.

    Guard distances:
        LOW  confidence → ±1 (integer °C→°F fallback can be off by 1)
        MEDIUM/HIGH     → ±0 (only flag if exactly on a bucket-low)

    International °C markets: confidence is already HIGH for integer °C
    (METAR integer °C IS the resolution value), so guard=0 — only flag
    exact boundary.
    """
    guard = _GUARD_BY_CONFIDENCE.get(confidence, 1)

    # Collect all bucket lower bounds that are finite (these are transition points).
    lows: list[int] = []
    for label in bucket_labels:
        low, _ = _parse_bucket_bounds(label)
        if not math.isnan(low) and not math.isinf(low):
            lows.append(int(low))

    for low in lows:
        if abs(temp_res - low) <= guard:
            return "NO_TRADE_NEAR_BOUNDARY"

    return "CLEAR"


# ── Observation dataclass ─────────────────────────────────────────────────────

@dataclass
class Observation:
    icao: str
    obs_time_utc: str          # ISO-8601
    raw_ob: str
    parsed: ParsedTemp | None
    metar_type: str = "METAR"  # "METAR" or "SPECI"

    @property
    def temp_c(self) -> float | None:
        return self.parsed.temp_c if self.parsed else None

    @property
    def temp_resolution(self) -> int | None:
        return self.parsed.temp_resolution if self.parsed else None


# ── AWC HTTP client ───────────────────────────────────────────────────────────

class AWCClient:
    """Async client for aviationweather.gov METAR data API."""

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None

    def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                headers={"User-Agent": AWC_UA},
                timeout=AWC_TIMEOUT,
            )
        return self._http

    async def fetch_observations(self, icaos: list[str]) -> list[Observation]:
        """Fetch latest METAR for all supplied ICAOs in a single request."""
        if not icaos:
            return []

        ids_str = ",".join(icaos)
        try:
            resp = await self._client().get(
                AWC_URL,
                params={"ids": ids_str, "format": "json"},
            )
            resp.raise_for_status()
            records: list[dict] = resp.json()
        except Exception as exc:
            log.error("AWC fetch failed: %s", exc)
            return []

        observations: list[Observation] = []
        for rec in records:
            icao = rec.get("icaoId", "")
            if not icao:
                continue

            station = STATIONS.get(icao)
            if not station:
                continue

            raw_ob   = rec.get("rawOb", "") or ""
            awc_temp = rec.get("temp")
            unit     = station.get("resolution_unit", "C")

            obs_time = rec.get("reportTime") or rec.get("receiptTime", "")

            parsed = parse_metar_temp(raw_ob, awc_temp, unit)

            observations.append(Observation(
                icao=icao,
                obs_time_utc=obs_time,
                raw_ob=raw_ob,
                parsed=parsed,
                metar_type=rec.get("metarType", "METAR"),
            ))

        return observations

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()


# ── Daily high tracker (persistent) ──────────────────────────────────────────

@dataclass
class DailyHighRecord:
    icao: str
    local_date: str            # "YYYY-MM-DD" in station local time
    unit: str                  # "F" or "C"
    high_c: float              # best °C seen today
    high_resolution: int       # best whole-unit seen today
    confidence: str            # of the observation that set the high
    source: str                # "T-group" | "awc_field" | "main_body"
    obs_count: int = 0
    last_obs_time: str = ""
    last_updated: str = ""
    observations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["observations"] = d["observations"][-200:]  # cap stored obs
        return d


class DailyHighTracker:
    """Thread-safe (single-process) tracker of daily high temperatures.

    State is loaded from / saved to a JSON file at `path` so that bot restarts
    mid-day do not lose the day's maximum.
    """

    def __init__(self, path: Path = DAILY_HIGHS_PATH) -> None:
        self._path = path
        self._data: dict[str, dict] = {}
        self._load()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except Exception:
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(self._data, separators=(",", ":"))
        fd, tmp = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp"
        )
        try:
            os.write(fd, blob.encode())
            os.fsync(fd)
            os.close(fd)
            os.replace(tmp, str(self._path))
        except BaseException:
            os.close(fd)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _local_date(icao: str) -> str:
        tz_name = STATIONS.get(icao, {}).get("timezone", "UTC")
        tz = pytz.timezone(tz_name)
        return datetime.now(tz).strftime("%Y-%m-%d")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, obs: Observation) -> DailyHighRecord | None:
        """Update daily high for the station.  Returns updated record or None."""
        if obs.parsed is None:
            return None

        icao       = obs.icao
        local_date = self._local_date(icao)
        key        = f"{icao}:{local_date}"
        unit       = obs.parsed.unit
        temp_c     = obs.parsed.temp_c
        temp_res   = obs.parsed.temp_resolution
        now_iso    = datetime.now(UTC).isoformat()

        new_obs_entry = {
            "time":       obs.obs_time_utc,
            "temp_c":     temp_c,
            "temp_res":   temp_res,
            "source":     obs.parsed.source,
            "confidence": obs.parsed.confidence,
            "raw_segment": obs.parsed.raw_segment,
        }

        if key not in self._data:
            self._data[key] = DailyHighRecord(
                icao=icao,
                local_date=local_date,
                unit=unit,
                high_c=temp_c,
                high_resolution=temp_res,
                confidence=obs.parsed.confidence,
                source=obs.parsed.source,
                obs_count=1,
                last_obs_time=obs.obs_time_utc,
                last_updated=now_iso,
                observations=[new_obs_entry],
            ).to_dict()
        else:
            rec = self._data[key]
            rec["obs_count"] = rec.get("obs_count", 0) + 1
            rec["last_obs_time"] = obs.obs_time_utc
            rec["last_updated"] = now_iso
            rec.setdefault("observations", []).append(new_obs_entry)
            # Update high if this observation exceeds the current max
            if temp_res > rec["high_resolution"]:
                rec["high_c"] = temp_c
                rec["high_resolution"] = temp_res
                rec["confidence"] = obs.parsed.confidence
                rec["source"] = obs.parsed.source

        return self._record_obj(key)

    def save(self) -> None:
        """Persist current state to disk.  Call once after a batch of update()s."""
        self._save()

    def get_high(self, icao: str, local_date: str | None = None) -> DailyHighRecord | None:
        if local_date is None:
            local_date = self._local_date(icao)
        key = f"{icao}:{local_date}"
        d = self._data.get(key)
        if d is None:
            return None
        return self._record_obj(key)

    def get_all_today(self) -> dict[str, DailyHighRecord]:
        """Return all records whose local_date == today for that station."""
        out: dict[str, DailyHighRecord] = {}
        for icao in STATIONS:
            rec = self.get_high(icao)
            if rec is not None:
                out[icao] = rec
        return out

    def cleanup_old(self, keep_days: int = 7) -> int:
        """Remove records older than keep_days days.  Returns count deleted."""
        cutoff = date.today().toordinal() - keep_days
        stale = [
            k for k, v in self._data.items()
            if _date_ordinal(v.get("local_date", "")) < cutoff
        ]
        for k in stale:
            del self._data[k]
        if stale:
            self._save()
        return len(stale)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _record_obj(self, key: str) -> DailyHighRecord:
        d = self._data[key]
        return DailyHighRecord(
            icao=d["icao"],
            local_date=d["local_date"],
            unit=d["unit"],
            high_c=d["high_c"],
            high_resolution=d["high_resolution"],
            confidence=d.get("confidence", "LOW"),
            source=d.get("source", "unknown"),
            obs_count=d.get("obs_count", 0),
            last_obs_time=d.get("last_obs_time", ""),
            last_updated=d.get("last_updated", ""),
            observations=d.get("observations", []),
        )


def _date_ordinal(date_str: str) -> int:
    try:
        d = date.fromisoformat(date_str)
        return d.toordinal()
    except Exception:
        return 0


# ── Module-level singletons (for import convenience) ─────────────────────────
_tracker: DailyHighTracker | None = None


def get_tracker() -> DailyHighTracker:
    global _tracker
    if _tracker is None:
        _tracker = DailyHighTracker()
    return _tracker
