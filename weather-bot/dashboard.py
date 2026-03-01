from __future__ import annotations

import json
import re
import subprocess
import time as _time
from collections import defaultdict
from datetime import UTC, date as _date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover
    st_autorefresh = None

VM_NAME = "weather-bot"
VM_ZONE = "us-east1-b"
VM_PROJECT = "weather-488111"
VM_REMOTE_USER = "tombaxter"
VM_WORKDIR = f"/home/{VM_REMOTE_USER}/weather-bot"

# Mirror of settings.py MODEL_RUN_TRIGGER_TIMES_UTC — kept in sync manually
_TRIGGER_SCHEDULE: list[tuple[int, int, str]] = [
    (6, 10, "GFS_00Z+ECMWF_00Z"),
    (10, 0, "GFS_06Z"),
    (13, 0, "MIDDAY_DISCOVERY"),
    (18, 30, "GFS_12Z+ECMWF_12Z"),
    (23, 0, "GFS_18Z"),
]


ROOT = Path(__file__).resolve().parent
TRADES_CSV = ROOT / "logs" / "trades.csv"
SIGNALS_CSV = ROOT / "logs" / "signals.csv"
RESOLVED_CSV = ROOT / "logs" / "resolved.csv"
POSITIONS_JSON = ROOT / "data" / "positions.json"
DEFAULT_ENV = ROOT / ".env"
VM_ENV = Path("/etc/weather-bot.env")

BG = "#0E1117"
GREEN = "#00FF88"
RED = "#FF4444"
BLUE = "#4DA6FF"
GRAY = "#888888"
PANEL = "#141A22"
TEXT = "#E6EDF3"

# ---------------------------------------------------------------------------
# Commercial forecast providers (AccuWeather + Weather.com/IBM)
# These are logged daily to build a backtestable historical dataset.
# ---------------------------------------------------------------------------

# WU/weather.com embedded API key (public, loaded in every wunderground.com session)
_WU_FORECAST_API_KEY = "6532d6454b8aa370768e63d6ba5a832e"
_WU_FORECAST_API_URL = "https://api.weather.com/v3/wx/forecast/daily/10day"
# Separate key used for the historical observations endpoint (Polymarket resolution source)
_WU_OBS_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
_WU_OBS_API_URL = "https://api.weather.com/v1/location/{station}/observations/historical.json"
_ACCU_API_BASE = "https://dataservice.accuweather.com"

# City → WU observation station ID (ICAO:9:COUNTRY_CODE format)
# These are the SAME stations Polymarket resolves against — live temp = resolution floor.
_CITY_WU_STATION: dict[str, tuple[str, str]] = {
    # (station_id, units)  units: "m"=metric(°C)  "e"=english(°F)
    "Seoul":        ("RKSI:9:KR", "m"),
    "London":       ("EGLC:9:GB", "m"),
    "New York":     ("KLGA:9:US", "e"),
    "Atlanta":      ("KATL:9:US", "e"),
    "Chicago":      ("KORD:9:US", "e"),
    "Miami":        ("KMIA:9:US", "e"),
    "Dallas":       ("KDFW:9:US", "e"),
    "Buenos Aires": ("SAEZ:9:AR", "m"),
    "Paris":        ("LFPG:9:FR", "m"),
    "Toronto":      ("CYYZ:9:CA", "m"),
    "Seattle":      ("KSEA:9:US", "e"),
    "Ankara":       ("LTAC:9:TR", "m"),
    "Wellington":   ("NZWN:9:NZ", "m"),
}

# ICAO → AccuWeather location key (stable, no geoposition lookup needed)
_ACCU_LOCATION_KEYS: dict[str, str] = {
    "EGLC": "2532754",   # London City
    "KATL": "2140212",   # Atlanta Hartsfield
    "KDFW": "336107",    # Dallas/Fort Worth
    "KLGA": "2627477",   # New York LaGuardia
    "KMIA": "3593859",   # Miami International
    "KORD": "2626577",   # Chicago O'Hare
    "KSEA": "341357",    # Seattle-Tacoma (already present)
    "LFPG": "159190",    # Paris CDG
    "RKSI": "2331998",   # Seoul Incheon
    "SBGR": "36369",     # São Paulo Guarulhos
    "CYYZ": "55488",     # Toronto Pearson
    "LTAC": "1294331",  # Ankara Esenboğa
    "SAEZ": "7894",      # Buenos Aires (city — closest to Ezeiza resolution station)
    "NZWN": "250938",    # Wellington NZ
}

# Dashboard city name → ICAO code
_CITY_ICAO: dict[str, str] = {
    "Seoul":        "RKSI",
    "London":       "EGLC",
    "New York":     "KLGA",
    "Atlanta":      "KATL",
    "Chicago":      "KORD",
    "Miami":        "KMIA",
    "Dallas":       "KDFW",
    "Buenos Aires": "SAEZ",
    "Paris":        "LFPG",
    "Toronto":      "CYYZ",
    "Seattle":      "KSEA",
    "Ankara":       "LTAC",
    "Wellington":   "NZWN",
}

_COMMERCIAL_LOG_PATH = ROOT / "data" / "commercial_forecast_log.json"


def _read_env_key(name: str) -> str:
    """Read a single key from the local .env file.

    Strips inline comments (e.g. KEY=value  # comment → returns 'value').
    """
    for env_path in (DEFAULT_ENV, VM_ENV):
        try:
            text = env_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if line.startswith(f"{name}="):
                    value = line.split("=", 1)[1].strip()
                    value = value.split("#")[0].strip()   # drop inline comments
                    return value
        except Exception:
            pass
    return ""


_COMM_LOG_LOCK_HOUR_UTC = 19  # After this UTC hour the snapshot is frozen for backtesting


def _log_commercial_forecast(
    city: str, date_str: str, accu: float | None, wu: float | None, unit: str,
    accu_d2: float | None = None, wu_d2: float | None = None,
    accu_d3: float | None = None, wu_d3: float | None = None,
) -> None:
    """Write a commercial forecast snapshot to disk.

    Locking rule:
    - Before 19:00 UTC: overwrite freely — morning reads are noisy drafts.
    - At/after 19:00 UTC: write-once — the 19:05 cron run is the canonical
      backtesting snapshot (post-12Z, after commercial forecasters have updated).

    This prevents an early dashboard load (e.g. 16:00 UTC) from freezing the
    snapshot and blocking the cron's more accurate evening reading.
    """
    if accu is None and wu is None:
        return
    try:
        _COMMERCIAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log: dict = {}
        if _COMMERCIAL_LOG_PATH.exists():
            try:
                log = json.loads(_COMMERCIAL_LOG_PATH.read_text(encoding="utf-8"))
            except Exception:
                log = {}
        if city not in log:
            log[city] = {}
        now_utc = datetime.now(UTC)
        existing = log[city].get(date_str)
        if existing:
            # Check if the existing entry was written before the lock hour
            try:
                logged_at = datetime.fromisoformat(existing["logged_at"])
                already_locked = logged_at.hour >= _COMM_LOG_LOCK_HOUR_UTC
            except Exception:
                already_locked = True  # be conservative if parse fails
            if already_locked:
                # Allow filling in null slots even after the lock —
                # e.g. AccuWeather was 429 when the entry was first written.
                existing_accu    = existing.get("accu")
                existing_wu      = existing.get("wu")
                existing_accu_d2 = existing.get("accu_d2")
                existing_wu_d2   = existing.get("wu_d2")
                existing_accu_d3 = existing.get("accu_d3")
                existing_wu_d3   = existing.get("wu_d3")
                needs_patch = (
                    (accu    is not None and existing_accu    is None) or
                    (wu      is not None and existing_wu      is None) or
                    (accu_d2 is not None and existing_accu_d2 is None) or
                    (wu_d2   is not None and existing_wu_d2   is None) or
                    (accu_d3 is not None and existing_accu_d3 is None) or
                    (wu_d3   is not None and existing_wu_d3   is None)
                )
                if needs_patch:
                    log[city][date_str] = {
                        "accu":      accu    if existing_accu    is None else existing_accu,
                        "wu":        wu      if existing_wu      is None else existing_wu,
                        "accu_d2":   accu_d2 if existing_accu_d2 is None else existing_accu_d2,
                        "wu_d2":     wu_d2   if existing_wu_d2   is None else existing_wu_d2,
                        "accu_d3":   accu_d3 if existing_accu_d3 is None else existing_accu_d3,
                        "wu_d3":     wu_d3   if existing_wu_d3   is None else existing_wu_d3,
                        "unit":      unit,
                        "logged_at": existing.get("logged_at", now_utc.isoformat()),
                    }
                    _COMMERCIAL_LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
                return
        # Write (new entry or pre-lock update)
        log[city][date_str] = {
            "accu":    accu,
            "wu":      wu,
            "accu_d2": accu_d2,
            "wu_d2":   wu_d2,
            "accu_d3": accu_d3,
            "wu_d3":   wu_d3,
            "unit":    unit,
            "logged_at": now_utc.isoformat(),
        }
        _COMMERCIAL_LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    except Exception:
        pass


@st.cache_data(ttl=60, show_spinner=False)
def _load_commercial_log() -> dict:
    """Load the full commercial forecast log from disk."""
    try:
        if _COMMERCIAL_LOG_PATH.exists():
            d = json.loads(_COMMERCIAL_LOG_PATH.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# 12Z model snapshot log
# Saves D+1 predictions at the 19:05 UTC cron window so the dashboard always
# shows the 12Z run (the most accurate) — even in the morning when the cache
# has refreshed with the less reliable 00Z values.
# ---------------------------------------------------------------------------
_MODEL_SNAPSHOT_PATH = ROOT / "data" / "model_snapshot_log.json"
_SNAP_LOCK_HOUR_UTC  = 19   # same as commercial log — cron locks at 19:05


def _log_model_snapshot(city: str, target_date: str, preds: dict) -> None:
    """Write model predictions snapshot for target_date to disk.

    Same locking rule as commercial log:
    - Before 19:00 UTC: overwrite (morning/noon reads are pre-12Z drafts).
    - At/after 19:00 UTC: write-once (19:05 cron is the canonical 12Z snapshot).
    """
    clean = {k: v for k, v in preds.items() if not k.startswith("__")}
    if not clean:
        return
    try:
        _MODEL_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        snap: dict = {}
        if _MODEL_SNAPSHOT_PATH.exists():
            try:
                snap = json.loads(_MODEL_SNAPSHOT_PATH.read_text(encoding="utf-8"))
            except Exception:
                snap = {}
        if city not in snap:
            snap[city] = {}
        now_utc = datetime.now(UTC)
        existing = snap[city].get(target_date)
        if existing:
            try:
                logged_hour = datetime.fromisoformat(existing["logged_at"]).hour
                if logged_hour >= _SNAP_LOCK_HOUR_UTC:
                    return  # canonical snapshot already locked
            except Exception:
                pass
        snap[city][target_date] = {
            "preds":     clean,
            "logged_at": now_utc.isoformat(),
        }
        _MODEL_SNAPSHOT_PATH.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_model_snapshot(city: str, target_date: str) -> dict | None:
    """Return stored 12Z model predictions for target_date, or None if not saved yet."""
    try:
        if _MODEL_SNAPSHOT_PATH.exists():
            snap = json.loads(_MODEL_SNAPSHOT_PATH.read_text(encoding="utf-8"))
            entry = snap.get(city, {}).get(target_date)
            if entry and entry.get("preds"):
                return entry["preds"], entry.get("logged_at", "?")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Disk-backed accuracy row cache (avoids re-fetching resolved historical data)
# ---------------------------------------------------------------------------
_ACCURACY_CACHE_PATH = ROOT / "data" / "accuracy_rows_cache.json"


def _load_accuracy_disk_cache(city: str) -> list[dict]:
    try:
        if _ACCURACY_CACHE_PATH.exists():
            raw = json.loads(_ACCURACY_CACHE_PATH.read_text(encoding="utf-8"))
            return raw.get(city, [])
    except Exception:
        pass
    return []


def _save_accuracy_disk_cache(city: str, rows: list[dict]) -> None:
    try:
        _ACCURACY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache: dict = {}
        if _ACCURACY_CACHE_PATH.exists():
            try:
                cache = json.loads(_ACCURACY_CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                cache = {}
        cache[city] = rows
        _ACCURACY_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


@st.cache_data(ttl=600, show_spinner=False)
def fetch_wu_live_obs(city: str) -> dict | None:
    """Fetch today's live WU observations for the city's resolution station.

    Returns latest current temp, today's running maximum (= Polymarket resolution floor),
    number of readings, and the timestamp of the most recent reading.
    These are the ACTUAL sensor readings — not forecasts.
    """
    station_info = _CITY_WU_STATION.get(city)
    if not station_info:
        return None
    station_id, units = station_info
    from datetime import date as _d
    today = _d.today().strftime("%Y%m%d")
    try:
        r = requests.get(
            _WU_OBS_API_URL.format(station=station_id),
            params={"apiKey": _WU_OBS_API_KEY, "units": units,
                    "startDate": today, "endDate": today},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            return None
        temps = [o["temp"] for o in obs if o.get("temp") is not None]
        if not temps:
            return None
        last = obs[-1]
        last_time = last.get("valid_time_gmt", 0)
        last_dt = datetime.fromtimestamp(last_time, tz=UTC).strftime("%H:%M UTC") if last_time else "?"
        return {
            "latest_temp": last.get("temp"),
            "running_max": max(temps),
            "n_obs": len(temps),
            "last_time": last_dt,
            "unit": "C" if units == "m" else "F",
            "station_id": station_id,
        }
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_commercial_forecasts(city: str) -> dict:
    """Fetch AccuWeather and Weather.com (WU) D+1/D+2/D+3 forecasts for a city.

    Both APIs return multi-day forecasts in one call, so D+2 and D+3 come for free.
    Result includes accu_d2/wu_d2 and accu_d3/wu_d3 keys for the pending rows.

    Refresh logic:
      - Before 19:00 UTC: serve from disk if we have both values already logged.
      - After 19:00 UTC: locked — serve canonical value from disk, no more
        API calls until next day.
    """
    from datetime import date as _d, timedelta
    tomorrow  = (_d.today() + timedelta(days=1)).isoformat()
    day2      = (_d.today() + timedelta(days=2)).isoformat()
    day3      = (_d.today() + timedelta(days=3)).isoformat()
    cfg  = ACCURACY_CITIES.get(city, {})
    lat  = cfg.get("lat")
    lon  = cfg.get("lon")
    icao = _CITY_ICAO.get(city)
    unit = "F" if cfg.get("temperature_unit", "celsius") != "celsius" else "C"
    result: dict = {"target_date": tomorrow, "accu": None, "wu": None,
                    "accu_d2": None, "wu_d2": None,
                    "accu_d3": None, "wu_d3": None,
                    "unit": unit, "errors": [], "source": "api"}

    logged      = _load_commercial_log().get(city, {}).get(tomorrow, {})
    now_utc_h   = datetime.now(UTC).hour
    now_utc     = datetime.now(UTC)

    # After 19:00 UTC: locked — always serve from disk if available
    if now_utc_h >= _COMM_LOG_LOCK_HOUR_UTC:
        if logged.get("accu") is not None or logged.get("wu") is not None:
            logged_ts = ""
            try:
                logged_ts = datetime.fromisoformat(logged["logged_at"]).strftime("%H:%M UTC")
            except Exception:
                pass
            return {
                "target_date": tomorrow,
                "accu":    logged.get("accu"),
                "wu":      logged.get("wu"),
                "accu_d2": logged.get("accu_d2"),
                "wu_d2":   logged.get("wu_d2"),
                "accu_d3": logged.get("accu_d3"),
                "wu_d3":   logged.get("wu_d3"),
                "unit":    unit,
                "errors":  [],
                "source":  f"🔒 locked {logged_ts}",
            }

    # Before 19:00 UTC: serve from disk if we already have both values logged.
    # Forecasts don't change meaningfully hour-to-hour; only fetch live when
    # we have no data at all (e.g. early in the morning before the 19:00 cron).
    if logged.get("accu") is not None and logged.get("wu") is not None:
        logged_ts = ""
        try:
            logged_ts = datetime.fromisoformat(logged["logged_at"]).strftime("%H:%M UTC")
        except Exception:
            pass
        return {
            "target_date": tomorrow,
            "accu":    logged["accu"],
            "wu":      logged["wu"],
            "accu_d2": logged.get("accu_d2"),
            "wu_d2":   logged.get("wu_d2"),
            "accu_d3": logged.get("accu_d3"),
            "wu_d3":   logged.get("wu_d3"),
            "unit":    unit,
            "errors":  [],
            "source":  f"disk {logged_ts}",
        }

    # --- Weather.com/IBM forecast (returns ~15 days — parse D+1 and D+2) ---
    if lat is not None and lon is not None:
        try:
            wu_units = "m" if unit == "C" else "e"
            r = requests.get(
                _WU_FORECAST_API_URL,
                params={"geocode": f"{lat},{lon}", "format": "json",
                        "units": wu_units, "language": "en-US",
                        "apiKey": _WU_FORECAST_API_KEY},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=12,
            )
            r.raise_for_status()
            d = r.json()
            valid_times = d.get("validTimeLocal", [])
            highs = d.get("calendarDayTemperatureMax", [])
            for ts, high in zip(valid_times, highs):
                if high is None:
                    continue
                try:
                    day = datetime.fromisoformat(str(ts)).date().isoformat()
                except (ValueError, TypeError):
                    continue
                if day == tomorrow:
                    result["wu"] = float(high)
                elif day == day2:
                    result["wu_d2"] = float(high)
                elif day == day3:
                    result["wu_d3"] = float(high)
        except Exception as exc:
            result["errors"].append(f"WU: {exc}")

    # --- AccuWeather forecast (5-day response includes D+2) ---
    accu_key = _read_env_key("ACCUWEATHER_API_KEY")
    if accu_key and icao:
        loc_key = _ACCU_LOCATION_KEYS.get(icao)
        if loc_key:
            try:
                metric = "true" if unit == "C" else "false"
                r = requests.get(
                    f"{_ACCU_API_BASE}/forecasts/v1/daily/5day/{loc_key}",
                    params={"apikey": accu_key, "metric": metric},
                    timeout=12,
                )
                if r.status_code in (403, 429):
                    result["errors"].append(f"AccuWeather rate-limited ({r.status_code})")
                else:
                    r.raise_for_status()
                    payload = r.json()
                    for fc in payload.get("DailyForecasts", []):
                        fc_date = str(fc.get("Date", ""))[:10]
                        temp = fc.get("Temperature", {}).get("Maximum", {}).get("Value")
                        if fc_date == tomorrow and temp is not None:
                            result["accu"] = float(temp)
                        elif fc_date == day2 and temp is not None:
                            result["accu_d2"] = float(temp)
                        elif fc_date == day3 and temp is not None:
                            result["accu_d3"] = float(temp)
            except Exception as exc:
                result["errors"].append(f"AccuWeather: {exc}")
    elif not accu_key:
        result["errors"].append("AccuWeather: ACCUWEATHER_API_KEY not set in .env")

    # Persist snapshot for backtesting (write-once per date), including D+2/D+3 values
    _log_commercial_forecast(city, tomorrow, result.get("accu"), result.get("wu"), unit,
                             accu_d2=result.get("accu_d2"), wu_d2=result.get("wu_d2"),
                             accu_d3=result.get("accu_d3"), wu_d3=result.get("wu_d3"))

    # If live fetch failed, fall back to the most recent logged value for today's target date
    if result["accu"] is None or result["wu"] is None:
        logged_fb = _load_commercial_log().get(city, {}).get(tomorrow, {})
        if result["accu"] is None and logged_fb.get("accu") is not None:
            result["accu"] = logged_fb["accu"]
            result["errors"] = [e for e in result["errors"] if "AccuWeather" not in e]
            result["source"] = "disk (API failed)"
        if result["wu"] is None and logged_fb.get("wu") is not None:
            result["wu"] = logged_fb["wu"]
        if result["accu_d2"] is None and logged_fb.get("accu_d2") is not None:
            result["accu_d2"] = logged_fb["accu_d2"]
        if result["wu_d2"] is None and logged_fb.get("wu_d2") is not None:
            result["wu_d2"] = logged_fb["wu_d2"]
        if result["accu_d3"] is None and logged_fb.get("accu_d3") is not None:
            result["accu_d3"] = logged_fb["accu_d3"]
        if result["wu_d3"] is None and logged_fb.get("wu_d3") is not None:
            result["wu_d3"] = logged_fb["wu_d3"]

    return result


def _empty_trades_df() -> pd.DataFrame:
    cols = [
        "timestamp",
        "city",
        "date",
        "bucket",
        "side",
        "forecast_prob",
        "market_prob",
        "edge",
        "size_usd",
        "fill_price",
        "outcome",
        "pnl",
    ]
    return pd.DataFrame(columns=cols)


def _empty_signals_df() -> pd.DataFrame:
    cols = ["timestamp", "city", "date", "bucket", "forecast_prob", "market_prob", "edge", "action_taken"]
    return pd.DataFrame(columns=cols)


@st.cache_data(ttl=15)
def load_trades_df() -> pd.DataFrame:
    try:
        df = pd.read_csv(TRADES_CSV)
    except Exception:
        return _empty_trades_df()
    for col in ("timestamp", "date", "city", "bucket", "side", "outcome"):
        if col not in df.columns:
            df[col] = ""
    for col in ("pnl", "fill_price", "size_usd", "edge", "market_prob", "forecast_prob"):
        if col not in df.columns:
            df[col] = 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    return df


@st.cache_data(ttl=15)
def load_signals_df() -> pd.DataFrame:
    try:
        df = pd.read_csv(SIGNALS_CSV)
    except Exception:
        return _empty_signals_df()
    for col in ("timestamp", "city", "date", "bucket", "action_taken"):
        if col not in df.columns:
            df[col] = ""
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


@st.cache_data(ttl=15)
def load_resolved_df() -> pd.DataFrame:
    """Load from logs/resolved.csv — written by scripts/daily_resolver.py."""
    try:
        df = pd.read_csv(RESOLVED_CSV)
    except Exception:
        return pd.DataFrame()
    for col in ("resolved_at", "target_date", "city", "bucket", "outcome"):
        if col not in df.columns:
            df[col] = ""
    for col in ("pnl_usd", "entry_price", "size_usd", "ev_per_bet"):
        if col not in df.columns:
            df[col] = 0.0
    df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce", utc=True)
    df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce").fillna(0.0)
    df = df.sort_values("resolved_at", ascending=True)
    df["cum_pnl"] = df["pnl_usd"].cumsum()
    return df


def load_positions() -> list[dict]:
    try:
        payload = json.loads(POSITIONS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        return [p for p in payload.values() if isinstance(p, dict)]
    return []


def load_mode_from_env(path: Path) -> tuple[bool, bool]:
    paper = True
    live = False
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return paper, live
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        val = value.strip().lower()
        truthy = val in {"1", "true", "yes", "on"}
        if key == "PAPER_TRADING":
            paper = truthy
        elif key == "LIVE_TRADING":
            live = truthy
    return paper, live


def write_mode_to_env(path: Path, target_live: bool) -> tuple[bool, str]:
    paper_val = "false" if target_live else "true"
    live_val = "true" if target_live else "false"
    lines: list[str] = []
    try:
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = []

    found_paper = False
    found_live = False
    out: list[str] = []
    for raw in lines:
        if raw.startswith("PAPER_TRADING="):
            out.append(f"PAPER_TRADING={paper_val}")
            found_paper = True
        elif raw.startswith("LIVE_TRADING="):
            out.append(f"LIVE_TRADING={live_val}")
            found_live = True
        else:
            out.append(raw)
    if not found_paper:
        out.append(f"PAPER_TRADING={paper_val}")
    if not found_live:
        out.append(f"LIVE_TRADING={live_val}")

    try:
        path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
        return True, f"Updated mode flags in {path}"
    except Exception as exc:
        return False, f"Could not write {path}: {exc}"


def realized_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()
    mask = trades_df["outcome"].astype(str).str.lower().isin({"won", "lost"})
    out = trades_df.loc[mask].copy()
    out = out.sort_values("timestamp", ascending=True)
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce").fillna(0.0)
    out["cum_pnl"] = out["pnl"].cumsum()
    return out


def kpis(trades_df: pd.DataFrame, positions: list[dict]) -> dict[str, float]:
    resolved = realized_trades(trades_df)
    wins = 0
    losses = 0
    if not resolved.empty:
        outcomes = resolved["outcome"].astype(str).str.lower()
        wins = int((outcomes == "won").sum())
        losses = int((outcomes == "lost").sum())

    exposure = sum(float(p.get("cost", 0.0) or 0.0) for p in positions)
    open_count = len(positions)

    now_utc = datetime.now(UTC).date()
    today = now_utc.isoformat()
    tomorrow = (now_utc + timedelta(days=1)).isoformat()
    resolving_today = sum(1 for p in positions if str(p.get("date", "")) == today)
    resolving_tomorrow = sum(1 for p in positions if str(p.get("date", "")) == tomorrow)

    total_pnl = float(resolved["pnl"].sum()) if not resolved.empty else 0.0
    total_resolved = wins + losses
    win_rate = (wins / total_resolved) if total_resolved else 0.0

    return {
        "realized_pnl": total_pnl,
        "win_rate": win_rate,
        "open_positions": float(open_count),
        "open_exposure": exposure,
        "resolving_today": float(resolving_today),
        "resolving_tomorrow": float(resolving_tomorrow),
        "wins": float(wins),
        "losses": float(losses),
    }


def apply_style() -> None:
    st.markdown(
        f"""
<style>
.stApp {{
    background-color: {BG};
    color: {TEXT};
}}
.block-container {{
    padding-top: 1rem;
    max-width: 1200px;
}}
.panel {{
    background: {PANEL};
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
}}
.kpi-card {{
    background: {PANEL};
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 10px 12px;
    text-align: center;
}}
.kpi-label {{
    color: {GRAY};
    font-size: 0.78rem;
    margin-top: 4px;
}}
.kpi-value {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 1.35rem;
    font-weight: 700;
}}
.banner-title {{
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}}
.muted {{
    color: {GRAY};
    font-size: 0.82rem;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def next_model_run_trigger(now_utc: datetime) -> tuple[datetime, str]:
    candidates: list[tuple[datetime, str]] = []
    for delta_days in (0, 1):
        base = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=delta_days)
        for hour, minute, label in _TRIGGER_SCHEDULE:
            candidate = base.replace(hour=hour, minute=minute)
            if candidate > now_utc:
                candidates.append((candidate, label))
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def sync_from_vm() -> tuple[bool, str]:
    files = [
        ("logs/trades.csv",                f"{VM_WORKDIR}/logs/trades.csv"),
        ("logs/signals.csv",               f"{VM_WORKDIR}/logs/signals.csv"),
        ("data/positions.json",            f"{VM_WORKDIR}/data/positions.json"),
        ("logs/calibration.json",          f"{VM_WORKDIR}/logs/calibration.json"),
        ("data/commercial_forecast_log.json", f"{VM_WORKDIR}/data/commercial_forecast_log.json"),
        ("data/model_snapshot_log.json",   f"{VM_WORKDIR}/data/model_snapshot_log.json"),
    ]
    messages: list[str] = []
    for local_rel, remote_path in files:
        local_abs = ROOT / local_rel
        local_abs.parent.mkdir(parents=True, exist_ok=True)
        src = f"{VM_NAME}:{remote_path}"
        result = subprocess.run(
            [
                "gcloud", "compute", "scp", src, str(local_abs),
                "--zone", VM_ZONE, "--project", VM_PROJECT,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            messages.append(f"✓ {local_rel}")
        else:
            messages.append(f"— {local_rel} (not found or error)")
    return True, "\n".join(messages)


# ---------------------------------------------------------------------------
# Model Accuracy — data layer
# ---------------------------------------------------------------------------

ACCURACY_CITIES: dict[str, dict] = {
    "Seoul": {
        "lat": 37.4492, "lon": 126.451,
        "timezone": "Asia/Seoul",
        "polymarket_slug": "highest-temperature-in-seoul-on",
        "models": {
            "ncep_aigfs025":               ("NCEP AI-GFS",       "🤖"),
            "gfs_graphcast025":            ("GFS GraphCast",     "🌐"),
            "ecmwf_ifs025":                ("ECMWF IFS",         "🌍"),
            "ncep_hgefs025_ensemble_mean": ("NCEP HGEFS Ens",    "📊"),
            "kma_gdps":                    ("KMA GDPS",          "🇰🇷"),
        },
        "best_ensemble": {
            "short":      "AVG(NCEP+GC)",
            "label":      "AVG(NCEP AI-GFS + GraphCast)",
            "model_keys": ["ncep_aigfs025", "gfs_graphcast025"],
        },
        "top_model_key":   "ncep_aigfs025",
        "top_model_label": "NCEP AI-GFS D1",
        "chart_models":    ["ncep_aigfs025", "gfs_graphcast025", "ecmwf_ifs025"],
        "notes": (
            "**Best signal:** AVG(NCEP AI-GFS + GFS GraphCast) D1 — exhaustive search confirmed this "
            "as the accuracy ceiling for Seoul.\n\n"
            "**Coverage:** `ncep_aigfs025` data starts ~Jan 7 2026. HGEFS Ensemble starts ~Jan 22 2026."
        ),
        "polymarket": {
            "2026-01-07": ("5+",   5, True),  "2026-01-08": ("-2",  -2, False),
            "2026-01-09": ("7",    7, False),  "2026-01-10": ("8",   8, False),
            "2026-01-11": ("-3",  -3, False),  "2026-01-12": ("4",   4, False),
            "2026-01-13": ("1+",   1, True),   "2026-01-14": ("2+",  2, True),
            "2026-01-15": ("7+",   7, True),   "2026-01-16": ("6",   6, False),
            "2026-01-17": ("2",    2, False),  "2026-01-18": ("2",   2, False),
            "2026-01-19": ("0",    0, False),  "2026-01-20": ("-6", -6, False),
            "2026-01-21": ("-6",  -6, False),  "2026-01-22": ("-5", -5, False),
            "2026-01-23": ("-1",  -1, False),  "2026-01-24": ("-2", -2, False),
            "2026-01-25": ("-3",  -3, False),  "2026-01-26": ("-1", -1, False),
            "2026-01-27": ("-1",  -1, False),  "2026-01-28": ("-2", -2, False),
            "2026-01-29": ("-1",  -1, False),  "2026-01-30": ("-3", -3, False),
            "2026-01-31": ("-1",  -1, False),
            "2026-02-01": ("0",    0, False),  "2026-02-02": ("0",   0, False),
            "2026-02-03": ("3",    3, False),  "2026-02-04": ("7+",  7, True),
            "2026-02-05": ("8+",   8, True),   "2026-02-06": ("-4", -4, False),
            "2026-02-07": ("-5",  -5, False),  "2026-02-08": ("-4", -4, False),
            "2026-02-09": ("2",    2, False),  "2026-02-10": ("3",   3, False),
            "2026-02-11": ("4",    4, False),  "2026-02-12": ("7+",  7, True),
            "2026-02-13": ("9+",   9, True),   "2026-02-14": ("11+",11, True),
            "2026-02-15": ("8",    8, False),  "2026-02-16": ("5",   5, False),
            "2026-02-17": ("4",    4, False),  "2026-02-18": ("6",   6, False),
            "2026-02-19": ("5",    5, False),  "2026-02-20": ("11+",11, True),
            "2026-02-21": ("14+", 14, True),   "2026-02-22": ("11", 11, False),
            "2026-02-23": ("4",    4, False),
        },
    },
    "London": {
        "lat": 51.5053, "lon": 0.0553,
        "timezone": "Europe/London",
        "polymarket_slug": "highest-temperature-in-london-on",
        "models": {
            "meteofrance_arome_france":       ("MF AROME France",    "🇫🇷"),
            "meteofrance_seamless":           ("MF Seamless",        "🇫🇷"),
            "meteofrance_arome_france_hd":    ("MF AROME HD",        "🇫🇷"),
            "icon_seamless":                  ("ICON Seamless",      "🇩🇪"),
            "dmi_seamless":                   ("DMI Seamless",       "🇩🇰"),
            "ecmwf_ifs025":                   ("ECMWF IFS",          "🌍"),
            "kma_seamless":                   ("KMA Seamless",       "🇰🇷"),
            "ukmo_uk_deterministic_2km":      ("UKMO 2km",           "🇬🇧"),
            "ukmo_seamless":                  ("UKMO Seamless",      "🇬🇧"),
            "ukmo_global_deterministic_10km": ("UKMO Global 10km",   "🇬🇧"),
            "ncep_aigfs025":                  ("NCEP AIGFS",         "🤖"),
        },
        "best_ensemble": {
            "short":      "MF AROME France",
            "label":      "MF AROME France D1 (primary signal)",
            "model_keys": ["meteofrance_arome_france"],
        },
        "hypothesis_ensembles": [
            {
                "key":        "h1_5model",
                "short":      "H1: 5-model (MF3+ICON+DMI)",
                "model_keys": ["meteofrance_arome_france","meteofrance_seamless",
                               "meteofrance_arome_france_hd","icon_seamless","dmi_seamless"],
                "weights":    None,
            },
        ],
        "top_model_key":   "meteofrance_arome_france",
        "top_model_label": "MF AROME France D1",
        # Pre-registered spread filter (2026-02-24). Threshold 1.0°C chosen once, forward-testing only.
        # In-sample: ≤1.0°C → 80.8% (52 days), >1.0°C → 47.6% (21 days). Christmas contamination noted.
        "spread_filter": {
            "model_keys": [
                "meteofrance_arome_france", "meteofrance_seamless",
                "meteofrance_arome_france_hd", "icon_seamless", "dmi_seamless",
            ],
            "threshold": 1.0,
            "label": "5-model spread (MF3+ICON+DMI)",
        },
        "chart_models": [
            "meteofrance_arome_france",
            "meteofrance_seamless",
            "meteofrance_arome_france_hd",
            "ukmo_uk_deterministic_2km",
            "ecmwf_ifs025",
        ],
        "notes": (
            "**Trading signal: MF AROME France D1 — 70.7%** (53/75, Dec 2025–Feb 2026).\n\n"
            "**Why MF AROME alone, not an ensemble:**\n"
            "Exhaustive permutation over all 1,585 model combos found AVG(MF+MF HD+ICON+DMI) at 77.3% (58/75). "
            "But 1,585 hypotheses on 75 days = textbook data dredging. Bonferroni-corrected threshold requires "
            "~65+ wins to be significant — 58 doesn't clear it. The 5-win gap is noise. "
            "MF AROME was not selected by searching — it has a clear causal story: 1.3km NW Europe, "
            "post-Oct 2024 3DEnVar upgrade, 3× observation intake. Use it alone.\n\n"
            "**Forward test protocol:** All models accumulate fresh daily data automatically. "
            "After 30+ new days, evaluate all combinations on that unseen window — no pre-selected candidates. "
            "Until then, MF AROME alone.\n\n"
            "**Month-by-month (MF AROME D1):**\n"
            "- Dec 2025: ~48% (21 days)\n"
            "- Jan 2026: ~81% (31 days) ← strongest month\n"
            "- Feb 2026: ~70% (23 days)\n\n"
            "**⚠ SEASONAL EDGE — winter only.** Atlantic frontal systems Dec–Feb predictable; "
            "spring/summer convection is not. Conservative winter prior: **65–75%**. "
            "Size up Dec–Feb, scale back or skip Apr–Sep.\n\n"
            "**UKMO for comparison** (data from Aug 2024): ~44% accuracy — consistently worse than MF. "
            "NCEP AIGFS025 starts Jan 7 2026. London Polymarket started Feb 2025 — no 2024 data exists.\n\n"
            "**🔬 Spread Filter (pre-registered 2026-02-24 — forward test only):**\n"
            "In-sample finding: when MF AROME + MF Seamless + MF HD + ICON + DMI all agree within "
            "**1.0°C**, MF AROME hit 80.8% (42/52 days). When spread >1.0°C, only 47.6% (10/21 days). "
            "Caveat: threshold found in-sample; 1.1–1.5°C 'death zone' heavily contaminated by "
            "Christmas week (5/14 days). **Do not change position sizing yet.** "
            "Track for 30+ forward days, then evaluate."
        ),
        "polymarket": {
            "2025-12-11": ("13°C", 13, False), "2025-12-12": ("13°C", 13, False),
            "2025-12-13": ("11°C", 11, False), "2025-12-14": ("11°C", 11, False),
            "2025-12-15": ("11°C", 11, False), "2025-12-16": ("13°C", 13, False),
            "2025-12-17": ("≥11°C", 11, True), "2025-12-18": ("13°C", 13, False),
            "2025-12-19": ("≥12°C", 12, True), "2025-12-20": ("10°C", 10, False),
            "2025-12-21": ("11°C", 11, False), "2025-12-22": ("10°C", 10, False),
            "2025-12-23": ("9°C",   9, False), "2025-12-24": ("7°C",   7, False),
            "2025-12-25": ("6°C",   6, False), "2025-12-26": ("6°C",   6, False),
            "2025-12-27": ("8°C",   8, False), "2025-12-28": ("7°C",   7, False),
            "2025-12-29": ("7°C",   7, False), "2025-12-30": ("7°C",   7, False),
            "2025-12-31": ("5°C",   5, False),
            "2026-01-01": ("6",   6, False), "2026-01-02": ("4",   4, False),
            "2026-01-03": ("3",   3, False), "2026-01-04": ("3",   3, False),
            "2026-01-05": ("2",   2, False), "2026-01-06": ("4+",  4, True),
            "2026-01-07": ("6",   6, False), "2026-01-08": ("8+",  8, True),
            "2026-01-09": ("6+",  6, True),  "2026-01-10": ("3",   3, False),
            "2026-01-11": ("8+",  8, True),  "2026-01-12": ("11", 11, False),
            "2026-01-13": ("11", 11, False), "2026-01-14": ("9+",  9, True),
            "2026-01-15": ("10", 10, False), "2026-01-16": ("10", 10, False),
            "2026-01-17": ("11", 11, False), "2026-01-18": ("11+",11, True),
            "2026-01-19": ("11", 11, False), "2026-01-20": ("10", 10, False),
            "2026-01-21": ("10", 10, False), "2026-01-22": ("9",   9, False),
            "2026-01-23": ("9",   9, False), "2026-01-24": ("10", 10, False),
            "2026-01-25": ("8",   8, False), "2026-01-26": ("6",   6, False),
            "2026-01-27": ("9+",  9, True),  "2026-01-28": ("10", 10, False),
            "2026-01-29": ("7",   7, False), "2026-01-30": ("11", 11, False),
            "2026-01-31": ("11+",11, True),
            "2026-02-01": ("9",   9, False), "2026-02-02": ("8",   8, False),
            "2026-02-03": ("7",   7, False), "2026-02-04": ("10", 10, False),
            "2026-02-05": ("8",   8, False), "2026-02-06": ("12", 12, False),
            "2026-02-07": ("11", 11, False), "2026-02-08": ("11", 11, False),
            "2026-02-09": ("9",   9, False), "2026-02-10": ("12", 12, False),
            "2026-02-11": ("12", 12, False), "2026-02-12": ("11", 11, False),
            "2026-02-13": ("8",   8, False), "2026-02-14": ("6",   6, False),
            "2026-02-15": ("9",   9, False), "2026-02-16": ("10", 10, False),
            "2026-02-17": ("7",   7, False), "2026-02-18": ("6",   6, False),
            "2026-02-19": ("7",   7, False), "2026-02-20": ("12", 12, False),
            "2026-02-21": ("14", 14, False), "2026-02-22": ("14", 14, False),
            "2026-02-23": ("13", 13, False), "2026-02-24": ("16", 16, False),
        },
    },
    "New York": {
        "lat": 40.7769, "lon": -73.8740,
        "timezone": "America/New_York",
        "temperature_unit": "fahrenheit",
        "bucket_style": "range_2f",
        "temp_unit_display": "°F",
        "polymarket_slug": "highest-temperature-in-new-york-on",
        "models": {
            "ncep_nbm_conus":   ("NCEP NBM",      "🇺🇸"),
            "gfs_graphcast025": ("GFS GraphCast", "🌐"),
            "icon_seamless":    ("ICON Seamless", "🇩🇪"),
            "kma_seamless":     ("KMA Seamless",  "🇰🇷"),
            "gem_seamless":     ("GEM Seamless",  "🇨🇦"),
            "kma_gdps":         ("KMA GDPS",      "🇰🇷"),
            "gem_hrdps_continental": ("GEM HRDPS", "🇨🇦"),
            "ncep_aigfs025":    ("NCEP AIGFS",    "🤖"),
        },
        "best_ensemble": {
            "short":      "AVG(NBM+ICON+GraphCast+KMA+GEM)",
            "label":      "AVG(NCEP NBM + ICON Seamless + GFS GraphCast + KMA Seamless + GEM Seamless)",
            "model_keys": ["ncep_nbm_conus", "icon_seamless", "gfs_graphcast025", "kma_seamless", "gem_seamless"],
        },
        "top_model_key":   "ncep_nbm_conus",
        "top_model_label": "NCEP NBM D1",
        "chart_models":    ["ncep_nbm_conus", "gfs_graphcast025", "icon_seamless", "gem_seamless"],
        "notes": (
            "**Best signal:** AVG(NCEP NBM + ICON Seamless + GFS GraphCast + KMA Seamless + GEM Seamless) D1 — "
            "exhaustive search over all 43,795 subsets of top models across **363 dates** (Feb 2025 → Feb 2026) "
            "found **44.1% bucket accuracy** (+6.1pp over single-model best of 37.7%).\n\n"
            "**Station:** LaGuardia Airport (KLGA) — same source as Polymarket (Wunderground KLGA).\n\n"
            "**Bucket:** 2°F wide pairs (e.g. 38-39°F, 40-41°F) measured in whole degrees Fahrenheit. "
            "20 of 48 Open-Meteo models cover NYC.\n\n"
            "**Notable:** NCEP NBM (National Blend of Models) is the top single model at 37.7% — a US-specific "
            "blend that outperforms all global models. GFS GraphCast consistently boosts ensemble accuracy."
        ),
        "polymarket": {
            "2026-01-07": ("48-49°F", 48, 49, 41, 52),
            "2026-01-08": ("48-49°F", 48, 49, 41, 52),
            "2026-01-09": ("≥48°F",   48, None, 37, 48),
            "2026-01-10": ("52-53°F", 52, 53, 43, 54),
            "2026-01-11": ("46-47°F", 46, 47, 41, 52),
            "2026-01-12": ("40-41°F", 40, 41, 37, 48),
            "2026-01-13": ("46-47°F", 46, 47, 41, 52),
            "2026-01-14": ("50-51°F", 50, 51, 41, 52),
            "2026-01-15": ("≥46°F",   46, None, 35, 46),
            "2026-01-16": ("34-35°F", 34, 35, 25, 36),
            "2026-01-17": ("40-41°F", 40, 41, 33, 44),
            "2026-01-18": ("34-35°F", 34, 35, 31, 42),
            "2026-01-19": ("32-33°F", 32, 33, 25, 36),
            "2026-01-20": ("24-25°F", 24, 25, 19, 30),
            "2026-01-21": ("≥36°F",   36, None, 25, 36),
            "2026-01-22": ("≥46°F",   46, None, 35, 46),
            "2026-01-23": ("≥36°F",   36, None, 25, 36),
            "2026-01-24": ("18-19°F", 18, 19, 15, 26),
            "2026-01-25": ("≥22°F",   22, None, 11, 22),
            "2026-01-26": ("26-27°F", 26, 27, 25, 36),
            "2026-01-27": ("22-23°F", 22, 23, 15, 26),
            "2026-01-28": ("24-25°F", 24, 25, 23, 34),
            "2026-01-29": ("24-25°F", 24, 25, 17, 28),
            "2026-01-30": ("18-19°F", 18, 19,  9, 20),
            "2026-01-31": ("24-25°F", 24, 25, 17, 28),
            "2026-02-03": ("32-33°F", 32, 33, 29, 40),
            "2026-02-04": ("32-33°F", 32, 33, 31, 42),
            "2026-02-05": ("32-33°F", 32, 33, 23, 34),
            "2026-02-06": ("32-33°F", 32, 33, 23, 34),
            "2026-02-07": ("≥26°F",   26, None, 15, 26),
            "2026-02-08": ("18-19°F", 18, 19, 15, 26),
            "2026-02-09": ("30-31°F", 30, 31, 23, 34),
            "2026-02-10": ("≥36°F",   36, None, 25, 36),
            "2026-02-11": ("40-41°F", 40, 41, 33, 44),
            "2026-02-12": ("36-37°F", 36, 37, 33, 44),
            "2026-02-13": ("38-39°F", 38, 39, 33, 44),
            "2026-02-14": ("44-45°F", 44, 45, 35, 50),
            "2026-02-15": ("38-39°F", 38, 39, 31, 46),
            "2026-02-16": ("38-39°F", 38, 39, 31, 46),
            "2026-02-17": ("46-47°F", 46, 47, 41, 56),
            "2026-02-18": ("38-39°F", 38, 39, 31, 46),
            "2026-02-19": ("36-37°F", 36, 37, 31, 46),
            "2026-02-20": ("38-39°F", 38, 39, 31, 46),
            "2026-02-21": ("46-47°F", 46, 47, 39, 54),
            "2026-02-22": ("34-35°F", 34, 35, 29, 44),
            "2026-02-23": ("≥36°F",   36, None, 28, 36),
            "2026-02-24": ("32-33°F", 32, 33,   25, 40),
            "2026-02-25": ("42-43°F", 42, 43,   33, 50),
            "2026-02-26": ("46-47°F", 46, 47,   37, 54),
            "2026-02-27": ("40-41°F", 40, 41,   33, 50),
            "2026-02-28": ("≥50°F",   50, None, 37, 50),
        },
    },
    "Atlanta": {
        "lat": 33.6407, "lon": -84.4277,
        "timezone": "America/New_York",
        "temperature_unit": "fahrenheit",
        "bucket_style": "range_2f",
        "temp_unit_display": "°F",
        "polymarket_slug": "highest-temperature-in-atlanta-on",
        "models": {
            "ncep_nbm_conus": ("NCEP NBM",      "🇺🇸"),
            "icon_seamless":  ("ICON Seamless", "🇩🇪"),
            "gem_global":     ("GEM Global",    "🇨🇦"),
            "ncep_aigfs025":  ("NCEP AIGFS",    "🤖"),
            "gem_seamless":   ("GEM Seamless",  "🇨🇦"),
        },
        "best_ensemble": {
            "short":      "NCEP NBM D1",
            "label":      "NCEP NBM (National Blend of Models)",
            "model_keys": ["ncep_nbm_conus"],
        },
        "top_model_key":   "ncep_nbm_conus",
        "top_model_label": "NCEP NBM D1",
        "chart_models":    ["ncep_nbm_conus", "icon_seamless", "gem_global", "ncep_aigfs025"],
        "notes": (
            "**Best signal:** NCEP NBM D1 — **43.8%** accuracy over 48 days (Jan 6–Feb 22 2026). "
            "Exhaustive ensemble search found no combination beats the single model; averaging others in "
            "dilutes NCEP NBM's US-station calibration.\n\n"
            "**Station:** Hartsfield-Jackson Atlanta International Airport (KATL) — Wunderground KATL.\n\n"
            "**Bucket:** 2°F wide pairs in Fahrenheit. 22 of 48 Open-Meteo models cover KATL. "
            "Regional European models don't reach this location.\n\n"
            "**Notable:** NCEP NBM is a US-specific blend specifically calibrated for American airport "
            "stations. It outperforms global models here by ~10 percentage points."
        ),
        "polymarket": {
            "2026-01-06": ("≥60°F",   60, None, None, None),
            "2026-01-07": ("≥68°F",   68, None, None, None),
            "2026-01-08": ("≥66°F",   66, None, None, None),
            "2026-01-09": ("70-71°F", 70,   71, None, None),
            "2026-01-10": ("70-71°F", 70,   71, None, None),
            "2026-01-11": ("54-55°F", 54,   55, None, None),
            "2026-01-12": ("52-53°F", 52,   53, None, None),
            "2026-01-13": ("62-63°F", 62,   63, None, None),
            "2026-01-14": ("52-53°F", 52,   53, None, None),
            "2026-01-15": ("36-37°F", 36,   37, None, None),
            "2026-01-16": ("≥50°F",   50, None, None, None),
            "2026-01-17": ("≥52°F",   52, None, None, None),
            "2026-01-18": ("≥40°F",   40, None, None, None),
            "2026-01-19": ("46-47°F", 46,   47, None, None),
            "2026-01-20": ("≥46°F",   46, None, None, None),
            "2026-01-21": ("54-55°F", 54,   55, None, None),
            "2026-01-22": ("50-51°F", 50,   51, None, None),
            "2026-01-23": ("≥52°F",   52, None, None, None),
            "2026-01-24": ("≥46°F",   46, None, None, None),
            "2026-01-25": ("38-39°F", 38,   39, None, None),
            "2026-01-26": ("36-37°F", 36,   37, None, None),
            "2026-01-27": ("42-43°F", 42,   43, None, None),
            "2026-01-28": ("≥48°F",   48, None, None, None),
            "2026-01-29": ("46-47°F", 46,   47, None, None),
            "2026-01-30": ("≥52°F",   52, None, None, None),
            "2026-01-31": ("≥32°F",   32, None, None, None),
            "2026-02-01": ("36-37°F", 36,   37, None, None),
            "2026-02-02": ("≥50°F",   50, None, None, None),
            "2026-02-03": ("≥56°F",   56, None, None, None),
            "2026-02-04": ("≥56°F",   56, None, None, None),
            "2026-02-05": ("44-45°F", 44,   45, None, None),
            "2026-02-06": ("64-65°F", 64,   65, None, None),
            "2026-02-07": ("≥54°F",   54, None, None, None),
            "2026-02-08": ("≥60°F",   60, None, None, None),
            "2026-02-09": ("66-67°F", 66,   67, None, None),
            "2026-02-10": ("≥66°F",   66, None, None, None),
            "2026-02-11": ("64-65°F", 64,   65, None, None),
            "2026-02-12": ("64-65°F", 64,   65, None, None),
            "2026-02-13": ("≥62°F",   62, None, None, None),
            "2026-02-14": ("62-63°F", 62,   63, None, None),
            "2026-02-15": ("60-61°F", 60,   61, None, None),
            "2026-02-16": ("≥64°F",   64, None, None, None),
            "2026-02-17": ("68-69°F", 68,   69, None, None),
            "2026-02-18": ("≥68°F",   68, None, None, None),
            "2026-02-19": ("≥78°F",   78, None, None, None),
            "2026-02-20": ("78-79°F", 78,   79, None, None),
            "2026-02-21": ("66-67°F", 66,   67, None, None),
            "2026-02-22": ("52-53°F", 52,   53, None, None),
        },
    },
    "Chicago": {
        "lat": 41.9742, "lon": -87.9073,
        "timezone": "America/Chicago",
        "temperature_unit": "fahrenheit",
        "bucket_style": "range_2f",
        "temp_unit_display": "°F",
        "polymarket_slug": "highest-temperature-in-chicago-on",
        "models": {
            "ncep_nbm_conus":       ("NCEP NBM",      "🇺🇸"),
            "ncep_aigfs025":        ("NCEP AIGFS",    "🤖"),
            "gem_seamless":         ("GEM Seamless",  "🇨🇦"),
            "best_match":           ("Best Match",    "🌍"),
            "icon_seamless":        ("ICON Seamless", "🇩🇪"),
        },
        "best_ensemble": {
            "short":      "AVG(NBM+AIGFS+GEM+BestMatch+ICON)",
            "label":      "AVG(NCEP NBM + NCEP AIGFS + GEM Seamless + Best Match + ICON Seamless)",
            "model_keys": ["ncep_nbm_conus", "ncep_aigfs025", "gem_seamless", "best_match", "icon_seamless"],
        },
        "top_model_key":   "ncep_nbm_conus",
        "top_model_label": "NCEP NBM D1",
        "chart_models":    ["ncep_nbm_conus", "ncep_aigfs025", "gem_seamless", "icon_seamless"],
        "notes": (
            "**Best signal:** AVG(NCEP NBM + NCEP AIGFS + GEM Seamless + Best Match + ICON Seamless) D1 — "
            "exhaustive search confirmed **71.9%** (23/32 days) as the accuracy ceiling for Chicago Jan–Feb 2026.\n\n"
            "**Station:** Chicago O'Hare International Airport (KORD) — Wunderground KORD.\n\n"
            "**Bucket:** 2°F wide pairs in Fahrenheit. 24 of 48 Open-Meteo models cover KORD. "
            "Markets started Jan 22, 2026 (32 days total).\n\n"
            "**Notable:** NCEP NBM leads at 62.5% single-model, and the 5-model blend adds +6.3pp. "
            "Chicago's cold January temps (as low as 2°F!) provide strong model differentiation."
        ),
        "polymarket": {
            "2026-01-22": ("24-25°F", 24,   25, None, None),
            "2026-01-23": ("2-3°F",    2,    3, None, None),
            "2026-01-24": ("6-7°F",    6,    7, None, None),
            "2026-01-25": ("≤17°F",  None,  17, None, None),
            "2026-01-26": ("≤11°F",  None,  11, None, None),
            "2026-01-27": ("14-15°F", 14,   15, None, None),
            "2026-01-28": ("≤19°F",  None,  19, None, None),
            "2026-01-29": ("16-17°F", 16,   17, None, None),
            "2026-01-30": ("18-19°F", 18,   19, None, None),
            "2026-01-31": ("24-25°F", 24,   25, None, None),
            "2026-02-01": ("28-29°F", 28,   29, None, None),
            "2026-02-02": ("26-27°F", 26,   27, None, None),
            "2026-02-03": ("28-29°F", 28,   29, None, None),
            "2026-02-04": ("26-27°F", 26,   27, None, None),
            "2026-02-05": ("30-31°F", 30,   31, None, None),
            "2026-02-06": ("38-39°F", 38,   39, None, None),
            "2026-02-07": ("24-25°F", 24,   25, None, None),
            "2026-02-08": ("30-31°F", 30,   31, None, None),
            "2026-02-09": ("38-39°F", 38,   39, None, None),
            "2026-02-10": ("≥42°F",   42, None, None, None),
            "2026-02-11": ("42-43°F", 42,   43, None, None),
            "2026-02-12": ("38-39°F", 38,   39, None, None),
            "2026-02-13": ("≥50°F",   50, None, None, None),
            "2026-02-14": ("50-51°F", 50,   51, None, None),
            "2026-02-15": ("≥46°F",   46, None, None, None),
            "2026-02-16": ("≥52°F",   52, None, None, None),
            "2026-02-17": ("≥54°F",   54, None, None, None),
            "2026-02-18": ("62-63°F", 62,   63, None, None),
            "2026-02-19": ("≥54°F",   54, None, None, None),
            "2026-02-20": ("44-45°F", 44,   45, None, None),
            "2026-02-21": ("≤29°F",  None,  29, None, None),
            "2026-02-22": ("26-27°F", 26,   27, None, None),
        },
    },
    "Miami": {
        "lat": 25.7959, "lon": -80.2870,
        "timezone": "America/New_York",
        "temperature_unit": "fahrenheit",
        "bucket_style": "range_2f",
        "temp_unit_display": "°F",
        "polymarket_slug": "highest-temperature-in-miami-on",
        "models": {
            "gem_global":    ("GEM Global",    "🇨🇦"),
            "ncep_aigfs025": ("NCEP AIGFS",    "🤖"),
            "gem_seamless":  ("GEM Seamless",  "🇨🇦"),
            "gem_regional":  ("GEM Regional",  "🇨🇦"),
            "gfs_graphcast025": ("GFS GraphCast", "🌐"),
        },
        "best_ensemble": {
            "short":      "GEM Global D1",
            "label":      "GEM Global (Canadian Global Model)",
            "model_keys": ["gem_global"],
        },
        "top_model_key":   "gem_global",
        "top_model_label": "GEM Global D1",
        "chart_models":    ["gem_global", "ncep_aigfs025", "gem_seamless", "gem_regional"],
        "notes": (
            "**Best signal:** GEM Global D1 — **59.4%** accuracy over 32 days (Jan 22–Feb 22 2026). "
            "MET Norway Seamless ensemble adds 0% improvement; GEM Global alone is the cleanest signal.\n\n"
            "**Station:** Miami International Airport (KMIA) — Wunderground KMIA.\n\n"
            "**Bucket:** 2°F wide pairs in Fahrenheit. 23 of 48 Open-Meteo models cover KMIA. "
            "Markets started Jan 22, 2026 (32 days total).\n\n"
            "**Notable:** Miami's high and stable temperatures (62–89°F range) make it theoretically "
            "easier to predict, and GEM (Canadian) models dominate here as in NYC. "
            "NCEP NBM, which dominates Chicago/Atlanta, ranks only 3rd here."
        ),
        "polymarket": {
            "2026-01-22": ("76-77°F",  76,   77, None, None),
            "2026-01-23": ("≥82°F",    82, None, None, None),
            "2026-01-24": ("82-83°F",  82,   83, None, None),
            "2026-01-25": ("≥80°F",    80, None, None, None),
            "2026-01-26": ("86-87°F",  86,   87, None, None),
            "2026-01-27": ("66-67°F",  66,   67, None, None),
            "2026-01-28": ("62-63°F",  62,   63, None, None),
            "2026-01-29": ("72-73°F",  72,   73, None, None),
            "2026-01-30": ("68-69°F",  68,   69, None, None),
            "2026-01-31": ("64-65°F",  64,   65, None, None),
            "2026-02-01": ("≤53°F",  None,   53, None, None),
            "2026-02-02": ("≤59°F",  None,   59, None, None),
            "2026-02-03": ("68-69°F",  68,   69, None, None),
            "2026-02-04": ("≤75°F",  None,   75, None, None),
            "2026-02-05": ("≥70°F",    70, None, None, None),
            "2026-02-06": ("66-67°F",  66,   67, None, None),
            "2026-02-07": ("78-79°F",  78,   79, None, None),
            "2026-02-08": ("≥70°F",    70, None, None, None),
            "2026-02-09": ("74-75°F",  74,   75, None, None),
            "2026-02-10": ("≥76°F",    76, None, None, None),
            "2026-02-11": ("76-77°F",  76,   77, None, None),
            "2026-02-12": ("80-81°F",  80,   81, None, None),
            "2026-02-13": ("80-81°F",  80,   81, None, None),
            "2026-02-14": ("78-79°F",  78,   79, None, None),
            "2026-02-15": ("80-81°F",  80,   81, None, None),
            "2026-02-16": ("84-85°F",  84,   85, None, None),
            "2026-02-17": ("80-81°F",  80,   81, None, None),
            "2026-02-18": ("78-79°F",  78,   79, None, None),
            "2026-02-19": ("80-81°F",  80,   81, None, None),
            "2026-02-20": ("82-83°F",  82,   83, None, None),
            "2026-02-21": ("82-83°F",  82,   83, None, None),
            "2026-02-22": ("88-89°F",  88,   89, None, None),
        },
    },
    "Dallas": {
        "lat": 32.8481, "lon": -96.8518,
        "timezone": "America/Chicago",
        "temperature_unit": "fahrenheit",
        "bucket_style": "range_2f",
        "temp_unit_display": "°F",
        "polymarket_slug": "highest-temperature-in-dallas-on",
        "models": {
            "icon_seamless":  ("ICON Seamless", "🇩🇪"),
            "gem_seamless":   ("GEM Seamless",  "🇨🇦"),
            "gem_regional":   ("GEM Regional",  "🇨🇦"),
            "gfs_seamless":   ("GFS Seamless",  "🇺🇸"),
            "gfs_hrrr":       ("GFS HRRR",      "🇺🇸"),
            "ncep_aigfs025":  ("NCEP AIGFS",    "🤖"),
        },
        "best_ensemble": {
            "short":      "AVG(ICON+GEM+GEM Reg+GFS)",
            "label":      "AVG(ICON Seamless + GEM Seamless + GEM Regional + GFS Seamless)",
            "model_keys": ["icon_seamless", "gem_seamless", "gem_regional", "gfs_seamless"],
        },
        "top_model_key":   "icon_seamless",
        "top_model_label": "ICON Seamless D1",
        "chart_models":    ["icon_seamless", "gem_seamless", "gfs_seamless", "gfs_hrrr"],
        "notes": (
            "**Best signal:** AVG(ICON Seamless + GEM Seamless + GEM Regional + GFS Seamless) D1 — "
            "**54.4%** (43/79) over 79 days (Dec 4 2025–Feb 22 2026). "
            "Best single: ICON Seamless 54.4% (43/79), tied with ensemble.\n\n"
            "**Station:** Dallas Love Field (KDAL) — Wunderground KDAL.\n\n"
            "**Bucket:** 2°F wide pairs (e.g. 60-61°F) with ≤ and ≥ edge buckets. Fahrenheit. "
            "MF AROME/UKMO UK 2km don't cover Dallas (HTTP 400). North American & ICON models dominate.\n\n"
            "**Why full Dec 4 window:** NCEP AIGFS only starts Jan 6 and ranks 19/23 at 29.2% here — "
            "not a leader. Full Dec 4 Polymarket history used."
        ),
        "polymarket": {
            "2025-12-04": ("≤54°F",   None, 54,   54, None),
            "2025-12-05": ("≤54°F",   None, 54,   54, None),
            "2025-12-06": ("≤54°F",   None, 54,   54, None),
            "2025-12-09": ("65-66°F",   65,  66, None, None),
            "2025-12-10": ("64-65°F",   64,  65, None, None),
            "2025-12-11": ("66-67°F",   66,  67, None, None),
            "2025-12-12": ("69-70°F",   69,  70, None, None),
            "2025-12-13": ("59-60°F",   59,  60, None, None),
            "2025-12-14": ("44-45°F",   44,  45, None, None),
            "2025-12-15": ("52-53°F",   52,  53, None, None),
            "2025-12-16": ("60-61°F",   60,  61, None, None),
            "2025-12-17": ("≥60°F",     60, None, None,  60),
            "2025-12-18": ("≥70°F",     70, None, None,  70),
            "2025-12-19": ("64-65°F",   64,  65, None, None),
            "2025-12-20": ("≥72°F",     72, None, None,  72),
            "2025-12-21": ("≥60°F",     60, None, None,  60),
            "2025-12-22": ("≥66°F",     66, None, None,  66),
            "2025-12-23": ("80-81°F",   80,  81, None, None),
            "2025-12-24": ("≥74°F",     74, None, None,  74),
            "2025-12-25": ("78-79°F",   78,  79, None, None),
            "2025-12-26": ("86-87°F",   86,  87, None, None),
            "2025-12-27": ("≥74°F",     74, None, None,  74),
            "2025-12-28": ("≥80°F",     80, None, None,  80),
            "2025-12-29": ("≤45°F",   None, 45,   45, None),
            "2025-12-30": ("52-53°F",   52,  53, None, None),
            "2025-12-31": ("66-67°F",   66,  67, None, None),
            "2026-01-01": ("70-71°F",   70,  71, None, None),
            "2026-01-02": ("76-77°F",   76,  77, None, None),
            "2026-01-03": ("64-65°F",   64,  65, None, None),
            "2026-01-04": ("62-63°F",   62,  63, None, None),
            "2026-01-05": ("≥62°F",     62, None, None,  62),
            "2026-01-06": ("76-77°F",   76,  77, None, None),
            "2026-01-07": ("80-81°F",   80,  81, None, None),
            "2026-01-08": ("78-79°F",   78,  79, None, None),
            "2026-01-09": ("70-71°F",   70,  71, None, None),
            "2026-01-10": ("54-55°F",   54,  55, None, None),
            "2026-01-11": ("60-61°F",   60,  61, None, None),
            "2026-01-12": ("≤65°F",   None, 65,   65, None),
            "2026-01-13": ("66-67°F",   66,  67, None, None),
            "2026-01-14": ("62-63°F",   62,  63, None, None),
            "2026-01-15": ("58-59°F",   58,  59, None, None),
            "2026-01-16": ("≤59°F",   None, 59,   59, None),
            "2026-01-17": ("≤43°F",   None, 43,   43, None),
            "2026-01-18": ("60-61°F",   60,  61, None, None),
            "2026-01-19": ("52-53°F",   52,  53, None, None),
            "2026-01-20": ("54-55°F",   54,  55, None, None),
            "2026-01-21": ("64-65°F",   64,  65, None, None),
            "2026-01-22": ("52-53°F",   52,  53, None, None),
            "2026-01-23": ("≥50°F",     50, None, None,  50),
            "2026-01-24": ("32-33°F",   32,  33, None, None),
            "2026-01-25": ("20-21°F",   20,  21, None, None),
            "2026-01-26": ("32-33°F",   32,  33, None, None),
            "2026-01-27": ("40-41°F",   40,  41, None, None),
            "2026-01-28": ("≥40°F",     40, None, None,  40),
            "2026-01-29": ("≥46°F",     46, None, None,  46),
            "2026-01-30": ("48-49°F",   48,  49, None, None),
            "2026-01-31": ("32-33°F",   32,  33, None, None),
            "2026-02-01": ("50-51°F",   50,  51, None, None),
            "2026-02-02": ("≥62°F",     62, None, None,  62),
            "2026-02-03": ("≥62°F",     62, None, None,  62),
            "2026-02-04": ("≥54°F",     54, None, None,  54),
            "2026-02-05": ("≥58°F",     58, None, None,  58),
            "2026-02-06": ("≥82°F",     82, None, None,  82),
            "2026-02-07": ("72-73°F",   72,  73, None, None),
            "2026-02-08": ("≥76°F",     76, None, None,  76),
            "2026-02-09": ("≥74°F",     74, None, None,  74),
            "2026-02-10": ("≥72°F",     72, None, None,  72),
            "2026-02-11": ("≥62°F",     62, None, None,  62),
            "2026-02-12": ("74-75°F",   74,  75, None, None),
            "2026-02-13": ("≥68°F",     68, None, None,  68),
            "2026-02-14": ("68-69°F",   68,  69, None, None),
            "2026-02-15": ("≥68°F",     68, None, None,  68),
            "2026-02-16": ("70-71°F",   70,  71, None, None),
            "2026-02-17": ("76-77°F",   76,  77, None, None),
            "2026-02-18": ("76-77°F",   76,  77, None, None),
            "2026-02-19": ("78-79°F",   78,  79, None, None),
            "2026-02-20": ("≤65°F",   None, 65,   65, None),
            "2026-02-21": ("62-63°F",   62,  63, None, None),
            "2026-02-22": ("58-59°F",   58,  59, None, None),
        },
    },
    "Seattle": {
        "lat": 47.4502, "lon": -122.3088,
        "timezone": "America/Los_Angeles",
        "temperature_unit": "fahrenheit",
        "bucket_style": "range_2f",
        "temp_unit_display": "°F",
        "polymarket_slug": "highest-temperature-in-seattle-on",
        # Top 8 models from exhaustive 37-model sweep, 81 resolved markets Dec 5 2025–Feb 24 2026
        "models": {
            "ncep_nbm_conus":       ("NCEP NBM",        "🇺🇸"),
            "gem_seamless":         ("GEM Seamless",    "🇨🇦"),
            "gem_hrdps_continental":("GEM HRDPS",       "🇨🇦"),
            "knmi_seamless":        ("KNMI Seamless",   "🇳🇱"),
            "dmi_seamless":         ("DMI Seamless",    "🇩🇰"),
            "icon_seamless":        ("ICON Seamless",   "🇩🇪"),
            "ecmwf_ifs025":         ("ECMWF IFS",       "🌍"),
            "gfs_seamless":         ("GFS Seamless",    "🇺🇸"),
        },
        "best_ensemble": {
            "short":      "AVG(ICON+GEM+NBM)",
            "label":      "AVG(ICON Seamless + GEM Seamless + NCEP NBM)",
            "model_keys": ["icon_seamless", "gem_seamless", "ncep_nbm_conus"],
        },
        "top_model_key":   "gem_seamless",
        "top_model_label": "GEM Seamless D1",
        "chart_models":    ["ncep_nbm_conus", "gem_seamless", "icon_seamless", "ecmwf_ifs025"],
        "notes": (
            "**Best signal:** AVG(ICON Seamless + GEM Seamless + NCEP NBM) — bucket accuracy **58.0%** "
            "(47/81 days), MAE **1.07°F**, ≤1°F **55.6%** over 81 resolved markets Dec 5 2025–Feb 24 2026.\n\n"
            "**How found:** Exhaustive search over 263,949 subsets (size 1–8) of 20 valid models. "
            "Size-3 and size-7 combos both hit the ceiling of 58.0% bucket accuracy — the simplest "
            "(ICON + GEM + NBM) is used as it matches the maximum with fewest parameters.\n\n"
            "**⚠ Data-dredging caveat:** 263K combinations on 81 days. In-sample ceiling. "
            "Pre-registered; evaluate on 30+ forward days before sizing up.\n\n"
            "**Single-model baseline:** GEM Seamless — **46.9%** bucket accuracy.\n"
            "**Ensemble gain:** +11.1 pp vs best single (in-sample).\n\n"
            "**37-model sweep singles ranking (81 days, by MAE, in °F):**\n"
            "1. NCEP NBM CONUS — MAE 1.19°F, bucket 39.5%\n"
            "2. GEM Seamless / GEM HRDPS — MAE 1.28°F, bucket **46.9%** ← best single bucket\n"
            "3. KNMI / DMI / MetNO Seamless — MAE 1.28°F, bucket 44.4%\n"
            "4. ECMWF IFS — MAE 1.44°F\n"
            "5. ICON Seamless / Global — MAE 1.45°F, bucket 42.0%\n"
            "Models NOT covering Seattle (KSEA): AROME France, UK Met Office, jma_msm, BOM, SMHI, CFS.\n\n"
            "**Station:** Seattle-Tacoma International Airport (KSEA) — same as Polymarket Wunderground.\n\n"
            "**Bucket:** 2°F wide pairs (e.g. 40-41°F, 42-43°F) with lower/upper boundary buckets. "
            "Markets started Dec 5 2025."
        ),
        "polymarket": {
            "2025-12-05": ("≥54°F",   54, None, None,   54),
            "2025-12-06": ("50-51°F", 50,   51, None, None),
            "2025-12-07": ("54-55°F", 54,   55, None, None),
            "2025-12-08": ("55-56°F", 55,   56, None, None),
            "2025-12-09": ("52-53°F", 52,   53, None, None),
            "2025-12-10": ("56-57°F", 56,   57, None, None),
            "2025-12-11": ("55-56°F", 55,   56, None, None),
            "2025-12-12": ("53-54°F", 53,   54, None, None),
            "2025-12-13": ("52-53°F", 52,   53, None, None),
            "2025-12-14": ("52-53°F", 52,   53, None, None),
            "2025-12-15": ("56-57°F", 56,   57, None, None),
            "2025-12-16": ("54-55°F", 54,   55, None, None),
            "2025-12-17": ("≥52°F",   52, None, None,   52),
            "2025-12-18": ("50-51°F", 50,   51, None, None),
            "2025-12-19": ("42-43°F", 42,   43, None, None),
            "2025-12-20": ("44-45°F", 44,   45, None, None),
            "2025-12-21": ("46-47°F", 46,   47, None, None),
            "2025-12-22": ("50-51°F", 50,   51, None, None),
            "2025-12-23": ("44-45°F", 44,   45, None, None),
            "2025-12-24": ("≥46°F",   46, None, None,   46),
            "2025-12-25": ("42-43°F", 42,   43, None, None),
            "2025-12-26": ("44-45°F", 44,   45, None, None),
            "2025-12-27": ("40-41°F", 40,   41, None, None),
            "2025-12-28": ("40-41°F", 40,   41, None, None),
            "2025-12-29": ("42-43°F", 42,   43, None, None),
            "2025-12-30": ("≥48°F",   48, None, None,   48),
            "2025-12-31": ("≤41°F", None,   41,   41, None),
            "2026-01-01": ("42-43°F", 42,   43, None, None),
            "2026-01-02": ("48-49°F", 48,   49, None, None),
            "2026-01-03": ("≥52°F",   52, None, None,   52),
            "2026-01-04": ("48-49°F", 48,   49, None, None),
            "2026-01-05": ("42-43°F", 42,   43, None, None),
            "2026-01-06": ("44-45°F", 44,   45, None, None),
            "2026-01-07": ("42-43°F", 42,   43, None, None),
            "2026-01-08": ("44-45°F", 44,   45, None, None),
            "2026-01-09": ("50-51°F", 50,   51, None, None),
            "2026-01-11": ("50-51°F", 50,   51, None, None),
            "2026-01-12": ("52-53°F", 52,   53, None, None),
            "2026-01-13": ("56-57°F", 56,   57, None, None),
            "2026-01-14": ("52-53°F", 52,   53, None, None),
            "2026-01-15": ("48-49°F", 48,   49, None, None),
            "2026-01-16": ("52-53°F", 52,   53, None, None),
            "2026-01-17": ("52-53°F", 52,   53, None, None),
            "2026-01-18": ("48-49°F", 48,   49, None, None),
            "2026-01-19": ("48-49°F", 48,   49, None, None),
            "2026-01-20": ("50-51°F", 50,   51, None, None),
            "2026-01-21": ("≤45°F", None,   45,   45, None),
            "2026-01-22": ("40-41°F", 40,   41, None, None),
            "2026-01-23": ("≥42°F",   42, None, None,   42),
            "2026-01-24": ("44-45°F", 44,   45, None, None),
            "2026-01-25": ("44-45°F", 44,   45, None, None),
            "2026-01-26": ("46-47°F", 46,   47, None, None),
            "2026-01-27": ("50-51°F", 50,   51, None, None),
            "2026-01-28": ("50-51°F", 50,   51, None, None),
            "2026-01-29": ("50-51°F", 50,   51, None, None),
            "2026-01-30": ("52-53°F", 52,   53, None, None),
            "2026-01-31": ("56-57°F", 56,   57, None, None),
            "2026-02-01": ("≥52°F",   52, None, None,   52),
            "2026-02-02": ("52-53°F", 52,   53, None, None),
            "2026-02-03": ("≤55°F", None,   55,   55, None),
            "2026-02-04": ("60-61°F", 60,   61, None, None),
            "2026-02-05": ("≤59°F", None,   59,   59, None),
            "2026-02-06": ("56-57°F", 56,   57, None, None),
            "2026-02-07": ("50-51°F", 50,   51, None, None),
            "2026-02-08": ("48-49°F", 48,   49, None, None),
            "2026-02-09": ("50-51°F", 50,   51, None, None),
            "2026-02-10": ("≥52°F",   52, None, None,   52),
            "2026-02-11": ("≥48°F",   48, None, None,   48),
            "2026-02-12": ("≥52°F",   52, None, None,   52),
            "2026-02-13": ("46-47°F", 46,   47, None, None),
            "2026-02-14": ("48-49°F", 48,   49, None, None),
            "2026-02-15": ("46-47°F", 46,   47, None, None),
            "2026-02-16": ("42-43°F", 42,   43, None, None),
            "2026-02-17": ("46-47°F", 46,   47, None, None),
            "2026-02-18": ("40-41°F", 40,   41, None, None),
            "2026-02-19": ("40-41°F", 40,   41, None, None),
            "2026-02-20": ("42-43°F", 42,   43, None, None),
            "2026-02-21": ("48-49°F", 48,   49, None, None),
            "2026-02-22": ("50-51°F", 50,   51, None, None),
            "2026-02-23": ("48-49°F", 48,   49, None, None),
            "2026-02-24": ("46-47°F", 46,   47, None, None),
        },
    },
    "Buenos Aires": {
        "lat": -34.8222, "lon": -58.5358,
        "timezone": "America/Argentina/Buenos_Aires",
        "temperature_unit": "celsius",
        "bucket_style": "exact_1c",
        "temp_unit_display": "°C",
        "polymarket_slug": "highest-temperature-in-buenos-aires-on",
        "models": {
            "ukmo_seamless":  ("UKMO Seamless", "🇬🇧"),
            "ecmwf_ifs025":   ("ECMWF IFS",     "🌍"),
            "best_match":     ("Best Match",    "🌐"),
            "icon_seamless":  ("ICON Seamless", "🇩🇪"),
            "ncep_aigfs025":  ("NCEP AIGFS",    "🤖"),
        },
        "best_ensemble": {
            "short":      "UKMO Seamless D1",
            "label":      "UKMO Seamless (UK Met Office Global)",
            "model_keys": ["ukmo_seamless"],
        },
        "top_model_key":   "ukmo_seamless",
        "top_model_label": "UKMO Seamless D1",
        "chart_models":    ["ukmo_seamless", "ecmwf_ifs025", "best_match", "icon_seamless"],
        "notes": (
            "**Best signal:** UKMO Seamless D1 — **65.4%** accuracy over 78 days (Dec 6 2025–Feb 22 2026). "
            "Exhaustive ensemble search found no combination beats the single model; UKMO Global excels "
            "in the Southern Hemisphere where it is specifically well-calibrated.\n\n"
            "**Station:** Minister Pistarini International Airport (SAEZ/Ezeiza) — Wunderground SAEZ.\n\n"
            "**Bucket:** Exact 1°C integers with ≤ and ≥ edge buckets (e.g. 30°C, 31°C, ≥32°C, ≤29°C). "
            "Temperatures in Celsius. ~20 of 48 Open-Meteo models cover SAEZ.\n\n"
            "**Why full Dec 6 window:** NCEP AIGFS (our best model elsewhere) only starts Jan 6 at SAEZ "
            "and ranks 6th here (50.0%/48 days). UKMO leads from Dec 6 with 29 extra days of data. "
            "NCEP is NOT the leader here — UK Met Office dominates the Southern Hemisphere."
        ),
        "polymarket": {
            "2025-12-06": ("30°C",   30, False),
            "2025-12-07": ("≥28°C",  28, True),   # confirmed from Polymarket (slug API quirk)
            "2025-12-09": ("≥26°C",  26, True),
            "2025-12-10": ("31°C",   31, False),
            "2025-12-11": ("31°C",   31, False),
            "2025-12-12": ("34°C",   34, False),
            "2025-12-13": ("34°C",   34, False),
            "2025-12-14": ("≥25°C",  25, True),
            "2025-12-15": ("27°C",   27, False),
            "2025-12-16": ("25°C",   25, False),
            "2025-12-17": ("≤31°C",  31, None),
            "2025-12-18": ("≤36°C",  36, None),
            "2025-12-19": ("≤33°C",  33, None),
            "2025-12-20": ("28°C",   28, False),
            "2025-12-21": ("≥33°C",  33, True),
            "2025-12-22": ("31°C",   31, False),
            "2025-12-23": ("≥33°C",  33, True),
            "2025-12-24": ("35°C",   35, False),
            "2025-12-25": ("31°C",   31, False),
            "2025-12-26": ("35°C",   35, False),
            "2025-12-27": ("≤35°C",  35, None),
            "2025-12-28": ("≤35°C",  35, None),
            "2025-12-29": ("≤37°C",  37, None),
            "2025-12-30": ("≤38°C",  38, None),
            "2025-12-31": ("≤40°C",  40, None),
            "2026-01-01": ("31°C",   31, False),
            "2026-01-02": ("29°C",   29, False),
            "2026-01-03": ("≤24°C",  24, None),
            "2026-01-04": ("≤26°C",  26, None),
            "2026-01-05": ("≤29°C",  29, None),
            "2026-01-06": ("32°C",   32, False),
            "2026-01-07": ("≤33°C",  33, None),
            "2026-01-08": ("≤26°C",  26, None),
            "2026-01-09": ("26°C",   26, False),
            "2026-01-10": ("≤26°C",  26, None),
            "2026-01-11": ("35°C",   35, False),
            "2026-01-12": ("39°C",   39, False),
            "2026-01-13": ("33°C",   33, False),
            "2026-01-14": ("≤36°C",  36, None),
            "2026-01-15": ("35°C",   35, False),
            "2026-01-16": ("31°C",   31, False),
            "2026-01-17": ("≤35°C",  35, None),
            "2026-01-18": ("≤29°C",  29, None),
            "2026-01-19": ("25°C",   25, False),
            "2026-01-20": ("29°C",   29, False),
            "2026-01-21": ("≤32°C",  32, None),
            "2026-01-22": ("≤32°C",  32, None),
            "2026-01-23": ("≤34°C",  34, None),
            "2026-01-24": ("≤35°C",  35, None),
            "2026-01-25": ("≤35°C",  35, None),
            "2026-01-26": ("≤38°C",  38, None),
            "2026-01-27": ("≤35°C",  35, None),
            "2026-01-28": ("≤34°C",  34, None),
            "2026-01-29": ("35°C",   35, False),
            "2026-01-30": ("31°C",   31, False),
            "2026-01-31": ("32°C",   32, False),
            "2026-02-01": ("≤34°C",  34, None),
            "2026-02-02": ("≤38°C",  38, None),
            "2026-02-03": ("≤37°C",  37, None),
            "2026-02-04": ("≤31°C",  31, None),
            "2026-02-05": ("30°C",   30, False),
            "2026-02-06": ("31°C",   31, False),
            "2026-02-07": ("33°C",   33, False),
            "2026-02-08": ("≤34°C",  34, None),
            "2026-02-09": ("34°C",   34, False),
            "2026-02-10": ("36°C",   36, False),
            "2026-02-11": ("≤32°C",  32, None),
            "2026-02-12": ("31°C",   31, False),
            "2026-02-13": ("29°C",   29, False),
            "2026-02-14": ("29°C",   29, False),
            "2026-02-15": ("30°C",   30, False),
            "2026-02-16": ("33°C",   33, False),
            "2026-02-17": ("31°C",   31, False),
            "2026-02-18": ("≤35°C",  35, None),
            "2026-02-19": ("32°C",   32, False),
            "2026-02-20": ("31°C",   31, False),
            "2026-02-21": ("31°C",   31, False),
            "2026-02-22": ("31°C",   31, False),
        },
    },
    "Paris": {
        "lat": 49.0097, "lon": 2.5479,
        "timezone": "Europe/Paris",
        "temperature_unit": "celsius",
        "bucket_style": "exact_1c",
        "temp_unit_display": "°C",
        "polymarket_slug": "highest-temperature-in-paris-on",
        "models": {
            "ukmo_seamless":              ("UKMO Seamless",    "🇬🇧"),
            "ukmo_uk_deterministic_2km":  ("UKMO UK 2km",      "🇬🇧"),
            "metno_seamless":             ("MET Norway",       "🇳🇴"),
            "ecmwf_ifs025":               ("ECMWF IFS",        "🌍"),
            "meteofrance_seamless":       ("MeteoFrance",      "🇫🇷"),
            "jma_seamless":               ("JMA Seamless",     "🇯🇵"),
            "ncep_aigfs025":              ("NCEP AIGFS",       "🤖"),
        },
        "best_ensemble": {
            "short":      "AVG(UKMO UK 2km + MET Norway)",
            "label":      "Ensemble: UKMO UK 2km + MET Norway Seamless",
            "model_keys": ["ukmo_uk_deterministic_2km", "metno_seamless"],
        },
        "top_model_key":   "ukmo_seamless",
        "top_model_label": "UKMO Seamless D1",
        "chart_models":    ["ukmo_seamless", "ukmo_uk_deterministic_2km", "metno_seamless", "ecmwf_ifs025"],
        "notes": (
            "**Best signal:** AVG(UKMO UK 2km + MET Norway) — **75.0%** (6/8) over 8 resolved days "
            "(Feb 11–21 2026). All three top singles tie at 62.5% (5/8): UKMO Seamless, UKMO UK 2km, "
            "MET Norway. Full 47-model sweep run; MeteoFrance's own models (50%) beaten by UK & Nordic models.\n\n"
            "**Station:** Charles de Gaulle Airport (LFPG) — Wunderground LFPG.\n\n"
            "**Bucket:** Exact 1°C integers with ≤ and ≥ edge buckets. Temperatures in Celsius. "
            "NCEP AIGFS only 25% here — European models dominate.\n\n"
            "**Note:** Paris markets only started on Polymarket on Feb 11 2026. Very small sample (10 days) "
            "— accuracy figures will stabilise as more data accumulates."
        ),
        "polymarket": {
            "2026-02-11": ("13°C",  13, False),
            "2026-02-15": ("≥7°C",   7, True),
            "2026-02-16": ("11°C",  11, False),
            "2026-02-17": ("8°C",    8, False),
            "2026-02-18": ("≥8°C",   8, True),
            "2026-02-19": ("≥10°C", 10, True),
            "2026-02-20": ("11°C",  11, False),
            "2026-02-21": ("16°C",  16, False),
            "2026-02-22": ("14°C",  14, False),
            "2026-02-23": ("16°C",  16, False),
        },
    },
    "Toronto": {
        "lat": 43.6772, "lon": -79.6306,
        "timezone": "America/Toronto",
        "temperature_unit": "celsius",
        "bucket_style": "exact_1c",
        "temp_unit_display": "°C",
        "polymarket_slug": "highest-temperature-in-toronto-on",
        # Top 8 models from exhaustive 38-model sweep, 81 resolved markets Dec 6 2025–Feb 24 2026
        # NCEP NBM CONUS covers Toronto (close to US border) and is the clear #1
        # kma_seamless = kma_gdps (identical scores); meteofrance_seamless = meteofrance_arpege_world
        "models": {
            "ncep_nbm_conus":          ("NCEP NBM",           "🇺🇸"),
            "kma_gdps":                ("KMA GDPS",           "🇰🇷"),
            "meteofrance_arpege_world": ("MF ARPEGE World",   "🇫🇷"),
            "gem_global":              ("GEM Global",         "🇨🇦"),
            "dmi_seamless":            ("DMI Seamless",       "🇩🇰"),
            "ncep_aigfs025":           ("NCEP AI-GFS",        "🤖"),
            "gem_regional":            ("GEM Regional",       "🇨🇦"),
            "gfs_seamless":            ("GFS Seamless",       "🇺🇸"),
            "ecmwf_ifs025":            ("ECMWF IFS",          "🌍"),
        },
        "best_ensemble": {
            "short":      "AVG(KMA+MF+GEM+DMI)",
            "label":      "AVG(KMA GDPS + MF ARPEGE World + GEM Global + DMI Seamless)",
            "model_keys": ["kma_gdps", "meteofrance_arpege_world", "gem_global", "dmi_seamless"],
        },
        "top_model_key":   "ncep_nbm_conus",
        "top_model_label": "NCEP NBM D1",
        "chart_models":    ["ncep_nbm_conus", "kma_gdps", "meteofrance_arpege_world", "gem_global"],
        "notes": (
            "**Best signal:** AVG(KMA GDPS + MF ARPEGE World + GEM Global + DMI Seamless) — "
            "bucket accuracy **53.1%** (43/81 days), MAE **0.691°C**, ≤1°C **70.4%**.\n\n"
            "**How found:** Exhaustive permutation search over all 263,949 subsets (size 1–8) "
            "of 20 valid models, tested on 81 resolved markets Dec 6 2025–Feb 24 2026. "
            "Ceiling was 53.1%, hit by multiple size-3, size-5, and size-6 combos — all sharing "
            "the same KMA + GEM core. Adding more models did not improve bucket accuracy beyond 53.1%.\n\n"
            "**⚠ Data-dredging caveat:** 263K hypotheses on 81 days is significant in-sample "
            "fitting. Treat 53.1% as the ceiling, not a forward-test guarantee. "
            "Pre-registering this ensemble now; evaluate after 30+ new forward days.\n\n"
            "**Single-model baseline:** KMA GDPS — 45.7% bucket accuracy.\n"
            "**Ensemble gain:** +7.4 pp vs best single (in-sample).\n\n"
            "**38-model sweep singles ranking (81 days, by MAE):**\n"
            "1. NCEP NBM CONUS — MAE 0.846°C, ≤1°C 66.7% ← lowest MAE, best for ≤1°C\n"
            "2. KMA GDPS — MAE 0.878°C, bucket 45.7% ← best single by bucket accuracy\n"
            "3. MF ARPEGE World — MAE 0.940°C\n"
            "4. GEM Global — MAE 0.983°C\n"
            "Models NOT covering Toronto: UK Met Office, AROME France/HD, icon_eu/d2, BOM, SMHI.\n\n"
            "**Station:** Toronto Pearson International Airport (CYYZ) — same as Polymarket Wunderground.\n\n"
            "**Bucket:** Exact 1°C integers with lower/upper boundary buckets. Markets started Dec 6 2025."
        ),
        "polymarket": {
            "2025-12-06": ("1°C",    1,  False),
            "2025-12-07": ("-1°C",  -1,  False),
            "2025-12-08": ("-4°C",  -4,  False),
            "2025-12-09": ("0°C",    0,  False),
            "2025-12-10": ("1°C",    1,  False),
            "2025-12-11": ("-6°C",  -6,  False),
            "2025-12-12": ("-1°C",  -1,  False),
            "2025-12-13": ("-2°C",  -2,  False),
            "2025-12-14": ("-7°C",  -7,  False),
            "2025-12-15": ("-5°C",  -5,  False),
            "2025-12-16": ("≥1°C",   1,  True),
            "2025-12-17": ("4°C",    4,  False),
            "2025-12-18": ("≥7°C",   7,  True),
            "2025-12-19": ("≥6°C",   6,  True),
            "2025-12-20": ("≥2°C",   2,  True),
            "2025-12-21": ("≥2°C",   2,  True),
            "2025-12-22": ("≥1°C",   1,  True),
            "2025-12-23": ("4°C",    4,  False),
            "2025-12-24": ("≥3°C",   3,  True),
            "2025-12-25": ("2°C",    2,  False),
            "2025-12-26": ("-2°C",  -2,  False),
            "2025-12-27": ("-3°C",  -3,  False),
            "2025-12-28": ("2°C",    2,  False),
            "2025-12-29": ("5°C",    5,  False),
            "2025-12-30": ("-4°C",  -4,  False),
            "2025-12-31": ("≤-4°C", -4,  None),
            "2026-01-01": ("-7°C",  -7,  False),
            "2026-01-02": ("-4°C",  -4,  False),
            "2026-01-03": ("-3°C",  -3,  False),
            "2026-01-04": ("-1°C",  -1,  False),
            "2026-01-05": ("0°C",    0,  False),
            "2026-01-06": ("1°C",    1,  False),
            "2026-01-07": ("3°C",    3,  False),
            "2026-01-08": ("≥5°C",   5,  True),
            "2026-01-09": ("≥10°C", 10,  True),
            "2026-01-10": ("≥3°C",   3,  True),
            "2026-01-11": ("2°C",    2,  False),
            "2026-01-12": ("0°C",    0,  False),
            "2026-01-13": ("5°C",    5,  False),
            "2026-01-14": ("5°C",    5,  False),
            "2026-01-15": ("-9°C",  -9,  False),
            "2026-01-16": ("≥-1°C", -1,  True),
            "2026-01-17": ("2°C",    2,  False),
            "2026-01-18": ("-6°C",  -6,  False),
            "2026-01-19": ("-5°C",  -5,  False),
            "2026-01-20": ("≥-11°C",-11, True),
            "2026-01-21": ("-1°C",  -1,  False),
            "2026-01-22": ("-1°C",  -1,  False),
            "2026-01-23": ("-9°C",  -9,  False),
            "2026-01-24": ("-11°C", -11, False),
            "2026-01-25": ("-9°C",  -9,  False),
            "2026-01-26": ("-9°C",  -9,  False),
            "2026-01-27": ("-9°C",  -9,  False),
            "2026-01-28": ("-10°C", -10, False),
            "2026-01-29": ("≤-12°C",-12, None),
            "2026-01-30": ("≥-10°C",-10, True),
            "2026-01-31": ("≥-10°C",-10, True),
            "2026-02-01": ("≥-5°C", -5,  True),
            "2026-02-02": ("≥-5°C", -5,  True),
            "2026-02-03": ("-3°C",  -3,  False),
            "2026-02-04": ("-5°C",  -5,  False),
            "2026-02-05": ("-5°C",  -5,  False),
            "2026-02-06": ("-3°C",  -3,  False),
            "2026-02-07": ("-13°C", -13, False),
            "2026-02-08": ("≤-12°C",-12, None),
            "2026-02-09": ("-7°C",  -7,  False),
            "2026-02-10": ("2°C",    2,  False),
            "2026-02-11": ("1°C",    1,  False),
            "2026-02-12": ("-3°C",  -3,  False),
            "2026-02-13": ("-1°C",  -1,  False),
            "2026-02-14": ("2°C",    2,  False),
            "2026-02-15": ("1°C",    1,  False),
            "2026-02-16": ("2°C",    2,  False),
            "2026-02-17": ("≥7°C",   7,  True),
            "2026-02-18": ("3°C",    3,  False),
            "2026-02-19": ("2°C",    2,  False),
            "2026-02-20": ("2°C",    2,  False),
            "2026-02-21": ("2°C",    2,  False),
            "2026-02-22": ("0°C",    0,  False),
            "2026-02-23": ("-1°C",  -1,  False),
            "2026-02-24": ("-4°C",  -4,  False),
            "2026-02-25": ("2°C",    2,  False),
            "2026-02-26": ("-2°C",  -2,  False),
            "2026-02-27": ("7°C",    7,  False),
            "2026-02-28": ("≥6°C",   6,  True),
        },
    },
    "Ankara": {
        "lat": 40.1281, "lon": 32.9951,
        "timezone": "Europe/Istanbul",
        "temperature_unit": "celsius",
        "bucket_style": "exact_1c",
        "temp_unit_display": "°C",
        "polymarket_slug": "highest-temperature-in-ankara-on",
        # Best 6-model ensemble from exhaustive permutation search over 37 resolved days (Jan 22–Feb 27 2026)
        # icon_global, meteofrance_arpege_world, jma_seamless, icon_seamless, ecmwf_ifs025, gfs_graphcast025 → 62.2% bucket accuracy
        "models": {
            "icon_seamless":             ("ICON Seamless",       "🌐"),
            "icon_global":               ("ICON Global",         "🌍"),
            "ecmwf_ifs025":              ("ECMWF IFS",           "🇪🇺"),
            "gfs_graphcast025":          ("GFS GraphCast",       "🤖"),
            "jma_seamless":              ("JMA Seamless",        "🇯🇵"),
            "meteofrance_arpege_world":  ("MF ARPEGE World",     "🇫🇷"),
        },
        "best_ensemble": {
            "short":      "AVG(ICON+MF+JMA+EC+GC)",
            "label":      "AVG(ICON Global + MF ARPEGE + JMA + ICON Seamless + ECMWF + GraphCast)",
            "model_keys": ["icon_global", "meteofrance_arpege_world", "jma_seamless",
                           "icon_seamless", "ecmwf_ifs025", "gfs_graphcast025"],
        },
        "top_model_key":   "icon_seamless",
        "top_model_label": "ICON Seamless D1",
        "chart_models":    ["icon_seamless", "icon_global", "ecmwf_ifs025", "gfs_graphcast025"],
        "notes": (
            "**Best signal:** AVG(ICON Global + MF ARPEGE World + JMA Seamless + ICON Seamless + ECMWF IFS + GFS GraphCast) — "
            "**62.2%** bucket accuracy (23/37 days, Jan 22–Feb 27 2026).\n\n"
            "**How found:** Exhaustive permutation search over all 145,498 subsets (size 1–6) of "
            "**23 Open-Meteo models** that cover LTAC, tested on 37 resolved markets. "
            "Full 37-model sweep run; remaining 14 models returned no data for Ankara. "
            "Best per size: Size 1 → JMA Seamless 46.0%; Size 2 → ICON+JMA 51.4%; "
            "Size 3 → 56.8%; Size 4 → 56.8%; Size 5 → 59.5%; Size 6 → 62.2% (ceiling).\n\n"
            "**Single-model baseline:** JMA Seamless 45.9% (best single by bucket accuracy). "
            "Best by MAE: ICON Seamless & ICON EU (0.822°C, within-1°C 78.4%). "
            "GEM Seamless/Global: 43.2% bucket. UKMO models tested — only 21.6% for Ankara (poor coverage).\n\n"
            "**Ensemble gain:** +16.3 pp vs best single (in-sample).\n\n"
            "**Station:** Esenboğa International Airport (LTAC) — Wunderground LTAC.\n\n"
            "**Bucket:** Exact 1°C integers with ≤ and ≥ edge buckets. Temperatures in Celsius. "
            "Markets started Jan 22, 2026."
        ),
        "polymarket": {
            "2026-01-22": ("≤2°C",   2, None),
            "2026-01-23": ("5°C",    5, False),
            "2026-01-24": ("7°C",    7, False),
            "2026-01-25": ("≥9°C",   9, True),
            "2026-01-26": ("≥7°C",   7, True),
            "2026-01-27": ("≥10°C", 10, True),
            "2026-01-28": ("7°C",    7, False),
            "2026-01-29": ("8°C",    8, False),
            "2026-01-30": ("11°C",  11, False),
            "2026-01-31": ("8°C",    8, False),
            "2026-02-01": ("12°C",  12, False),
            "2026-02-02": ("≥8°C",   8, True),
            "2026-02-03": ("6°C",    6, False),
            "2026-02-04": ("7°C",    7, False),
            "2026-02-05": ("8°C",    8, False),
            "2026-02-06": ("8°C",    8, False),
            "2026-02-07": ("10°C",  10, False),
            "2026-02-08": ("≥10°C", 10, True),
            "2026-02-09": ("9°C",    9, False),
            "2026-02-10": ("6°C",    6, False),
            "2026-02-11": ("7°C",    7, False),
            "2026-02-12": ("10°C",  10, False),
            "2026-02-13": ("11°C",  11, False),
            "2026-02-14": ("11°C",  11, False),
            "2026-02-15": ("14°C",  14, False),
            "2026-02-16": ("15°C",  15, False),
            "2026-02-17": ("13°C",  13, False),
            "2026-02-18": ("9°C",    9, False),
            "2026-02-19": ("5°C",    5, False),
            "2026-02-20": ("11°C",  11, False),
            "2026-02-21": ("13°C",  13, False),
            "2026-02-22": ("7°C",    7, False),
            "2026-02-23": ("7°C",    7, False),
            "2026-02-24": ("8°C",    8, False),
            "2026-02-25": ("6°C",    6, False),
            "2026-02-26": ("5°C",    5, False),
            "2026-02-27": ("5°C",    5, False),
        },
    },
    "Wellington": {
        "lat": -41.3272, "lon": 174.8050,
        "timezone": "Pacific/Auckland",
        "temperature_unit": "celsius",
        "bucket_style": "exact_1c",
        "temp_unit_display": "°C",
        "polymarket_slug": "highest-temperature-in-wellington-on",
        "polymarket_since": "2026-03-01",  # First possible resolution date (markets created Feb 28 for Mar 2+)
        # Top models from Open-Meteo for New Zealand region
        "models": {
            "bom_access_global": ("BOM ACCESS-G",  "🇦🇺"),
            "ukmo_seamless":     ("UKMO Seamless",  "🇬🇧"),
            "ecmwf_ifs025":      ("ECMWF IFS",      "🌍"),
            "icon_seamless":     ("ICON Seamless",  "🇩🇪"),
            "gfs_seamless":      ("GFS Seamless",   "🇺🇸"),
        },
        "best_ensemble": {
            "short":      "AVG(BOM+UKMO+ECMWF+ICON)",
            "label":      "AVG(BOM ACCESS-G + UKMO Seamless + ECMWF IFS + ICON Seamless)",
            "model_keys": ["bom_access_global", "ukmo_seamless", "ecmwf_ifs025", "icon_seamless"],
        },
        "top_model_key":   "bom_access_global",
        "top_model_label": "BOM ACCESS-G D1",
        "chart_models":    ["bom_access_global", "ukmo_seamless", "ecmwf_ifs025", "icon_seamless"],
        "notes": (
            "**Best signal:** Not yet backtested — using BOM ACCESS-G as anchor (Australian Bureau of Meteorology "
            "global model provides excellent coverage of New Zealand). UKMO Global also strong in Southern Hemisphere.\n\n"
            "**Station:** Wellington International Airport (NZWN) — Wunderground NZWN.\n\n"
            "**Bucket:** Exact 1°C integers with ≤ and ≥ edge buckets. Temperatures in Celsius.\n\n"
            "**TODO:** Run backtest once resolved markets are collected."
        ),
        "polymarket": {
        },
    },
}

_OM_PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
_PM_CACHE_PATH = ROOT / "data" / "polymarket_cache.json"
_PM_MONTHS = ["january","february","march","april","may","june","july",
              "august","september","october","november","december"]


def _hround(x: float) -> int:
    """Standard half-up rounding. Python's built-in round() uses banker's rounding
    (round-half-to-even), so e.g. round(6.5)==6. For temperature bucket matching
    we want conventional rounding: 6.5 → 7."""
    import math
    return math.floor(x + 0.5)


def _wins(pred: float, res_int: int, is_plus) -> bool:
    """is_plus=True → ≥ (or higher), False → exact, None → ≤ (or below)"""
    p = _hround(pred)
    if is_plus is True:  return p >= res_int
    if is_plus is None:  return p <= res_int
    return p == res_int


def _wins_nyc(pred_f: float, low, high, bottom_thresh, top_thresh) -> bool:
    """2°F bucket win check for NYC (Fahrenheit markets)."""
    p = _hround(pred_f)
    if low is None:   return p <= (bottom_thresh or high or 999)
    if high is None:  return p >= (top_thresh or low or -999)
    return low <= p <= high


def _parse_pm_celsius(markets: list) -> tuple | None:
    """Parse winning bucket from a Celsius exact_1c Polymarket market list."""
    for mkt in markets:
        raw = mkt.get("outcomePrices", "[]")
        prices = json.loads(raw) if isinstance(raw, str) else raw
        if not prices or float(prices[0]) < 0.9:
            continue
        q = mkt.get("question", "").lower()
        m = re.search(r'(\d+)\s*°c\s*or\s*(higher|above)', q)
        if m:
            t = int(m.group(1))
            return (f"≥{t}°C", t, True)
        m = re.search(r'(\d+)\s*°c\s*or\s*below', q)
        if m:
            t = int(m.group(1))
            return (f"≤{t}°C", t, None)
        m = re.search(r'be\s+(\d+)\s*°c\b', q)
        if m:
            t = int(m.group(1))
            return (f"{t}°C", t, False)
    return None


def _parse_pm_fahrenheit(markets: list) -> tuple | None:
    """Parse winning bucket from a Fahrenheit range_2f Polymarket market list."""
    for mkt in markets:
        raw = mkt.get("outcomePrices", "[]")
        prices = json.loads(raw) if isinstance(raw, str) else raw
        if not prices or float(prices[0]) < 0.9:
            continue
        q = mkt.get("question", "").lower()
        m = re.search(r'(\d+)[-–](\d+)\s*°f', q)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            return (f"{lo}-{hi}°F", lo, hi, None, None)
        m = re.search(r'(\d+)\s*°f\s*or\s*below', q)
        if m:
            t = int(m.group(1))
            return (f"≤{t}°F", None, t, None, None)
        m = re.search(r'(\d+)\s*°f\s*or\s*(higher|above)', q)
        if m:
            t = int(m.group(1))
            return (f"≥{t}°F", t, None, None, None)
    return None


def _pm_fetch_new_resolutions(city: str, slug: str, bucket_style: str,
                               from_date: _date, to_date: _date) -> dict:
    """
    Fetch Polymarket resolutions for dates not yet in the disk cache.
    Returns only newly resolved entries (permanent — written to disk by caller).
    """
    new: dict = {}
    d = from_date
    while d <= to_date:
        ds = d.strftime("%Y-%m-%d")
        mn = _PM_MONTHS[d.month - 1]
        slugs = [
            f"{slug}-{mn}-{d.day}-{d.year}",  # year-specific first (unambiguous)
            f"{slug}-{mn}-{d.day}",
        ]
        for sl in slugs:
            try:
                r = requests.get("https://gamma-api.polymarket.com/events",
                                 params={"slug": sl}, timeout=8)
                if r.status_code != 200 or not r.json():
                    continue
                e = r.json()[0]
                created = e.get("createdAt", "")[:10]
                # Verify this market is for the right date (not a year-collision)
                if created:
                    cdate = datetime.strptime(created, "%Y-%m-%d").date()
                    if not (0 <= (d - cdate).days <= 7):
                        continue
                mkts = e.get("markets", [])
                if bucket_style == "exact_1c":
                    result = _parse_pm_celsius(mkts)
                else:
                    result = _parse_pm_fahrenheit(mkts)
                if result:
                    new[ds] = list(result)  # store as list for JSON serialisation
                break
            except Exception:
                pass
        _time.sleep(0.1)
        d += timedelta(days=1)
    return new


def get_polymarket_for_city(city: str) -> dict:
    """
    Return the full polymarket resolution dict for a city.
    - Hardcoded entries in ACCURACY_CITIES are the historical seed.
    - Any dates after the last hardcoded entry are fetched from Polymarket API
      and persisted to disk so they are never re-fetched.
    Returns tuples suitable for the accuracy computation.
    """
    cfg = ACCURACY_CITIES[city]
    hardcoded: dict = cfg.get("polymarket", {})
    slug: str | None = cfg.get("polymarket_slug")

    if not slug:
        return hardcoded

    # For brand-new cities with no hardcoded data, use polymarket_since as the
    # starting point for auto-fetch (avoids crashing on max({}.keys())).
    since_str: str | None = cfg.get("polymarket_since")
    if not hardcoded and not since_str:
        return hardcoded  # no seed and no since date — nothing to fetch

    # Load disk cache
    disk: dict = {}
    if _PM_CACHE_PATH.exists():
        try:
            disk = json.loads(_PM_CACHE_PATH.read_text()).get(city, {})
        except Exception:
            disk = {}

    # Fetch all dates after the last hardcoded entry up to and including today.
    # We include today because markets often resolve by early evening; the
    # Polymarket API simply returns nothing if not yet resolved, so it's safe.
    if hardcoded:
        last_seed = _date.fromisoformat(max(hardcoded.keys()))
    else:
        # New city: start from polymarket_since (one day before so fetch_start = since)
        last_seed = _date.fromisoformat(since_str) - timedelta(days=1)  # type: ignore[arg-type]
    today = datetime.now(UTC).date()
    fetch_start = last_seed + timedelta(days=1)

    if fetch_start <= today:
        missing = [
            d for d in (
                fetch_start + timedelta(n)
                for n in range((today - fetch_start).days + 1)
            )
            if d.strftime("%Y-%m-%d") not in disk
        ]
        if missing:
            bucket_style = cfg.get("bucket_style", "exact_1c")
            new = _pm_fetch_new_resolutions(
                city, slug, bucket_style, missing[0], missing[-1]
            )
            if new:
                disk.update(new)
                # Persist to disk — resolved markets never change
                all_cache: dict = {}
                if _PM_CACHE_PATH.exists():
                    try:
                        all_cache = json.loads(_PM_CACHE_PATH.read_text())
                    except Exception:
                        pass
                all_cache[city] = disk
                _PM_CACHE_PATH.write_text(json.dumps(all_cache, indent=2))

    # Merge: hardcoded seed + disk cache (disk can override if we ever need to correct)
    merged = dict(hardcoded)
    for ds, entry in disk.items():
        merged[ds] = tuple(entry)  # JSON stored as list; convert back to tuple

    return merged


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pm_resolutions(city: str) -> dict:
    """Short-TTL (5 min) wrapper around get_polymarket_for_city.

    Used by the table renderer to check whether today/yesterday has actually resolved
    *without* waiting for the 1-hour fetch_accuracy_data cache to expire.
    Returns the full merged dict {date_str: (label, res, is_plus, ...)} ready for scoring.
    """
    return get_polymarket_for_city(city)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_accuracy_data(city: str) -> dict:
    """Fetch previous_day1 + previous_day2 for all city-specific models, from first PM date.

    Uses a disk cache to avoid re-fetching already-resolved historical rows.
    Only dates not yet in the disk cache are fetched from the API.
    """
    cfg = ACCURACY_CITIES[city]
    now = datetime.now(UTC)
    end = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    bucket_style = cfg.get("bucket_style", "exact_1c")
    temp_unit    = cfg.get("temperature_unit", "celsius")
    _pm_all = get_polymarket_for_city(city)

    # Load what's already on disk, dropping rows where ALL model predictions are None
    # (these were cached before the previous-runs API had data — re-fetch them)
    _model_keys = list(cfg["models"].keys())
    def _row_has_any_preds(row: dict) -> bool:
        return any(row.get(f"{mk}_d1") is not None for mk in _model_keys)

    cached_rows  = [r for r in _load_accuracy_disk_cache(city) if _row_has_any_preds(r)]
    cached_dates = {r["date"] for r in cached_rows}

    # Determine which dates need fetching
    all_pm_dates = sorted(_pm_all.keys())
    new_dates    = [d for d in all_pm_dates if d not in cached_dates and d <= end]

    if not new_dates:
        return {"rows": cached_rows, "fetched_at": datetime.now(UTC).isoformat()}

    start = min(new_dates)

    raw: dict[str, tuple[dict, dict]] = {}
    for model_key in cfg["models"]:
        params = {
            "latitude": cfg["lat"], "longitude": cfg["lon"],
            "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
            "models": model_key,
            "timezone": cfg["timezone"],
            "start_date": start,
            "end_date": end,
        }
        if temp_unit != "celsius":
            params["temperature_unit"] = temp_unit
        try:
            r = requests.get(_OM_PREV_URL, params=params, timeout=30)
            d = r.json()
            if "error" in d:
                raw[model_key] = ({}, {})
                continue
            times = d["hourly"]["time"]
            v1 = d["hourly"].get("temperature_2m_previous_day1", [])
            v2 = d["hourly"].get("temperature_2m_previous_day2", [])
            daily1: dict[str, list] = defaultdict(list)
            daily2: dict[str, list] = defaultdict(list)
            for t, a, b in zip(times, v1, v2):
                dt = t[:10]
                if a is not None: daily1[dt].append(a)
                if b is not None: daily2[dt].append(b)
            raw[model_key] = (dict(daily1), dict(daily2))
        except Exception:
            raw[model_key] = ({}, {})

    polymarket = get_polymarket_for_city(city)
    ens_keys = cfg["best_ensemble"]["model_keys"]
    rows = []

    # Load snapshot log for this city — used as fallback when Open-Meteo
    # previous_day1 API has a lag (typically 1-2 days behind for recent dates)
    try:
        _snap_raw = json.loads(_MODEL_SNAPSHOT_PATH.read_text(encoding="utf-8")) if _MODEL_SNAPSHOT_PATH.exists() else {}
    except Exception:
        _snap_raw = {}
    _city_snap: dict[str, dict] = _snap_raw.get(city, {})   # {date: {preds: {model_key: temp}}}

    # Only build rows for NEW dates — not for dates already in the cache.
    # Previously the loop ran over ALL polymarket dates, which created duplicate
    # rows for already-cached dates (empty row + cached row → duplicates in UI).
    for date in sorted(new_dates):
        pm_entry = polymarket.get(date)
        if pm_entry is None:
            continue
        if bucket_style == "range_2f":
            lbl, low, high, bottom_thresh, top_thresh = pm_entry
            row: dict = {"date": date, "resolved": lbl}
        else:
            lbl, res_int, is_plus = pm_entry
            row = {"date": date, "resolved": lbl, "res_int": res_int, "is_plus": is_plus}

        def compute_win(pred):
            if pred is None:
                return None
            if bucket_style == "range_2f":
                return _wins_nyc(pred, low, high, bottom_thresh, top_thresh)
            return _wins(pred, res_int, is_plus)  # type: ignore[name-defined]

        # Snapshot fallback: bot's 12Z prediction for this date (from model_snapshot_log.json)
        _snap_preds: dict = _city_snap.get(date, {}).get("preds", {}) if isinstance(_city_snap.get(date), dict) else {}

        for mk in cfg["models"]:
            d1_map, d2_map = raw.get(mk, ({}, {}))
            p1 = _hround(max(d1_map[date]) * 10) / 10 if d1_map.get(date) else None
            p2 = _hround(max(d2_map[date]) * 10) / 10 if d2_map.get(date) else None
            # Fallback to snapshot log when Open-Meteo has no data for this date/model
            if p1 is None and mk in _snap_preds:
                try:
                    p1 = _hround(float(_snap_preds[mk]) * 10) / 10
                except (TypeError, ValueError):
                    pass
            row[f"{mk}_d1"] = p1
            row[f"{mk}_d2"] = p2
            row[f"{mk}_d1_win"] = compute_win(p1)
            row[f"{mk}_d2_win"] = compute_win(p2)

        # Best ensemble — D1
        ens_d1 = [row[f"{k}_d1"] for k in ens_keys if row.get(f"{k}_d1") is not None]
        best_ens_d1 = (_hround(sum(ens_d1) / len(ens_d1) * 10) / 10) if len(ens_d1) == len(ens_keys) else None
        row["best_ens_d1"] = best_ens_d1
        row["best_ens_d1_win"] = compute_win(best_ens_d1)

        # Best ensemble — D2
        ens_d2 = [row[f"{k}_d2"] for k in ens_keys if row.get(f"{k}_d2") is not None]
        best_ens_d2 = (_hround(sum(ens_d2) / len(ens_d2) * 10) / 10) if len(ens_d2) == len(ens_keys) else None
        row["best_ens_d2"] = best_ens_d2
        row["best_ens_d2_win"] = compute_win(best_ens_d2)

        # Hypothesis ensembles (forward-test candidates — not trading signals)
        for hyp in cfg.get("hypothesis_ensembles", []):
            hkeys = hyp["model_keys"]
            hweights = hyp.get("weights")
            hpreds = [row[f"{k}_d1"] for k in hkeys if row.get(f"{k}_d1") is not None]
            if len(hpreds) == len(hkeys):
                if hweights:
                    wavg = sum(p * w for p, w in zip(hpreds, hweights))
                else:
                    wavg = sum(hpreds) / len(hpreds)
                hval = _hround(wavg * 10) / 10
            else:
                hval = None
            row[f"{hyp['key']}_d1"] = hval
            row[f"{hyp['key']}_d1_win"] = compute_win(hval)

        # Spread filter (pre-registered threshold — forward-test instrument)
        sf = cfg.get("spread_filter")
        if sf:
            sf_preds = [row[f"{k}_d1"] for k in sf["model_keys"] if row.get(f"{k}_d1") is not None]
            if len(sf_preds) == len(sf["model_keys"]):
                row["spread_d1"] = round(max(sf_preds) - min(sf_preds), 1)
                row["spread_green"] = row["spread_d1"] <= sf["threshold"]
            else:
                row["spread_d1"] = None
                row["spread_green"] = None

        rows.append(row)

    # Merge new rows with disk cache, deduplicate by date (keep row with most predictions)
    merged: dict[str, dict] = {}
    for r in cached_rows + rows:
        d = r["date"]
        existing = merged.get(d)
        if existing is None:
            merged[d] = r
        else:
            # Prefer whichever row has more non-None model predictions
            def _n_preds(row: dict) -> int:
                return sum(1 for k, v in row.items() if k.endswith("_d1") and v is not None)
            if _n_preds(r) > _n_preds(existing):
                merged[d] = r
    all_rows = sorted(merged.values(), key=lambda r: r["date"])
    _save_accuracy_disk_cache(city, all_rows)
    return {"rows": all_rows, "fetched_at": datetime.now(UTC).isoformat()}


def _build_leaderboard(rows: list[dict], city: str) -> pd.DataFrame:
    cfg = ACCURACY_CITIES[city]
    ens_cfg = cfg["best_ensemble"]

    strategies: list[tuple[str, str, str]] = [
        ("best_ens_d1", f"{ens_cfg['short']} D1", "🏆"),
        ("best_ens_d2", f"{ens_cfg['short']} D2", "📅"),
    ]
    # Hypothesis ensembles (forward-test candidates)
    for hyp in cfg.get("hypothesis_ensembles", []):
        strategies.append((f"{hyp['key']}_d1", f"{hyp['short']} D1", "🧪"))
    for mk, (label, icon) in cfg["models"].items():
        strategies.append((f"{mk}_d1", f"{label} D1", icon))
        strategies.append((f"{mk}_d2", f"{label} D2", icon))

    records = []
    for key, label, icon in strategies:
        win_key = f"{key}_win"
        wins = sum(1 for r in rows if r.get(win_key) is True)
        total = sum(1 for r in rows if r.get(win_key) is not None)
        if total == 0:
            continue
        pct = wins / total * 100
        records.append({
            "Strategy": f"{icon} {label}",
            "Wins": wins,
            "Days": total,
            "Accuracy": pct,
            "_sort": pct,
        })
    df = pd.DataFrame(records).sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
    df["Accuracy"] = df["Accuracy"].map(lambda x: f"{x:.1f}%")
    return df


# ---------------------------------------------------------------------------
# Live spread signal (today's D1 forecast, not historical)
# ---------------------------------------------------------------------------
# Live position prices from Polymarket CLOB
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_position_prices(token_ids: tuple) -> dict[str, float]:
    """Fetch current mark prices for open positions from Polymarket CLOB API.

    Uses POST /prices (batch, up to 100 token IDs per request) which returns the
    best available SELL price for each token — this is the mark-to-market exit
    value for our long YES and long NO positions.

    Returns {token_id: sell_price}. Cached for 5 min.
    """
    prices: dict[str, float] = {}
    if not token_ids:
        return prices
    batch_size = 100
    for i in range(0, len(token_ids), batch_size):
        batch = list(token_ids)[i : i + batch_size]
        try:
            r = requests.post(
                "https://clob.polymarket.com/prices",
                json=[{"token_id": tid} for tid in batch],
                timeout=15,
            )
            if not r.ok:
                continue
            data = r.json()
            for tid, info in data.items():
                try:
                    # SELL key = best price we can exit our position at
                    sell = info.get("SELL") or info.get("BUY")
                    if sell is not None:
                        prices[tid] = float(sell)
                except (ValueError, TypeError, AttributeError):
                    pass
        except Exception:
            pass
    return prices


# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_spread(city: str) -> dict | None:
    """Fetch live D1 (tomorrow's) spread across the spread-filter models.
    Used to show today's GREEN/RED signal before market close."""
    cfg = ACCURACY_CITIES.get(city)
    sf = cfg.get("spread_filter") if cfg else None
    if not sf:
        return None
    from datetime import date, timedelta
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    preds = {}
    for mk in sf["model_keys"]:
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude": cfg["lat"], "longitude": cfg["lon"],
                        "hourly": "temperature_2m", "models": mk,
                        "start_date": tomorrow, "end_date": tomorrow,
                        "timezone": cfg["timezone"]}, timeout=15)
            d = r.json()
            vals = [v for v in d.get("hourly", {}).get("temperature_2m", []) if v is not None]
            if vals:
                preds[mk] = _hround(max(vals) * 10) / 10
        except Exception:
            pass
    fetched_at = datetime.now(UTC).strftime("%H:%M UTC")
    if len(preds) < len(sf["model_keys"]):
        return {"error": f"Only {len(preds)}/{len(sf['model_keys'])} models returned data",
                "preds": preds, "fetched_at": fetched_at}
    spread = round(max(preds.values()) - min(preds.values()), 1)
    # Save 12Z snapshot to disk (locked in at 19:00 UTC)
    _log_model_snapshot(city, tomorrow, preds)
    return {
        "spread": spread,
        "green": spread <= sf["threshold"],
        "threshold": sf["threshold"],
        "preds": preds,
        "target_date": tomorrow,
        "n_models": len(preds),
        "fetched_at": fetched_at,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_models_for_date(city: str, date_str: str) -> dict[str, float]:
    """Fetch what all models *predicted* for a specific date using the previous-runs API.

    Uses `temperature_2m_previous_day1` — the D+1 forecast that was issued the day before
    `date_str`.  Used to populate the "today" pending row so it shows yesterday's D+1
    predictions (what was actually predicted for today) rather than today's D+1 predictions
    (what is predicted for tomorrow).
    """
    cfg = ACCURACY_CITIES.get(city)
    if not cfg:
        return {}
    temp_unit = cfg.get("temperature_unit", "celsius")
    preds: dict[str, float] = {}

    for mk in cfg.get("models", {}):
        params: dict = {
            "latitude": cfg["lat"], "longitude": cfg["lon"],
            "hourly": "temperature_2m_previous_day1",
            "models": mk,
            "start_date": date_str, "end_date": date_str,
            "timezone": cfg["timezone"],
        }
        if temp_unit != "celsius":
            params["temperature_unit"] = temp_unit
        try:
            r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast",
                             params=params, timeout=12)
            d = r.json()
            if "error" in d:
                continue
            vals = [v for v in d.get("hourly", {}).get("temperature_2m_previous_day1", []) if v is not None]
            if vals:
                preds[mk] = _hround(max(vals) * 10) / 10
        except Exception:
            pass

    preds["__ts__"] = datetime.now(UTC).strftime("%H:%M UTC")
    return preds


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_models_d2(city: str) -> dict[str, float]:
    """Fetch live D+2 (48h) predictions for ALL models configured for a city.

    Returns {model_key: predicted_max} using api.open-meteo.com/v1/forecast.
    Used to populate the D+2 pending row as soon as Polymarket spawns the market.
    """
    cfg = ACCURACY_CITIES.get(city)
    if not cfg:
        return {}
    from datetime import date, timedelta
    day2 = (date.today() + timedelta(days=2)).isoformat()
    temp_unit = cfg.get("temperature_unit", "celsius")
    preds: dict[str, float] = {}

    for mk in cfg.get("models", {}):
        params: dict = {
            "latitude": cfg["lat"], "longitude": cfg["lon"],
            "hourly": "temperature_2m",
            "models": mk,
            "start_date": day2, "end_date": day2,
            "timezone": cfg["timezone"],
        }
        if temp_unit != "celsius":
            params["temperature_unit"] = temp_unit
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast",
                             params=params, timeout=12)
            d = r.json()
            if "error" in d:
                continue
            vals = [v for v in d.get("hourly", {}).get("temperature_2m", []) if v is not None]
            if vals:
                preds[mk] = _hround(max(vals) * 10) / 10
        except Exception:
            pass

    preds["__ts__"] = datetime.now(UTC).strftime("%H:%M UTC")
    return preds


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_models_d3(city: str) -> dict[str, float]:
    """Fetch live D+3 (72h) predictions for ALL models configured for a city.

    Returns {model_key: predicted_max} using api.open-meteo.com/v1/forecast.
    Used to populate the D+3 pending row so you can see early pricing opportunities.
    """
    cfg = ACCURACY_CITIES.get(city)
    if not cfg:
        return {}
    from datetime import date, timedelta
    day3 = (date.today() + timedelta(days=3)).isoformat()
    temp_unit = cfg.get("temperature_unit", "celsius")
    preds: dict[str, float] = {}

    for mk in cfg.get("models", {}):
        params: dict = {
            "latitude": cfg["lat"], "longitude": cfg["lon"],
            "hourly": "temperature_2m",
            "models": mk,
            "start_date": day3, "end_date": day3,
            "timezone": cfg["timezone"],
        }
        if temp_unit != "celsius":
            params["temperature_unit"] = temp_unit
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast",
                             params=params, timeout=12)
            d = r.json()
            if "error" in d:
                continue
            vals = [v for v in d.get("hourly", {}).get("temperature_2m", []) if v is not None]
            if vals:
                preds[mk] = _hround(max(vals) * 10) / 10
        except Exception:
            pass

    preds["__ts__"] = datetime.now(UTC).strftime("%H:%M UTC")
    return preds


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_models_live(city: str) -> dict[str, float]:
    """Fetch live D+1 predictions for ALL models configured for a city.

    Returns {model_key: predicted_max} using api.open-meteo.com/v1/forecast.
    Used to populate the pending rows in the day-by-day table.
    """
    cfg = ACCURACY_CITIES.get(city)
    if not cfg:
        return {}
    from datetime import date, timedelta
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    temp_unit = cfg.get("temperature_unit", "celsius")
    preds: dict[str, float] = {}

    # Start with spread-filter preds (already cached) — no extra API calls
    sf = cfg.get("spread_filter")
    if sf:
        live = fetch_live_spread(city)
        if live and "preds" in live:
            preds.update(live["preds"])

    # Fetch remaining models not already covered
    for mk in cfg.get("models", {}):
        if mk in preds:
            continue
        params: dict = {
            "latitude": cfg["lat"], "longitude": cfg["lon"],
            "hourly": "temperature_2m",
            "models": mk,
            "start_date": tomorrow, "end_date": tomorrow,
            "timezone": cfg["timezone"],
        }
        if temp_unit != "celsius":
            params["temperature_unit"] = temp_unit
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast",
                             params=params, timeout=12)
            d = r.json()
            if "error" in d:
                continue
            vals = [v for v in d.get("hourly", {}).get("temperature_2m", []) if v is not None]
            if vals:
                preds[mk] = _hround(max(vals) * 10) / 10
        except Exception:
            pass

    # Save 12Z snapshot to disk (locked in at 19:00 UTC)
    _log_model_snapshot(city, tomorrow, preds)
    # Store fetch timestamp as sentinel key (ignored by model-key lookups)
    preds["__ts__"] = datetime.now(UTC).strftime("%H:%M UTC")
    return preds


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_top_model_live(city: str) -> float | None:
    """Fetch the city's top Open-Meteo model D+1 prediction for any city.

    Works for all cities, whether or not they have a spread_filter configured.
    Falls back to the spread filter preds if available (avoids double fetch).
    """
    cfg = ACCURACY_CITIES.get(city)
    if not cfg:
        return None

    # If city has spread_filter, reuse that data (no extra API call)
    sf = cfg.get("spread_filter")
    if sf:
        live = fetch_live_spread(city)
        if live and "preds" in live:
            top_mk = cfg.get("top_model_key")
            return live["preds"].get(top_mk)

    # No spread filter — fetch the top model directly
    from datetime import date, timedelta
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    top_mk = cfg.get("top_model_key")
    if not top_mk:
        return None
    temp_unit = cfg.get("temperature_unit", "celsius")
    try:
        params: dict = {
            "latitude": cfg["lat"], "longitude": cfg["lon"],
            "hourly": "temperature_2m",
            "models": top_mk,
            "start_date": tomorrow, "end_date": tomorrow,
            "timezone": cfg["timezone"],
        }
        if temp_unit != "celsius":
            params["temperature_unit"] = temp_unit
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
        d = r.json()
        vals = [v for v in d.get("hourly", {}).get("temperature_2m", []) if v is not None]
        if vals:
            return _hround(max(vals) * 10) / 10
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Weather Edge Terminal", layout="wide")
    apply_style()

    with st.sidebar:
        # Next model run trigger
        now_utc = datetime.now(UTC)
        next_dt, next_label = next_model_run_trigger(now_utc)
        mins_away = int((next_dt - now_utc).total_seconds() / 60)
        hours_away, mins_part = divmod(mins_away, 60)
        time_str = f"{hours_away}h {mins_part}m" if hours_away else f"{mins_part}m"
        st.markdown("### Next Model Run")
        st.markdown(
            f"""
<div style="background:#141A22;border:1px solid #1f2937;border-radius:8px;padding:10px 12px;margin-bottom:8px;">
  <div style="font-family:monospace;font-size:0.95rem;color:#4DA6FF;font-weight:700;">{next_label}</div>
  <div style="font-family:monospace;font-size:0.85rem;color:#E6EDF3;">{next_dt.strftime('%H:%M UTC')}</div>
  <div style="color:#888888;font-size:0.78rem;">in {time_str}</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Bot fires immediately at each trigger. Fast scans run for 30 min after each.")
        st.divider()
        st.markdown("### Data Sync")
        st.caption(f"Bot runs on `{VM_NAME}` ({VM_PROJECT}).")
        if st.button("🔄 Sync from VM + Refresh", use_container_width=True,
                     help="Pull latest logs/trades from VM, then clear all caches and reload"):
            with st.spinner("Syncing from VM..."):
                try:
                    _, msg = sync_from_vm()
                    st.success(msg)
                except Exception as exc:
                    st.warning(f"VM sync partial: {exc}")
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.markdown("### Auto-refresh")

    refresh_opt = st.sidebar.selectbox("Interval", options=["off", "30s", "60s", "5m"], index=1)
    if refresh_opt != "off" and st_autorefresh is not None:
        interval_ms = {"30s": 30_000, "60s": 60_000, "5m": 300_000}[refresh_opt]
        st_autorefresh(interval=interval_ms, key="dashboard-refresh")
    st.sidebar.caption("Live position prices refresh every 5 min regardless of page interval.")

    now_stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    mode_env_path = DEFAULT_ENV
    paper_mode, live_mode = load_mode_from_env(mode_env_path)
    mode_text = "LIVE MODE" if live_mode else "PAPER MODE"
    mode_color = RED if live_mode else GREEN

    st.markdown(
        f"""
<div class="panel" style="display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div class="banner-title">⚡ WEATHER EDGE</div>
    <div class="muted">Ensemble Mispricing Terminal</div>
  </div>
  <div style="text-align:right;">
    <div style="font-weight:700;color:{mode_color};">● {mode_text}</div>
    <div class="muted">auto-refresh: {refresh_opt} · updated: {now_stamp}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    tab_trading, tab_accuracy = st.tabs(["⚡ Trading", "📊 Model Accuracy"])

    with tab_trading:
        _render_trading_tab()

    with tab_accuracy:
        _render_accuracy_tab()


def _render_trading_tab() -> None:
    trades_df = load_trades_df()
    resolved_df = load_resolved_df()   # from daily_resolver.py
    _ = load_signals_df()
    positions = load_positions()
    metrics = kpis(trades_df, positions)
    _, live_mode = load_mode_from_env(DEFAULT_ENV)

    # ── KPI bar ──────────────────────────────────────────────────────────────
    # Prefer resolved_df for P&L/win-rate metrics when trades.csv is sparse
    if not resolved_df.empty:
        res_wins   = int((resolved_df["outcome"] == "WIN").sum())
        res_losses = int((resolved_df["outcome"] == "LOSS").sum())
        res_total  = res_wins + res_losses
        res_pnl    = float(resolved_df["pnl_usd"].sum())
        res_wr     = res_wins / res_total if res_total else 0.0
    else:
        res_wins, res_losses, res_total = int(metrics["wins"]), int(metrics["losses"]), 0
        res_pnl = metrics["realized_pnl"]
        res_wr  = metrics["win_rate"]

    # ── Fetch live unrealized PnL from Polymarket CLOB ──────────────────────
    _token_ids = tuple(p["token_id"] for p in positions if p.get("token_id"))
    _live_prices = fetch_live_position_prices(_token_ids) if _token_ids else {}
    _live_ts = datetime.now(UTC).strftime("%H:%M UTC")

    unreal_pnl = 0.0
    for p in positions:
        tid = p.get("token_id")
        if not tid or tid not in _live_prices:
            continue
        cur  = _live_prices[tid]
        fill = float(p.get("fill_price", 0) or 0)
        size = float(p.get("fill_size",  0) or 0)
        unreal_pnl += (cur - fill) * size

    total_pnl = res_pnl + unreal_pnl

    cols = st.columns(7)
    kpi_items = [
        ("Total P&L",         f"${total_pnl:+.2f}",            GREEN if total_pnl >= 0 else RED),
        ("Realized P&L",      f"${res_pnl:+.2f}",              GREEN if res_pnl >= 0 else RED),
        ("Unrealized P&L",    f"${unreal_pnl:+.2f}",           GREEN if unreal_pnl >= 0 else RED),
        ("Win Rate",          f"{res_wr*100:.1f}%",             BLUE),
        ("Open Positions",    f"{int(metrics['open_positions'])}", BLUE),
        ("Resolving Today",   f"{int(metrics['resolving_today'])}", BLUE),
        ("Resolving Tomorrow",f"{int(metrics['resolving_tomorrow'])}", BLUE),
    ]
    for col, (label, value, color) in zip(cols, kpi_items):
        with col:
            st.markdown(
                f"""
<div class="kpi-card">
  <div class="kpi-value" style="color:{color};">{value}</div>
  <div class="kpi-label">{label}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

    # ── Live P&L chart with pulsing dot ──────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # Choose best data source: resolved_df > trades_df
    pnl_df = resolved_df.rename(columns={"resolved_at": "ts", "pnl_usd": "pnl"}) if not resolved_df.empty else pd.DataFrame()
    if pnl_df.empty:
        # Fallback: trades.csv resolved trades
        fallback = realized_trades(trades_df)
        if not fallback.empty:
            pnl_df = fallback.rename(columns={"timestamp": "ts"})

    has_data = not pnl_df.empty and "cum_pnl" in pnl_df.columns

    # Header row with live indicator
    last_pnl   = float(pnl_df["cum_pnl"].iloc[-1]) if has_data else 0.0
    last_ts    = pnl_df["ts"].iloc[-1].strftime("%H:%M UTC") if has_data else "—"
    total_color = "#00FF88" if total_pnl >= 0 else "#FF4444"
    pnl_color   = "#00FF88" if last_pnl  >= 0 else "#FF4444"
    pnl_sign    = "+" if last_pnl >= 0 else ""
    unreal_color = "#00FF88" if unreal_pnl >= 0 else "#FF4444"
    unreal_sign  = "+" if unreal_pnl >= 0 else ""

    st.markdown(
        f"""
<style>
@keyframes pulse {{
  0%   {{ box-shadow: 0 0 0 0 rgba(0,255,136,0.7); opacity:1; }}
  70%  {{ box-shadow: 0 0 0 8px rgba(0,255,136,0); opacity:0.7; }}
  100% {{ box-shadow: 0 0 0 0 rgba(0,255,136,0); opacity:1; }}
}}
.live-dot {{
  display:inline-block; width:10px; height:10px; border-radius:50%;
  background:{total_color}; animation:pulse 1.8s infinite;
  margin-right:7px; vertical-align:middle;
}}
.pnl-header {{
  display:flex; align-items:center; gap:16px; margin-bottom:6px; flex-wrap:wrap;
}}
.pnl-title   {{ color:#E6EDF3; font-size:1.05rem; font-weight:600; letter-spacing:.04em; }}
.pnl-total   {{ color:{total_color}; font-size:1.35rem; font-weight:700; }}
.pnl-sub     {{ color:#aaa; font-size:0.85rem; }}
.pnl-sub-val {{ font-weight:600; }}
.pnl-stamp   {{ color:#888; font-size:0.8rem; margin-left:auto; }}
</style>
<div class="pnl-header">
  <span class="live-dot"></span>
  <span class="pnl-title">TOTAL P&amp;L</span>
  <span class="pnl-total">${total_pnl:+.2f}</span>
  <span class="pnl-sub">
    Realized: <span class="pnl-sub-val" style="color:{pnl_color};">${last_pnl:+.2f}</span>
    &nbsp;+&nbsp;
    Unrealized: <span class="pnl-sub-val" style="color:{unreal_color};">${unreal_pnl:+.2f}</span>
  </span>
  <span class="pnl-stamp">live prices: {_live_ts} · refreshes every 5 min</span>
</div>
        """,
        unsafe_allow_html=True,
    )

    now_ts = datetime.now(UTC).replace(tzinfo=None)
    n_live = len([p for p in positions if p.get("token_id") in _live_prices])

    if not has_data:
        # ── No resolved history yet: live unrealized P&L chart ───────────────
        # Build per-position rows
        _bar_rows = []
        for p in positions:
            tid = p.get("token_id")
            if not tid or tid not in _live_prices:
                continue
            cur  = _live_prices[tid]
            fill = float(p.get("fill_price", 0) or 0)
            size = float(p.get("fill_size",  0) or 0)
            upnl = round((cur - fill) * size, 2)
            _bar_rows.append({
                "label": f"{p.get('city','?')} {p.get('date','?')} {p.get('bucket','?')}",
                "city":  p.get("city", "?"),
                "date":  p.get("date", "?"),
                "bucket": p.get("bucket", "?"),
                "side":  p.get("side", "?"),
                "fill":  fill,
                "cur":   cur,
                "upnl":  upnl,
            })

        if _bar_rows:
            # Toggle
            chart_view = st.radio(
                "View",
                options=["📊 Waterfall", "📈 Cumulative line"],
                horizontal=True,
                label_visibility="collapsed",
                key="unreal_chart_toggle",
            )

            _bar_rows_sorted = sorted(_bar_rows, key=lambda r: r["upnl"])
            bar_labels = [r["label"] for r in _bar_rows_sorted]
            bar_vals   = [r["upnl"]  for r in _bar_rows_sorted]
            bar_colors = ["#00FF88" if v >= 0 else "#FF4444" for v in bar_vals]

            if chart_view == "📊 Waterfall":
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=bar_vals,
                    y=bar_labels,
                    orientation="h",
                    marker_color=bar_colors,
                    text=[f"${v:+.2f}" for v in bar_vals],
                    textposition="outside",
                    hovertemplate="%{y}<br>Unrealized: $%{x:.2f}<extra></extra>",
                    showlegend=False,
                ))
                fig.add_vline(x=0, line_color="#444", line_width=1)
                fig.update_layout(
                    plot_bgcolor=BG,
                    paper_bgcolor=PANEL,
                    font={"color": TEXT, "family": "Inter, Arial, sans-serif"},
                    margin={"l": 10, "r": 60, "t": 10, "b": 20},
                    xaxis={"gridcolor": "#1f2937", "title": "USD"},
                    yaxis={"gridcolor": "#1f2937", "tickfont": {"size": 10}},
                    height=max(300, 22 * len(_bar_rows_sorted)),
                )

            else:
                # Cumulative unrealized PnL line — sorted by date then city
                _line_rows = sorted(_bar_rows, key=lambda r: (r["date"], r["city"]))
                cum = 0.0
                cum_vals, labels, colors, hovers = [], [], [], []
                for r in _line_rows:
                    cum += r["upnl"]
                    cum_vals.append(round(cum, 2))
                    labels.append(r["label"])
                    colors.append("#00FF88" if r["upnl"] >= 0 else "#FF4444")
                    hovers.append(
                        f"{r['city']} · {r['date']} · {r['bucket']}<br>"
                        f"Side: {r['side']}  Entry: {r['fill']:.3f}  Now: {r['cur']:.3f}<br>"
                        f"This position: ${r['upnl']:+.2f}<br>"
                        f"Running total: ${cum:.2f}"
                    )

                line_color = "#00FF88" if cum_vals[-1] >= 0 else "#FF4444"
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(cum_vals))),
                    y=cum_vals,
                    mode="lines+markers",
                    line={"color": line_color, "width": 2},
                    fill="tozeroy",
                    fillcolor="rgba(0,255,136,0.10)" if cum_vals[-1] >= 0 else "rgba(255,68,68,0.10)",
                    marker={
                        "color": colors,
                        "size": 8,
                        "line": {"color": "#141A22", "width": 1},
                    },
                    text=labels,
                    hovertext=hovers,
                    hoverinfo="text",
                    showlegend=False,
                ))
                # Final pulsing dot
                fig.add_trace(go.Scatter(
                    x=[len(cum_vals) - 1],
                    y=[cum_vals[-1]],
                    mode="markers",
                    marker={
                        "color": line_color,
                        "size": 14,
                        "symbol": "circle",
                        "line": {"color": "#141A22", "width": 2},
                    },
                    hovertemplate=f"Total unrealized: ${cum_vals[-1]:+.2f}<extra></extra>",
                    showlegend=False,
                ))
                fig.add_hline(y=0, line_dash="dot", line_color="#333", line_width=1)
                fig.update_layout(
                    plot_bgcolor=BG,
                    paper_bgcolor=PANEL,
                    font={"color": TEXT, "family": "Inter, Arial, sans-serif"},
                    margin={"l": 20, "r": 10, "t": 10, "b": 20},
                    xaxis={
                        "gridcolor": "#1f2937",
                        "tickmode": "array",
                        "tickvals": list(range(len(labels))),
                        "ticktext": [lb.split(" ")[0] for lb in labels],  # just city name
                        "tickangle": -45,
                        "tickfont": {"size": 9},
                    },
                    yaxis={"gridcolor": "#1f2937", "title": "USD", "zeroline": False},
                    height=340,
                )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Live unrealized P&L · {n_live}/{len(positions)} prices · "
                f"total: **${sum(r['upnl'] for r in _bar_rows):+.2f}** · "
                f"resolves start appearing at 10:00 UTC · prices refresh every 5 min"
            )
        else:
            st.info("Fetching live prices… refresh in a moment.")

    else:
        n_total = len(pnl_df)
        fig = go.Figure()

        # Fill area + line
        fig.add_trace(
            go.Scatter(
                x=pnl_df["ts"],
                y=pnl_df["cum_pnl"],
                mode="lines",
                line={"color": pnl_color, "width": 2},
                fill="tozeroy",
                fillcolor="rgba(0,255,136,0.12)" if last_pnl >= 0 else "rgba(255,68,68,0.12)",
                hovertemplate="%{x|%b %d %H:%M}<br>Cumulative P&L: $%{y:.2f}<extra></extra>",
                showlegend=False,
                name="Realized",
            )
        )

        # Individual trade markers (WIN=green dot, LOSS=red dot)
        if "outcome" in pnl_df.columns:
            for outcome, dot_color, symbol in [("WIN", "#00FF88", "circle"), ("LOSS", "#FF4444", "x")]:
                mask = pnl_df["outcome"] == outcome
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=pnl_df.loc[mask, "ts"],
                        y=pnl_df.loc[mask, "cum_pnl"],
                        mode="markers",
                        marker={"color": dot_color, "size": 7, "symbol": symbol, "opacity": 0.85},
                        name=outcome,
                        hovertemplate=(
                            "%{x|%b %d}<br>"
                            + ("city" in pnl_df.columns and "%{customdata[0]} %{customdata[1]}<br>" or "")
                            + f"{outcome}: $%{{y:.2f}}<extra></extra>"
                        ),
                        customdata=pnl_df.loc[mask, ["city", "bucket"]].values if "city" in pnl_df.columns else None,
                    ))

        # Realized endpoint dot
        fig.add_trace(go.Scatter(
            x=[pnl_df["ts"].iloc[-1]],
            y=[last_pnl],
            mode="markers",
            marker={
                "color": pnl_color,
                "size": 12,
                "symbol": "circle",
                "line": {"color": "#141A22", "width": 2},
            },
            hovertemplate=f"Realized: ${last_pnl:+.2f}<extra></extra>",
            showlegend=False,
        ))

        # Live unrealized extension: dashed line from last realized point → total P&L now
        if unreal_pnl != 0.0:
            fig.add_trace(go.Scatter(
                x=[pnl_df["ts"].iloc[-1], now_ts],
                y=[last_pnl, total_pnl],
                mode="lines",
                line={"color": total_color, "width": 2, "dash": "dot"},
                hovertemplate="%{x|%H:%M}<br>Total (incl. unrealized): $%{y:.2f}<extra></extra>",
                name="Unrealized",
                showlegend=True,
            ))
            # Pulsing dot at current total P&L
            fig.add_trace(go.Scatter(
                x=[now_ts],
                y=[total_pnl],
                mode="markers",
                marker={
                    "color": total_color,
                    "size": 14,
                    "symbol": "circle",
                    "line": {"color": "#141A22", "width": 2},
                },
                hovertemplate=f"Now (total): ${total_pnl:+.2f}<extra></extra>",
                showlegend=False,
            ))

        # Zero baseline
        fig.add_hline(y=0, line_dash="dot", line_color="#333", line_width=1)

        fig.update_layout(
            plot_bgcolor=BG,
            paper_bgcolor=PANEL,
            font={"color": TEXT, "family": "Inter, Arial, sans-serif"},
            margin={"l": 20, "r": 10, "t": 10, "b": 20},
            xaxis={"gridcolor": "#1f2937", "tickformat": "%b %d"},
            yaxis={"gridcolor": "#1f2937", "title": "USD", "zeroline": False},
            legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Breakdown strip below chart
        n_w = int((pnl_df["outcome"] == "WIN").sum()) if "outcome" in pnl_df.columns else res_wins
        n_l = int((pnl_df["outcome"] == "LOSS").sum()) if "outcome" in pnl_df.columns else res_losses
        acc = n_w / (n_w + n_l) * 100 if (n_w + n_l) else 0
        st.caption(
            f"**{n_w}W / {n_l}L · {acc:.0f}% accuracy · {n_total} total trades**  "
            f"· live prices fetched for {n_live}/{len(positions)} open positions · prices refresh every 5 min"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Open Positions ────────────────────────────────────────────────────────
    _today_str = datetime.now(UTC).strftime("%Y-%m-%d")
    pos_df = pd.DataFrame(positions)
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    if pos_df.empty:
        st.subheader("OPEN POSITIONS (0)")
        st.info("No open positions — bot is scanning, bets logged when signal fires.")
    else:
        for col in ["city", "station_icao", "date", "bucket", "side", "fill_price", "cost", "fill_size", "token_id"]:
            if col not in pos_df.columns:
                pos_df[col] = ""
        pos_df["fill_price"] = pd.to_numeric(pos_df["fill_price"], errors="coerce").fillna(0.0)
        pos_df["cost"]       = pd.to_numeric(pos_df["cost"],       errors="coerce").fillna(0.0)
        pos_df["fill_size"]  = pd.to_numeric(pos_df["fill_size"],  errors="coerce").fillna(0.0)

        # Live current price and unrealized P&L
        pos_df["live_price"] = pos_df["token_id"].map(lambda t: _live_prices.get(t))
        pos_df["unreal_pnl"] = (
            (pos_df["live_price"].fillna(pos_df["fill_price"]) - pos_df["fill_price"])
            * pos_df["fill_size"]
        ).round(2)

        # Split: active = date >= today OR has a live price; expired = past date with no price
        # (Polymarket stops serving prices for closed markets — positions.json may lag resolver)
        pos_df["_expired"] = (pos_df["date"] < _today_str) & pos_df["live_price"].isna()
        active_df  = pos_df[~pos_df["_expired"]].copy()
        expired_df = pos_df[ pos_df["_expired"]].copy()

        # Show expired banner first if any exist
        if not expired_df.empty:
            exp_dates = sorted(expired_df["date"].unique())
            st.warning(
                f"⚠️ **{len(expired_df)} positions from {len(exp_dates)} past date(s) "
                f"({', '.join(exp_dates)}) are still in positions.json** — "
                f"their markets have closed on Polymarket so no live price is available. "
                f"Run `daily_resolver.py` on the VM to prune them automatically.",
                icon=None,
            )

        st.subheader(f"OPEN POSITIONS ({len(active_df)})")

        if active_df.empty:
            st.info("No active open positions.")
        else:
            active_df["live_price_fmt"] = active_df["live_price"].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "—"
            )
            active_df["unreal_pnl_fmt"] = active_df["unreal_pnl"].apply(
                lambda x: f"${x:+.2f}"
            )
            # Profit if all resolve in our favour: shares × $1 payout − cost paid
            active_df["to_win"] = (active_df["fill_size"] - active_df["cost"]).round(2)
            active_df = active_df.sort_values(["date", "city", "bucket"], ascending=True)

            show_cols = ["city", "station_icao", "date", "bucket", "side",
                         "fill_price", "live_price_fmt", "unreal_pnl_fmt", "cost", "to_win"]

            display_df = active_df[show_cols].rename(columns={
                "city":           "City",
                "station_icao":   "Station",
                "date":           "Date",
                "bucket":         "Bucket",
                "side":           "Side",
                "fill_price":     "Entry",
                "live_price_fmt": "Live Price",
                "unreal_pnl_fmt": "Unreal. P&L",
                "cost":           "Cost ($)",
                "to_win":         "To Win ($)",
            })

            def _colour_unreal(val: str):
                if val.startswith("$+") or (val.startswith("$") and not val.startswith("$-")):
                    return "color: #00FF88; font-weight:600"
                elif val.startswith("$-"):
                    return "color: #FF4444; font-weight:600"
                return ""

            styled = display_df.style.applymap(_colour_unreal, subset=["Unreal. P&L"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            total_to_win = active_df["to_win"].sum()
            total_unreal = active_df["unreal_pnl"].sum()
            n_priced     = int(active_df["live_price"].notna().sum())
            st.caption(
                f"Total exposure: **${active_df['cost'].sum():.2f}** · "
                f"Unrealized P&L: **${total_unreal:+.2f}** ({n_priced}/{len(active_df)} live prices) · "
                f"To win if all ✅: **${total_to_win:.2f}**  ·  prices updated {_live_ts}"
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── A/B Strategy Comparison: LADDER vs CONVICTION ─────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("A/B: LADDER vs CONVICTION")
    if resolved_df.empty or "strategy" not in resolved_df.columns:
        st.info(
            "A/B data starts accumulating after the first daily resolution (10:00 UTC).  \n"
            "**CONVICTION** fires only when its score ≥ 0.38 — computed from: "
            "hot-hand weighted model consensus (40%) · raw consensus count (25%) · "
            "Weather.com + AccuWeather agreement (20%) · GREEN spread (15%).  \n"
            "Days where CONVICTION doesn't fire (score too low) the LADDER still runs as normal."
        )
    else:
        # Separate executed bets from shadow conviction signals
        ladder_df     = resolved_df[resolved_df["strategy"] == "LADDER"].copy()
        conviction_df = resolved_df[resolved_df["strategy"] == "CONVICTION"].copy()
        single_df     = resolved_df[resolved_df["strategy"] == "SINGLE"].copy()
        # Combine LADDER + SINGLE as "executed" for overall stats
        executed_df = pd.concat([ladder_df, single_df], ignore_index=True)

        def _strat_metrics(df: pd.DataFrame) -> dict:
            if df.empty:
                return {"n": 0, "wins": 0, "acc": 0.0, "pnl": 0.0, "staked": 0.0, "roi": 0.0}
            n  = len(df)
            w  = int((df["outcome"] == "WIN").sum())
            p  = float(df["pnl_usd"].sum())
            s  = float(df.get("size_usd", pd.Series(dtype=float)).fillna(0).sum())
            return {"n": n, "wins": w, "acc": w / n * 100, "pnl": p,
                    "staked": s, "roi": p / s * 100 if s else 0.0}

        lm = _strat_metrics(ladder_df)
        cm = _strat_metrics(conviction_df)
        sm = _strat_metrics(single_df)

        # KPI comparison cards
        ab_cols = st.columns(3)
        for col, (label, m, color_if_better) in zip(
            ab_cols,
            [
                ("⚡ LADDER (executed)", lm, None),
                ("🎯 CONVICTION (shadow)", cm, None),
                ("🔹 SINGLE (executed)", sm, None),
            ],
        ):
            pnl_c = GREEN if m["pnl"] >= 0 else RED
            with col:
                st.markdown(
                    f"""
<div class="panel" style="text-align:center; padding:12px 8px;">
  <div style="font-size:0.85rem; font-weight:700; color:#aaa; letter-spacing:.06em; margin-bottom:8px;">{label}</div>
  <div style="font-size:1.5rem; font-weight:700; color:{pnl_c};">${m['pnl']:+.2f}</div>
  <div style="color:#888; font-size:0.8rem; margin-top:4px;">
    {m['wins']}/{m['n']} trades · {m['acc']:.0f}% acc · ROI {m['roi']:+.1f}%
  </div>
</div>""",
                    unsafe_allow_html=True,
                )

        # Head-to-head chart: cumulative PnL for LADDER and CONVICTION over time
        if not ladder_df.empty and not conviction_df.empty:
            ladder_df  = ladder_df.sort_values("resolved_at")
            conviction_df = conviction_df.sort_values("resolved_at")
            ladder_df["cum_pnl_l"]  = ladder_df["pnl_usd"].cumsum()
            conviction_df["cum_pnl_c"] = conviction_df["pnl_usd"].cumsum()

            fig_ab = go.Figure()
            fig_ab.add_trace(go.Scatter(
                x=ladder_df["resolved_at"],
                y=ladder_df["cum_pnl_l"],
                mode="lines+markers",
                name="LADDER",
                line={"color": "#4CC9F0", "width": 2},
                marker={"size": 6},
                hovertemplate="%{x|%b %d}<br>LADDER cum: $%{y:.2f}<extra></extra>",
            ))
            fig_ab.add_trace(go.Scatter(
                x=conviction_df["resolved_at"],
                y=conviction_df["cum_pnl_c"],
                mode="lines+markers",
                name="CONVICTION",
                line={"color": "#F72585", "width": 2, "dash": "dash"},
                marker={"size": 6},
                hovertemplate="%{x|%b %d}<br>CONVICTION cum: $%{y:.2f}<extra></extra>",
            ))
            fig_ab.add_hline(y=0, line_dash="dot", line_color="#333", line_width=1)
            fig_ab.update_layout(
                plot_bgcolor=BG,
                paper_bgcolor=PANEL,
                font={"color": TEXT, "family": "Inter, Arial, sans-serif"},
                margin={"l": 20, "r": 10, "t": 10, "b": 20},
                xaxis={"gridcolor": "#1f2937", "tickformat": "%b %d"},
                yaxis={"gridcolor": "#1f2937", "title": "Cumulative P&L ($)", "zeroline": False},
                legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 12}},
                height=300,
            )
            st.plotly_chart(fig_ab, use_container_width=True)

            # City-level head-to-head table
            l_by_city = ladder_df.groupby("city").agg(
                L_n=("pnl_usd", "count"), L_pnl=("pnl_usd", "sum"),
                L_wins=("outcome", lambda x: (x == "WIN").sum())
            ).reset_index()
            c_by_city = conviction_df.groupby("city").agg(
                C_n=("pnl_usd", "count"), C_pnl=("pnl_usd", "sum"),
                C_wins=("outcome", lambda x: (x == "WIN").sum())
            ).reset_index()
            merged = l_by_city.merge(c_by_city, on="city", how="outer").fillna(0)
            merged["L_acc"]   = (merged["L_wins"] / merged["L_n"].replace(0, 1) * 100).round(0)
            merged["C_acc"]   = (merged["C_wins"] / merged["C_n"].replace(0, 1) * 100).round(0)
            merged["winner"]  = merged.apply(
                lambda r: "🎯 CONVICTION" if r["C_pnl"] > r["L_pnl"] else "⚡ LADDER", axis=1
            )
            merged["L_pnl"]   = merged["L_pnl"].round(2)
            merged["C_pnl"]   = merged["C_pnl"].round(2)
            st.dataframe(
                merged[["city", "L_n", "L_acc", "L_pnl", "C_n", "C_acc", "C_pnl", "winner"]].rename(columns={
                    "city":    "City",
                    "L_n":     "Ladder N", "L_acc": "Ladder Acc%", "L_pnl": "Ladder P&L",
                    "C_n":     "Conv N",   "C_acc": "Conv Acc%",   "C_pnl": "Conv P&L",
                    "winner":  "Better?",
                }),
                use_container_width=True,
                hide_index=True,
            )
            n_l_wins = int((merged["L_pnl"] > merged["C_pnl"]).sum())
            n_c_wins = int((merged["C_pnl"] > merged["L_pnl"]).sum())
            st.caption(
                f"City-level score: LADDER wins {n_l_wins} cities · CONVICTION wins {n_c_wins} cities  "
                f"· resolves update at 10:00 UTC daily"
            )
        else:
            st.caption("Both LADDER and CONVICTION data needed before head-to-head chart appears.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Recent resolutions ────────────────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(f"RECENT RESOLVED TRADES  ({res_wins}W / {res_losses}L)")

    # Use resolved_df first, fall back to trades_df
    if not resolved_df.empty:
        show = resolved_df.sort_values("resolved_at", ascending=False).head(50).copy()
        show["resolved_at"] = show["resolved_at"].dt.strftime("%m-%d %H:%M")
        show_cols = [c for c in ["resolved_at", "city", "target_date", "bucket", "side",
                                  "entry_price", "actual_temp", "outcome", "pnl_usd"] if c in show.columns]
        st.dataframe(
            show[show_cols].rename(columns={
                "resolved_at":  "Resolved",
                "city":         "City",
                "target_date":  "Date",
                "bucket":       "Bucket",
                "side":         "Side",
                "entry_price":  "Entry",
                "actual_temp":  "Actual°",
                "outcome":      "Result",
                "pnl_usd":      "P&L",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        resolved = realized_trades(trades_df)
        if resolved.empty:
            st.info("Awaiting first resolution — resolver runs daily at 10:00 UTC.")
        else:
            latest = resolved.sort_values("timestamp", ascending=False).head(50).copy()
            latest["timestamp"] = latest["timestamp"].dt.strftime("%m-%d %H:%M")
            show_cols = ["timestamp", "city", "date", "bucket", "side", "fill_price", "edge", "pnl", "outcome"]
            st.dataframe(
                latest[show_cols].rename(columns={
                    "timestamp": "Time", "city": "City", "date": "Date",
                    "bucket": "Bucket", "side": "Side", "fill_price": "Entry",
                    "edge": "Edge", "pnl": "PnL", "outcome": "Result",
                }),
                use_container_width=True,
                hide_index=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("TRADING MODE")
    mode_choice = st.radio("Mode", options=["Paper", "Live"], index=1 if live_mode else 0, horizontal=True)
    env_target = st.selectbox("Env file target", options=[str(DEFAULT_ENV), str(VM_ENV)], index=0)
    if env_target == str(VM_ENV):
        st.warning("Writing /etc/weather-bot.env usually requires sudo/root privileges.")
    st.warning("Live mode is still blocked by main.py safety guard unless code is changed.")

    if st.button("Apply Mode Change"):
        target_live = mode_choice == "Live"
        ok, msg = write_mode_to_env(Path(env_target), target_live=target_live)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    restart_cmd = "sudo systemctl restart weather-bot" if env_target == str(VM_ENV) else "python3 main.py"
    st.code(restart_cmd, language="bash")
    st.caption("Manual restart required. Dashboard does not execute privileged restart automatically.")

    st.markdown("### Wallet")
    st.button("Connect Polygon Wallet (Coming Soon)", disabled=True, help="Wallet integration will be added later.")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_accuracy_tab() -> None:
    st.markdown(
        f"""
<div class="panel" style="display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div class="banner-title">📊 MODEL ACCURACY BACKTEST</div>
    <div class="muted">Bucket hit rate vs Polymarket resolved temperatures · Jan–Feb 2026</div>
  </div>
  <div style="text-align:right;">
    <div style="color:{BLUE};font-weight:700;font-size:0.85rem;">Data: Open-Meteo Previous Runs API</div>
    <div class="muted">Refreshes every hour · D1 = T-24h · D2 = T-48h</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    _city_options = list(ACCURACY_CITIES.keys())
    _london_idx = _city_options.index("London") if "London" in _city_options else 0
    city = st.selectbox("City", options=_city_options, index=_london_idx, key="acc_city")
    cfg = ACCURACY_CITIES[city]
    ens_cfg = cfg["best_ensemble"]
    temp_unit_disp = cfg.get("temp_unit_display", "°C")

    with st.spinner(f"Loading model predictions for {city}..."):
        acc = fetch_accuracy_data(city)

    rows = acc["rows"]
    fetched_at = acc.get("fetched_at", "")
    st.caption(f"Last fetched: {fetched_at[:19].replace('T', ' ')} UTC · {len(rows)} market days")

    if not rows:
        since_str = cfg.get("polymarket_since")
        if since_str:
            st.info(
                f"**{city}** markets launched recently — no resolved markets yet. "
                f"First data expected from **{since_str}** onward. "
                "This table will populate automatically once markets resolve."
            )
        else:
            st.warning("No data available. Check API connectivity.")
        return

    # ── Bet Window Indicator ───────────────────────────────────────────────
    _now_utc = datetime.now(UTC)
    _h, _m = _now_utc.hour, _now_utc.minute
    _now_mins = _h * 60 + _m  # minutes since midnight UTC
    _WIN_OPEN  = 18 * 60 + 30   # 18:30 UTC — 12Z models land on Open-Meteo
    _WIN_CLOSE = 19 * 60 + 30   # 19:30 UTC — smart money starts acting
    _LATE      = 21 * 60        # 21:00 UTC — market usually repriced

    if _now_mins < _WIN_OPEN:
        _win_colour = "#f59e0b"   # amber
        _win_icon   = "🟡"
        _win_label  = "PRE-12Z — morning forecast only"
        _win_sub    = f"12Z models land at 18:30 UTC ({(_WIN_OPEN - _now_mins) // 60}h {(_WIN_OPEN - _now_mins) % 60}m away). Predictions are 06Z run — less accurate."
    elif _now_mins < _WIN_CLOSE:
        _win_colour = "#00FF88"   # green
        _win_icon   = "🟢"
        _win_label  = "BET WINDOW OPEN — 12Z landed"
        _win_sub    = "Best forecast of the day. Place your bet now before smart money reprices. Window closes ~19:30 UTC."
    elif _now_mins < _LATE:
        _win_colour = "#f97316"   # orange
        _win_icon   = "🟠"
        _win_label  = "POST-WINDOW — still actionable"
        _win_sub    = "Smart money is starting to act. Edge likely intact but prices tightening. Move quickly."
    else:
        _win_colour = "#ef4444"   # red
        _win_icon   = "🔴"
        _win_label  = "LATE — market likely repriced"
        _win_sub    = "Smart money has usually acted by 21:00 UTC. Check prices carefully — edge may be compressed."

    _utc_str = _now_utc.strftime("%H:%M UTC")
    st.markdown(
        f"""<div style="background:#1a1f2e;border-left:4px solid {_win_colour};border-radius:6px;
padding:10px 16px;margin-bottom:12px;display:flex;align-items:center;gap:16px;">
  <div style="font-size:1.6rem;">{_win_icon}</div>
  <div>
    <div style="color:{_win_colour};font-weight:700;font-size:0.95rem;">{_win_label}</div>
    <div style="color:#9ca3af;font-size:0.78rem;margin-top:2px;">{_win_sub}</div>
  </div>
  <div style="margin-left:auto;color:#6b7280;font-size:0.78rem;white-space:nowrap;">{_utc_str}</div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Section 0: Today's Signal (spread filter, cities that have it) ─────
    sf = cfg.get("spread_filter")
    if sf:
        live = fetch_live_spread(city)
        # Compute ensemble consensus for the signal box
        _sig_ens_cfg = cfg.get("best_ensemble", {})
        _sig_ens_keys = _sig_ens_cfg.get("model_keys", [])
        _live_all = fetch_all_models_live(city) if _sig_ens_keys else {}
        _sig_preds = [_live_all[k] for k in _sig_ens_keys if k in _live_all]
        _sig_cons_text = ""
        if len(_sig_preds) >= 2:
            _sig_mean = sum(_sig_preds) / len(_sig_preds)
            _sig_bucket = _hround(_sig_mean)
            _sig_agree = sum(1 for v in _sig_preds if _hround(v) == _sig_bucket)
            _sig_frac = _sig_agree / len(_sig_preds)
            _sig_icon = "🔵" if _sig_frac >= 0.67 else "🟡" if _sig_frac >= 0.50 else "🔴"
            _sig_cons_text = f"{_sig_icon} Consensus: {_sig_agree}/{len(_sig_preds)} models agree on bucket {_sig_bucket}"

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        sig_col, detail_col = st.columns([1, 2])
        with sig_col:
            if live and "error" not in live:
                colour   = GREEN if live["green"] else "#ef4444"
                signal   = "🟢 GREEN — BET" if live["green"] else "🔴 RED — SIZE DOWN"
                sp_text  = f"{live['spread']:.1f}°C spread (threshold ≤{live['threshold']}°C)"
                mf_pred  = live["preds"].get("meteofrance_arome_france")
                mf_text  = f"MF AROME D1: **{mf_pred:.1f}°C**" if mf_pred else ""
                fetched_ts = live.get("fetched_at", "unknown")
                st.markdown(
                    f"""<div class="kpi-card" style="border-left:4px solid {colour};">
  <div class="kpi-value" style="color:{colour};font-size:1.3rem;">{signal}</div>
  <div class="kpi-label">{sp_text}</div>
  <div class="kpi-label" style="margin-top:4px;">{mf_text}</div>
  <div class="kpi-label" style="margin-top:4px;color:#9ca3af;">{_sig_cons_text}</div>
  <div class="kpi-label" style="margin-top:4px;color:#6b7280;">For {live['target_date']}</div>
  <div class="kpi-label" style="margin-top:6px;color:#f59e0b;font-size:0.75rem;">🕐 Predictions fetched {fetched_ts} — hit ⋮ → Clear cache if stale</div>
</div>""", unsafe_allow_html=True)
            elif live and "error" in live:
                st.warning(f"Spread: {live['error']}")
            else:
                st.info("Spread data unavailable")
        with detail_col:
            if live and "preds" in live:
                st.caption(f"**{sf['label']}** — live D1 forecasts for {live.get('target_date','tomorrow')}")
                model_names = {
                    "meteofrance_arome_france":    "MF AROME",
                    "meteofrance_seamless":        "MF Seamless",
                    "meteofrance_arome_france_hd": "MF AROME HD",
                    "icon_seamless":               "ICON",
                    "dmi_seamless":                "DMI",
                }
                pred_rows = [{"Model": model_names.get(k, k), "D1 Prediction": f"{v:.1f}°C"}
                             for k, v in sorted(live["preds"].items(), key=lambda x: -x[1])]
                st.dataframe(pd.DataFrame(pred_rows), hide_index=True, use_container_width=True, height=210)
                st.caption("⚠ Pre-registered forward test (2026-02-24). Do not resize positions until 30+ days of out-of-sample data.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 0.5: Live WU Station + Commercial Forecasts ─────────────────
    if city in _CITY_ICAO or city in _CITY_WU_STATION:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("🌤 Live Station + Commercial Forecast Comparison")

        # ── Row 1: Live WU station data (actual sensor readings, NOT forecasts) ──
        live_wu = fetch_wu_live_obs(city)
        temp_unit_str = cfg.get("temp_unit_display", "°C")
        u_letter = temp_unit_str.replace("°", "")

        st.markdown("**📡 WU Station — Today's Actual Readings** *(these are what Polymarket resolves against)*")
        lc1, lc2, lc3 = st.columns(3)

        with lc1:
            if live_wu:
                lt = live_wu["latest_temp"]
                st.markdown(
                    f"""<div class="kpi-card" style="border-left:4px solid {GREEN};">
  <div class="kpi-value" style="color:{GREEN};">{lt}{temp_unit_str}</div>
  <div class="kpi-label">🌡 Current Temp (live)</div>
  <div class="kpi-label" style="color:#6b7280;">as of {live_wu['last_time']} · station {live_wu['station_id']}</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="kpi-card" style="border-left:4px solid #6b7280;">
  <div class="kpi-value" style="color:#6b7280;">—</div>
  <div class="kpi-label">🌡 Current Temp (live)</div>
  <div class="kpi-label" style="color:#6b7280;">station unavailable</div>
</div>""", unsafe_allow_html=True)

        with lc2:
            if live_wu:
                mx = live_wu["running_max"]
                st.markdown(
                    f"""<div class="kpi-card" style="border-left:4px solid #FF9F40;">
  <div class="kpi-value" style="color:#FF9F40;">{mx}{temp_unit_str}</div>
  <div class="kpi-label">📈 Today's Max so far</div>
  <div class="kpi-label" style="color:#6b7280;">Polymarket resolution floor · {live_wu['n_obs']} readings today</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="kpi-card" style="border-left:4px solid #6b7280;">
  <div class="kpi-value" style="color:#6b7280;">—</div>
  <div class="kpi-label">📈 Today's Max so far</div>
  <div class="kpi-label" style="color:#6b7280;">—</div>
</div>""", unsafe_allow_html=True)

        with lc3:
            # Show top Open-Meteo model for comparison (works for all cities)
            top_label = cfg.get("top_model_label", "Top Model")
            top_om_val = fetch_top_model_live(city)
            from datetime import date as _d2, timedelta
            tomorrow = (_d2.today() + timedelta(days=1)).isoformat()
            if top_om_val is not None:
                st.markdown(
                    f"""<div class="kpi-card" style="border-left:4px solid {BLUE};">
  <div class="kpi-value" style="color:{BLUE};">{top_om_val:.1f}{temp_unit_str}</div>
  <div class="kpi-label">🔵 {top_label} (D+1 forecast)</div>
  <div class="kpi-label" style="color:#6b7280;">Open-Meteo · predicts {tomorrow}</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""<div class="kpi-card" style="border-left:4px solid {BLUE};">
  <div class="kpi-value" style="color:{BLUE};">—</div>
  <div class="kpi-label">🔵 {top_label} (D+1 forecast)</div>
  <div class="kpi-label" style="color:#6b7280;">Open-Meteo · predicts {tomorrow}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Commercial D+1 forecasts (predictions, NOT sensor readings) ──
        if city in _CITY_ICAO:
            st.markdown(f"**🔮 Commercial D+1 Forecasts** *(independent predictions for tomorrow {tomorrow} — NOT the live station)*")
            with st.spinner("Fetching commercial forecasts..."):
                comm = fetch_commercial_forecasts(city)

            target_date = comm.get("target_date", tomorrow)
            accu_val = comm.get("accu")
            wu_forecast_val = comm.get("wu")
            errors = comm.get("errors", [])
            comm_source = comm.get("source", "api")

            fc1, fc2 = st.columns(2)

            with fc1:
                if accu_val is not None:
                    accu_rounded = _hround(accu_val * 10) / 10
                    st.markdown(
                        f"""<div class="kpi-card" style="border-left:4px solid #FF9F40;">
  <div class="kpi-value" style="color:#FF9F40;">{accu_val:.1f}{temp_unit_str}</div>
  <div class="kpi-label">🔶 AccuWeather FORECAST D+1</div>
  <div class="kpi-label" style="color:#6b7280;">rounds → {accu_rounded:.0f}{temp_unit_str} · for {target_date}</div>
  <div class="kpi-label" style="color:#6b7280;font-size:0.75rem;">source: {comm_source}</div>
</div>""", unsafe_allow_html=True)
                else:
                    error_msgs = " · ".join(errors) if errors else "API unavailable"
                    st.markdown(f"""<div class="kpi-card" style="border-left:4px solid #ef4444;">
  <div class="kpi-value" style="color:#ef4444;">⚠ NO DATA</div>
  <div class="kpi-label">🔶 AccuWeather FORECAST D+1</div>
  <div class="kpi-label" style="color:#ef4444;font-size:0.75rem;">{error_msgs}</div>
</div>""", unsafe_allow_html=True)

            with fc2:
                if wu_forecast_val is not None:
                    wu_rounded = _hround(wu_forecast_val * 10) / 10
                    st.markdown(
                        f"""<div class="kpi-card" style="border-left:4px solid #A855F7;">
  <div class="kpi-value" style="color:#A855F7;">{wu_forecast_val:.1f}{temp_unit_str}</div>
  <div class="kpi-label">🔷 Weather.com/IBM FORECAST D+1</div>
  <div class="kpi-label" style="color:#6b7280;">rounds → {wu_rounded:.0f}{temp_unit_str} · for {target_date}</div>
  <div class="kpi-label" style="color:#6b7280;font-size:0.75rem;">source: {comm_source}</div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="kpi-card" style="border-left:4px solid #ef4444;">
  <div class="kpi-value" style="color:#ef4444;">⚠ NO DATA</div>
  <div class="kpi-label">🔷 Weather.com/IBM FORECAST D+1</div>
  <div class="kpi-label" style="color:#ef4444;font-size:0.75rem;">API unavailable</div>
</div>""", unsafe_allow_html=True)

            # Consensus / divergence signals
            if accu_val is not None and wu_forecast_val is not None:
                diff = abs(accu_val - wu_forecast_val)
                avg = (accu_val + wu_forecast_val) / 2
                agree = diff <= 1.0
                agree_colour = GREEN if agree else "#ef4444"
                agree_text = (
                    f"✅ Forecast agreement (Δ={diff:.1f}{temp_unit_str}) — both predict ~{avg:.1f}{temp_unit_str} → {_hround(avg * 10) / 10:.0f}{temp_unit_str}"
                    if agree else
                    f"⚠️ Forecast disagreement (Δ={diff:.1f}{temp_unit_str}) — AccuWeather {accu_val:.1f}{temp_unit_str} vs Weather.com {wu_forecast_val:.1f}{temp_unit_str}"
                )
                st.markdown(f"<div style='margin-top:8px;font-size:0.85rem;color:{agree_colour};'>{agree_text}</div>", unsafe_allow_html=True)
                if top_om_val is not None:
                    om_diff = abs(avg - top_om_val)
                    if om_diff >= 2.0:
                        st.markdown(
                            f"<div style='margin-top:4px;font-size:0.85rem;color:#FF9F40;'>⚡ Commercial forecasts diverge from {top_label} by {om_diff:.1f}{temp_unit_str} — commercial providers may have better data assimilation for tomorrow.</div>",
                            unsafe_allow_html=True,
                        )

            if errors:
                for err in errors:
                    st.caption(f"⚠ {err}")
            st.caption(f"📝 D+1 forecasts logged to `data/commercial_forecast_log.json` — building dataset from {datetime.now(UTC).strftime('%Y-%m-%d')} onwards.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 1: KPI cards ────────────────────────────────────────────────
    best_wins = sum(1 for r in rows if r.get("best_ens_d1_win") is True)
    best_n    = sum(1 for r in rows if r.get("best_ens_d1_win") is not None)
    best_pct  = best_wins / best_n * 100 if best_n else 0

    top_mk   = cfg["top_model_key"]
    top_wins = sum(1 for r in rows if r.get(f"{top_mk}_d1_win") is True)
    top_n    = sum(1 for r in rows if r.get(f"{top_mk}_d1_win") is not None)
    top_pct  = top_wins / top_n * 100 if top_n else 0

    # Spread-filtered accuracy (for cities with spread_filter)
    sf_low_wins = sum(1 for r in rows if r.get("spread_green") is True  and r.get(f"{top_mk}_d1_win") is True)
    sf_low_n    = sum(1 for r in rows if r.get("spread_green") is True  and r.get(f"{top_mk}_d1_win") is not None)
    sf_hi_wins  = sum(1 for r in rows if r.get("spread_green") is False and r.get(f"{top_mk}_d1_win") is True)
    sf_hi_n     = sum(1 for r in rows if r.get("spread_green") is False and r.get(f"{top_mk}_d1_win") is not None)
    sf_low_pct  = sf_low_wins / sf_low_n * 100 if sf_low_n else 0
    sf_hi_pct   = sf_hi_wins  / sf_hi_n  * 100 if sf_hi_n  else 0

    if sf and sf_low_n > 0:
        kpi_cols = st.columns(5)
        kpi_data = [
            (kpi_cols[0], f"Best Signal ({ens_cfg['short']})", f"{best_pct:.1f}%", GREEN),
            (kpi_cols[1], cfg["top_model_label"],              f"{top_pct:.1f}%",  BLUE),
            (kpi_cols[2], "Market Days Tested",                str(best_n),        BLUE),
            (kpi_cols[3], f"🟢 Spread ≤{sf['threshold']}°C ({sf_low_n}d)",  f"{sf_low_pct:.1f}%", GREEN),
            (kpi_cols[4], f"🔴 Spread >{sf['threshold']}°C ({sf_hi_n}d)",   f"{sf_hi_pct:.1f}%",  "#ef4444"),
        ]
    else:
        kpi_cols = st.columns(4)
        kpi_data = [
            (kpi_cols[0], f"Best Signal ({ens_cfg['short']})", f"{best_pct:.1f}%", GREEN),
            (kpi_cols[1], cfg["top_model_label"],              f"{top_pct:.1f}%",  BLUE),
            (kpi_cols[2], "Market Days Tested",                str(best_n),        BLUE),
            (kpi_cols[3], "Signal Lead Time",                  "T-24h / T-48h",   GRAY),
        ]

    for col, label, val, color in kpi_data:
        with col:
            st.markdown(
                f"""<div class="kpi-card">
  <div class="kpi-value" style="color:{color};">{val}</div>
  <div class="kpi-label">{label}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Leaderboard ──────────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🏆 Model Leaderboard — Bucket Accuracy")
    st.caption("Bucket hit = model's rounded prediction matches the Polymarket winning bucket")

    lb_df = _build_leaderboard(rows, city)

    bar_colors = [GREEN if i == 0 else BLUE if i < 3 else GRAY for i in range(len(lb_df))]
    max_acc = lb_df["Accuracy"].str.replace("%", "").astype(float).max() if not lb_df.empty else 70
    fig_bar = go.Figure(go.Bar(
        x=lb_df["Strategy"],
        y=lb_df["Accuracy"].str.replace("%", "").astype(float),
        marker_color=bar_colors,
        text=lb_df["Accuracy"],
        textposition="outside",
        hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
    ))
    fig_bar.add_hline(y=50, line_dash="dot", line_color=GRAY, annotation_text="50% baseline", annotation_font_color=GRAY)
    fig_bar.update_layout(
        plot_bgcolor=BG, paper_bgcolor=PANEL,
        font={"color": TEXT, "family": "Inter, Arial, sans-serif"},
        margin={"l": 10, "r": 10, "t": 10, "b": 130},
        xaxis={"gridcolor": "#1f2937", "tickangle": -35},
        yaxis={"gridcolor": "#1f2937", "title": "Bucket Accuracy %", "range": [0, min(max_acc + 15, 100)]},
        height=420, showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(lb_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 3: Day-by-day table ─────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📅 Day-by-Day Predictions vs Polymarket (D1 / T-24h)")

    # Model filter
    all_model_options = {f"{icon} {label}": mk for mk, (label, icon) in cfg["models"].items()}
    filter_key = f"model_filter_{city}"
    selected_labels = st.multiselect(
        "Show models (leave empty = show all)",
        options=list(all_model_options.keys()),
        default=[],
        key=filter_key,
        placeholder="Filter models...",
    )
    active_models = {lbl: all_model_options[lbl] for lbl in selected_labels} if selected_labels else all_model_options

    fmt_val = lambda v: f"{v:.0f}{temp_unit_disp}" if cfg.get("bucket_style") == "range_2f" else f"{v:.1f}{temp_unit_disp}"

    # Load commercial forecast log for this city
    comm_log = _load_commercial_log().get(city, {})
    has_comm_data = bool(comm_log)

    # ── Pending rows (today + tomorrow, no Polymarket resolution yet) ─────────
    from datetime import date as _today_date
    # Use the 5-min-TTL polymarket fetch so newly resolved days appear immediately
    # without waiting for the 1-hour fetch_accuracy_data cache to expire.
    _fresh_pm     = fetch_pm_resolutions(city)
    resolved_dates = {r["date"] for r in rows} | set(_fresh_pm.keys())
    _today_str    = _today_date.today().isoformat()
    _tomorrow_str = (_today_date.today() + timedelta(days=1)).isoformat()
    _day2_str     = (_today_date.today() + timedelta(days=2)).isoformat()
    _day3_str     = (_today_date.today() + timedelta(days=3)).isoformat()

    def _make_pending_row(date_str: str, label: str) -> dict | None:
        """Build a pending/live row using Open-Meteo forecasts appropriate for date_str.

        D+3 row → live 72h forecast (fetch_all_models_d3)
        D+2 row → live 48h forecast (fetch_all_models_d2)
        D+1 row → today's D+1 run (fetch_all_models_live)
        D+0 row → yesterday's D+1 run via previous-runs API
        """
        from datetime import date as _d, timedelta
        _now_utc_pr = datetime.now(UTC)
        _is_tomorrow = date_str == (_d.today() + timedelta(days=1)).isoformat()
        _is_day2     = date_str == (_d.today() + timedelta(days=2)).isoformat()
        _is_day3     = date_str == (_d.today() + timedelta(days=3)).isoformat()
        _in_12z_window = _now_utc_pr.hour * 60 + _now_utc_pr.minute >= 18 * 60 + 30

        if _is_day3:
            all_preds = fetch_all_models_d3(city)
            fetch_ts  = all_preds.get("__ts__", "?") + " D+3 est."
        elif _is_day2:
            all_preds = fetch_all_models_d2(city)
            fetch_ts  = all_preds.get("__ts__", "?") + " D+2 est."
        elif _is_tomorrow:
            if _in_12z_window:
                # After 18:30 UTC: live API has 12Z data — use it and snapshot it
                all_preds = fetch_all_models_live(city)
                fetch_ts  = all_preds.get("__ts__", "?")
            else:
                # Before 18:30 UTC: only trust snapshot if it was itself logged after 18:30 UTC
                # (i.e., it contains real 12Z data, not morning 00Z/06Z junk).
                snap = _load_model_snapshot(city, date_str)
                snap_is_valid_12z = False
                if snap:
                    snap_preds, snap_logged = snap
                    try:
                        snap_dt = datetime.fromisoformat(snap_logged)
                        snap_is_valid_12z = (snap_dt.hour * 60 + snap_dt.minute) >= (18 * 60 + 30)
                    except Exception:
                        snap_is_valid_12z = False

                if snap and snap_is_valid_12z:
                    all_preds = dict(snap_preds)
                    snap_time = datetime.fromisoformat(snap_logged).strftime("%H:%M UTC")
                    fetch_ts  = f"12Z snap {snap_time}"
                    # Fill in any models the snapshot didn't capture (e.g. old cron version)
                    all_model_keys = set(cfg["models"].keys())
                    missing_keys = all_model_keys - set(snap_preds.keys())
                    if missing_keys:
                        live_fill = fetch_all_models_live(city)
                        for mk in missing_keys:
                            if mk in live_fill:
                                all_preds[mk] = live_fill[mk]
                else:
                    # No valid 12Z snapshot — fall back to live (will be 00Z, warn user)
                    all_preds = fetch_all_models_live(city)
                    fetch_ts  = all_preds.get("__ts__", "?") + " ⚠pre-12Z"
        else:
            # Today (or any other past/current date): use yesterday's D+1 run
            all_preds = fetch_models_for_date(city, date_str)
            fetch_ts  = all_preds.get("__ts__", "?")

        live = fetch_live_spread(city) if cfg.get("spread_filter") else None

        top_mk = cfg.get("top_model_key")
        top_val = all_preds.get(top_mk) if top_mk else None
        if top_val is None:
            return None  # no data at all — skip

        ens_mk_list = cfg["best_ensemble"]["model_keys"]
        ens_preds = [all_preds[k] for k in ens_mk_list if k in all_preds]
        if len(ens_preds) == len(ens_mk_list):
            ens_val = _hround(sum(ens_preds) / len(ens_preds) * 10) / 10
        else:
            ens_val = top_val

        # Consensus: fraction of best-ensemble models landing in same bucket as ensemble mean
        ens_bucket = _hround(ens_val)
        n_agree = sum(1 for v in ens_preds if _hround(v) == ens_bucket)
        n_total = len(ens_preds)
        consensus = n_agree / n_total if n_total > 0 else 0.0
        cons_icon = "🔵" if consensus >= 0.67 else "🟡" if consensus >= 0.50 else "🔴"
        cons_cell = f"{cons_icon} {n_agree}/{n_total} ({consensus:.0%})"

        row_d: dict = {
            "Date":           f"{date_str} 📍 🕐{fetch_ts}",
            "Resolved":       label,
            ens_cfg["short"]: f"{fmt_val(ens_val)} ⏳",
        }

        # Spread column — only show live spread for tomorrow (today's run predicting tomorrow).
        # For today's row we computed spread from yesterday's run (all_preds), so derive it.
        if cfg.get("spread_filter"):
            from datetime import date as _d2, timedelta
            sf_keys = cfg["spread_filter"]["model_keys"]
            sf_thr  = cfg["spread_filter"]["threshold"]
            if date_str == (_d2.today() + timedelta(days=1)).isoformat() and live and "spread" in live:
                sp = live["spread"]
                row_d["Spread"] = f"{'🟢' if live['green'] else '🔴'} {sp:.1f}°C"
            else:
                sf_vals = [all_preds[k] for k in sf_keys if k in all_preds]
                if len(sf_vals) >= 2:
                    sp = max(sf_vals) - min(sf_vals)
                    row_d["Spread"] = f"{'🟢' if sp <= sf_thr else '🔴'} {sp:.1f}°C"

        # Consensus column — always shown for pending rows
        if n_total >= 2:
            row_d["Consensus"] = cons_cell

        # Hypothesis ensembles — all constituent models now available
        for hyp in cfg.get("hypothesis_ensembles", []):
            hkeys = hyp["model_keys"]
            hweights = hyp.get("weights")
            hpreds = [all_preds[k] for k in hkeys if k in all_preds]
            if len(hpreds) == len(hkeys):
                if hweights:
                    hval = sum(p * w for p, w in zip(hpreds, hweights))
                else:
                    hval = sum(hpreds) / len(hpreds)
                row_d[f"🧪 {hyp['short']}"] = f"{fmt_val(_hround(hval * 10) / 10)} ⏳"

        # Commercial forecasts — D+3/D+2 come from fetch_commercial_forecasts (accu_d3/wu_d3, accu_d2/wu_d2),
        # D+1 falls back to a live fetch when the disk log has no entry yet (cron not run today),
        # D+0 reads strictly from disk (yesterday's cron snapshot); shows "—" if missing.
        if _is_day3:
            comm_live = fetch_commercial_forecasts(city)
            _accu_val = comm_live.get("accu_d3")
            _wu_val   = comm_live.get("wu_d3")
            row_d["🔶 AccuWeather"] = (fmt_val(_hround(_accu_val * 10) / 10) + " ⏳") if _accu_val is not None else "—"
            row_d["🔷 Weather.com"] = (fmt_val(_hround(_wu_val   * 10) / 10) + " ⏳") if _wu_val   is not None else "—"
        elif _is_day2:
            comm_live = fetch_commercial_forecasts(city)
            _accu_val = comm_live.get("accu_d2")
            _wu_val   = comm_live.get("wu_d2")
            row_d["🔶 AccuWeather"] = (fmt_val(_hround(_accu_val * 10) / 10) + " ⏳") if _accu_val is not None else "—"
            row_d["🔷 Weather.com"] = (fmt_val(_hround(_wu_val   * 10) / 10) + " ⏳") if _wu_val   is not None else "—"
        elif _is_tomorrow:
            # D+1: prefer disk snapshot, but if missing do a live fetch so columns always appear.
            comm_entry = comm_log.get(date_str)
            if comm_entry:
                _accu_val = comm_entry.get("accu")
                _wu_val   = comm_entry.get("wu")
            else:
                comm_live = fetch_commercial_forecasts(city)
                _accu_val = comm_live.get("accu")
                _wu_val   = comm_live.get("wu")
            row_d["🔶 AccuWeather"] = (fmt_val(_hround(_accu_val * 10) / 10) + " ⏳") if _accu_val is not None else "—"
            row_d["🔷 Weather.com"] = (fmt_val(_hround(_wu_val   * 10) / 10) + " ⏳") if _wu_val   is not None else "—"
        else:
            # D+0 (today): read strictly from yesterday's cron snapshot; always emit columns.
            comm_entry = comm_log.get(date_str)
            _accu_val = comm_entry.get("accu") if comm_entry else None
            _wu_val   = comm_entry.get("wu")   if comm_entry else None
            row_d["🔶 AccuWeather"] = (fmt_val(_hround(_accu_val * 10) / 10) + " ⏳") if _accu_val is not None else "—"
            row_d["🔷 Weather.com"] = (fmt_val(_hround(_wu_val   * 10) / 10) + " ⏳") if _wu_val   is not None else "—"

        # All individual model columns
        for col_label, mk in active_models.items():
            val = all_preds.get(mk)
            row_d[col_label] = f"{fmt_val(val)} ⏳" if val is not None else "—"

        return row_d

    display_rows = []

    # Dates that are in the fresh PM fetch but NOT yet in the cached accuracy rows.
    # These are recently resolved markets that the 1-hour cache hasn't picked up yet.
    # We build a live resolved row for them so they appear immediately.
    cached_dates = {r["date"] for r in rows}
    _newly_resolved = sorted(
        [d for d in _fresh_pm if d not in cached_dates and d <= _today_str],
        reverse=True,
    )

    def _make_live_resolved_row(date_str: str) -> dict | None:
        """Resolved-but-not-yet-cached row: real resolution + live model predictions."""
        res_data = _fresh_pm.get(date_str)
        if not res_data:
            return None
        res_label = res_data[0]

        # Build a scorer that handles both exact_1c 3-tuples and range_2f 5-tuples.
        # range_2f tuple: (label, lo, hi, bottom_thresh, top_thresh)
        #   ≥X  → lo=X,   hi=None → win if actual >= lo
        #   ≤X  → lo=None, hi=X   → win if actual <= hi
        #   A-B → lo=A,   hi=B    → win if lo <= actual <= hi
        # exact_1c tuple: (label, res_int, is_plus)
        bucket_style = cfg.get("bucket_style", "exact_1c")
        if bucket_style == "range_2f" and len(res_data) >= 5:
            _lo, _hi = res_data[1], res_data[2]
            def _score(v):
                if v is None: return None
                r = _hround(v)
                if _lo is not None and _hi is not None:
                    return _lo <= r <= _hi
                if _lo is not None:
                    return r >= _lo
                if _hi is not None:
                    return r <= _hi
                return False
        else:
            res_int = res_data[1]
            is_plus = res_data[2] if len(res_data) > 2 else False
            def _score(v):
                if v is None: return None
                r = _hround(v)
                return r >= res_int if is_plus else r == res_int

        all_preds = fetch_models_for_date(city, date_str)
        top_mk  = cfg.get("top_model_key")
        top_val = all_preds.get(top_mk) if top_mk else None
        if top_val is None:
            return None

        top_win  = _score(top_val)
        tick     = "✅" if top_win else "❌"

        ens_mk_list = cfg["best_ensemble"]["model_keys"]
        ens_preds   = [all_preds[k] for k in ens_mk_list if k in all_preds]
        if len(ens_preds) == len(ens_mk_list):
            ens_val = _hround(sum(ens_preds) / len(ens_preds) * 10) / 10
            ens_win = _score(ens_val)
            ens_cell = f"{fmt_val(ens_val)} {'✅' if ens_win else '❌'}"
        else:
            ens_cell = "—"

        row_d: dict = {
            "Date":           f"{date_str} 🆕",
            "Resolved":       f"{res_label} 🆕",
            ens_cfg["short"]: ens_cell,
        }

        if cfg.get("spread_filter"):
            sf_keys = cfg["spread_filter"]["model_keys"]
            sf_thr  = cfg["spread_filter"]["threshold"]
            sf_vals = [all_preds[k] for k in sf_keys if k in all_preds]
            if len(sf_vals) >= 2:
                sp = max(sf_vals) - min(sf_vals)
                row_d["Spread"] = f"{'🟢' if sp <= sf_thr else '🔴'} {sp:.1f}°C"

        for hyp in cfg.get("hypothesis_ensembles", []):
            hkeys = hyp["model_keys"]
            hpreds = [all_preds[k] for k in hkeys if k in all_preds]
            if len(hpreds) == len(hkeys):
                hval = _hround(sum(hpreds) / len(hpreds) * 10) / 10
                hwin = _score(hval)
                row_d[f"🧪 {hyp['short']}"] = f"{fmt_val(hval)} {'✅' if hwin else '❌'}"

        comm_entry = comm_log.get(date_str)
        _accu_comm = comm_entry.get("accu") if comm_entry else None
        _wu_comm   = comm_entry.get("wu")   if comm_entry else None
        row_d["🔶 AccuWeather"] = (fmt_val(_hround(_accu_comm * 10) / 10)) if _accu_comm is not None else "—"
        row_d["🔷 Weather.com"] = (fmt_val(_hround(_wu_comm   * 10) / 10)) if _wu_comm   is not None else "—"

        for col_label, mk in active_models.items():
            val = all_preds.get(mk)
            if val is not None:
                w = _score(val)
                row_d[col_label] = f"{fmt_val(val)} {'✅' if w else '❌'}"
            else:
                row_d[col_label] = "—"

        return row_d

    # Inject pending rows at the top (most recent first): D+3, D+2, D+1, D+0
    # D+3 is shown so you can spot early mispricing before markets fill up
    # (Polymarket often prices D+3 buckets at 5-15¢ — prime entry territory).
    for pending_date, pending_label in [
        (_day3_str,     "⏳ Not resolved"),
        (_day2_str,     "⏳ Not resolved"),
        (_tomorrow_str, "⏳ Not resolved"),
        (_today_str,    "⏳ Not resolved"),
    ]:
        if pending_date not in resolved_dates:
            pr = _make_pending_row(pending_date, pending_label)
            if pr:
                display_rows.append(pr)

    # Inject newly-resolved rows (resolved in fresh PM but cache hasn't caught up yet)
    for nr_date in _newly_resolved:
        nr_row = _make_live_resolved_row(nr_date)
        if nr_row:
            display_rows.append(nr_row)

    for r in reversed(rows):  # most recent first
        ens_val = r.get("best_ens_d1")
        ens_win = r.get("best_ens_d1_win")
        ens_cell = (f"{fmt_val(ens_val)} {'✅' if ens_win else '❌'}") if ens_val is not None else "—"

        # Consensus for historical rows: count how many ens models hit same bucket as ens_val
        hist_cons_cell = None
        if ens_val is not None:
            ens_bucket_h = _hround(ens_val)
            ens_mk_list_h = cfg["best_ensemble"]["model_keys"]
            hist_preds = [r.get(f"{mk}_d1") for mk in ens_mk_list_h if r.get(f"{mk}_d1") is not None]
            if len(hist_preds) >= 2:
                n_agree_h = sum(1 for v in hist_preds if _hround(v) == ens_bucket_h)
                n_total_h = len(hist_preds)
                cons_h = n_agree_h / n_total_h
                icon_h = "🔵" if cons_h >= 0.67 else "🟡" if cons_h >= 0.50 else "🔴"
                hist_cons_cell = f"{icon_h} {n_agree_h}/{n_total_h} ({cons_h:.0%})"

        row_d: dict = {
            "Date":           r["date"],
            "Resolved":       r["resolved"],
            ens_cfg["short"]: ens_cell,
        }

        # Spread column (only for cities with spread_filter configured)
        if cfg.get("spread_filter") and r.get("spread_d1") is not None:
            sp = r["spread_d1"]
            row_d["Spread"] = f"{'🟢' if r['spread_green'] else '🔴'} {sp:.1f}°C"

        if hist_cons_cell:
            row_d["Consensus"] = hist_cons_cell

        # Hypothesis ensemble columns (🧪 forward-test candidates)
        for hyp in cfg.get("hypothesis_ensembles", []):
            hval = r.get(f"{hyp['key']}_d1")
            hwin = r.get(f"{hyp['key']}_d1_win")
            row_d[f"🧪 {hyp['short']}"] = (f"{fmt_val(hval)} {'✅' if hwin else '❌'}") if hval is not None else "—"

        # Commercial forecast columns (from log, where available)
        if has_comm_data:
            comm_entry = comm_log.get(r["date"])
            accu_logged = comm_entry.get("accu") if comm_entry else None
            wu_logged = comm_entry.get("wu") if comm_entry else None

            def _comm_cell(val: float | None) -> str:
                if val is None:
                    return "—"
                rounded = _hround(val * 10) / 10
                # Determine win against the resolved bucket (reuse compute_win logic inline)
                resolved_entry = cfg.get("polymarket", {}).get(r["date"])
                if resolved_entry and cfg.get("bucket_style", "exact_1c") != "range_2f":
                    _, res_int, is_plus = resolved_entry if len(resolved_entry) == 3 else (resolved_entry[0], resolved_entry[1], False)
                    won = _wins(rounded, res_int, is_plus)
                    marker = " ✅" if won else " ❌"
                else:
                    marker = ""
                return f"{fmt_val(rounded)}{marker}"

            row_d["🔶 AccuWeather"] = _comm_cell(accu_logged)
            row_d["🔷 Weather.com"] = _comm_cell(wu_logged)

        for col_label, mk in active_models.items():
            val = r.get(f"{mk}_d1")
            win = r.get(f"{mk}_d1_win")
            row_d[col_label] = (f"{fmt_val(val)} {'✅' if win else '❌'}") if val is not None else "—"

        display_rows.append(row_d)

    detail_df = pd.DataFrame(display_rows)
    st.dataframe(detail_df, use_container_width=True, hide_index=True, height=520)
    if has_comm_data:
        st.caption(f"🔶 AccuWeather / 🔷 Weather.com columns show logged D+1 forecasts from `data/commercial_forecast_log.json` — logging started {min(comm_log.keys())}.")
    else:
        st.caption("🔶🔷 AccuWeather / Weather.com columns will appear once daily forecasts are logged. Data collection starts today.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 4: Rolling 10-day accuracy ──────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📈 Rolling 10-Day Accuracy")
    st.caption("How model accuracy evolves over the backtest period")

    window = 10
    _chart_palette = [BLUE, "#FF9F40", "#A855F7", GRAY, "#FF6B6B"]
    chart_model_keys = cfg.get("chart_models", list(cfg["models"].keys())[:3])

    trace_configs: list[tuple[str, str, str]] = [
        ("best_ens_d1", f"🏆 {ens_cfg['short']} D1", GREEN),
    ]
    for i, mk in enumerate(chart_model_keys):
        lbl, icon = cfg["models"][mk]
        trace_configs.append((f"{mk}_d1", f"{icon} {lbl} D1", _chart_palette[i % len(_chart_palette)]))

    fig_roll = go.Figure()
    for key, label, color in trace_configs:
        win_key = f"{key}_win"
        valid = [(r["date"], r.get(win_key)) for r in rows if r.get(win_key) is not None]
        if len(valid) < window:
            continue
        dates_v = [v[0] for v in valid]
        wins_v  = [1 if v[1] else 0 for v in valid]
        rolling = [
            sum(wins_v[max(0, i - window + 1): i + 1]) / min(i + 1, window) * 100
            for i in range(len(wins_v))
        ]
        fig_roll.add_trace(go.Scatter(
            x=dates_v, y=rolling,
            mode="lines+markers", name=label,
            line={"color": color, "width": 2},
            marker={"size": 5},
            hovertemplate=f"{label}<br>%{{x}}<br>Rolling acc: %{{y:.1f}}%<extra></extra>",
        ))

    fig_roll.add_hline(y=50, line_dash="dot", line_color=GRAY, annotation_text="50%", annotation_font_color=GRAY)
    fig_roll.update_layout(
        plot_bgcolor=BG, paper_bgcolor=PANEL,
        font={"color": TEXT, "family": "Inter, Arial, sans-serif"},
        margin={"l": 20, "r": 10, "t": 10, "b": 20},
        xaxis={"gridcolor": "#1f2937"},
        yaxis={"gridcolor": "#1f2937", "title": "Rolling accuracy %", "range": [0, 100]},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
        height=340,
    )
    st.plotly_chart(fig_roll, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 5: Notes ────────────────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📌 Notes & Methodology")
    st.markdown(f"""
**Bucket rule:** Model prediction is rounded to nearest integer. Wins if it matches the Polymarket winning bucket.
For "X or higher" top buckets, any prediction ≥ X is a win.

**T-24h (D1):** Forecast made the day before the resolution date — this is the primary trading window.

**T-48h (D2):** Forecast made two days before — useful for earlier entry at better prices.

**Data source:** [Open-Meteo Previous Runs API](https://previous-runs-api.open-meteo.com) — `temperature_2m_previous_day1/2`

{cfg.get('notes', '')}
    """)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
