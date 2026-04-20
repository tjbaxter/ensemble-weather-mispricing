#!/usr/bin/env python3
"""Read-only dashboard data API for VM-backed Streamlit Cloud usage."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import secrets
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Final

from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.dashboard_overview import build_dashboard_overview_payload  # noqa: E402
from data.wu_observations import fetch_wu_observed_max  # noqa: E402

SETTLEMENT_STATUS_PATH = ROOT / "data" / "settlement_status.json"
MODEL_SNAPSHOT_LOG_PATH = ROOT / "data" / "model_snapshot_log.json"
COMMERCIAL_FORECAST_LOG_PATH = ROOT / "data" / "commercial_forecast_log.json"
ACCURACY_CACHE_PATH = ROOT / "data" / "accuracy_rows_cache.json"

# All tradeable cities with live weather stations
LIVE_WEATHER_CITIES: dict[str, dict[str, Any]] = {
    "Ankara": {"icao": "LTAC", "unit": "C"},
    "Atlanta": {"icao": "KATL", "unit": "F"},
    "Buenos Aires": {"icao": "SAEZ", "unit": "C"},
    "Chicago": {"icao": "KORD", "unit": "F"},
    "Dallas": {"icao": "KDFW", "unit": "F"},
    "London": {"icao": "EGLC", "unit": "C"},
    "Miami": {"icao": "KMIA", "unit": "F"},
    "New York": {"icao": "KLGA", "unit": "F"},
    "Paris": {"icao": "LFPG", "unit": "C"},
    "Seattle": {"icao": "KSEA", "unit": "F"},
    "Seoul": {"icao": "RKSI", "unit": "C"},
    "Toronto": {"icao": "CYYZ", "unit": "C"},
    "Wellington": {"icao": "NZWN", "unit": "C"},
}

# Key models to show in forecasts
DISPLAY_MODELS = [
    "meteofrance_arome_france",
    "icon_seamless",
    "ecmwf_ifs025",
    "ncep_aigfs025",
    "gfs_graphcast025",
    "dmi_seamless",
]

DATA_FILE_RE: Final = re.compile(r"^data/[^/]+\.(json|csv)$")
LOG_FILE_RE: Final = re.compile(
    r"^logs/(resolved(?:_archive_\d{8})?\.csv|trades\.csv|signals\.csv|calibration\.json)$"
)
SHADOW_LOG_RE: Final = re.compile(r"^logs/shadow_[^/]+/resolved(?:_archive_\d{8})?\.csv$")
BACKTEST_FILE_RE: Final = re.compile(r"^backtest/data/resolved_markets\.(json|csv)$")
TRUTHY = {"1", "true", "yes", "on"}
OVERVIEW_CACHE_KEY = "dashboard_overview_cache"


def _parse_truthy(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in TRUTHY


def _api_host() -> str:
    return os.getenv("DASHBOARD_API_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _api_port() -> int:
    try:
        return int(os.getenv("DASHBOARD_API_PORT", "8510"))
    except Exception:
        return 8510


def _api_token() -> str:
    return os.getenv("DASHBOARD_API_TOKEN", "").strip()


def _public_health() -> bool:
    return _parse_truthy(os.getenv("DASHBOARD_API_PUBLIC_HEALTH"), default=True)


def _overview_refresh_seconds() -> int:
    try:
        return max(15, int(os.getenv("DASHBOARD_API_OVERVIEW_REFRESH_SEC", "60")))
    except Exception:
        return 60


def _is_loopback_request(request: web.Request) -> bool:
    host = request.remote or ""
    return host in {"127.0.0.1", "::1", "localhost"}


def _is_allowed_rel_path(rel_path: str) -> bool:
    return bool(
        DATA_FILE_RE.fullmatch(rel_path)
        or LOG_FILE_RE.fullmatch(rel_path)
        or SHADOW_LOG_RE.fullmatch(rel_path)
        or BACKTEST_FILE_RE.fullmatch(rel_path)
    )


def _resolve_rel_path(raw_path: str) -> tuple[str, Path]:
    rel = PurePosixPath(raw_path.lstrip("/"))
    if rel.is_absolute() or any(part in {"..", ""} for part in rel.parts):
        raise web.HTTPForbidden(text="Invalid path.")
    rel_path = rel.as_posix()
    if not _is_allowed_rel_path(rel_path):
        raise web.HTTPForbidden(text="Path not allowed.")
    abs_path = (ROOT / Path(rel_path)).resolve()
    try:
        abs_path.relative_to(ROOT)
    except Exception as exc:  # pragma: no cover - defense in depth
        raise web.HTTPForbidden(text="Path escapes repository root.") from exc
    if not abs_path.is_file():
        raise web.HTTPNotFound(text="File not found.")
    return rel_path, abs_path


def _dashboard_sync_status_payload() -> dict:
    now_iso = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    watcher_payload: dict = {}
    try:
        if SETTLEMENT_STATUS_PATH.exists():
            loaded = json.loads(SETTLEMENT_STATUS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                watcher_payload = loaded
    except Exception:
        watcher_payload = {}
    return {
        "last_fast_sync_utc": now_iso,
        "last_api_refresh_utc": now_iso,
        "api_mode": True,
        "settlement_watcher_last_success_utc": watcher_payload.get("last_success_utc", ""),
        "settlement_watcher_last_heartbeat_utc": watcher_payload.get("last_heartbeat_utc", ""),
    }


async def _refresh_overview_cache_once(app: web.Application) -> None:
    cache = app.setdefault(OVERVIEW_CACHE_KEY, {})
    try:
        payload = await asyncio.to_thread(build_dashboard_overview_payload, ROOT)
        cache["payload"] = payload
        cache["last_refresh_utc"] = datetime.now(UTC).isoformat()
        cache["last_error"] = ""
    except Exception as exc:  # pragma: no cover - defensive
        cache["last_error"] = str(exc)
        cache["last_refresh_utc"] = datetime.now(UTC).isoformat()


async def _overview_cache_ctx(app: web.Application):
    app[OVERVIEW_CACHE_KEY] = {
        "payload": {},
        "last_error": "",
        "last_refresh_utc": "",
    }
    async def _loop() -> None:
        while True:
            await _refresh_overview_cache_once(app)
            await asyncio.sleep(_overview_refresh_seconds())

    task = asyncio.create_task(_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def _dashboard_overview_payload(app: web.Application) -> dict:
    cache = app.get(OVERVIEW_CACHE_KEY, {})
    payload = cache.get("payload")
    if isinstance(payload, dict) and payload.get("strategies"):
        return payload
    return {
        "generated_at_utc": cache.get("last_refresh_utc", ""),
        "strategies": [],
        "last_error": cache.get("last_error", ""),
        "warming": True,
    }


@web.middleware
async def auth_middleware(request: web.Request, handler):
    # Public endpoints that don't require auth
    if request.path == "/health" and _public_health():
        return await handler(request)
    if request.path == "/raw/data/live_weather.json":
        return await handler(request)
    if request.path == "/raw/data/dashboard_overview.json":
        return await handler(request)

    token = _api_token()
    if not token:
        if _is_loopback_request(request):
            return await handler(request)
        raise web.HTTPServiceUnavailable(
            text="Dashboard API token is not configured for non-loopback access."
        )

    auth_header = request.headers.get("Authorization", "")
    supplied = ""
    if auth_header.startswith("Bearer "):
        supplied = auth_header.split(" ", 1)[1].strip()
    elif request.headers.get("X-Dashboard-Api-Key"):
        supplied = request.headers.get("X-Dashboard-Api-Key", "").strip()

    if not supplied or not secrets.compare_digest(supplied, token):
        raise web.HTTPUnauthorized(text="Unauthorized.")
    return await handler(request)


async def health(_: web.Request) -> web.Response:
    payload = {
        "ok": True,
        "service": "dashboard_api",
        "generated_at": datetime.now(UTC).isoformat(),
        "host": _api_host(),
        "port": _api_port(),
        "root": str(ROOT),
    }
    return web.json_response(payload)


async def logs_index(_: web.Request) -> web.Response:
    files = sorted(
        path.name
        for path in (ROOT / "logs").glob("*")
        if path.is_file() and LOG_FILE_RE.fullmatch(f"logs/{path.name}")
    )
    return web.json_response({"files": files, "generated_at": datetime.now(UTC).isoformat()})


def _build_live_weather_payload() -> dict[str, Any]:
    """Build aggregated live weather data for all tradeable cities."""
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    today_str = date.today().isoformat()
    
    # Load model snapshot log
    model_snapshots: dict = {}
    if MODEL_SNAPSHOT_LOG_PATH.exists():
        try:
            model_snapshots = json.loads(MODEL_SNAPSHOT_LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    # Load commercial forecasts
    commercial_log: dict = {}
    if COMMERCIAL_FORECAST_LOG_PATH.exists():
        try:
            commercial_log = json.loads(COMMERCIAL_FORECAST_LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    # Load accuracy cache
    accuracy_cache: dict = {}
    if ACCURACY_CACHE_PATH.exists():
        try:
            accuracy_cache = json.loads(ACCURACY_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    cities_data: dict[str, dict] = {}
    
    for city, cfg in LIVE_WEATHER_CITIES.items():
        icao = cfg["icao"]
        unit = cfg["unit"]
        
        # 1. Live station data
        live_station: dict[str, Any] = {
            "current_temp": None,
            "today_max": None,
            "n_readings": 0,
            "last_reading_utc": None,
            "station_id": None,
            "unit": unit,
        }
        try:
            wu_data = fetch_wu_observed_max(icao)
            if wu_data:
                live_station["current_temp"] = wu_data.get("latest_temp")
                live_station["today_max"] = wu_data.get("running_max")
                live_station["n_readings"] = wu_data.get("n_obs", 0)
                live_station["last_reading_utc"] = wu_data.get("last_obs_utc")
                live_station["station_id"] = wu_data.get("station_id")
        except Exception:
            pass
        
        # 2. Model forecasts for tomorrow
        model_forecasts: dict[str, Any] = {
            "target_date": tomorrow,
            "models": {},
            "spread": None,
            "spread_color": "UNKNOWN",
        }
        city_snapshots = model_snapshots.get(city, {})
        tomorrow_preds = city_snapshots.get(tomorrow, {}).get("preds", {})
        
        if tomorrow_preds:
            # Get key model forecasts
            for model in DISPLAY_MODELS:
                if model in tomorrow_preds and tomorrow_preds[model] is not None:
                    model_forecasts["models"][model] = round(float(tomorrow_preds[model]), 1)
            
            # Calculate spread (max - min)
            all_vals = [v for v in tomorrow_preds.values() if v is not None]
            if all_vals:
                spread = max(all_vals) - min(all_vals)
                model_forecasts["spread"] = round(spread, 1)
                # Spread thresholds: ≤1°C for Celsius, ≤2°F for Fahrenheit
                threshold = 1.0 if unit == "C" else 2.0
                model_forecasts["spread_color"] = "GREEN" if spread <= threshold else "RED"
        
        # 3. Commercial forecasts for tomorrow
        commercial: dict[str, Any] = {
            "accuweather": None,
            "weather_com": None,
            "target_date": tomorrow,
        }
        city_commercial = commercial_log.get(city, {})
        if tomorrow in city_commercial:
            entry = city_commercial[tomorrow]
            commercial["accuweather"] = entry.get("accu")
            commercial["weather_com"] = entry.get("wu")
        
        # 4. Accuracy stats
        accuracy: dict[str, Any] = {
            "best_model": None,
            "best_accuracy_pct": None,
            "market_days": 0,
            "city_accuracy_pct": None,
        }
        city_accuracy_rows = accuracy_cache.get(city, [])
        if city_accuracy_rows:
            accuracy["market_days"] = len(city_accuracy_rows)
            
            # Calculate city-level accuracy (best_ens_d1_win)
            wins = sum(1 for r in city_accuracy_rows if r.get("best_ens_d1_win") is True)
            total = sum(1 for r in city_accuracy_rows if r.get("best_ens_d1_win") is not None)
            if total > 0:
                accuracy["city_accuracy_pct"] = round(100 * wins / total, 1)
            
            # Find best individual model
            model_wins: dict[str, tuple[int, int]] = {}
            for row in city_accuracy_rows:
                for key in row:
                    if key.endswith("_d1_win") and row[key] is not None:
                        model = key.replace("_d1_win", "")
                        if model not in model_wins:
                            model_wins[model] = (0, 0)
                        w, t = model_wins[model]
                        model_wins[model] = (w + (1 if row[key] else 0), t + 1)
            
            if model_wins:
                best_model = max(model_wins.items(), key=lambda x: x[1][0] / x[1][1] if x[1][1] > 0 else 0)
                model_name, (w, t) = best_model
                if t > 0:
                    accuracy["best_model"] = model_name
                    accuracy["best_accuracy_pct"] = round(100 * w / t, 1)
        
        cities_data[city] = {
            "icao": icao,
            "unit": unit,
            "live_station": live_station,
            "model_forecasts": model_forecasts,
            "commercial_forecasts": commercial,
            "accuracy": accuracy,
        }
    
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "target_date": tomorrow,
        "today": today_str,
        "cities": cities_data,
    }


def _cache_freshness_payload() -> dict:
    """Return mtime-based freshness info for key data caches."""
    files = {
        "polymarket_cache": ROOT / "data" / "polymarket_cache.json",
        "accuracy_rows_cache": ROOT / "data" / "accuracy_rows_cache.json",
        "model_snapshot_log": ROOT / "data" / "model_snapshot_log.json",
        "settlement_snapshot": ROOT / "data" / "settlement_snapshot.json",
    }
    result: dict = {}
    for key, path in files.items():
        if path.exists():
            mtime = path.stat().st_mtime
            result[f"{key}_updated_at"] = datetime.fromtimestamp(mtime, tz=UTC).isoformat()
        else:
            result[f"{key}_updated_at"] = None
    result["generated_at"] = datetime.now(UTC).isoformat()
    return result


async def raw_file(request: web.Request) -> web.Response:
    raw_target = request.match_info.get("tail", "")
    rel_path = PurePosixPath(raw_target.lstrip("/")).as_posix()
    if rel_path == "data/dashboard_sync_status.json":
        return web.json_response(_dashboard_sync_status_payload())
    if rel_path == "data/dashboard_overview.json":
        return web.json_response(_dashboard_overview_payload(request.app))
    if rel_path == "data/cache_freshness.json":
        return web.json_response(_cache_freshness_payload())
    if rel_path == "data/live_weather.json":
        return web.json_response(_build_live_weather_payload())

    rel_path, abs_path = _resolve_rel_path(raw_target)
    try:
        text = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise web.HTTPUnsupportedMediaType(text="Only UTF-8 text files are supported.") from exc

    content_type = "text/plain"
    if rel_path.endswith(".json"):
        content_type = "application/json"
    elif rel_path.endswith(".csv"):
        content_type = "text/csv"
    return web.Response(text=text, content_type=content_type)


def build_app() -> web.Application:
    app = web.Application(middlewares=[auth_middleware])
    app.cleanup_ctx.append(_overview_cache_ctx)
    app.router.add_get("/health", health)
    app.router.add_get("/logs/index", logs_index)
    app.router.add_get("/raw/{tail:.*}", raw_file)
    return app


def main() -> None:
    web.run_app(build_app(), host=_api_host(), port=_api_port(), access_log=None)


if __name__ == "__main__":
    main()
