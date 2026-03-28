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
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.dashboard_overview import build_dashboard_overview_payload  # noqa: E402

SETTLEMENT_STATUS_PATH = ROOT / "data" / "settlement_status.json"

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


@contextlib.asynccontextmanager
async def _overview_cache_ctx(app: web.Application):
    app[OVERVIEW_CACHE_KEY] = {
        "payload": {},
        "last_error": "",
        "last_refresh_utc": "",
    }
    await _refresh_overview_cache_once(app)

    async def _loop() -> None:
        while True:
            await asyncio.sleep(_overview_refresh_seconds())
            await _refresh_overview_cache_once(app)

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
    if request.path == "/health" and _public_health():
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


async def raw_file(request: web.Request) -> web.Response:
    raw_target = request.match_info.get("tail", "")
    rel_path = PurePosixPath(raw_target.lstrip("/")).as_posix()
    if rel_path == "data/dashboard_sync_status.json":
        return web.json_response(_dashboard_sync_status_payload())
    if rel_path == "data/dashboard_overview.json":
        return web.json_response(_dashboard_overview_payload(request.app))

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
