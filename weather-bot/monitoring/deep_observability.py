"""Deep observability logging (append-only JSONL).

Instrumentation-only module: no trading logic, no execution behavior changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


class DeepObservability:
    """Append structured decision telemetry to daily JSONL files."""

    def __init__(self, root: str | None = None, enabled: bool | None = None) -> None:
        base = Path(__file__).resolve().parents[1]
        self.root = Path(root) if root else (base / "logs" / "deep")
        self.root.mkdir(parents=True, exist_ok=True)
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("DEEP_OBSERVABILITY_ENABLED", "true").lower() in {"1", "true", "yes"}
        )
        self._git_sha = self._get_git_sha()
        self._config_hash = self._get_config_hash()
        self._host = socket.gethostname()
        self._schema_version = "1.0"

    def _get_git_sha(self) -> str:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=2,
                text=True,
            ).strip()
            return out
        except Exception:
            return "unknown"

    def _get_config_hash(self) -> str:
        try:
            from config import settings as cfg  # lazy import

            keys = [k for k in dir(cfg) if k.isupper()]
            payload = {k: getattr(cfg, k) for k in sorted(keys)}
            raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            return hashlib.sha256(raw).hexdigest()[:16]
        except Exception:
            return "unknown"

    def _path_for_layer(self, layer: str) -> Path:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        return self.root / f"{layer}_{day}.jsonl"

    def _meta(self, mode: str = "paper") -> dict[str, Any]:
        return {
            "schema_version": self._schema_version,
            "git_sha": self._git_sha,
            "config_hash": self._config_hash,
            "host": self._host,
            "mode": mode,
        }

    def _append(self, layer: str, row: dict[str, Any], mode: str = "paper") -> str:
        if not self.enabled:
            return row.get("decision_id") or row.get("id") or ""
        payload = {**row, **self._meta(mode=mode)}
        path = self._path_for_layer(layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
        return str(payload.get("decision_id") or payload.get("id") or "")

    def log_model_snapshot(self, row: dict[str, Any], mode: str = "paper") -> str:
        row = {"id": row.get("snapshot_id") or uuid4().hex, **row}
        if "snapshot_id" not in row:
            row["snapshot_id"] = row["id"]
        return self._append("model_snapshots", row, mode=mode)

    def log_bucket_probs(self, row: dict[str, Any], mode: str = "paper") -> str:
        row = {"id": row.get("prob_calc_id") or uuid4().hex, **row}
        if "prob_calc_id" not in row:
            row["prob_calc_id"] = row["id"]
        return self._append("bucket_probs", row, mode=mode)

    def log_market_state(self, row: dict[str, Any], mode: str = "paper") -> str:
        row = {"id": row.get("market_scan_id") or uuid4().hex, **row}
        if "market_scan_id" not in row:
            row["market_scan_id"] = row["id"]
        return self._append("market_state", row, mode=mode)

    def log_signal_eval(self, row: dict[str, Any], mode: str = "paper") -> str:
        row = {"decision_id": row.get("decision_id") or uuid4().hex, **row}
        return self._append("signal_eval", row, mode=mode)

    def log_execution(self, row: dict[str, Any], mode: str = "paper") -> str:
        row = {"id": row.get("execution_id") or uuid4().hex, **row}
        if "execution_id" not in row:
            row["execution_id"] = row["id"]
        return self._append("executions", row, mode=mode)

    def log_resolution(self, row: dict[str, Any], mode: str = "paper") -> str:
        row = {"id": row.get("resolution_id") or uuid4().hex, **row}
        if "resolution_id" not in row:
            row["resolution_id"] = row["id"]
        return self._append("resolutions", row, mode=mode)


_DEEP_OBS = DeepObservability()


def get_deep_observability() -> DeepObservability:
    return _DEEP_OBS

