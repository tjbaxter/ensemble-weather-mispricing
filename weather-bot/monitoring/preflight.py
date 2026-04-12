"""
Pre-flight checks before bot startup.

Validates:
- Disk space (must have > 10% free)
- Database integrity
- Critical file existence
- Network connectivity
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path
from typing import NamedTuple


class PreflightResult(NamedTuple):
    """Result of a preflight check."""
    name: str
    passed: bool
    message: str
    critical: bool = True


def check_disk_space(min_free_pct: float = 10.0) -> PreflightResult:
    """Ensure disk has at least min_free_pct free space."""
    total, used, free = shutil.disk_usage("/")
    free_pct = (free / total) * 100
    
    if free_pct < min_free_pct:
        return PreflightResult(
            name="disk_space",
            passed=False,
            message=f"Disk only {free_pct:.1f}% free (need > {min_free_pct}%)",
            critical=True,
        )
    return PreflightResult(
        name="disk_space",
        passed=True,
        message=f"Disk {free_pct:.1f}% free ({free / (1024**3):.2f} GB)",
    )


def check_database_integrity(db_path: Path) -> PreflightResult:
    """Check SQLite database integrity."""
    if not db_path.exists():
        return PreflightResult(
            name=f"db_integrity:{db_path.name}",
            passed=True,
            message="Database does not exist (will be created)",
            critical=False,
        )
    
    try:
        conn = sqlite3.connect(db_path)
        result = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()
        
        if result[0] == "ok":
            return PreflightResult(
                name=f"db_integrity:{db_path.name}",
                passed=True,
                message="Database integrity OK",
            )
        else:
            return PreflightResult(
                name=f"db_integrity:{db_path.name}",
                passed=False,
                message=f"Database corrupted: {result[0]}",
                critical=True,
            )
    except Exception as e:
        return PreflightResult(
            name=f"db_integrity:{db_path.name}",
            passed=False,
            message=f"Database check failed: {e}",
            critical=True,
        )


def check_critical_files(data_dir: Path) -> PreflightResult:
    """Check that critical data files exist and are readable."""
    critical_files = [
        "polymarket_cache.json",
        "accuracy_rows_cache.json",
    ]
    
    missing = []
    for fname in critical_files:
        fpath = data_dir / fname
        if not fpath.exists():
            missing.append(fname)
        elif fpath.stat().st_size == 0:
            missing.append(f"{fname} (empty)")
    
    if missing:
        return PreflightResult(
            name="critical_files",
            passed=False,
            message=f"Missing/empty: {', '.join(missing)}",
            critical=True,
        )
    return PreflightResult(
        name="critical_files",
        passed=True,
        message="All critical files present",
    )


def check_network_connectivity() -> PreflightResult:
    """Check basic network connectivity."""
    import socket
    
    hosts = [
        ("polymarket.com", 443),
        ("api.open-meteo.com", 443),
    ]
    
    failed = []
    for host, port in hosts:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
        except (socket.timeout, socket.error):
            failed.append(host)
    
    if failed:
        return PreflightResult(
            name="network",
            passed=False,
            message=f"Cannot reach: {', '.join(failed)}",
            critical=True,
        )
    return PreflightResult(
        name="network",
        passed=True,
        message="Network connectivity OK",
    )


def run_preflight_checks(
    data_dir: Path,
    skip_network: bool = False,
    min_disk_free_pct: float = 10.0,
) -> tuple[bool, list[PreflightResult]]:
    """
    Run all preflight checks.
    
    Returns:
        (all_passed, list of results)
    """
    results: list[PreflightResult] = []
    
    # Disk space (most critical)
    results.append(check_disk_space(min_disk_free_pct))
    
    # Database integrity
    for db_name in ["settlement_watcher.db", "trade_observability.db"]:
        results.append(check_database_integrity(data_dir / db_name))
    
    # Critical files
    results.append(check_critical_files(data_dir))
    
    # Network (optional)
    if not skip_network:
        results.append(check_network_connectivity())
    
    # Check if any critical checks failed
    critical_failures = [r for r in results if not r.passed and r.critical]
    all_passed = len(critical_failures) == 0
    
    return all_passed, results


def print_preflight_report(results: list[PreflightResult]) -> None:
    """Print a formatted preflight report."""
    print("=" * 60)
    print("PREFLIGHT CHECKS")
    print("=" * 60)
    
    for result in results:
        status = "✅" if result.passed else ("❌" if result.critical else "⚠️")
        print(f"{status} {result.name}: {result.message}")
    
    print("=" * 60)


def require_preflight(data_dir: Path, exit_on_failure: bool = True) -> bool:
    """
    Run preflight checks and optionally exit on failure.
    
    Use at bot startup:
        from monitoring.preflight import require_preflight
        require_preflight(Path("data"))
    """
    passed, results = run_preflight_checks(data_dir)
    print_preflight_report(results)
    
    if not passed:
        print("\n❌ PREFLIGHT FAILED - refusing to start")
        if exit_on_failure:
            sys.exit(1)
    else:
        print("\n✅ All preflight checks passed")
    
    return passed
