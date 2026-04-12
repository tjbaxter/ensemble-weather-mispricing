#!/usr/bin/env python3
"""
Production data retention and maintenance script.

Retention policies:
- SQLite databases: 90 days of records
- CSV logs: 30 days
- JSON model snapshots: 60 days of entries
- Deep observability: 7 days
- Polymarket/accuracy caches: Permanent (source of truth)

Run daily via cron/systemd timer:
    0 3 * * * /home/tombaxter/weather-bot/venv/bin/python scripts/data_retention.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


# Retention periods
SQLITE_RETENTION_DAYS = 90
CSV_RETENTION_DAYS = 30
SNAPSHOT_RETENTION_DAYS = 60
DEEP_LOG_RETENTION_DAYS = 7

# Root directory
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}")


def check_disk_space() -> dict:
    """Check disk space and return stats."""
    import shutil
    total, used, free = shutil.disk_usage("/")
    return {
        "total_gb": total / (1024**3),
        "used_gb": used / (1024**3),
        "free_gb": free / (1024**3),
        "used_pct": (used / total) * 100,
    }


def cleanup_sqlite_database(db_path: Path, retention_days: int, dry_run: bool = False) -> dict:
    """Purge old records from SQLite database and vacuum."""
    if not db_path.exists():
        return {"status": "not_found", "path": str(db_path)}
    
    size_before = db_path.stat().st_size
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_str = cutoff.isoformat()
    
    stats = {
        "path": str(db_path),
        "size_before_mb": size_before / (1024**2),
        "retention_days": retention_days,
        "cutoff": cutoff_str,
        "dry_run": dry_run,
    }
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        stats["tables"] = tables
        
        # Common timestamp columns to check
        ts_columns = ["timestamp", "created_at", "resolved_at", "recorded_at"]
        
        deleted_total = 0
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = {row[1] for row in cursor.fetchall()}
            
            for ts_col in ts_columns:
                if ts_col in columns:
                    if dry_run:
                        cursor.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE {ts_col} < ?",
                            (cutoff_str,)
                        )
                        count = cursor.fetchone()[0]
                        _log(f"  [DRY-RUN] Would delete {count} rows from {table}.{ts_col}")
                        deleted_total += count
                    else:
                        cursor.execute(
                            f"DELETE FROM {table} WHERE {ts_col} < ?",
                            (cutoff_str,)
                        )
                        deleted = cursor.rowcount
                        deleted_total += deleted
                        if deleted > 0:
                            _log(f"  Deleted {deleted} rows from {table}.{ts_col}")
                    break
        
        stats["deleted_rows"] = deleted_total
        
        if not dry_run and deleted_total > 0:
            conn.commit()
            _log(f"  Running VACUUM on {db_path.name}...")
            conn.execute("VACUUM")
            _log(f"  Running ANALYZE on {db_path.name}...")
            conn.execute("ANALYZE")
        
        conn.close()
        
        size_after = db_path.stat().st_size
        stats["size_after_mb"] = size_after / (1024**2)
        stats["freed_mb"] = (size_before - size_after) / (1024**2)
        stats["status"] = "ok"
        
    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
    
    return stats


def cleanup_old_files(directory: Path, pattern: str, retention_days: int, dry_run: bool = False) -> dict:
    """Delete files older than retention period."""
    if not directory.exists():
        return {"status": "not_found", "path": str(directory)}
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_ts = cutoff.timestamp()
    
    stats = {
        "path": str(directory),
        "pattern": pattern,
        "retention_days": retention_days,
        "dry_run": dry_run,
        "files_deleted": 0,
        "bytes_freed": 0,
    }
    
    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.stat().st_mtime < cutoff_ts:
            size = file_path.stat().st_size
            if dry_run:
                _log(f"  [DRY-RUN] Would delete {file_path.name} ({size / 1024:.1f} KB)")
            else:
                file_path.unlink()
                _log(f"  Deleted {file_path.name} ({size / 1024:.1f} KB)")
            stats["files_deleted"] += 1
            stats["bytes_freed"] += size
    
    stats["freed_mb"] = stats["bytes_freed"] / (1024**2)
    stats["status"] = "ok"
    return stats


def cleanup_json_entries(json_path: Path, retention_days: int, dry_run: bool = False) -> dict:
    """Remove old entries from JSON log files (keyed by date)."""
    if not json_path.exists():
        return {"status": "not_found", "path": str(json_path)}
    
    size_before = json_path.stat().st_size
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).date().isoformat()
    
    stats = {
        "path": str(json_path),
        "retention_days": retention_days,
        "cutoff": cutoff,
        "dry_run": dry_run,
        "size_before_mb": size_before / (1024**2),
    }
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Structure: {city: {date: value, ...}, ...}
            entries_before = sum(len(v) if isinstance(v, dict) else 1 for v in data.values())
            
            pruned_data = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    pruned_value = {k: v for k, v in value.items() if k >= cutoff}
                    if pruned_value:
                        pruned_data[key] = pruned_value
                else:
                    pruned_data[key] = value
            
            entries_after = sum(len(v) if isinstance(v, dict) else 1 for v in pruned_data.values())
            stats["entries_before"] = entries_before
            stats["entries_after"] = entries_after
            stats["entries_removed"] = entries_before - entries_after
            
            if not dry_run and entries_before != entries_after:
                with open(json_path, "w") as f:
                    json.dump(pruned_data, f, indent=2)
                _log(f"  Pruned {entries_before - entries_after} entries from {json_path.name}")
        
        size_after = json_path.stat().st_size
        stats["size_after_mb"] = size_after / (1024**2)
        stats["freed_mb"] = (size_before - size_after) / (1024**2)
        stats["status"] = "ok"
        
    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
    
    return stats


def run_retention(dry_run: bool = False) -> dict:
    """Run all retention tasks."""
    _log("=" * 60)
    _log("DATA RETENTION AND MAINTENANCE")
    _log(f"Dry run: {dry_run}")
    _log("=" * 60)
    
    disk_before = check_disk_space()
    _log(f"Disk before: {disk_before['used_pct']:.1f}% used ({disk_before['free_gb']:.2f} GB free)")
    
    results = {"disk_before": disk_before, "tasks": []}
    
    # 1. SQLite databases
    _log("\n[1/5] SQLite database retention...")
    for db_name in ["settlement_watcher.db", "trade_observability.db"]:
        db_path = DATA_DIR / db_name
        result = cleanup_sqlite_database(db_path, SQLITE_RETENTION_DAYS, dry_run)
        _log(f"  {db_name}: {result.get('status')} (freed {result.get('freed_mb', 0):.2f} MB)")
        results["tasks"].append({"task": f"sqlite_{db_name}", **result})
    
    # 2. Deep observability logs
    _log("\n[2/5] Deep observability logs...")
    deep_dir = LOGS_DIR / "deep"
    for pattern in ["*.jsonl", "*.json"]:
        result = cleanup_old_files(deep_dir, pattern, DEEP_LOG_RETENTION_DAYS, dry_run)
        _log(f"  {pattern}: {result['files_deleted']} files, {result['freed_mb']:.2f} MB freed")
        results["tasks"].append({"task": f"deep_{pattern}", **result})
    
    # 3. CSV logs
    _log("\n[3/5] CSV log rotation...")
    for pattern in ["*.csv"]:
        result = cleanup_old_files(LOGS_DIR, pattern, CSV_RETENTION_DAYS, dry_run)
        _log(f"  {pattern}: {result['files_deleted']} files, {result['freed_mb']:.2f} MB freed")
        results["tasks"].append({"task": f"csv_{pattern}", **result})
    
    # 4. JSON snapshots (prune entries, don't delete file)
    _log("\n[4/5] JSON snapshot pruning...")
    for json_name in ["model_snapshot_log.json", "commercial_forecast_log.json"]:
        json_path = DATA_DIR / json_name
        result = cleanup_json_entries(json_path, SNAPSHOT_RETENTION_DAYS, dry_run)
        _log(f"  {json_name}: {result.get('entries_removed', 0)} entries removed")
        results["tasks"].append({"task": f"json_{json_name}", **result})
    
    # 5. JSONL observability
    _log("\n[5/5] JSONL observability cleanup...")
    result = cleanup_old_files(DATA_DIR, "*.jsonl", CSV_RETENTION_DAYS, dry_run)
    _log(f"  *.jsonl: {result['files_deleted']} files, {result['freed_mb']:.2f} MB freed")
    results["tasks"].append({"task": "jsonl_data", **result})
    
    disk_after = check_disk_space()
    results["disk_after"] = disk_after
    freed = disk_before["used_gb"] - disk_after["used_gb"]
    
    _log("\n" + "=" * 60)
    _log(f"Disk after: {disk_after['used_pct']:.1f}% used ({disk_after['free_gb']:.2f} GB free)")
    _log(f"Total freed: {freed:.2f} GB")
    _log("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Data retention and maintenance")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()
    
    results = run_retention(dry_run=args.dry_run)
    
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    
    # Exit with error if disk still > 80% full
    if results["disk_after"]["used_pct"] > 80:
        _log("WARNING: Disk still > 80% full!")
        sys.exit(1)


if __name__ == "__main__":
    main()
