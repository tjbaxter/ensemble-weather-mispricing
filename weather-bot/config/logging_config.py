"""
Production-grade logging configuration.

Implements:
- Size-based rotation (50MB max, 10 backups)
- Time-based rotation at midnight (30 days retention)  
- JSON structured logging for observability tooling
- Separate log files per component
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import json


class JSONFormatter(logging.Formatter):
    """Structured JSON logging for ELK/Datadog compatibility."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "city"):
            log_data["city"] = record.city
        if hasattr(record, "strategy"):
            log_data["strategy"] = record.strategy
            
        for key in ("extra_data", "context"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        
        return json.dumps(log_data, default=str)


class CompactFormatter(logging.Formatter):
    """Human-readable format for console/simple file output."""
    
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def _get_log_dir() -> Path:
    """Get log directory, defaulting to weather-bot/logs."""
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    if not log_dir.is_absolute():
        log_dir = Path(__file__).parent.parent / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def configure_logging(
    component: str = "weather-bot",
    level: int = logging.INFO,
    json_format: bool = True,
    console: bool = True,
) -> logging.Logger:
    """
    Configure production logging for a component.
    
    Args:
        component: Name of component (bot, dashboard, settlement_watcher, etc.)
        level: Logging level
        json_format: Use JSON structured logging
        console: Also log to console
    
    Returns:
        Configured logger
    """
    log_dir = _get_log_dir()
    logger = logging.getLogger(component)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = JSONFormatter() if json_format else CompactFormatter()
    
    # Size-based rotation: 50MB max, 10 backups
    size_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"{component}.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10,
        encoding="utf-8",
    )
    size_handler.setFormatter(formatter)
    size_handler.setLevel(level)
    logger.addHandler(size_handler)
    
    # Time-based rotation at midnight, 30 days retention
    time_handler = logging.handlers.TimedRotatingFileHandler(
        log_dir / f"{component}_daily.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
        utc=True,
    )
    time_handler.setFormatter(formatter)
    time_handler.setLevel(level)
    logger.addHandler(time_handler)
    
    # Error-only log for quick triage
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"{component}_errors.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CompactFormatter())
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger with trace ID support."""
    return logging.getLogger(f"weather-bot.{name}")


def with_trace(logger: logging.Logger, trace_id: str):
    """Create a logger adapter with trace ID context."""
    return logging.LoggerAdapter(logger, {"trace_id": trace_id})
