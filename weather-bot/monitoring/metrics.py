"""
Production metrics collection for observability.

Implements Prometheus-style metrics for:
- Trade execution tracking
- API latency monitoring
- System health checks
- Model performance metrics
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager


@dataclass
class Counter:
    """Prometheus-style counter metric."""
    name: str
    help: str
    value: int = 0
    labels: dict[str, str] = field(default_factory=dict)
    
    def inc(self, amount: int = 1) -> None:
        self.value += amount
    
    def to_prometheus(self) -> str:
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        label_part = f"{{{labels_str}}}" if labels_str else ""
        return f"{self.name}{label_part} {self.value}"


@dataclass 
class Gauge:
    """Prometheus-style gauge metric."""
    name: str
    help: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    
    def set(self, value: float) -> None:
        self.value = value
    
    def inc(self, amount: float = 1.0) -> None:
        self.value += amount
    
    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount
    
    def to_prometheus(self) -> str:
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        label_part = f"{{{labels_str}}}" if labels_str else ""
        return f"{self.name}{label_part} {self.value}"


@dataclass
class Histogram:
    """Prometheus-style histogram for latency tracking."""
    name: str
    help: str
    buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    _observations: list[float] = field(default_factory=list)
    _sum: float = 0.0
    _count: int = 0
    
    def observe(self, value: float) -> None:
        self._observations.append(value)
        self._sum += value
        self._count += 1
    
    @contextmanager
    def time(self):
        """Context manager to time operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - start)
    
    def to_prometheus(self) -> str:
        lines = []
        for bucket in self.buckets:
            count = sum(1 for v in self._observations if v <= bucket)
            lines.append(f'{self.name}_bucket{{le="{bucket}"}} {count}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._count}')
        lines.append(f"{self.name}_sum {self._sum}")
        lines.append(f"{self.name}_count {self._count}")
        return "\n".join(lines)


class TradingMetrics:
    """Centralized metrics collection for the trading system."""
    
    def __init__(self):
        # Counters
        self.trades_placed = Counter("weather_bot_trades_placed_total", "Total trades placed")
        self.trades_won = Counter("weather_bot_trades_won_total", "Total winning trades")
        self.trades_lost = Counter("weather_bot_trades_lost_total", "Total losing trades")
        self.api_requests = Counter("weather_bot_api_requests_total", "Total API requests")
        self.api_errors = Counter("weather_bot_api_errors_total", "Total API errors")
        
        # Gauges
        self.open_positions = Gauge("weather_bot_open_positions", "Current open positions")
        self.total_pnl = Gauge("weather_bot_total_pnl_usd", "Total PnL in USD")
        self.disk_usage_pct = Gauge("weather_bot_disk_usage_percent", "Disk usage percentage")
        self.memory_usage_mb = Gauge("weather_bot_memory_usage_mb", "Memory usage in MB")
        
        # Histograms
        self.api_latency = Histogram("weather_bot_api_latency_seconds", "API request latency")
        self.model_inference_time = Histogram(
            "weather_bot_model_inference_seconds", 
            "Model inference time",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
        )
        
        # Strategy-specific counters
        self._strategy_trades: dict[str, Counter] = {}
        self._strategy_wins: dict[str, Counter] = {}
        
        self._last_heartbeat: Optional[datetime] = None
    
    def heartbeat(self) -> None:
        """Record a heartbeat."""
        self._last_heartbeat = datetime.now(timezone.utc)
    
    def record_trade(self, strategy: str, won: bool, pnl: float) -> None:
        """Record a trade execution."""
        self.trades_placed.inc()
        if won:
            self.trades_won.inc()
        else:
            self.trades_lost.inc()
        self.total_pnl.set(self.total_pnl.value + pnl)
        
        # Strategy-specific tracking
        if strategy not in self._strategy_trades:
            self._strategy_trades[strategy] = Counter(
                f"weather_bot_strategy_trades_total",
                f"Trades for {strategy}",
                labels={"strategy": strategy}
            )
            self._strategy_wins[strategy] = Counter(
                f"weather_bot_strategy_wins_total",
                f"Wins for {strategy}",
                labels={"strategy": strategy}
            )
        self._strategy_trades[strategy].inc()
        if won:
            self._strategy_wins[strategy].inc()
    
    def record_api_call(self, endpoint: str, latency: float, error: bool = False) -> None:
        """Record an API call."""
        self.api_requests.inc()
        self.api_latency.observe(latency)
        if error:
            self.api_errors.inc()
    
    def update_system_metrics(self) -> None:
        """Update system resource metrics."""
        import shutil
        
        # Disk usage
        total, used, _ = shutil.disk_usage("/")
        self.disk_usage_pct.set((used / total) * 100)
        
        # Memory usage (if psutil available)
        try:
            import psutil
            process = psutil.Process()
            self.memory_usage_mb.set(process.memory_info().rss / (1024**2))
        except ImportError:
            pass
    
    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus exposition format."""
        lines = []
        
        for metric in [
            self.trades_placed, self.trades_won, self.trades_lost,
            self.api_requests, self.api_errors,
            self.open_positions, self.total_pnl,
            self.disk_usage_pct, self.memory_usage_mb,
        ]:
            lines.append(f"# HELP {metric.name} {metric.help}")
            lines.append(f"# TYPE {metric.name} {'counter' if isinstance(metric, Counter) else 'gauge'}")
            lines.append(metric.to_prometheus())
            lines.append("")
        
        # Histograms
        for hist in [self.api_latency, self.model_inference_time]:
            lines.append(f"# HELP {hist.name} {hist.help}")
            lines.append(f"# TYPE {hist.name} histogram")
            lines.append(hist.to_prometheus())
            lines.append("")
        
        # Strategy-specific
        for counter in list(self._strategy_trades.values()) + list(self._strategy_wins.values()):
            lines.append(counter.to_prometheus())
        
        # Heartbeat
        if self._last_heartbeat:
            lines.append(f"weather_bot_last_heartbeat_timestamp {self._last_heartbeat.timestamp()}")
        
        return "\n".join(lines)
    
    def to_json(self) -> dict[str, Any]:
        """Export metrics as JSON for dashboard consumption."""
        self.update_system_metrics()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trades": {
                "placed": self.trades_placed.value,
                "won": self.trades_won.value,
                "lost": self.trades_lost.value,
                "win_rate": self.trades_won.value / max(1, self.trades_placed.value),
            },
            "pnl_usd": self.total_pnl.value,
            "positions": {
                "open": int(self.open_positions.value),
            },
            "api": {
                "requests": self.api_requests.value,
                "errors": self.api_errors.value,
                "avg_latency_ms": (self.api_latency._sum / max(1, self.api_latency._count)) * 1000,
            },
            "system": {
                "disk_usage_pct": self.disk_usage_pct.value,
                "memory_mb": self.memory_usage_mb.value,
            },
            "strategies": {
                name: {
                    "trades": self._strategy_trades[name].value,
                    "wins": self._strategy_wins[name].value,
                }
                for name in self._strategy_trades
            },
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
        }


# Global metrics instance
_metrics: Optional[TradingMetrics] = None


def get_metrics() -> TradingMetrics:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = TradingMetrics()
    return _metrics
