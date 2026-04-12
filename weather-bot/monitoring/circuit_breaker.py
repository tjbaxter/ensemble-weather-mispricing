"""
Circuit breaker pattern for resilient external service calls.

States:
- CLOSED: Normal operation, requests flow through
- OPEN: Service is down, fail fast without calling
- HALF_OPEN: Testing if service recovered

Usage:
    breaker = CircuitBreaker("polymarket_api")
    
    async def call_api():
        if not breaker.can_execute():
            raise CircuitOpenError("Polymarket API circuit is open")
        try:
            result = await polymarket_client.get_markets()
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            raise
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Callable, Any
from functools import wraps


logger = logging.getLogger("weather-bot.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when circuit is open and request cannot proceed."""
    pass


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for external service calls.
    
    Args:
        name: Identifier for this circuit
        failure_threshold: Number of failures before opening
        success_threshold: Successes needed to close from half-open
        reset_timeout: Seconds before trying half-open
        half_open_max_calls: Max calls allowed in half-open state
    """
    name: str
    failure_threshold: int = 5
    success_threshold: int = 2
    reset_timeout: float = 30.0
    half_open_max_calls: int = 3
    
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    
    @property
    def state(self) -> CircuitState:
        """Get current state, potentially transitioning from OPEN to HALF_OPEN."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_try_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
            return self._state
    
    def _should_try_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self._last_failure_time is None:
            return True
        return (time.monotonic() - self._last_failure_time) >= self.reset_timeout
    
    def can_execute(self) -> bool:
        """Check if a request can proceed."""
        current_state = self.state
        
        if current_state == CircuitState.CLOSED:
            return True
        
        if current_state == CircuitState.OPEN:
            return False
        
        # HALF_OPEN: allow limited calls
        with self._lock:
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
    
    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(f"Circuit '{self.name}' CLOSED after successful recovery")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning(f"Circuit '{self.name}' OPEN again after half-open failure")
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit '{self.name}' OPENED after {self._failure_count} failures"
                    )
    
    def reset(self) -> None:
        """Manually reset the circuit to CLOSED."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info(f"Circuit '{self.name}' manually RESET")
    
    def get_status(self) -> dict[str, Any]:
        """Get circuit status for monitoring."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": (
                datetime.fromtimestamp(self._last_failure_time, tz=timezone.utc).isoformat()
                if self._last_failure_time else None
            ),
        }


def circuit_protected(breaker: CircuitBreaker, fallback: Optional[Callable] = None):
    """
    Decorator to protect a function with a circuit breaker.
    
    Args:
        breaker: CircuitBreaker instance
        fallback: Optional fallback function to call when circuit is open
    
    Usage:
        polymarket_breaker = CircuitBreaker("polymarket")
        
        @circuit_protected(polymarket_breaker)
        async def get_markets():
            return await client.get_markets()
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not breaker.can_execute():
                if fallback:
                    return fallback(*args, **kwargs)
                raise CircuitOpenError(f"Circuit '{breaker.name}' is OPEN")
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not breaker.can_execute():
                if fallback:
                    return fallback(*args, **kwargs)
                raise CircuitOpenError(f"Circuit '{breaker.name}' is OPEN")
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Global circuit breakers for main services
_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create a named circuit breaker."""
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _breakers[name]


def get_all_circuit_status() -> list[dict[str, Any]]:
    """Get status of all circuit breakers for monitoring."""
    return [breaker.get_status() for breaker in _breakers.values()]
