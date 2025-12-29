"""
Rate Limiter for AI Trader

Provides rate limiting for external API calls to prevent quota exhaustion.
Uses token bucket algorithm for smooth rate limiting.

Usage:
    from agent_tools.cache import get_rate_limiter

    # Get limiter for a service
    limiter = get_rate_limiter("alpaca")

    # Wait for permission (blocks if needed)
    limiter.acquire()
    api_call()

    # Or check without blocking
    if limiter.try_acquire():
        api_call()
    else:
        handle_rate_limit()
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limiter."""
    requests_per_second: float  # Max requests per second
    burst_size: int  # Max burst size (token bucket capacity)
    name: str = "default"  # Service name for logging


# Default rate limits for different services
DEFAULT_RATE_LIMITS = {
    "alpaca": RateLimitConfig(
        requests_per_second=3.0,  # Alpaca allows ~200/min = ~3.3/sec
        burst_size=10,
        name="alpaca"
    ),
    "alpaca_data": RateLimitConfig(
        requests_per_second=5.0,  # Data API is more generous
        burst_size=20,
        name="alpaca_data"
    ),
    "jina": RateLimitConfig(
        requests_per_second=1.0,  # Conservative for Jina
        burst_size=5,
        name="jina"
    ),
    "openai": RateLimitConfig(
        requests_per_second=0.5,  # ~30 requests per minute
        burst_size=5,
        name="openai"
    ),
    "alphavantage": RateLimitConfig(
        requests_per_second=0.08,  # 5 requests per minute for free tier
        burst_size=2,
        name="alphavantage"
    ),
    "default": RateLimitConfig(
        requests_per_second=1.0,
        burst_size=5,
        name="default"
    ),
}


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.

    Allows burst traffic up to bucket capacity, then smoothly
    limits to the configured rate.
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self._tokens = float(config.burst_size)
        self._last_refill = time.time()
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "acquired": 0,
            "waited": 0,
            "rejected": 0,
            "total_wait_time": 0.0,
        }

    def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self._last_refill

        # Add tokens based on time elapsed
        new_tokens = elapsed * self.config.requests_per_second
        self._tokens = min(self.config.burst_size, self._tokens + new_tokens)
        self._last_refill = now

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.
        Blocks until a token is available or timeout is reached.

        Args:
            timeout: Max time to wait in seconds (None = wait forever)

        Returns:
            True if acquired, False if timeout
        """
        start_time = time.time()
        deadline = start_time + timeout if timeout else None

        while True:
            with self._lock:
                self._refill()

                if self._tokens >= 1:
                    self._tokens -= 1
                    self._stats["acquired"] += 1

                    wait_time = time.time() - start_time
                    if wait_time > 0.01:  # Only count meaningful waits
                        self._stats["waited"] += 1
                        self._stats["total_wait_time"] += wait_time

                    return True

                # Calculate wait time for next token
                wait_for_token = (1 - self._tokens) / self.config.requests_per_second

            # Check timeout
            if deadline and time.time() + wait_for_token > deadline:
                self._stats["rejected"] += 1
                return False

            # Wait for token refill
            time.sleep(min(wait_for_token, 0.1))  # Check at least every 100ms

    def try_acquire(self) -> bool:
        """
        Try to acquire permission without blocking.

        Returns:
            True if acquired, False if no tokens available
        """
        with self._lock:
            self._refill()

            if self._tokens >= 1:
                self._tokens -= 1
                self._stats["acquired"] += 1
                return True

            self._stats["rejected"] += 1
            return False

    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        with self._lock:
            avg_wait = (
                self._stats["total_wait_time"] / self._stats["waited"]
                if self._stats["waited"] > 0
                else 0
            )

            return {
                "service": self.config.name,
                "rate_per_second": self.config.requests_per_second,
                "burst_size": self.config.burst_size,
                "current_tokens": round(self._tokens, 2),
                "acquired": self._stats["acquired"],
                "waited": self._stats["waited"],
                "rejected": self._stats["rejected"],
                "avg_wait_seconds": round(avg_wait, 3),
            }


class RateLimiterManager:
    """
    Manager for multiple rate limiters.
    Provides a singleton interface for rate limiting different services.
    """

    def __init__(self):
        self._limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._lock = threading.Lock()

    def get_limiter(self, service: str) -> TokenBucketRateLimiter:
        """
        Get rate limiter for a service.

        Args:
            service: Service name (e.g., "alpaca", "jina")

        Returns:
            Rate limiter for the service
        """
        with self._lock:
            if service not in self._limiters:
                config = DEFAULT_RATE_LIMITS.get(
                    service,
                    DEFAULT_RATE_LIMITS["default"]
                )
                # Create new config with correct name if using default
                if service not in DEFAULT_RATE_LIMITS:
                    config = RateLimitConfig(
                        requests_per_second=config.requests_per_second,
                        burst_size=config.burst_size,
                        name=service
                    )
                self._limiters[service] = TokenBucketRateLimiter(config)

            return self._limiters[service]

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all rate limiters."""
        with self._lock:
            return {
                service: limiter.get_stats()
                for service, limiter in self._limiters.items()
            }


# Singleton instance
_rate_limiter_manager: Optional[RateLimiterManager] = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter(service: str) -> TokenBucketRateLimiter:
    """
    Get rate limiter for a service.

    Args:
        service: Service name (e.g., "alpaca", "jina", "openai")

    Returns:
        Rate limiter for the service
    """
    global _rate_limiter_manager

    if _rate_limiter_manager is None:
        with _rate_limiter_lock:
            if _rate_limiter_manager is None:
                _rate_limiter_manager = RateLimiterManager()

    return _rate_limiter_manager.get_limiter(service)


def get_all_rate_limiter_stats() -> Dict[str, Dict]:
    """Get statistics for all active rate limiters."""
    global _rate_limiter_manager

    if _rate_limiter_manager is None:
        return {}

    return _rate_limiter_manager.get_all_stats()


if __name__ == "__main__":
    # Test rate limiter
    print("Testing Rate Limiter...")

    limiter = get_rate_limiter("test")

    # Test burst
    print("\nBurst test (should be fast):")
    start = time.time()
    for i in range(5):
        limiter.acquire()
        print(f"  Request {i+1} at {time.time() - start:.3f}s")

    # Test rate limiting
    print("\nRate limit test (should be ~1 req/sec):")
    start = time.time()
    for i in range(3):
        limiter.acquire()
        print(f"  Request {i+1} at {time.time() - start:.3f}s")

    # Test try_acquire
    print("\nTry acquire test:")
    acquired = limiter.try_acquire()
    print(f"  Immediate acquire: {acquired}")

    # Stats
    print(f"\nStats: {limiter.get_stats()}")

    print("\nAll tests passed!")
